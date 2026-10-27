"""十条消息窗口的接收、调度及发送预留回归，不启动应用或模型。"""

from contextlib import suppress
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import asyncio
import sys

from pytests.maisaka.test_dynamic_reply_gate import ReplyCreditGate
from pytests.maisaka.test_hourly_fallback_release import HourlyFallbackController


ROOT = Path(__file__).resolve().parents[2]
TIMING = SimpleNamespace(inevitable_at_reply=True, mentioned_bot_reply=False)
CONFIG = SimpleNamespace(chat=SimpleNamespace(reply_timing=TIMING), experimental=SimpleNamespace(planner_message_steering=False))


class TextComponent:
    pass

class AtComponent:
    def __init__(self, target_user_id):
        self.target_user_id = target_user_id

class ReplyComponent:
    def __init__(self, target_message_id="source", target_message_sender_id=None):
        self.target_message_id = target_message_id
        self.target_message_sender_id = target_message_sender_id


def _load_class(path: str, name: str, namespace: dict, methods: set[str] | None = None):
    """直接编译生产类；仅排除会启动整个插件/配置系统的模块级导入。"""

    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = []
    if methods is not None:
        node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in methods]
    module = ModuleType(f"dynamic_reply_regression_{name}")
    sys.modules[module.__name__] = module
    module.__dict__.update(namespace)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), module.__dict__)
    return module.__dict__[name]


TurnGateResult = _load_class("src/maisaka/turn_trigger/gates.py", "TurnGateResult", {"dataclass": dataclass})
CLOCK = {"now": 100.0}
PROBABILITY = {"value": 0.2}
ONLINE_ACCOUNTS = {"qq": {"bot"}}
Gate = _load_class("src/maisaka/turn_trigger/gates.py", "DynamicReplyTurnGate", {
    "ReplyCreditGate": ReplyCreditGate,
    "global_config": CONFIG,
    "AtComponent": AtComponent, "ReplyComponent": ReplyComponent,
    "TurnGateResult": TurnGateResult,
    "is_bot_self": lambda platform, user_id: user_id == "bot",
    "get_online_bot_accounts": lambda platform: ONLINE_ACCOUNTS.get(platform, set()),
    "time": SimpleNamespace(time=lambda: CLOCK["now"], monotonic=lambda: CLOCK["now"]),
    "SOFT_GATE_MAX_DELAY_SECONDS": 5.0,
    "estimate_reply_probability": lambda value: PROBABILITY["value"],
    "MIN_REPLY_PROBABILITY": 0.15,
    "LAST_REPLY_SLOT_MIN_PROBABILITY": 0.60,
})
Scheduler = _load_class("src/maisaka/turn_trigger/scheduler.py", "MessageTurnScheduler", {
    "DynamicReplyTurnGate": lambda runtime: Mock(),
    "HourlyFallbackController": HourlyFallbackController,
    "HOURLY_FALLBACK_SECONDS": 3600.0,
    "time": SimpleNamespace(time=lambda: CLOCK["now"], monotonic=lambda: CLOCK["now"]),
    "global_config": CONFIG,
    "is_bot_self": lambda platform, user_id: user_id == "bot",
    "is_bare_ai_label": lambda text: text.strip().lower() == "ai",
    "TextComponent": TextComponent,
    "logger": SimpleNamespace(info=lambda *args: None, debug=lambda *args: None),
})
Engine = _load_class("src/maisaka/reasoning_engine.py", "MaisakaReasoningEngine", {
    "global_config": CONFIG,
    "deque": deque,
}, {"__init__", "_begin_reply_accounting", "_consume_ingested_forced_turn_reason", "_start_turn_reply_accounting", "_ingest_messages"})
PlannerReplyReservation = _load_class("src/maisaka/runtime.py", "PlannerReplyReservation", {"dataclass": dataclass})
Runtime = _load_class("src/maisaka/runtime.py", "MaisakaHeartFlowChatting", {
    "asyncio": asyncio, "PlannerReplyReservation": PlannerReplyReservation,
}, {
    "_cancel_deferred_message_turn_task", "_defer_message_turn_check", "_schedule_deferred_message_turn",
    "_uses_reply_window_quota", "reserve_planner_reply", "finish_planner_reply",
    "_arm_forced_turn_state",
})


def _message(message_id: str, *, at: bool = False, user_id: str = "user", notify=False, command=False):
    return SimpleNamespace(message_id=message_id, received_at=None, is_at=at, is_mentioned=at,
                           is_notify=notify, is_command=command, platform="qq",
                           processed_plain_text="ordinary message", raw_message=SimpleNamespace(components=[TextComponent()]),
                           message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user_id)))


def _scheduler_runtime(messages, *, backoff_seconds: float = 0.0, forced: bool = False,
                       group: bool = True, frequency: float = 0.4, quiet_seconds: float = 0.0,
                       silent: bool = False):
    return SimpleNamespace(
        _running=True, chat_stream=SimpleNamespace(is_group_session=group),
        _agent_state="running", _STATE_WAIT="wait", _message_turn_scheduled=False,
        _get_pending_message_count=Mock(return_value=len(messages)), _get_effective_reply_frequency=lambda: frequency,
        _is_reply_frequency_silent=lambda: silent, _has_forced_turn_trigger=lambda: forced,
        _get_message_quiet_period_remaining_seconds=Mock(return_value=quiet_seconds),
        _fixed_message_debouncer=SimpleNamespace(consume=Mock()),
        _has_pending_messages=lambda: bool(messages),
        _idle_backoff=SimpleNamespace(get_delay_seconds=Mock(return_value=backoff_seconds)),
        _get_message_trigger_threshold=lambda: 3,
        _get_recent_average_external_message_interval=lambda: 30.0,
        _last_external_message_received_at=100.0, _last_message_received_at=100.0,
        _cancel_wait_timeout_task=Mock(), _clear_message_debounce_required=Mock(), _enter_running_state=Mock(),
        _last_processed_index=0, message_cache=list(messages), log_prefix="test",
        _uses_reply_window_quota=lambda: group, _forced_reply_message_ids=deque(),
        _defer_message_turn_check=Mock(), _cancel_deferred_message_turn_task=Mock(), _enqueue_message_turn=Mock(),
        _mark_message_turn_unscheduled=Mock(),
    )


def _reservation_runtime():
    runtime = Runtime()
    runtime.chat_stream = SimpleNamespace(is_group_session=True)
    runtime._is_focus_mode_active_for_current_chat = lambda: False
    runtime._is_reply_frequency_silent = lambda: False
    runtime._forced_turn_enabled = False
    runtime._forced_reply_message_ids = deque(["at-msg"], maxlen=200)
    runtime._reserved_forced_reply_message_ids = set()
    runtime._consumed_forced_reply_message_ids = deque(maxlen=200)
    runtime._message_turn_scheduler = Mock()
    runtime._idle_backoff = Mock()
    runtime._message_turn_scheduler.hourly_turn_generation = None
    runtime._message_turn_scheduler.hourly = SimpleNamespace(active=False)
    runtime._message_turn_scheduler.reserve_reply.return_value = None
    return runtime


class TestDynamicReplyIntegration(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        CLOCK["now"] = 100.0
        PROBABILITY["value"] = 0.2
        ONLINE_ACCOUNTS.clear()
        ONLINE_ACCOUNTS["qq"] = {"bot"}

    async def test_queued_mention_signal_is_kept_until_its_message_is_ingested(self) -> None:
        runtime = SimpleNamespace(_forced_turn_message_id="new-at", _consume_forced_turn_reason=Mock(return_value="@"))
        engine = Engine(runtime)
        old = _message("old")
        new = _message("new-at", at=True)
        context = SimpleNamespace(logical_turn_id="current", cached_messages=[old], trigger_message=old, proactive_triggered=False)
        self.assertIsNone(engine._start_turn_reply_accounting(context))
        engine._build_history_message = AsyncMock(return_value=None)
        await engine._ingest_messages([old])
        runtime._consume_forced_turn_reason.assert_not_called()
        await engine._ingest_messages([new])
        runtime._consume_forced_turn_reason.assert_called_once()

    def test_turn_start_consumes_the_signal_of_an_ingested_mention(self) -> None:
        runtime = SimpleNamespace(_forced_turn_message_id="first-at", _consume_forced_turn_reason=Mock(return_value="@"))
        engine = Engine(runtime)
        batch = [_message("first-at", at=True)]
        context = SimpleNamespace(logical_turn_id="first", cached_messages=batch, trigger_message=batch[0], proactive_triggered=False)
        self.assertEqual(engine._start_turn_reply_accounting(context), "@")

    def test_group_gate_waits_for_more_messages_and_charges_only_real_replies(self) -> None:
        gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True)))
        gate._build_likelihood_input = lambda *args, **kwargs: None
        pending = [_message("one"), _message("two")]
        gate.observe_messages(pending, 0.4)
        self.assertEqual(gate.evaluate(pending_messages=pending, frequency=0.4).decision, "wait")
        CLOCK["now"] += 1000.0
        self.assertEqual(gate.evaluate(pending_messages=pending, frequency=0.4).decision, "wait")
        third = _message("three")
        gate.observe_messages([third], 0.4)
        self.assertTrue(gate.evaluate(pending_messages=[*pending, third], frequency=0.4).should_trigger)
        reservation = gate.reserve_reply("three")
        self.assertIsNotNone(reservation)
        gate.finish_reply(reservation, success=True)
        self.assertAlmostEqual(gate._gate.credit, 0.2)

    def test_group_gate_lets_a_conversation_with_the_bot_through_during_backoff(self) -> None:
        gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True)))
        gate._build_likelihood_input = lambda *args, **kwargs: None
        pending = [_message("a"), _message("b"), _message("c")]
        gate.observe_messages(pending, 0.4)
        blocked = gate.evaluate(pending_messages=pending, frequency=0.4, idle_backoff_seconds=30.0)
        self.assertEqual(blocked.decision, "delay")
        self.assertAlmostEqual(blocked.delay_seconds, 30.0)
        PROBABILITY["value"] = 0.99
        self.assertEqual(gate.evaluate(pending_messages=pending, frequency=0.4, idle_backoff_seconds=30.0).decision, "delay")
        pending[-1].is_mentioned = True
        self.assertTrue(gate.evaluate(pending_messages=pending, frequency=0.4, idle_backoff_seconds=30.0).should_trigger)

    def test_priority_requires_actual_bot_address_or_reply_target(self) -> None:
        runtime = SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True),
                                  find_source_message_by_id=Mock(return_value=_message("bot-source", user_id="bot")))
        gate = Gate(runtime)
        ordinary = _message("ordinary")
        self.assertFalse(gate._is_addressed_to_bot(ordinary))
        self.assertTrue(gate._is_addressed_to_bot(_message("at", at=True)))
        named = _message("nickname")
        named.is_mentioned = True
        self.assertTrue(gate._is_addressed_to_bot(named))
        for component in (AtComponent("bot"), ReplyComponent(target_message_sender_id="bot"), ReplyComponent()):
            ordinary.raw_message.components = [component]
            self.assertTrue(gate._is_addressed_to_bot(ordinary))
        for component in (AtComponent("someone-else"), ReplyComponent(target_message_sender_id="someone-else")):
            ordinary.raw_message.components = [component]
            self.assertFalse(gate._is_addressed_to_bot(ordinary))

    def test_offline_bot_history_does_not_filter_messages_until_reconnect(self) -> None:
        gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True)))
        former_bot = _message("former-bot", user_id="bot")
        ONLINE_ACCOUNTS["qq"] = {"current-bot"}
        self.assertEqual(gate._filter_external_messages([
            former_bot, _message("notice", notify=True), _message("command", command=True),
            _message("self", user_id="current-bot"),
        ]), [former_bot])
        ONLINE_ACCOUNTS["qq"].add("bot")
        self.assertEqual(gate._filter_external_messages([former_bot]), [])

    def test_gate_ignores_bot_messages_and_private_chats(self) -> None:
        group_gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True)))
        group_gate._build_likelihood_input = lambda *args, **kwargs: None
        self.assertEqual(group_gate.evaluate(pending_messages=[_message("self", user_id="bot")], frequency=0.4).decision, "wait")
        self.assertEqual(group_gate.evaluate(pending_messages=[_message("one")], frequency=0).decision, "wait")
        private_gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=False)))
        for index in range(3):
            self.assertTrue(private_gate.evaluate(pending_messages=[_message(f"p{index}")], frequency=0.1).should_trigger)

    def test_scheduler_rechecks_only_while_messages_are_pending(self) -> None:
        runtime = _scheduler_runtime([_message("one")])
        scheduler = Scheduler(runtime)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("delay", "pending", 45.0)
        scheduler.schedule_message_turn()
        runtime._defer_message_turn_check.assert_called_once_with(45.0)
        runtime._enqueue_message_turn.assert_not_called()
        runtime._get_pending_message_count.return_value = 0
        scheduler.schedule_message_turn()
        runtime._cancel_deferred_message_turn_task.assert_called_once()
        scheduler._dynamic_reply_gate.evaluate.assert_called_once()

    def test_scheduler_hands_idle_backoff_to_the_dynamic_gate(self) -> None:
        runtime = _scheduler_runtime([_message("one")], backoff_seconds=30.0)
        scheduler = Scheduler(runtime)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("trigger", "priority")
        scheduler.schedule_message_turn()
        self.assertEqual(scheduler._dynamic_reply_gate.evaluate.call_args.kwargs["idle_backoff_seconds"], 30.0)
        runtime._enqueue_message_turn.assert_called_once()
        runtime._defer_message_turn_check.assert_not_called()

    def test_forced_turn_skips_the_gate(self) -> None:
        runtime = _scheduler_runtime([_message("at", at=True)], forced=True)
        scheduler = Scheduler(runtime)
        scheduler.schedule_message_turn()
        runtime._enqueue_message_turn.assert_called_once()
        scheduler._dynamic_reply_gate.evaluate.assert_not_called()

    def test_forced_reply_has_only_one_successful_exemption(self) -> None:
        runtime = _reservation_runtime()
        reservation = runtime.reserve_planner_reply("at-msg")
        self.assertTrue(reservation.forced)
        runtime._message_turn_scheduler.reserve_reply.assert_not_called()
        self.assertIsNone(runtime.reserve_planner_reply("at-msg"))
        runtime.finish_planner_reply(reservation, success=True)
        runtime._arm_forced_turn_state(message_id="at-msg", reason="duplicate delivery")
        self.assertIsNone(runtime.reserve_planner_reply("at-msg"))
        self.assertEqual(list(runtime._consumed_forced_reply_message_ids), ["at-msg"])

    def test_failed_forced_send_returns_its_exemption(self) -> None:
        runtime = _reservation_runtime()
        first = runtime.reserve_planner_reply("at-msg")
        runtime.finish_planner_reply(first, success=False)
        second = runtime.reserve_planner_reply("at-msg")
        self.assertTrue(second.forced)
        runtime.finish_planner_reply(second, success=True)

    def test_ordinary_reply_reservation_is_finished_with_actual_send_outcome(self) -> None:
        runtime = _reservation_runtime()
        token = object()
        runtime._message_turn_scheduler.reserve_reply.return_value = token
        reservation = runtime.reserve_planner_reply("ordinary")
        runtime._message_turn_scheduler.reserve_reply.assert_called_once_with("ordinary")
        runtime.finish_planner_reply(reservation, success=False)
        runtime._message_turn_scheduler.finish_reply.assert_called_once_with(token, success=False)

    def test_quota_is_not_applied_to_private_chat_but_dynamic_zero_is_silent(self) -> None:
        runtime = _reservation_runtime()
        runtime.chat_stream.is_group_session = False
        self.assertIsNotNone(runtime.reserve_planner_reply("private"))
        runtime._message_turn_scheduler.reserve_reply.assert_not_called()
        runtime.chat_stream.is_group_session = True
        runtime._is_reply_frequency_silent = lambda: True
        self.assertIsNone(runtime.reserve_planner_reply("ordinary"))

    def test_hourly_admission_uses_one_separate_reservation_and_no_normal_credit(self) -> None:
        runtime = _reservation_runtime()
        hourly = HourlyFallbackController(quiet_seconds=0)
        token = hourly.begin_turn(["hourly-message"])
        runtime._message_turn_scheduler.hourly = hourly
        runtime._message_turn_scheduler.hourly_turn_generation = token

        reservation = runtime.reserve_planner_reply("hourly-message")
        self.assertEqual(reservation.hourly_generation, token)
        runtime._message_turn_scheduler.reserve_reply.assert_not_called()
        self.assertIsNone(runtime.reserve_planner_reply("hourly-message"))
        runtime.finish_planner_reply(reservation, success=True)
        self.assertTrue(hourly.reply_succeeded)
        runtime._message_turn_scheduler.finish_reply.assert_not_called()

    def test_invalidated_hourly_turn_never_falls_through_to_normal_reservation(self) -> None:
        runtime = _reservation_runtime()
        hourly = HourlyFallbackController(quiet_seconds=0)
        token = hourly.begin_turn(["old-message"])
        runtime._message_turn_scheduler.hourly = hourly
        runtime._message_turn_scheduler.hourly_turn_generation = token
        hourly.observe_message()

        self.assertIsNone(runtime.reserve_planner_reply("old-message"))
        runtime._message_turn_scheduler.reserve_reply.assert_not_called()

    def test_hourly_planner_cannot_select_an_old_at_target_to_bypass_draft_delay(self) -> None:
        runtime = _reservation_runtime()
        hourly = HourlyFallbackController(quiet_seconds=0)
        token = hourly.begin_turn(["latest-hourly-message"])
        runtime._message_turn_scheduler.hourly = hourly
        runtime._message_turn_scheduler.hourly_turn_generation = token

        self.assertIsNone(runtime.reserve_planner_reply("at-msg"))
        self.assertEqual(list(runtime._forced_reply_message_ids), ["at-msg"])
        self.assertFalse(runtime._reserved_forced_reply_message_ids)
        runtime._message_turn_scheduler.reserve_reply.assert_not_called()
        current = runtime.reserve_planner_reply("latest-hourly-message")
        self.assertEqual(current.hourly_generation, token)
        self.assertFalse(current.forced)

    def test_receipt_counts_ordinary_messages_in_a_mixed_batch_even_during_planner(self) -> None:
        runtime = _scheduler_runtime([])
        runtime._forced_reply_message_ids.append("at")
        scheduler = Scheduler(runtime)
        for message in (_message("one"), _message("at", at=True), _message("two"),
                        _message("notice", notify=True), _message("command", command=True),
                        _message("self", user_id="bot")):
            scheduler.observe_message(message)
        observed = scheduler._dynamic_reply_gate.observe_messages.call_args_list
        self.assertEqual([call.args[0][0].message_id for call in observed], ["one", "two"])
        self.assertTrue(all(call.args[1] == 0.4 for call in observed))

    def test_closed_window_deficit_does_not_schedule_without_new_messages(self) -> None:
        runtime = _scheduler_runtime([], backoff_seconds=999)
        scheduler = Scheduler(runtime)
        gate = Gate(SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True)))
        gate.observe_messages([_message(str(index)) for index in range(10)], 0.4)
        self.assertAlmostEqual(gate._gate.credit, 4.0)
        gate.evaluate = Mock(wraps=gate.evaluate)
        scheduler._dynamic_reply_gate = gate
        scheduler.schedule_message_turn()
        runtime._enqueue_message_turn.assert_not_called()
        runtime._cancel_deferred_message_turn_task.assert_called_once()
        gate.evaluate.assert_not_called()
        runtime._idle_backoff.get_delay_seconds.assert_not_called()

    def test_bare_ai_label_does_not_create_reply_credit_but_topics_and_media_do(self) -> None:
        runtime = _scheduler_runtime([])
        scheduler = Scheduler(runtime)
        bare = _message("bare")
        bare.processed_plain_text = "AI"
        topic = _message("topic")
        topic.processed_plain_text = "AI 最近有什么新进展？"
        caption = _message("caption")
        caption.processed_plain_text = "AI"
        caption.raw_message.components.append(object())
        mentioned = _message("mentioned", at=True)
        mentioned.processed_plain_text = "AI"
        for message in (bare, topic, caption, mentioned):
            scheduler.observe_message(message)
        calls = scheduler._dynamic_reply_gate.observe_messages.call_args_list
        self.assertEqual([call.args[0][0].message_id for call in calls], ["topic", "caption", "mentioned"])

    async def test_finished_recheck_does_not_clear_its_replacement(self) -> None:
        runtime = Runtime()
        runtime._running = True
        runtime._deferred_message_turn_task = None
        runtime._schedule_message_turn = lambda: runtime._defer_message_turn_check(60)
        first = asyncio.create_task(runtime._schedule_deferred_message_turn(0))
        runtime._deferred_message_turn_task = first
        await first
        replacement = runtime._deferred_message_turn_task
        self.assertIsNotNone(replacement)
        self.assertIsNot(first, replacement)
        self.assertFalse(replacement.done())
        runtime._cancel_deferred_message_turn_task()
        with suppress(asyncio.CancelledError):
            await replacement

    async def test_cancelled_old_recheck_does_not_clear_a_new_handle(self) -> None:
        runtime = Runtime()
        runtime._running = True
        runtime._deferred_message_turn_task = None
        runtime._schedule_message_turn = Mock()
        runtime._defer_message_turn_check(60)
        first = runtime._deferred_message_turn_task
        await asyncio.sleep(0)
        runtime._defer_message_turn_check(60)
        replacement = runtime._deferred_message_turn_task
        await first
        self.assertIs(runtime._deferred_message_turn_task, replacement)
        runtime._cancel_deferred_message_turn_task()
        with suppress(asyncio.CancelledError):
            await replacement
