"""实际发送成功后解除旧退避，草稿取消不能冒充已回复。"""

from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock
import ast
import asyncio

from pytests.maisaka.test_reply_liveness import _action, _engine, _message, _result, _start


ROOT = Path(__file__).resolve().parents[2]


def _load_method(path, owner, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    module = ModuleType(name)
    module.__dict__.update(namespace)
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), module.__dict__)
    return module.__dict__[name]


finish_reply = _load_method("src/maisaka/runtime.py", "MaisakaHeartFlowChatting", "finish_planner_reply", {})
finalize_cycle = _load_method("src/maisaka/reasoning_engine.py", "MaisakaReasoningEngine", "_finalize_cycle", {
    "asyncio": asyncio, "emit_planner_snapshot": AsyncMock(),
})


class TestReplySuccessBackoff(TestCase):
    def test_all_successful_send_tracks_clear_backoff_immediately(self):
        for track in ("ordinary", "forced", "hourly"):
            with self.subTest(track=track):
                runtime = SimpleNamespace(
                    _idle_backoff=Mock(), _message_turn_scheduler=Mock(),
                    _reserved_forced_reply_message_ids={"target"},
                    _forced_reply_message_ids=deque(), _consumed_forced_reply_message_ids=deque(),
                )
                reservation = SimpleNamespace(
                    target_message_id="target", forced=track == "forced",
                    hourly_generation=7 if track == "hourly" else None,
                    quota_reservation=object() if track == "ordinary" else None,
                )
                finish_reply(runtime, reservation, success=True)
                runtime._idle_backoff.reset.assert_called_once()

    def test_unsent_or_discarded_reply_keeps_backoff(self):
        runtime = SimpleNamespace(_idle_backoff=Mock(), _message_turn_scheduler=Mock())
        reservation = SimpleNamespace(hourly_generation=3, forced=False, quota_reservation=None)
        finish_reply(runtime, reservation, success=False)
        runtime._idle_backoff.reset.assert_not_called()


class TestReplySuccessCycle(IsolatedAsyncioTestCase):
    async def test_reply_then_stop_does_not_rearm_idle_backoff(self):
        for sent in (True, False):
            with self.subTest(sent=sent):
                runtime = SimpleNamespace(
                    _max_internal_rounds=16, _prepare_tool_prompt_records=Mock(return_value=[]),
                    _render_context_usage_panel=Mock(), _idle_backoff=Mock(),
                    session_id="test", _monitor_run_id="run", _agent_state="stop",
                )
                engine = SimpleNamespace(_runtime=runtime, _end_cycle=AsyncMock(return_value=SimpleNamespace(time_records={})))
                state = SimpleNamespace(
                    cycle_end=SimpleNamespace(reason="planner_no_tool_end", detail="stop"), response=None,
                    tool_monitor_results=[{"tool_name": "reply", "reply_sent": sent}],
                    tool_result_summaries=[], planner_extra_lines=[], planner_started_at=None,
                    planner_ended_at=None, planner_interrupted=False,
                )
                await finalize_cycle(engine, cycle_detail=SimpleNamespace(cycle_id=1), round_index=0, state=state)
                if sent:
                    runtime._idle_backoff.reset.assert_called_once()
                    runtime._idle_backoff.record_cycle_result.assert_not_called()
                else:
                    runtime._idle_backoff.record_cycle_result.assert_called_once_with("planner_no_tool_end")

    async def test_successful_cancellation_is_not_a_visible_reply(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        result = _result("reply")
        result.metadata = {"reply_sent": False, "hourly_reply_discarded": True}
        result.stop_after_execution = True
        runtime._tool_registry.invoke.side_effect = lambda *_: result
        await _action(engine, ["reply"])
        self.assertFalse(engine._reply_liveness_succeeded)
        self.assertEqual(engine._unanswered_user_turns, 1)

    async def test_partially_sent_reply_counts_even_if_tool_reports_failure(self):
        engine, runtime = _engine()
        _start(engine, _message(), after_unanswered=True)
        result = _result("reply", success=False)
        result.metadata = {"reply_sent": True}
        runtime._tool_registry.invoke.side_effect = lambda *_: result
        await _action(engine, ["reply"])
        self.assertTrue(engine._reply_liveness_succeeded)
        self.assertEqual(engine._unanswered_user_turns, 0)
