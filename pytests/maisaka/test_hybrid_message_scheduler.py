"""混合调度真实状态迁移回归；小时与防抖使用假时钟，不启动模型或平台。"""

from unittest import IsolatedAsyncioTestCase
from types import SimpleNamespace

from pytests.maisaka.test_dynamic_reply_integration import (
    CLOCK,
    Scheduler,
    TurnGateResult,
    _message,
    _scheduler_runtime,
)


class TestHybridMessageScheduler(IsolatedAsyncioTestCase):
    def test_group_wait_admits_normal_messages_without_waiting_for_hourly_deadline(self):
        runtime = _scheduler_runtime([_message("new-question")])
        runtime._agent_state = runtime._STATE_WAIT
        scheduler = Scheduler(runtime)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("trigger", "credit available")
        scheduler.schedule_message_turn()
        scheduler._dynamic_reply_gate.evaluate.assert_called_once()
        runtime._cancel_wait_timeout_task.assert_called_once()
        runtime._enter_running_state.assert_called_once()
        runtime._enqueue_message_turn.assert_called_once()

    def setUp(self):
        CLOCK["now"] = 100.0

    def test_debounce_finishes_before_probability_or_forced_admission(self):
        for forced in (False, True):
            with self.subTest(forced=forced):
                runtime = _scheduler_runtime([_message("new", at=forced)], forced=forced, quiet_seconds=0.8)
                scheduler = Scheduler(runtime)
                scheduler.schedule_message_turn()
                runtime._defer_message_turn_check.assert_called_once_with(0.8)
                runtime._enqueue_message_turn.assert_not_called()
                scheduler._dynamic_reply_gate.evaluate.assert_not_called()
                runtime._fixed_message_debouncer.consume.assert_not_called()

    def test_hourly_deadline_uses_last_consumption_and_only_admits_latest_quiet_batch(self):
        first, latest = _message("earlier-batch"), _message("latest-batch")
        runtime = _scheduler_runtime([first])
        scheduler = Scheduler(runtime)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "below credit")
        scheduler.observe_message(first)
        scheduler.schedule_message_turn()

        CLOCK["now"] = 130.0
        runtime.message_cache.append(latest)
        runtime._get_pending_message_count.return_value = 2
        scheduler.observe_message(latest)
        scheduler.schedule_message_turn()
        CLOCK["now"] = 3699.0
        scheduler.schedule_message_turn()
        runtime._enqueue_message_turn.assert_not_called()
        self.assertEqual(runtime._defer_message_turn_check.call_args.args, (1.0,))

        CLOCK["now"] = 3700.0
        scheduler.schedule_message_turn()
        runtime._enqueue_message_turn.assert_called_once()
        self.assertTrue(scheduler._hourly_scheduled)
        self.assertEqual(scheduler.consume_batch([first, latest]), [latest])
        self.assertIsNone(scheduler.hourly.reserve_reply(first.message_id))
        self.assertIsNotNone(scheduler.hourly.reserve_reply(latest.message_id))

    async def test_new_message_invalidates_hourly_draft_and_retries_without_another_hour(self):
        old, new = _message("old"), _message("new")
        runtime = _scheduler_runtime([old])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(old)
        scheduler._hourly_scheduled = True
        scheduler.consume_batch([old])
        old_token = scheduler.hourly.reserve_reply(old.message_id)

        scheduler.observe_message(new)
        self.assertFalse(await scheduler.hourly.wait_for_release(old_token))
        runtime.message_cache = [new]
        runtime._get_pending_message_count.return_value = 1
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "still below credit")
        scheduler.schedule_message_turn()

        runtime._enqueue_message_turn.assert_called_once()
        self.assertTrue(scheduler._hourly_scheduled)
        self.assertEqual(scheduler.consume_batch([new]), [new])
        self.assertNotEqual(scheduler.hourly.reserve_reply(new.message_id), old_token)

    def test_ordinary_admission_has_no_hourly_reservation(self):
        message = _message("ordinary")
        runtime = _scheduler_runtime([message])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(message)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("trigger", "credit and soft gate")

        scheduler.schedule_message_turn()
        self.assertEqual(scheduler.consume_batch([message]), [message])

        runtime._enqueue_message_turn.assert_called_once()
        self.assertFalse(scheduler.hourly.active)
        self.assertIsNone(scheduler.hourly.reserve_reply(message.message_id))

    def test_at_arriving_after_hourly_enqueue_upgrades_consumption_to_ordinary_reply(self):
        old, at = _message("older-batch"), _message("new-at", at=True)
        runtime = _scheduler_runtime([old])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(old)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "low credit")
        CLOCK["now"] = 3700.0
        scheduler.schedule_message_turn()
        self.assertTrue(scheduler._hourly_scheduled)

        runtime._has_forced_turn_trigger = lambda: True
        runtime._forced_reply_message_ids.append(at.message_id)
        scheduler.observe_message(at)
        consumed = scheduler.consume_batch([old, at])

        self.assertEqual(consumed, [old, at])
        self.assertFalse(scheduler._hourly_scheduled)
        self.assertIsNone(scheduler.hourly_turn_generation)
        self.assertFalse(scheduler.hourly.active)
        self.assertIsNone(scheduler.hourly.reserve_reply(at.message_id))

    def test_zero_frequency_does_not_start_an_hourly_reply(self):
        message = _message("silent")
        runtime = _scheduler_runtime([message], silent=True, frequency=0)
        scheduler = Scheduler(runtime)
        scheduler.observe_message(message)
        CLOCK["now"] = 5000.0
        scheduler.schedule_message_turn()

        runtime._enqueue_message_turn.assert_called_once()
        self.assertFalse(scheduler._hourly_scheduled)
        scheduler._dynamic_reply_gate.observe_messages.assert_not_called()
        scheduler._dynamic_reply_gate.evaluate.assert_not_called()

    def test_private_threshold_keeps_idle_compensation_without_group_ledger(self):
        runtime = _scheduler_runtime([_message("one"), _message("two")], group=False)
        scheduler = Scheduler(runtime)
        scheduler.schedule_message_turn()
        runtime._defer_message_turn_check.assert_called_once_with(30.0)
        runtime._enqueue_message_turn.assert_not_called()

        CLOCK["now"] = 130.0
        scheduler.schedule_message_turn()
        runtime._enqueue_message_turn.assert_called_once()
        scheduler._dynamic_reply_gate.evaluate.assert_not_called()
        self.assertFalse(scheduler._hourly_scheduled)

    async def test_stop_invalidates_hourly_permission_without_sending(self):
        message = _message("old")
        runtime = _scheduler_runtime([message])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(message)
        scheduler._hourly_scheduled = True
        scheduler.consume_batch([message])
        token = scheduler.hourly.reserve_reply(message.message_id)

        scheduler.stop()

        self.assertFalse(await scheduler.hourly.wait_for_release(token))
        self.assertFalse(scheduler.hourly.active)
        self.assertFalse(scheduler._hourly_scheduled)
        runtime._enqueue_message_turn.assert_not_called()

    def test_receipt_tracks_each_real_message_but_not_notice_command_or_self(self):
        runtime = _scheduler_runtime([])
        scheduler = Scheduler(runtime)
        messages = [_message("first"), _message("second"), _message("notice", notify=True),
                    _message("command", command=True), _message("self", user_id="bot")]
        for message in messages:
            scheduler.observe_message(message)

        calls = scheduler._dynamic_reply_gate.observe_messages.call_args_list
        self.assertEqual([call.args[0][0].message_id for call in calls], ["first", "second"])
        self.assertEqual(scheduler._open_batch_ids, ["first", "second"])

    async def test_raw_media_arrival_cancels_draft_before_preprocessing_and_prevents_old_cache_reentry(self):
        old, image = _message("old"), _message("slow-image")
        runtime = _scheduler_runtime([old])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(old)
        scheduler._hourly_scheduled = True
        scheduler.consume_batch([old])
        token = scheduler.hourly.reserve_reply(old.message_id)

        scheduler.incoming_started(image.message_id)
        self.assertFalse(await scheduler.hourly.wait_for_release(token))
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "low credit")
        CLOCK["now"] = 5000.0
        scheduler.schedule_message_turn()
        scheduler._dynamic_reply_gate.evaluate.assert_called_once()
        self.assertFalse(scheduler._hourly_scheduled)
        runtime._enqueue_message_turn.assert_not_called()

        runtime.message_cache = [image]
        scheduler.observe_message(image)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "low credit")
        scheduler.schedule_message_turn()
        self.assertTrue(scheduler._hourly_scheduled)
        runtime._enqueue_message_turn.assert_called_once()
        self.assertEqual(scheduler.consume_batch([image]), [image])

    def test_media_preprocessing_does_not_block_an_explicit_at(self):
        runtime = _scheduler_runtime([_message("at", at=True)], forced=True)
        scheduler = Scheduler(runtime)
        scheduler.incoming_started("slow-image")

        scheduler.schedule_message_turn()

        runtime._enqueue_message_turn.assert_called_once()
        scheduler._dynamic_reply_gate.evaluate.assert_not_called()
        self.assertFalse(scheduler._hourly_scheduled)

    def test_media_arriving_after_hourly_enqueue_pauses_before_consuming_old_batch(self):
        old = _message("quiet-old-batch")
        runtime = _scheduler_runtime([old])
        scheduler = Scheduler(runtime)
        scheduler.observe_message(old)
        scheduler._dynamic_reply_gate.evaluate.return_value = TurnGateResult("wait", "low credit")
        CLOCK["now"] = 3700.0
        scheduler.schedule_message_turn()
        self.assertTrue(scheduler._hourly_scheduled)
        runtime._message_turn_scheduled = True

        scheduler.incoming_started("slow-image")

        self.assertTrue(scheduler.pause_hourly_for_incoming())
        self.assertFalse(scheduler._hourly_scheduled)
        self.assertTrue(scheduler._retry_hourly)
        self.assertIsNone(scheduler.hourly_turn_generation)
        self.assertFalse(scheduler.hourly.active)
        self.assertEqual(scheduler._latest_batch_ids, (old.message_id,))
        runtime._mark_message_turn_unscheduled.assert_called_once()

    def test_hourly_tools_cannot_bypass_the_single_delayed_reply_exit(self):
        message = _message("latest")
        scheduler = Scheduler(_scheduler_runtime([message]))
        scheduler.observe_message(message)
        scheduler._hourly_scheduled = True
        scheduler.consume_batch([message])
        for name in ("reply", "stop"):
            self.assertTrue(scheduler.allows_tool(SimpleNamespace(name=name)))
        for name in ("wait", "wait_rest", "send_emoji"):
            self.assertFalse(scheduler.allows_tool(SimpleNamespace(name=name, annotation=SimpleNamespace(read_only=True))))
        self.assertTrue(scheduler.allows_tool(SimpleNamespace(name="knowledge_search", annotation=SimpleNamespace(read_only=True))))
        self.assertFalse(scheduler.allows_tool(SimpleNamespace(name="external_sender", annotation=SimpleNamespace(read_only=False))))
        self.assertFalse(scheduler.allows_tool(None))
        scheduler.finish_turn()
        self.assertTrue(scheduler.allows_tool(SimpleNamespace(name="external_sender", annotation=None)))

