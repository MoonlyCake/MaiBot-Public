"""小时草稿释放的并发与生命周期边界；不发送消息、不等待真实三十秒。"""

from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import asyncio
import importlib.util
import sys


MODULE_PATH = Path(__file__).resolve().parents[2] / "src/maisaka/turn_trigger/hourly_fallback.py"
SPEC = importlib.util.spec_from_file_location("maibot_hourly_fallback_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
HourlyFallbackController = MODULE.HourlyFallbackController


class TestHourlyFallbackRelease(IsolatedAsyncioTestCase):
    async def test_only_one_reply_to_the_current_batch_can_be_reserved(self):
        controller = HourlyFallbackController(quiet_seconds=0)
        token = controller.begin_turn(["latest-message"])
        self.assertIsNone(controller.reserve_reply("old-message"))
        self.assertEqual(controller.reserve_reply("latest-message"), token)
        self.assertIsNone(controller.reserve_reply("latest-message"))
        controller.finish_reply(token, success=False)
        self.assertIsNone(controller.reserve_reply("latest-message"))

    async def test_message_during_generation_discards_before_waiting(self):
        controller = HourlyFallbackController()
        token = controller.begin_turn(["old"])
        controller.reserve_reply("old")
        self.assertTrue(controller.observe_message())
        self.assertFalse(await controller.wait_for_release(token))

    async def test_new_message_interrupts_a_pending_quiet_wait(self):
        controller = HourlyFallbackController()
        token = controller.begin_turn(["old"])
        controller.reserve_reply("old")
        wait = asyncio.create_task(controller.wait_for_release(token))
        await asyncio.sleep(0)
        self.assertFalse(wait.done())
        controller.observe_message()
        self.assertFalse(await wait)

    async def test_stop_interrupts_a_pending_quiet_wait(self):
        controller = HourlyFallbackController()
        token = controller.begin_turn(["old"])
        controller.reserve_reply("old")
        wait = asyncio.create_task(controller.wait_for_release(token))
        await asyncio.sleep(0)
        controller.stop()
        self.assertFalse(await wait)

    async def test_quiet_wait_starts_after_reply_generation_not_turn_start(self):
        clock = [100.0]
        controller = HourlyFallbackController(clock=lambda: clock[0])
        token = controller.begin_turn(["message"])
        controller.reserve_reply("message")
        clock[0] += 120.0

        async def expire(awaitable, timeout):
            self.assertEqual(timeout, 30.0)
            awaitable.close()
            clock[0] += timeout
            raise asyncio.TimeoutError

        with patch.object(MODULE.asyncio, "wait_for", expire):
            self.assertTrue(await controller.wait_for_release(token))
        self.assertEqual(clock[0], 250.0)

    async def test_old_turn_token_never_revalidates_when_another_turn_starts(self):
        controller = HourlyFallbackController(quiet_seconds=0)
        old = controller.begin_turn(["message"])
        controller.reserve_reply("message")
        controller.finish_turn()
        new = controller.begin_turn(["message"])
        controller.reserve_reply("message")
        self.assertNotEqual(new, old)
        self.assertFalse(await controller.wait_for_release(old))
        self.assertTrue(await controller.wait_for_release(new))

    async def test_only_successful_actual_send_marks_the_hourly_reply_succeeded(self):
        controller = HourlyFallbackController(quiet_seconds=0)
        token = controller.begin_turn(["message"])
        controller.reserve_reply("message")
        self.assertTrue(await controller.wait_for_release(token))
        self.assertFalse(controller.reply_succeeded)
        controller.finish_reply(token, success=True)
        self.assertTrue(controller.reply_succeeded)
