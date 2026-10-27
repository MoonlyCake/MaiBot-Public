"""小时草稿在真正发送边界等待；取消时正文、附件和发送记账均不泄漏。"""

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

import asyncio

from pytests.maisaka.test_hourly_fallback_release import HourlyFallbackController
from pytests.maisaka.test_reply_window_sender import fixture


class TestHourlyReplySender(IsolatedAsyncioTestCase):
    async def test_regular_reply_never_enters_the_hourly_wait(self):
        case = fixture()
        reservation = SimpleNamespace(hourly_generation=None)
        case.runtime.reserve_planner_reply.return_value = reservation
        case.runtime.wait_for_planner_reply_release = AsyncMock()

        result = await case.run()

        self.assertTrue(result.metadata["reply_sent"])
        case.runtime.wait_for_planner_reply_release.assert_not_awaited()
        self.assertEqual(case.sender.await_count, 2)

    async def test_new_message_discards_all_generated_segments_before_transport(self):
        case = fixture()
        controller = HourlyFallbackController()
        token = controller.begin_turn([case.target.message_id])
        controller.reserve_reply(case.target.message_id)
        reservation = SimpleNamespace(hourly_generation=token)
        case.runtime.reserve_planner_reply.return_value = reservation
        draft_ready = asyncio.Event()

        async def wait_for_release(current):
            self.assertIs(current, reservation)
            case.replyer.generate_reply_with_context.assert_awaited_once()
            case.sender.assert_not_awaited()
            draft_ready.set()
            return await controller.wait_for_release(current.hourly_generation)

        case.runtime.wait_for_planner_reply_release = wait_for_release
        work = asyncio.create_task(case.run())
        await draft_ready.wait()
        controller.observe_message()
        result = await work

        self.assertTrue(result.stop_after_execution)
        self.assertTrue(result.metadata["hourly_reply_discarded"])
        self.assertFalse(result.metadata["reply_sent"])
        case.sender.assert_not_awaited()
        case.runtime.finish_planner_reply.assert_called_once_with(reservation, success=False)

    async def test_quiet_release_sends_the_same_draft_and_counts_split_reply_once(self):
        case = fixture()
        controller = HourlyFallbackController(quiet_seconds=0)
        token = controller.begin_turn([case.target.message_id])
        controller.reserve_reply(case.target.message_id)
        reservation = SimpleNamespace(hourly_generation=token)
        case.runtime.reserve_planner_reply.return_value = reservation

        async def wait_for_release(current):
            case.sender.assert_not_awaited()
            return await controller.wait_for_release(current.hourly_generation)

        case.runtime.wait_for_planner_reply_release = wait_for_release
        result = await case.run()

        self.assertTrue(result.metadata["reply_sent"])
        self.assertEqual(case.sender.await_count, 2)
        case.replyer.generate_reply_with_context.assert_awaited_once()
        case.runtime.finish_planner_reply.assert_called_once_with(reservation, success=True)
