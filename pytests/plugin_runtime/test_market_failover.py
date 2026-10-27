"""信息源超时、冷却及取消回归；无真实网络和八秒等待。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError, URLError
import asyncio
import json
import socket

from src.plugin_market import failover


class TestMarketSourceFailover(IsolatedAsyncioTestCase):
    def setUp(self):
        with failover._cooldown_lock:
            failover._cooldown_until.clear()
        self.now = 100.0
        self.clock = patch.object(failover, "monotonic", lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    async def test_timeout_uses_backup_without_changing_preference(self):
        loader = AsyncMock(side_effect=[TimeoutError(), {"source": "cnb"}])
        result, metadata = await failover.run_with_source_fallback("official", loader)
        self.assertEqual(result, {"source": "cnb"})
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["official", "cnb"])
        self.assertEqual(metadata, {
            "preferred_source": "official", "fallback_from": "official",
            "attempted_sources": ["official", "cnb"],
        })
        self.assertEqual(failover._cooldown_until["official"], 400.0)

    async def test_cooldown_skips_failed_source_then_recovers_preferred(self):
        calls = []
        async def load(source):
            calls.append(source)
            if calls == ["official"]:
                raise ConnectionRefusedError()
            return source
        await failover.run_with_source_fallback("official", load)
        result, metadata = await failover.run_with_source_fallback("official", load)
        self.assertEqual(result, "cnb")
        self.assertEqual(metadata["attempted_sources"], ["cnb"])
        self.now = 401.0
        result, metadata = await failover.run_with_source_fallback("official", load)
        self.assertEqual(result, "official")
        self.assertIsNone(metadata["fallback_from"])
        self.assertNotIn("official", failover._cooldown_until)

    async def test_each_preference_has_a_deterministic_backup_order(self):
        for preferred, expected in (("cnb", ["cnb", "github"]), ("github", ["github", "cnb"])):
            with self.subTest(preferred=preferred):
                failover._cooldown_until.clear()
                loader = AsyncMock(side_effect=[URLError(socket.gaierror("DNS unavailable")), "ok"])
                result, metadata = await failover.run_with_source_fallback(preferred, loader)
                self.assertEqual(result, "ok")
                self.assertEqual(metadata["attempted_sources"], expected)

    async def test_git_index_uses_only_real_git_sources_and_does_not_cool_official(self):
        loader = AsyncMock(side_effect=[TimeoutError(), "cnb-index"])
        result, metadata = await failover.run_with_source_fallback(
            "github", loader, candidates=("github", "cnb"),
        )
        self.assertEqual(result, "cnb-index")
        self.assertEqual(metadata["attempted_sources"], ["github", "cnb"])
        self.assertNotIn("official", failover._cooldown_until)

        failover._cooldown_until.update(github=200.0, cnb=300.0)
        loader = AsyncMock()
        with self.assertRaisesRegex(OSError, "100 秒"):
            await failover.run_with_source_fallback(
                "github", loader, candidates=("github", "cnb"),
            )
        loader.assert_not_awaited()
        self.assertNotIn("official", failover._cooldown_until)

    async def test_network_gateway_errors_switch_but_auth_and_format_errors_do_not(self):
        for code in (408, 502, 503, 504):
            with self.subTest(code=code):
                failover._cooldown_until.clear()
                loader = AsyncMock(side_effect=[HTTPError("https://market.invalid", code, "failed", {}, None), "ok"])
                result, _ = await failover.run_with_source_fallback("official", loader)
                self.assertEqual(result, "ok")
        for error in (
            HTTPError("https://market.invalid", 401, "auth", {}, None),
            HTTPError("https://market.invalid", 403, "forbidden", {}, None),
            HTTPError("https://market.invalid", 404, "not found", {}, None),
            json.JSONDecodeError("invalid", "", 0),
            ValueError("invalid release index"),
        ):
            with self.subTest(error=type(error).__name__):
                failover._cooldown_until.clear()
                loader = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)):
                    await failover.run_with_source_fallback("official", loader)
                loader.assert_awaited_once_with("official")
                self.assertEqual(failover._cooldown_until, {})

    async def test_real_async_deadline_cancels_slow_source_and_moves_on(self):
        cancelled = asyncio.Event()
        async def load(source):
            if source == "official":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return source
        with patch.object(failover, "REQUEST_DEADLINE_SECONDS", 0.001):
            result, _ = await failover.run_with_source_fallback("official", load)
        self.assertEqual(result, "cnb")
        self.assertTrue(cancelled.is_set())

    async def test_caller_cancellation_does_not_try_another_source(self):
        entered = asyncio.Event()
        calls = []
        async def load(source):
            calls.append(source)
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(failover.run_with_source_fallback("official", load))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(calls, ["official"])
        self.assertEqual(failover._cooldown_until, {})

    async def test_all_cooling_sources_fail_fast_with_retry_time(self):
        failover._cooldown_until.update(official=350.0, cnb=200.0, github=190.0)
        loader = AsyncMock()
        with self.assertRaisesRegex(OSError, "90 秒"):
            await failover.run_with_source_fallback("official", loader)
        loader.assert_not_awaited()

    async def test_all_network_failures_try_each_source_once(self):
        loader = AsyncMock(side_effect=TimeoutError())
        with self.assertRaisesRegex(OSError, "300 秒"):
            await failover.run_with_source_fallback("official", loader)
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["official", "cnb", "github"])

    async def test_shared_cooldown_is_usable_from_distinct_event_loops(self):
        barrier = Barrier(2)
        async def load(source):
            if source == "official":
                raise ConnectionRefusedError()
            # 两个独立线程/事件循环必须能同时进入loader；网络期间没有全局锁。
            await asyncio.to_thread(barrier.wait, 1)
            return source
        def run():
            return asyncio.run(failover.run_with_source_fallback("official", load))[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run) for _ in range(2)]
            results = await asyncio.gather(*(asyncio.wrap_future(future) for future in futures))
        self.assertEqual(results, ["cnb", "cnb"])

