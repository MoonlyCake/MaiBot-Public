"""目录、详情与安装版本查询的换源一致性；不启动应用或访问网络。"""

from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from urllib.error import HTTPError
import ast
import asyncio
import json

from src.plugin_market import failover


SERVICE_PATH = Path(__file__).resolve().parents[2] / "src/plugin_market/service.py"
FUNCTIONS = {
    "_snapshot_from_source", "_snapshot", "get_catalog", "_resolve",
    "_detail_from_source", "get_detail", "_release_entry_from_source",
    "load_plugin_release_entry", "load_release_index", "resolve_release",
}


class TestMarketFailoverIntegration(IsolatedAsyncioTestCase):
    def setUp(self):
        with failover._cooldown_lock:
            failover._cooldown_until.clear()
        self.settings = SimpleNamespace(information_source="official")
        self.entry = SimpleNamespace(
            id="repo.demo", manifest_id="plugin.demo", origin="cnb",
            sync_error=None, mode="branch",
        )
        self.plugin = {
            "id": "plugin.demo", "marketplace_id": "repo.demo", "name": "Demo",
            "repository_url": "https://github.com/example/demo",
            "manifest": {"id": "plugin.demo"}, "stats": None,
        }
        self.service = ModuleType("market_failover_integration_service")
        self.catalog_calls = []

        def catalog(items, source, index=None):
            self.catalog_calls.append((source, index))
            return [{**deepcopy(item), "catalog_source": source} for item in items]

        self.service.__dict__.update(
            asyncio=asyncio, json=json, HTTPError=HTTPError, MMC_VERSION="1.3.12",
            load_source_settings=lambda: self.settings,
            run_with_source_fallback=failover.run_with_source_fallback,
            _catalog=catalog,
            request_market=AsyncMock(return_value={
                "plugins": [self.plugin], "source_url": "official-summary", "cache": {},
            }),
            request_market_detail=AsyncMock(side_effect=TimeoutError()),
            fetch_information_file=AsyncMock(return_value={
                "data": json.dumps([self.plugin]), "url": "cnb-details", "cache": {},
            }),
            _git_releases=AsyncMock(return_value=SimpleNamespace(plugins=[self.entry])),
            _release_description=lambda entry, manifest: {"origin": entry.origin},
        )
        tree = ast.parse(SERVICE_PATH.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
                     and node.name in FUNCTIONS]
        unit = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0,
        ), *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), str(SERVICE_PATH), "exec"), self.service.__dict__)

    async def test_full_index_failure_tracks_real_git_source_not_official_service(self):
        expected = SimpleNamespace(plugins=[self.entry])
        self.service._git_releases.side_effect = [TimeoutError(), expected]

        self.assertIs(await self.service.load_release_index(True), expected)
        self.assertEqual(
            [(call.args[0], call.args[1]) for call in self.service._git_releases.await_args_list],
            [("github", True), ("cnb", True)],
        )
        self.assertNotIn("official", failover._cooldown_until)

    async def test_detail_timeout_reloads_catalog_and_releases_from_one_backup(self):
        result = await self.service.get_detail("plugin.demo")

        self.assertEqual(result["source"], "cnb")
        self.assertEqual(result["plugin"]["catalog_source"], "cnb")
        self.assertEqual(result["releases"]["origin"], "cnb")
        self.assertEqual([source for source, _ in self.catalog_calls], ["official", "cnb"])
        self.assertIsNotNone(self.catalog_calls[1][1])
        self.assertEqual([call.args[0] for call in self.service._git_releases.await_args_list], ["cnb", "cnb"])
        self.service.request_market_detail.assert_awaited_once_with("plugin.demo", "1.3.12", False)

    async def test_catalog_exposes_actual_source_and_unavailable_stats(self):
        self.service.request_market.side_effect = TimeoutError()
        result = await self.service.get_catalog()

        self.assertTrue(result["success"])
        self.assertEqual(result["source"], "cnb")
        self.assertEqual(result["source_url"], "cnb-details")
        self.assertEqual(result["features"], {"stats": False, "releases": True})
        self.assertEqual(result["preferred_source"], "official")
        self.assertEqual(result["fallback_from"], "official")
        self.assertEqual(result["attempted_sources"], ["official", "cnb"])
        self.assertEqual(self.settings.information_source, "official")

    async def test_install_resolution_refreshes_release_index_on_backup(self):
        entry, version = await self.service.resolve_release("plugin.demo", "latest")

        self.assertIs(entry, self.entry)
        self.assertIsNone(version)
        self.service.request_market_detail.assert_awaited_once_with("plugin.demo", "1.3.12", True)
        self.service._git_releases.assert_awaited_once_with("cnb", True)

    async def test_missing_official_plugin_does_not_install_backup_substitute(self):
        self.service.request_market_detail.side_effect = HTTPError(
            "https://market.invalid/plugin.demo", 404, "not found", {}, None,
        )
        result = await self.service.load_plugin_release_entry("plugin.demo", force_refresh=True)

        self.assertIsNone(result)
        self.service.request_market_detail.assert_awaited_once_with("plugin.demo", "1.3.12", True)
        self.service._git_releases.assert_not_awaited()
        self.assertEqual(failover._cooldown_until, {})

    async def test_invalid_catalog_json_is_not_hidden_by_another_source(self):
        error = json.JSONDecodeError("invalid catalog", "", 0)
        self.service.request_market.side_effect = error

        with self.assertRaises(json.JSONDecodeError) as caught:
            await self.service.get_catalog()
        self.assertIs(caught.exception, error)
        self.service.request_market.assert_awaited_once()
        self.service.fetch_information_file.assert_not_awaited()
        self.service._git_releases.assert_not_awaited()
        self.assertEqual(failover._cooldown_until, {})
