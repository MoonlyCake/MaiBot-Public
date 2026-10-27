"""完整识别按哈希只请求一次，取消等待和进程重启不丢复用结果。"""

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import ast
import asyncio
import json
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]


def load_cache(directory):
    path = ROOT / "src/chat/image_system/recognition_cache.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    body = [node for node in body if not (isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "_CACHE_ROOT" for target in node.targets))]
    module = ModuleType(__name__ + "_production")
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update(OrderedDict=OrderedDict, Future=Future, dataclass=dataclass, field=field, Path=Path,
               asyncio=asyncio, json=json, threading=threading, ANIMATION_DESCRIPTION_VERSION=2,
               _CACHE_ROOT=Path(directory))
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), env)
    env["_legacy_static_description"] = Mock(return_value=None)
    return env


class ImageRecognitionCacheTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.env = load_cache(self.directory.name)

    def tearDown(self):
        sys.modules.pop(self.env["__name__"], None)
        self.directory.cleanup()

    async def test_same_hash_in_parallel_and_after_restart_only_recognizes_once(self):
        env = self.env
        started, release = asyncio.Event(), asyncio.Event()
        result = env["RecognitionResult"]("先挥手然后转身，字幕从你好变成再见", ["告别", "挥手"])

        async def recognize():
            started.set()
            await release.wait()
            return result

        factory = AsyncMock(side_effect=recognize)
        first = asyncio.create_task(env["recognize_once"]("same-hash", b"GIF89a", factory))
        await asyncio.wait_for(started.wait(), 1)
        second = asyncio.create_task(env["recognize_once"]("same-hash", b"GIF89a", factory))
        await asyncio.sleep(0)
        release.set()
        self.assertEqual((await first).description, result.description)
        self.assertEqual((await second).emoji_tags, result.emoji_tags)
        factory.assert_awaited_once()
        env["_hot"].clear()
        self.assertEqual((await env["recognize_once"]("same-hash", b"GIF89a", factory)).description, result.description)
        factory.assert_awaited_once()
        self.assertEqual([path.name for path in Path(self.directory.name).iterdir()], ["same-hash.json"])

    async def test_cancelled_waiter_does_not_cancel_shared_first_recognition(self):
        env = self.env
        started, release = asyncio.Event(), asyncio.Event()
        result = env["RecognitionResult"]("动作完整描述", ["开心"])

        async def recognize():
            started.set()
            await release.wait()
            return result

        factory = AsyncMock(side_effect=recognize)
        waiter = asyncio.create_task(env["recognize_once"]("cancel-hash", b"GIF89a", factory))
        await asyncio.wait_for(started.wait(), 1)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertIn("cancel-hash", env["_inflight"])
        release.set()
        recovered = await env["recognize_once"]("cancel-hash", b"GIF89a", factory)
        self.assertEqual(recovered.description, result.description)
        factory.assert_awaited_once()

    async def test_new_strategy_lazily_replaces_only_requested_old_gif(self):
        env = self.env
        path = Path(self.directory.name) / "old-gif.json"
        path.write_text(json.dumps({"version": 1, "description": "旧首帧", "emoji_tags": []}))
        factory = AsyncMock(return_value=env["RecognitionResult"]("完整动作", ["惊讶"]))
        result = await env["recognize_once"]("old-gif", b"GIF89a", factory)
        self.assertEqual(result.description, "完整动作")
        self.assertEqual(json.loads(path.read_text())["version"], 2)
        factory.assert_awaited_once()

    async def test_empty_results_are_not_saved_as_success_and_payload_preserves_tags(self):
        env = self.env
        result = env["parse_recognition_response"]('```json\n{"description":"打招呼后逃跑","emoji_tags":["害羞","逃跑"]}\n```')
        self.assertEqual(result.description, "打招呼后逃跑")
        self.assertEqual(result.emoji_tags, ["害羞", "逃跑"])
        factory = AsyncMock(return_value=env["RecognitionResult"](""))
        await env["recognize_once"]("empty", b"image", factory)
        self.assertFalse((Path(self.directory.name) / "empty.json").exists())

    async def test_foreground_waiter_reuses_background_task_and_inbound_hash(self):
        path = ROOT / "src/chat/image_system/image_manager.py"
        tree = ast.parse(path.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageManager")
        names = {"get_image_description", "_schedule_description_build", "ensure_image_saved"}
        original.body = [node for node in original.body if getattr(node, "name", None) in names]
        from types import SimpleNamespace
        env = {"asyncio": asyncio, "_is_vlm_task_configured": lambda: True,
               "get_cached_recognition": Mock(return_value=""), "logger": Mock(),
               "hashlib": SimpleNamespace(sha256=Mock(side_effect=AssertionError("must reuse inbound hash")))}
        unit = ast.Module(body=ast.parse("from __future__ import annotations").body + [original], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), env)
        manager = env["ImageManager"]()
        saved = SimpleNamespace(file_hash="provided-hash")
        manager._ensure_image_saved = Mock(return_value=saved)
        manager._pending_description_tasks = {}
        manager._finalize_description_build = Mock()
        started, release = asyncio.Event(), asyncio.Event()

        async def build(*args, **kwargs):
            started.set()
            await release.wait()
            return "完整识别"

        manager._build_description_in_background = AsyncMock(side_effect=build)
        self.assertEqual(await manager.get_image_description(
            image_hash="provided-hash", image_bytes=b"image", wait_for_build=False), "")
        await asyncio.wait_for(started.wait(), 1)
        foreground = asyncio.create_task(manager.get_image_description(
            image_hash="provided-hash", image_bytes=b"image", wait_for_build=True))
        await asyncio.sleep(0)
        release.set()
        self.assertEqual(await foreground, "完整识别")
        manager._build_description_in_background.assert_awaited_once()
        env["hashlib"].sha256.assert_not_called()
        self.assertTrue(all(call.args == (b"image", "provided-hash") for call in manager._ensure_image_saved.call_args_list))
