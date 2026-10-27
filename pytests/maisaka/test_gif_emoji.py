"""GIF 表情分镜与已识别哈希复用，只装载实际方法，不初始化应用。"""

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from PIL import Image, ImageOps

import ast
import asyncio
import base64
import re
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Part:
    image_format: str
    image_base64: str


def image_bytes(size=(512, 512)):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format="JPEG")
    return output.getvalue()


def load(path, names, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names or (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                            for target in node.targets))]
    if methods is not None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                node.body = [method for method in node.body if getattr(method, "name", None) in methods]
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), namespace)
    return namespace


class EmojiStoryboardTests(TestCase):
    def setUp(self):
        self.prepare = Mock(return_value=None)
        self.env = load("src/maisaka/visual/emoji_image_cache.py", {
            "_prepare_emoji_image", "EmojiImageCache", "EMOJI_MAX_SIDE", "EMOJI_CACHE_MAX_BYTES",
            "EMOJI_CACHE_MAX_ENTRIES", "_ENCODING_VERSION"}, dict(
                Image=Image, ImageOps=ImageOps, BytesIO=BytesIO, base64=base64, sys=sys,
                threading=threading, OrderedDict=OrderedDict, Future=Future,
                ContextImagePart=Part, SUPPORTED_IMAGE_FORMATS={"jpeg", "png", "gif", "webp"},
                ANIMATION_DESCRIPTION_VERSION=2, prepare_animated_image=self.prepare))

    def test_storyboard_keeps_panel_resolution_and_uses_existing_hash(self):
        self.prepare.return_value = image_bytes((1152, 816))
        cache = self.env["EmojiImageCache"]()
        first = cache.get("existing-hash", b"gif-original")
        self.assertIs(first, cache.get("existing-hash", b"gif-original"))
        self.prepare.assert_called_once_with(b"gif-original", image_hash="existing-hash")
        with Image.open(BytesIO(base64.b64decode(first.image_base64))) as storyboard:
            self.assertEqual(storyboard.size, (1152, 816))

    def test_static_emoji_still_uses_256_pixel_thumbnail(self):
        part = self.env["_prepare_emoji_image"](image_bytes(), image_hash="static")
        with Image.open(BytesIO(base64.b64decode(part.image_base64))) as thumbnail:
            self.assertEqual(thumbnail.size, (256, 256))


class EmojiRecognitionTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.result = NS(description="举手后挥手告别", emoji_tags=["告别", "开心"])
        self.vlm = NS(generate_response_for_image=AsyncMock(return_value=NS(response="json-result")))
        self.prepare = Mock(return_value=b"storyboard-jpeg")
        self.prompt = NS(add_context=Mock())
        self.env = load("src/emoji_system/emoji_manager.py", {
            "EmojiManager", "_normalize_emoji_tag_text"}, dict(
                asyncio=asyncio, re=re, logger=Mock(), _is_vlm_task_configured=lambda: True,
                prepare_animated_image=self.prepare,
                parse_recognition_response=Mock(return_value=self.result),
                prompt_manager=NS(get_prompt=Mock(return_value=self.prompt), render_prompt=AsyncMock(return_value="prompt")),
                emoji_manager_vlm=self.vlm,
                ImageUtils=NS(image_bytes_to_base64=lambda data: base64.b64encode(data).decode("ascii")),
                _serialize_emoji_for_hook=lambda emoji: {"file_hash": emoji.file_hash},
                _get_runtime_manager=lambda: NS(invoke_hook=AsyncMock(return_value=NS(aborted=False, kwargs={})))),
            methods={"build_emoji_description", "get_emoji_description"})
        self.manager = self.env["EmojiManager"]()
        self.target = NS(file_hash="known-hash", image_format="gif", image_bytes=b"original-gif",
                         file_name="emoji.gif", description="")

    async def test_cached_full_recognition_does_not_decode_or_call_vlm_again(self):
        self.env["recognize_once"] = AsyncMock(return_value=self.result)
        success, target = await self.manager.build_emoji_description(self.target)
        self.assertTrue(success)
        self.assertEqual(target.description, "告别,开心")
        self.prepare.assert_not_called()
        self.vlm.generate_response_for_image.assert_not_awaited()

    async def test_first_recognition_sends_storyboard_once_and_keeps_tags(self):
        async def recognize_once(image_hash, data, builder):
            self.assertEqual((image_hash, data), ("known-hash", b"original-gif"))
            return await builder()

        self.env["recognize_once"] = recognize_once
        success, target = await self.manager.build_emoji_description(self.target)
        self.assertTrue(success)
        self.assertEqual(target.emotion, ["告别", "开心"])
        self.prepare.assert_called_once_with(b"original-gif", image_hash="known-hash")
        self.vlm.generate_response_for_image.assert_awaited_once_with(
            "prompt", base64.b64encode(b"storyboard-jpeg").decode("ascii"), "jpeg", session_id="")

    async def test_legacy_static_description_keeps_existing_emoji_tags_without_vlm(self):
        self.target.description = "开心,告别"
        self.env["recognize_once"] = AsyncMock(return_value=NS(description="旧静态图片描述", emoji_tags=[]))
        success, target = await self.manager.build_emoji_description(self.target)
        self.assertTrue(success)
        self.assertEqual(target.description, "开心,告别")
        self.vlm.generate_response_for_image.assert_not_awaited()

    async def test_received_known_hash_returns_full_description_without_rehash(self):
        self.env["get_cached_recognition_result"] = Mock(return_value=self.result)
        self.env["hashlib"] = NS(sha256=Mock(side_effect=AssertionError("must use supplied hash")))
        self.manager.ensure_emoji_saved = AsyncMock()
        description, tags = await self.manager.get_emoji_description(
            emoji_hash="known-hash", emoji_bytes=b"original-gif", wait_for_build=False)
        self.assertEqual((description, tags), (self.result.description, self.result.emoji_tags))
        self.manager.ensure_emoji_saved.assert_awaited_once_with(b"original-gif", emoji_hash="known-hash")
        self.vlm.generate_response_for_image.assert_not_awaited()
