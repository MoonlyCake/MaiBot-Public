"""GIF在上下文和客户端保留动作；缓存命中后不再次发送图片像素。"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import ast
import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

from src.common.utils.animated_image import prepare_animated_image

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Part:
    image_format: str
    image_base64: str

    @property
    def normalized_image_format(self):
        return self.image_format


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    unit = ast.Module(body=ast.parse("from __future__ import annotations").body + body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)


def animation():
    output = BytesIO()
    Image.new("RGB", (80, 60), "red").save(output, format="GIF", save_all=True,
        append_images=[Image.new("RGB", (80, 60), "blue")], duration=100)
    return output.getvalue()


class GifContextTests(TestCase):
    def assert_dynamic_sheet(self, encoded):
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            pixels = list(image.convert("RGB").getdata())
            self.assertTrue(any(red > 180 and blue < 100 for red, _, blue in pixels))
            self.assertTrue(any(blue > 180 and red < 100 for red, _, blue in pixels))

    def test_image_and_emoji_cache_hits_only_append_description(self):
        namespace = {"get_cached_recognition": lambda *args: "先抬手，随后挥手告别"}
        load_functions("src/maisaka/context/messages.py", {"_append_image_component", "_append_emoji_component"}, namespace)
        for name in ("_append_image_component", "_append_emoji_component"):
            with self.subTest(component=name):
                builder = Mock()
                component = SimpleNamespace(binary_hash="same-picture", binary_data=b"not-decoded", content="old")
                namespace[name](builder, component, enable_visual_message=True)
                builder.add_image_content.assert_not_called()
                self.assertIn("随后挥手", builder.add_text_content.call_args.args[0])

    def test_uncached_gif_context_uses_both_frames(self):
        namespace = {"prepare_animated_image": prepare_animated_image, "base64": base64,
                     "SUPPORTED_IMAGE_FORMATS": {"gif", "webp", "jpeg", "png"}}
        load_functions("src/maisaka/context/messages.py", {"_encode_image_for_context"}, namespace)
        image_format, encoded = namespace["_encode_image_for_context"](animation(), "gif", "context-gif")
        self.assertEqual(image_format, "jpeg")
        self.assert_dynamic_sheet(encoded)

    def test_normalizer_and_openai_transport_preserve_dynamic_content(self):
        namespace = {"prepare_animated_image": prepare_animated_image, "base64": base64,
                     "binascii": binascii, "Image": Image, "PILImage": Image, "BytesIO": BytesIO,
                     "io": io, "UnidentifiedImageError": UnidentifiedImageError,
                     "ContextImagePart": Part, "SUPPORTED_OPENAI_IMAGE_FORMATS": {"jpeg", "png", "webp"},
                     "logger": Mock()}
        load_functions("src/llm_models/image_normalizer.py", {"normalize_image_part"}, namespace)
        load_functions("src/llm_models/model_client/openai_client.py", {"_normalize_image_part_for_openai"}, namespace)
        original = Part("gif", base64.b64encode(animation()).decode("ascii"))
        normalized = namespace["normalize_image_part"](original)
        self.assertEqual(len(normalized), 1)
        self.assert_dynamic_sheet(normalized[0].image_base64)
        image_format, encoded = namespace["_normalize_image_part_for_openai"](original)
        self.assertEqual(image_format, "jpeg")
        self.assert_dynamic_sheet(encoded)

    def test_binary_component_reuses_hash_already_computed_by_adapter(self):
        path = ROOT / "src/common/data_models/message_component_data_model.py"
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ByteComponent")
        namespace = {"hashlib": SimpleNamespace(sha256=Mock(side_effect=AssertionError("duplicate hash")))}
        unit = ast.Module(body=ast.parse("from __future__ import annotations").body + [node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
        component = namespace["ByteComponent"](binary_hash="adapter-content-hash", binary_data=b"image-bytes")
        self.assertEqual(component.binary_hash, "adapter-content-hash")
        self.assertEqual(component.binary_data, b"image-bytes")
