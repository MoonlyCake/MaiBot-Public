"""离线覆盖动态分镜的内容顺序、时间抽样和接收端动画保留。"""

from io import BytesIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import ast
import io

from PIL import Image

from src.common.utils import animated_image


def gif(colors, *, durations=100, size=(64, 40)):
    output = BytesIO()
    frames = [Image.new("RGB", size, color) for color in colors]
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return output.getvalue()


def compression_method():
    path = Path(__file__).resolve().parents[1] / "src/common/utils/utils_image.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageUtils")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"compress_image_to_size", "_prepare_image_for_receive_compression"}]
    namespace = {"PILImage": Image, "io": io, "logger": Mock()}
    unit = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace["ImageUtils"].compress_image_to_size


class AnimatedImageTests(TestCase):
    def setUp(self):
        with animated_image._animation_cache_lock:
            animated_image._animation_cache.clear()
            animated_image._animation_cache_bytes = 0

    def test_storyboard_keeps_first_and_last_frames_in_order(self):
        encoded = animated_image.prepare_animated_image(gif(["red", "green", "blue"]))
        with Image.open(BytesIO(encoded)) as sheet:
            self.assertEqual(sheet.format, "JPEG")
            self.assertEqual(sheet.size, (384, 64))
            self.assertGreater(sheet.getpixel((64, 44))[0], 200)
            self.assertGreater(sheet.getpixel((320, 44))[2], 200)
            # 时间编号处包含文字，不是未标记的拼图。
            self.assertTrue(any(sheet.getpixel((x, 10))[0] < 100 for x in range(4, 70)))

    def test_sampling_uses_duration_and_covers_tail(self):
        indices = animated_image._animation_frame_indices([1000, 10, 10, 10, 10, 10, 10, 10])
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 7)
        self.assertLessEqual(len(indices), 6)
        self.assertNotIn(1, indices)

    def test_same_hash_shares_one_conversion_and_strategy_version(self):
        data = gif(["red", "blue"])
        original = animated_image._build_animation_storyboard
        with patch.object(animated_image, "_build_animation_storyboard", wraps=original) as build:
            first = animated_image.prepare_animated_image(data, image_hash="existing-image-id")
            self.assertIs(animated_image.prepare_animated_image(data, image_hash="existing-image-id"), first)
            self.assertEqual(build.call_count, 1)
            with patch.object(animated_image, "ANIMATION_DESCRIPTION_VERSION", 999):
                animated_image.prepare_animated_image(data, image_hash="existing-image-id")
            self.assertEqual(build.call_count, 2)

    def test_static_image_passthrough_and_animation_original_preserved(self):
        output = BytesIO()
        Image.new("RGB", (20, 20), "white").save(output, format="PNG")
        self.assertIsNone(animated_image.prepare_animated_image(output.getvalue()))
        data = gif(["red", "blue"])
        self.assertIs(compression_method()(data, 1), data)

    def test_large_frames_have_bounded_sheet_and_white_transparency(self):
        output = BytesIO()
        first = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
        second = first.copy()
        second.paste((255, 0, 0, 255), (400, 250, 500, 350))
        first.save(output, format="WEBP", save_all=True, append_images=[second], duration=100, lossless=True)
        with Image.open(BytesIO(animated_image.prepare_animated_image(output.getvalue()))) as sheet:
            self.assertLessEqual(sheet.width, 3 * 384)
            self.assertLessEqual(sheet.height, 2 * (384 + 24))
            self.assertGreater(min(sheet.getpixel((10, 50))), 245)
