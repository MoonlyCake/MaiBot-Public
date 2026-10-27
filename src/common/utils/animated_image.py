"""将动态图片转换为带时间顺序的模型分镜；原始图片保持不变。"""

from collections import OrderedDict
from io import BytesIO
from typing import List, Optional, Tuple, Union

import bisect
import math
import threading

from PIL import Image, ImageDraw

ANIMATION_DESCRIPTION_VERSION = 2
ANIMATION_FRAME_COUNT = 6
ANIMATION_CELL_MAX_SIDE = 384
ANIMATION_LABEL_HEIGHT = 24
ANIMATION_CACHE_MAX_ENTRIES = 32
ANIMATION_CACHE_MAX_BYTES = 32 * 1024 * 1024

CacheKey = Tuple[Union[str, bytes], int]
_animation_cache: OrderedDict[CacheKey, Tuple[Optional[bytes], int]] = OrderedDict()
_animation_cache_bytes = 0
_animation_cache_lock = threading.Lock()


def _animation_frame_indices(durations: List[int]) -> List[int]:
    """沿播放时间取样，去除长时间停留造成的重复帧，并保留首尾。"""
    ends: List[int] = []
    total = 0
    for duration in durations:
        total += duration
        ends.append(total)
    if len(durations) <= ANIMATION_FRAME_COUNT:
        return list(range(len(durations)))
    targets = [total * index / (ANIMATION_FRAME_COUNT - 1) for index in range(ANIMATION_FRAME_COUNT)]
    return sorted({min(bisect.bisect_right(ends, target), len(durations) - 1) for target in targets})


def _build_animation_storyboard(image_bytes: bytes) -> Optional[bytes]:
    with Image.open(BytesIO(image_bytes)) as image:
        if image.format not in {"GIF", "WEBP"} or not image.is_animated:
            return None

        # 第一遍只保留时间信息，不把全部帧解码后的像素常驻内存。
        durations: List[int] = []
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
            durations.append(max(1, int(image.info.get("duration", 100))))
        indices = _animation_frame_indices(durations)
        starts = [0]
        for duration in durations:
            starts.append(starts[-1] + duration)

        scale = min(1.0, ANIMATION_CELL_MAX_SIDE / max(image.size))
        frame_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        cell_width = max(128, frame_size[0])
        cell_height = frame_size[1] + ANIMATION_LABEL_HEIGHT
        columns = min(3, len(indices))
        canvas = Image.new("RGB", (columns * cell_width, math.ceil(len(indices) / columns) * cell_height), "white")
        labels = ImageDraw.Draw(canvas)
        for slot, index in enumerate(indices):
            # Pillow seek 会应用 GIF disposal/局部帧规则，避免把局部增量误当完整画面。
            image.seek(index)
            frame = image.convert("RGBA")
            frame.thumbnail((ANIMATION_CELL_MAX_SIDE, ANIMATION_CELL_MAX_SIDE), Image.Resampling.LANCZOS)
            x = slot % columns * cell_width
            y = slot // columns * cell_height
            canvas.paste(frame, (x + (cell_width - frame.width) // 2, y + ANIMATION_LABEL_HEIGHT), frame)
            labels.text((x + 4, y + 5), f"{slot + 1} | {starts[index] / 1000:.2f}s", fill="black")
        output = BytesIO()
        canvas.save(output, format="JPEG", quality=92)
        return output.getvalue()


def prepare_animated_image(image_bytes: bytes, *, image_hash: str = "") -> Optional[bytes]:
    """动画返回 JPEG 分镜，静态图返回 None；按现有哈希和策略版本复用结果。

    未传哈希的客户端使用原字节作为有界缓存键，不重复计算 SHA。调用方应在线程池中
    完成缓存未命中的抽帧工作；转换锁也使并发到达的同图只生成一次分镜。
    """
    global _animation_cache_bytes
    if not (image_bytes.startswith((b"GIF87a", b"GIF89a")) or (
        image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
    )):
        return None
    key = (image_hash if image_hash else image_bytes, ANIMATION_DESCRIPTION_VERSION)
    with _animation_cache_lock:
        cached = _animation_cache.get(key)
        if cached is not None:
            _animation_cache.move_to_end(key)
            return cached[0]
        result = _build_animation_storyboard(image_bytes)
        size = (len(image_bytes) if not image_hash else len(image_hash)) + (len(result) if result else 0)
        if size <= ANIMATION_CACHE_MAX_BYTES:
            while _animation_cache and (
                len(_animation_cache) >= ANIMATION_CACHE_MAX_ENTRIES
                or _animation_cache_bytes + size > ANIMATION_CACHE_MAX_BYTES
            ):
                _, (_, removed_size) = _animation_cache.popitem(last=False)
                _animation_cache_bytes -= removed_size
            _animation_cache[key] = (result, size)
            _animation_cache_bytes += size
        return result
