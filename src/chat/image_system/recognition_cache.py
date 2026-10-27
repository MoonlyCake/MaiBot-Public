"""按现有图片哈希复用完整识别；普通图片与表情共享同一次视觉请求。"""

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

import asyncio
import json
import threading

from src.common.paths import get_instance_root
from src.common.utils.animated_image import ANIMATION_DESCRIPTION_VERSION


@dataclass(frozen=True)
class RecognitionResult:
    description: str
    emoji_tags: List[str] = field(default_factory=list)


_CACHE_ROOT = get_instance_root() / "data" / "image-recognition"
_MAX_HOT_ENTRIES = 512
_hot: OrderedDict[str, RecognitionResult] = OrderedDict()
_inflight: Dict[str, Future[RecognitionResult]] = {}
_tasks: set[asyncio.Task[None]] = set()
_lock = threading.RLock()


def parse_recognition_response(response: str) -> RecognitionResult:
    """解析图片与表情共用的描述、情绪标签结构。"""
    text = str(response or "").strip()
    if not text:
        return RecognitionResult("")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    return RecognitionResult(payload["description"].strip(),
                             list(dict.fromkeys(tag.strip() for tag in payload["emoji_tags"] if tag.strip())))


def _remember(image_hash: str, result: RecognitionResult) -> None:
    with _lock:
        _hot[image_hash] = result
        _hot.move_to_end(image_hash)
        while len(_hot) > _MAX_HOT_ENTRIES:
            _hot.popitem(last=False)


def _save(
    image_hash: str, result: RecognitionResult, *, replace_existing: bool = True,
) -> RecognitionResult:
    """仅写当前被识别的哈希，不枚举旧图片或重新计算哈希。"""
    with _lock:
        if not replace_existing and image_hash in _hot:
            return _hot[image_hash]
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        path = _CACHE_ROOT / f"{image_hash}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": ANIMATION_DESCRIPTION_VERSION,
                                         "description": result.description,
                                         "emoji_tags": result.emoji_tags}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        _remember(image_hash, result)
        return result


def _legacy_static_description(image_hash: str, image_bytes: bytes) -> Optional[RecognitionResult]:
    """只信任原普通静态图片描述；旧 GIF 首帧和表情标签需要本次惰性更新。"""
    if image_bytes.startswith(b"GIF8") or (image_bytes.startswith(b"RIFF") and b"ANIM" in image_bytes[:256]):
        return None
    from sqlmodel import select
    from src.common.database.database import get_db_session
    from src.common.database.database_model import Images, ImageType

    with get_db_session() as session:
        record = session.exec(select(Images).filter_by(image_hash=image_hash, image_type=ImageType.IMAGE).limit(1)).first()
        if record is None or not record.vlm_processed or not record.description:
            return None
        if not image_bytes and Path(record.full_path).suffix.lower() in {".gif", ".webp"}:
            return None
        return RecognitionResult(str(record.description).strip())


def get_cached_recognition_result(image_hash: str, image_bytes: bytes = b"") -> Optional[RecognitionResult]:
    """同步读取一个哈希；调用方在工作线程使用未命中的磁盘/数据库查询。"""
    if not image_hash:
        return None
    with _lock:
        result = _hot.get(image_hash)
        if result is not None:
            _hot.move_to_end(image_hash)
            return result
    path = _CACHE_ROOT / f"{image_hash}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = None
    if payload is not None and payload["version"] == ANIMATION_DESCRIPTION_VERSION:
        result = RecognitionResult(payload["description"], payload["emoji_tags"])
        if result.description:
            _remember(image_hash, result)
            return result
    result = _legacy_static_description(image_hash, image_bytes)
    if result is not None and result.description:
        return _save(image_hash, result, replace_existing=False)
    return None


def get_cached_recognition(image_hash: str, image_bytes: bytes = b"") -> str:
    result = get_cached_recognition_result(image_hash, image_bytes)
    return result.description if result is not None else ""


async def recognize_once(
    image_hash: str,
    image_bytes: bytes,
    factory: Callable[[], Awaitable[RecognitionResult]],
) -> RecognitionResult:
    """跨事件循环共享同一哈希的请求；等待者取消不会取消首次完整识别。"""
    cached = await asyncio.to_thread(get_cached_recognition_result, image_hash, image_bytes)
    if cached is not None:
        return cached
    with _lock:
        cached = _hot.get(image_hash)
        if cached is not None:
            return cached
        shared = _inflight.get(image_hash)
        owns_request = shared is None
        if shared is None:
            shared = Future()
            _inflight[image_hash] = shared

    if owns_request:
        async def build() -> None:
            try:
                result = await factory()
                if result.description:
                    await asyncio.to_thread(_save, image_hash, result)
                shared.set_result(result)
            except BaseException as exc:
                shared.set_exception(exc)
            finally:
                with _lock:
                    _inflight.pop(image_hash, None)
                    _tasks.discard(asyncio.current_task())

        task = asyncio.create_task(build(), name=f"image-recognition:{image_hash[:12]}")
        with _lock:
            _tasks.add(task)
    return await asyncio.shield(asyncio.wrap_future(shared))
