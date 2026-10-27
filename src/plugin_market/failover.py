"""插件信息源的有界网络故障转移；不修改用户首选源和下载设置。"""

from math import ceil
from threading import RLock
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar
from urllib.error import HTTPError, URLError
import asyncio
import errno
import socket

from src.common.logger import get_logger

REQUEST_DEADLINE_SECONDS = 8.0
SOURCE_COOLDOWN_SECONDS = 300.0
_SOURCE_ORDER = {
    "official": ("official", "cnb", "github"),
    "cnb": ("cnb", "github", "official"),
    "github": ("github", "cnb", "official"),
}
_NETWORK_ERRNOS = {
    errno.ETIMEDOUT, errno.ECONNREFUSED, errno.ECONNRESET,
    errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENETDOWN, errno.EPIPE,
}
_cooldown_until: Dict[str, float] = {}
_cooldown_lock = RLock()
_T = TypeVar("_T")
logger = get_logger("plugin_market")


def _is_network_failure(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in {408, 502, 503, 504}
    if isinstance(error, URLError):
        return isinstance(error.reason, Exception) and _is_network_failure(error.reason)
    return (
        isinstance(error, (TimeoutError, ConnectionError, socket.gaierror))
        or isinstance(error, OSError) and error.errno in _NETWORK_ERRNOS
    )


async def run_with_source_fallback(
    preferred: str,
    loader: Callable[[str], Awaitable[_T]],
    *,
    candidates: Optional[Tuple[str, ...]] = None,
) -> Tuple[_T, Dict[str, Any]]:
    """每源最多尝试一次；网络失败冷却五分钟，语义错误和取消原样抛出。

    截止时间约束整个源操作，而非仅 socket 读取，因此 DNS 或共享缓存等待
    超时也能及时换源。锁只保护冷却表，不跨网络请求或 await 持有。
    """
    sources = _SOURCE_ORDER[preferred] if candidates is None else candidates
    attempted = []
    last_error = None
    for source in sources:
        with _cooldown_lock:
            if _cooldown_until.get(source, 0.0) > monotonic():
                continue
        attempted.append(source)
        try:
            result = await asyncio.wait_for(loader(source), timeout=REQUEST_DEADLINE_SECONDS)
        except Exception as error:
            if not _is_network_failure(error):
                raise
            last_error = error
            with _cooldown_lock:
                _cooldown_until[source] = monotonic() + SOURCE_COOLDOWN_SECONDS
            logger.warning(f"插件信息源网络失败，冷却五分钟: source={source} error={type(error).__name__}")
            continue
        with _cooldown_lock:
            _cooldown_until.pop(source, None)
        if source != preferred:
            logger.info(f"插件信息源自动切换: preferred={preferred} source={source}")
        return result, {
            "preferred_source": preferred,
            "fallback_from": preferred if source != preferred else None,
            "attempted_sources": attempted,
        }

    with _cooldown_lock:
        now = monotonic()
        # 另一事件循环可能已成功恢复某源；本请求仍须报告自身失败，而非读到缺失键。
        remaining = min(_cooldown_until.get(source, now) for source in sources) - now
    retry_after = max(1, ceil(remaining))
    raise OSError(f"插件信息源均处于网络故障冷却，最早 {retry_after} 秒后重试") from last_error
