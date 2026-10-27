"""小时低流量轮次的单次回复与静默释放，不持有消息正文或发送任务。"""

from time import monotonic
from typing import Callable, Optional, Sequence

import asyncio


HOURLY_REPLY_QUIET_SECONDS = 30.0


class HourlyFallbackController:
    """为小时轨回复绑定消息代次；新消息一到，旧草稿立即失效。"""

    def __init__(
        self,
        *,
        quiet_seconds: float = HOURLY_REPLY_QUIET_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._quiet_seconds = quiet_seconds
        self._clock = clock
        self._generation = 0
        self._active_generation: Optional[int] = None
        self._target_message_ids: set[str] = set()
        self._reply_reserved = False
        self._reply_succeeded = False
        self._stopped = False
        self._changed = asyncio.Event()

    @property
    def active(self) -> bool:
        return self._active_generation is not None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def reply_succeeded(self) -> bool:
        return self._reply_succeeded

    def _invalidate(self) -> None:
        self._generation += 1
        self._active_generation = None
        self._target_message_ids.clear()
        self._reply_reserved = False
        self._reply_succeeded = False
        self._changed.set()
        self._changed = asyncio.Event()

    def observe_message(self) -> bool:
        """使生成中或等待中的旧回复失效，返回是否中断了一个小时轮次。"""
        was_active = self.active
        self._invalidate()
        return was_active

    def begin_turn(self, message_ids: Sequence[str]) -> int:
        """防抖批次实际消费时绑定候选；下一小时的调度由外层负责。"""
        if self._stopped:
            raise RuntimeError("小时回复控制器已经停止")
        self._invalidate()
        self._active_generation = self._generation
        self._target_message_ids.update(message_ids)
        return self._generation

    def reserve_reply(self, target_message_id: str) -> Optional[int]:
        """小时轮次最多尝试一次回复，并且只允许回复这批用户消息。"""
        if (
            not self.active
            or self._reply_reserved
            or target_message_id not in self._target_message_ids
        ):
            return None
        self._reply_reserved = True
        return self._active_generation

    def finish_reply(self, generation: int, *, success: bool) -> None:
        """仅记录真正发送的结果；失败不自动重新预留或重发。"""
        if self._active_generation == generation:
            self._reply_succeeded = success

    def finish_turn(self) -> None:
        """Planner 主动不回或工具结束后，清理此轮的发送权限。"""
        self._invalidate()

    def stop(self) -> None:
        self._stopped = True
        self._invalidate()

    def _can_release(self, generation: int) -> bool:
        return (
            not self._stopped
            and self._active_generation == generation
            and self._reply_reserved
        )

    async def wait_for_release(self, generation: int) -> bool:
        """正文准备完成后等待连续三十秒；新消息或停机直接取消旧草稿。"""
        if not self._can_release(generation):
            return False
        changed = self._changed
        deadline = self._clock() + self._quiet_seconds
        while self._can_release(generation):
            remaining = deadline - self._clock()
            if remaining <= 0:
                return True
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return self._can_release(generation)
        return False
