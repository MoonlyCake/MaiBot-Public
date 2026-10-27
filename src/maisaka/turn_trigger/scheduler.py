"""Maisaka 消息触发调度：先收齐批次，再走账本或小时低流量轨道。"""

from typing import Optional, TYPE_CHECKING
import time

from src.chat.utils.identity_behavior import is_bare_ai_label
from src.chat.utils.utils import is_bot_self
from src.common.data_models.message_component_data_model import TextComponent
from src.common.logger import get_logger
from src.config.config import global_config

from .dynamic_gate import ReplyReservation
from .gates import DynamicReplyTurnGate
from .hourly_fallback import HourlyFallbackController

if TYPE_CHECKING:
    from src.chat.message_receive.message import SessionMessage
    from src.core.tooling import ToolSpec
    from src.maisaka.runtime import MaisakaHeartFlowChatting

logger = get_logger("maisaka_turn_scheduler")
HOURLY_FALLBACK_SECONDS = 3600.0


class MessageTurnScheduler:
    """普通群聊按十条账本调度；积压一小时的最新安静批次独立交给 Planner。"""

    def __init__(self, runtime: "MaisakaHeartFlowChatting") -> None:
        self._runtime = runtime
        self._dynamic_reply_gate = DynamicReplyTurnGate(runtime)
        self.hourly = HourlyFallbackController()
        self._last_user_turn_at = time.monotonic()
        self._open_batch_ids: list[str] = []
        self._latest_batch_ids: tuple[str, ...] = ()
        self._hourly_scheduled = False
        self._retry_hourly = False
        self.hourly_turn_generation: Optional[int] = None
        self._incoming_message_ids: set[str] = set()

    def incoming_started(self, message_id: str) -> None:
        self._incoming_message_ids.add(message_id)
        self._retry_hourly = self.hourly.observe_message() or self._retry_hourly

    def incoming_finished(self, message_id: str) -> None:
        self._incoming_message_ids.discard(message_id)
        if self._runtime._running and self._runtime._has_pending_messages():
            self.schedule_message_turn()

    def observe_message(self, message: "SessionMessage") -> None:
        """真实消息逐条记账；新输入也会立即作废旧小时草稿。"""
        runtime = self._runtime
        self._incoming_message_ids.discard(message.message_id)
        if (message.is_notify or message.is_command
                or is_bot_self(message.platform, message.message_info.user_info.user_id)):
            return
        self._retry_hourly = self.hourly.observe_message() or self._retry_hourly
        self._open_batch_ids.append(message.message_id)
        if not runtime._uses_reply_window_quota() or runtime._is_reply_frequency_silent():
            return
        if message.message_id in runtime._forced_reply_message_ids:
            return
        if (is_bare_ai_label(message.processed_plain_text)
                and not message.is_at and not message.is_mentioned
                and all(isinstance(component, TextComponent) for component in message.raw_message.components)):
            return
        self._dynamic_reply_gate.observe_messages([message], runtime._get_effective_reply_frequency())

    def reserve_reply(self, target_message_id: str) -> Optional[ReplyReservation]:
        return self._dynamic_reply_gate.reserve_reply(target_message_id)

    def finish_reply(self, reservation: ReplyReservation, *, success: bool) -> None:
        self._dynamic_reply_gate.finish_reply(reservation, success=success)

    def _settle_batch(self) -> None:
        if self._open_batch_ids:
            self._latest_batch_ids = tuple(self._open_batch_ids)
            self._open_batch_ids.clear()
            # 安静批次与 Planner 消费解耦：被门控挡住后，下一批也必须重新防抖。
            self._runtime._fixed_message_debouncer.consume()

    def pause_hourly_for_incoming(self) -> bool:
        """小时信号已入队后仍可能来媒体，消费前再次确认，不能生成旧缓存的新草稿。"""
        if not self._hourly_scheduled or not self._incoming_message_ids:
            return False
        self._hourly_scheduled = False
        self._retry_hourly = True
        self._runtime._mark_message_turn_unscheduled()
        return True

    def consume_batch(self, messages: list["SessionMessage"]) -> list["SessionMessage"]:
        """消费时固定真实批次，小时轨只选最后一个已收齐的批次。"""
        self._settle_batch()
        if self._hourly_scheduled and self._runtime._has_forced_turn_trigger():
            # 小时信号排队后才到达的明确 @ 升级为普通必回，不增加三十秒等待。
            self._hourly_scheduled = False
        if self._hourly_scheduled:
            latest_ids = set(self._latest_batch_ids)
            messages = [message for message in messages if message.message_id in latest_ids]
            self.hourly_turn_generation = self.hourly.begin_turn([message.message_id for message in messages])
        else:
            self.hourly.finish_turn()
            self.hourly_turn_generation = None
        self._hourly_scheduled = False
        self._retry_hourly = False
        self._latest_batch_ids = ()
        self._last_user_turn_at = time.monotonic()
        return messages

    def finish_turn(self) -> None:
        self.hourly.finish_turn()
        self.hourly_turn_generation = None
        if self._runtime._running and self._runtime._has_pending_messages():
            self.schedule_message_turn()

    def stop(self) -> None:
        self.hourly.stop()
        self._open_batch_ids.clear()
        self._latest_batch_ids = ()
        self._hourly_scheduled = False
        self._retry_hourly = False
        self.hourly_turn_generation = None
        self._incoming_message_ids.clear()

    def allows_tool(self, spec: Optional["ToolSpec"]) -> bool:
        """小时轨只读准备后通过一次 reply 出口，不能用独立发送或等待工具绕过草稿。"""
        if self.hourly_turn_generation is None:
            return True
        if spec is None:
            return False
        if spec.name in {"reply", "stop"}:
            return True
        if spec.name in {"wait", "wait_rest", "send_emoji"}:
            return False
        return spec.annotation is not None and spec.annotation.read_only is True

    def _schedule_private(self, pending_count: int) -> None:
        """私聊保留原消息阈值和空窗唤醒，不进入群聊十条账本。"""
        runtime = self._runtime
        threshold = runtime._get_message_trigger_threshold()
        if pending_count >= threshold:
            self._enqueue_ready_message_turn()
            return
        average_interval = runtime._get_recent_average_external_message_interval()
        if average_interval is None:
            return
        last_received = runtime._last_external_message_received_at or runtime._last_message_received_at
        delay = (threshold - pending_count) * average_interval - max(0.0, time.time() - last_received)
        if delay <= 0:
            self._enqueue_ready_message_turn()
        else:
            runtime._defer_message_turn_check(delay)

    def _enqueue_ready_message_turn(self) -> None:
        """消息通过门控即可结束 wait；一小时只属于低流量兜底，不能绑住正常对话。"""
        runtime = self._runtime
        if runtime._agent_state == runtime._STATE_WAIT:
            runtime._cancel_wait_timeout_task()
            runtime._clear_message_debounce_required()
            runtime._enter_running_state()
        runtime._enqueue_message_turn()

    def schedule_message_turn(self) -> None:
        runtime = self._runtime
        if runtime._message_turn_scheduled:
            return
        pending_count = runtime._get_pending_message_count()
        if pending_count <= 0:
            runtime._cancel_deferred_message_turn_task()
            return

        # 先防抖后门控，@ 同样遵守已有的最多一秒收批上限。
        quiet_seconds = runtime._get_message_quiet_period_remaining_seconds()
        if quiet_seconds > 0:
            runtime._defer_message_turn_check(quiet_seconds)
            return
        self._settle_batch()

        effective_frequency = runtime._get_effective_reply_frequency()
        if runtime._is_reply_frequency_silent():
            self._enqueue_ready_message_turn()
            return

        if runtime._agent_state == runtime._STATE_WAIT and global_config.experimental.planner_message_steering:
            self._enqueue_ready_message_turn()
            return

        if runtime._has_forced_turn_trigger():
            logger.info(f"{runtime.log_prefix} 调度=明确必回 待处理={pending_count}")
            self._enqueue_ready_message_turn()
            return

        idle_backoff_seconds = runtime._idle_backoff.get_delay_seconds(pending_count)
        if not runtime.chat_stream.is_group_session:
            if idle_backoff_seconds > 0:
                runtime._defer_message_turn_check(idle_backoff_seconds)
            else:
                self._schedule_private(pending_count)
            return

        result = self._dynamic_reply_gate.evaluate(
            pending_messages=runtime.message_cache[runtime._last_processed_index :],
            frequency=effective_frequency,
            idle_backoff_seconds=idle_backoff_seconds,
        )
        logger.info(f"{runtime.log_prefix} 回复调度 待处理={pending_count} 频率={effective_frequency:.3f} {result.detail}")
        if result.should_trigger:
            self._enqueue_ready_message_turn()
            return

        if self._incoming_message_ids:
            # 媒体尚未登记时只暂停小时兜底，普通消息和明确 @ 不被慢识别拖住。
            runtime._cancel_deferred_message_turn_task()
            return
        hourly_delay = max(0.0, HOURLY_FALLBACK_SECONDS - (time.monotonic() - self._last_user_turn_at))
        if self._retry_hourly or hourly_delay <= 0:
            self._hourly_scheduled = True
            logger.info(f"{runtime.log_prefix} 调度=小时低流量 待处理={pending_count}，交 Planner 自主决定，回复暂存至静默30秒")
            self._enqueue_ready_message_turn()
            return
        next_check = hourly_delay
        if result.delay_seconds is not None:
            next_check = min(next_check, result.delay_seconds)
        runtime._defer_message_turn_check(next_check)
