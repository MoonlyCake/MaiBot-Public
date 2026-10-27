"""Maisaka 消息触发门控。"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Sequence, TYPE_CHECKING
import time

from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.data_models.message_component_data_model import AtComponent, ReplyComponent
from src.services.bot_account_service import get_online_bot_accounts

from .dynamic_gate import ReplyCreditGate, ReplyReservation
from .reply_likelihood import (
    DEFAULT_SECONDS_SINCE_BOT_MESSAGE,
    ReplyLikelihoodInput,
    estimate_reply_probability,
    summarize_pending_texts,
)

if TYPE_CHECKING:
    from src.maisaka.runtime import MaisakaHeartFlowChatting


TurnGateDecision = Literal["trigger", "wait", "delay"]
RECENT_PRESENCE_WINDOW_SECONDS = 300.0
"""统计近期消息量与麦麦发言占比的时间窗口。"""
MIN_REPLY_PROBABILITY = 0.15
"""普通批次只排除明显低概率消息，不用概率重新定义十条消息的回复比例。"""
LAST_REPLY_SLOT_MIN_PROBABILITY = 0.60
"""完整窗口的规划名额只剩最后一份时，留给高概率批次；明确对话不受此限制。"""
SOFT_GATE_MAX_DELAY_SECONDS = 5.0
"""低分且已有额度时最多再等五秒；软筛选不能把正常对话挂到小时兜底。"""


@dataclass(frozen=True)
class TurnGateResult:
    """单个触发门控的判定结果。"""

    decision: TurnGateDecision
    detail: str
    delay_seconds: Optional[float] = None

    @property
    def should_trigger(self) -> bool:
        return self.decision == "trigger"


class DynamicReplyTurnGate:
    """结合成功回复账本与概率软门槛；明确对话保留优先。"""

    def __init__(self, runtime: "MaisakaHeartFlowChatting") -> None:
        self._runtime = runtime
        self._gate = ReplyCreditGate()
        self._soft_wait_started_at: Optional[float] = None

    def observe_messages(self, messages: Sequence[SessionMessage], frequency: float) -> None:
        """在真实接收边界计数，Planner 运行期间到来的消息也不会遗漏。"""
        self._gate.add_messages([message.message_id for message in messages], frequency)

    def reserve_reply(self, target_message_id: str) -> Optional[ReplyReservation]:
        return self._gate.reserve_reply(target_message_id)

    def finish_reply(self, reservation: ReplyReservation, *, success: bool) -> None:
        self._gate.finish_reply(reservation, success=success)

    def evaluate(
        self,
        *,
        pending_messages: Sequence[SessionMessage],
        frequency: float,
        idle_backoff_seconds: float = 0.0,
    ) -> TurnGateResult:
        """返回动态门控的触发判定。

        ``idle_backoff_seconds`` 是空闲退避剩余时间，只拦普通批次；在和麦麦说话的优先批次不受退避影响。
        """

        external_messages = self._filter_external_messages(pending_messages)
        if not external_messages:
            self._soft_wait_started_at = None
            return TurnGateResult(decision="wait", detail="没有新的外部消息 判定=等待更多消息")
        if not self._runtime.chat_stream.is_group_session:
            # 私聊里每条消息都是对麦麦说的，不设门控
            return TurnGateResult(decision="trigger", detail="私聊不设动态门控 判定=进入Planner")

        now = time.time()
        probability = estimate_reply_probability(
            self._build_likelihood_input(external_messages, now, pending_messages=external_messages)
        )
        priority = any(self._is_addressed_to_bot(message) for message in external_messages)
        gate_decision = self._gate.evaluate(
            probability, frequency, priority=priority, normal_blocked_seconds=idle_backoff_seconds,
        )
        if gate_decision.should_trigger:
            if not priority:
                window = self._gate.current_window_state(frequency)
                threshold = (
                    LAST_REPLY_SLOT_MIN_PROBABILITY
                    if window is not None and window.remaining_target_replies <= 1
                    else MIN_REPLY_PROBABILITY
                )
                if probability < threshold:
                    if self._soft_wait_started_at is None:
                        self._soft_wait_started_at = time.monotonic()
                    remaining = SOFT_GATE_MAX_DELAY_SECONDS - (time.monotonic() - self._soft_wait_started_at)
                    if remaining > 0:
                        return TurnGateResult(
                            decision="delay",
                            detail=(
                                f"动态门控: {gate_decision.detail} 软门槛={threshold:.2f} "
                                f"判定=短暂等待更合适的批次 最多再等={remaining:.1f}s"
                            ),
                            delay_seconds=remaining,
                        )
            self._soft_wait_started_at = None
            return TurnGateResult(decision="trigger", detail=f"动态门控: {gate_decision.detail} 判定=进入Planner")
        if gate_decision.delay_seconds is None:
            return TurnGateResult(decision="wait", detail=f"动态门控: {gate_decision.detail} 判定=等待更多消息")
        return TurnGateResult(
            decision="delay",
            detail=f"动态门控: {gate_decision.detail} 空闲退避中 判定=退避结束后复查",
            delay_seconds=gate_decision.delay_seconds,
        )

    def _is_addressed_to_bot(self, message: SessionMessage) -> bool:
        """只使用已确认的 @、昵称标记及真实回复关系，估计概率不代表有人在叫麦麦。"""
        if message.is_at or message.is_mentioned:
            return True
        for component in message.raw_message.components:
            if isinstance(component, AtComponent) and is_bot_self(message.platform, component.target_user_id):
                return True
            if not isinstance(component, ReplyComponent):
                continue
            if component.target_message_sender_id:
                if is_bot_self(message.platform, component.target_message_sender_id):
                    return True
                continue
            target = self._runtime.find_source_message_by_id(component.target_message_id)
            if target is not None and is_bot_self(message.platform, target.message_info.user_info.user_id):
                return True
        return False

    def _filter_external_messages(self, pending_messages: Sequence[SessionMessage]) -> List[SessionMessage]:
        online_accounts = {platform: get_online_bot_accounts(platform) for platform in {m.platform for m in pending_messages}}
        return [
            message
            for message in pending_messages
            if not message.is_notify and not message.is_command
            and str(message.message_info.user_info.user_id).strip() not in online_accounts[message.platform]
        ]

    def _build_likelihood_input(
        self,
        messages: Sequence[SessionMessage],
        now: float,
        *,
        pending_messages: Sequence[SessionMessage],
    ) -> ReplyLikelihoodInput:
        """按当前 runtime 快照为一批外部消息构造概率函数的输入。"""

        texts = [(message.processed_plain_text or "") for message in messages]
        at_other, has_question_mark, placeholder_only = summarize_pending_texts(
            texts,
            has_at_bot=any(message.is_at for message in messages),
        )
        recent_self_count, recent_history_count, seconds_since_bot = self._summarize_recent_history(now)
        # 待处理消息尚未写入内部历史，只补上最近 5 分钟内的消息，避免旧积压持续压低概率。
        # 近期活跃度属于整个聊天流，按全部待处理消息统计。
        current_time = datetime.fromtimestamp(now)
        recent_pending_count = sum(
            message.received_at is not None
            and (current_time - message.received_at).total_seconds() <= RECENT_PRESENCE_WINDOW_SECONDS
            for message in pending_messages
        )
        recent_message_count = recent_history_count + recent_pending_count
        return ReplyLikelihoodInput(
            mention_bot=any(message.is_mentioned for message in messages),
            at_other=at_other,
            has_question_mark=has_question_mark,
            placeholder_only=placeholder_only,
            recent_self_ratio=recent_self_count / max(1, recent_message_count),
            recent_message_count=recent_message_count,
            seconds_since_bot_message=seconds_since_bot,
            pending_count=len(messages),
        )

    def _summarize_recent_history(self, now: float) -> tuple[int, int, float]:
        """返回近期麦麦发言数、近期历史消息数，以及距麦麦上一次发言的秒数。"""

        current_time = datetime.fromtimestamp(now)
        recent_self_count = 0
        recent_total_count = 0
        seconds_since_bot: Optional[float] = None
        for message in reversed(self._runtime._chat_history):
            if not message.count_in_context:
                continue
            is_self_message = message.source == "guided_reply"
            # 外部消息按接收时刻统计活跃度；自身发言仍按本机发送时间计算。
            activity_time = message.timestamp if is_self_message else message.received_at
            if activity_time is None:
                continue
            elapsed_seconds = (current_time - activity_time).total_seconds()
            if is_self_message and seconds_since_bot is None:
                seconds_since_bot = max(0.0, elapsed_seconds)
            if elapsed_seconds > RECENT_PRESENCE_WINDOW_SECONDS:
                # 媒体处理可能乱序完成，历史列表不保证按接收时刻排序。
                continue
            recent_total_count += 1
            if is_self_message:
                recent_self_count += 1
        if seconds_since_bot is None:
            seconds_since_bot = DEFAULT_SECONDS_SINCE_BOT_MESSAGE
        return recent_self_count, recent_total_count, seconds_since_bot

