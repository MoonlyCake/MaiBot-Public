"""按每十条外部消息限制自然回复，允许额外一次，不强制补足次数。"""

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Sequence, Set

MESSAGE_WINDOW_SIZE = 10
REPLY_OVERFLOW_TOLERANCE = 1
PRIORITY_CREDIT_FLOOR = 0.0
CREDIT_EPSILON = 1e-6
MAX_COUNTED_MESSAGE_IDS = 2000
RETAINED_SETTLED_WINDOWS = 20


@dataclass(frozen=True, slots=True)
class ReplyCreditDecision:
    """一次 Planner 入场判定；计账发生在实际发送成功时。"""

    should_trigger: bool
    credit: float
    required_credit: float
    probability: float
    priority: bool
    delay_seconds: Optional[float]

    @property
    def detail(self) -> str:
        delay_text = "" if self.delay_seconds is None else f" 预计等待={self.delay_seconds:.1f}s"
        return (
            f"额度={self.credit:.2f} 所需={self.required_credit:.2f} 概率={self.probability:.2f} "
            f"优先={'是' if self.priority else '否'}{delay_text}"
        )


@dataclass(frozen=True, slots=True)
class ReplyReservation:
    """在任何模型或发送 await 之前占用一个窗口名额。"""

    window_id: int
    target_message_id: str
    reservation_id: int


@dataclass(frozen=True, slots=True)
class ReplyWindowState:
    """当前窗口的累计额度与完整十条消息的规划名额。"""

    window_id: int
    message_count: int
    planned_target: int
    replies: int
    in_flight: int
    credit: float

    @property
    def remaining_target_replies(self) -> int:
        return max(0, self.planned_target - self.replies - self.in_flight)


@dataclass(slots=True)
class _ReplyWindow:
    message_ids: List[str] = field(default_factory=list)
    entitlement: Decimal = Decimal(0)
    replies: int = 0
    reservations: Dict[int, ReplyReservation] = field(default_factory=dict)

    @property
    def target(self) -> int:
        return int(self.entitlement)

    @property
    def closed(self) -> bool:
        return len(self.message_ids) == MESSAGE_WINDOW_SIZE


class ReplyCreditGate:
    """单个聊天流的十条消息窗口与成功回复账本。

    窗口内没用完的额度不强制补答，也不转移到下个窗口。发送预留独立于
    成功计数，同时启动的多个 reply 也不能把允许的额外一次变成连续超发。
    """

    def __init__(self) -> None:
        self._windows: Dict[int, _ReplyWindow] = {}
        self._message_windows: Dict[str, int] = {}
        self._current_window_id: Optional[int] = None
        self._next_reservation_id = 0
        self._counted_message_ids: Set[str] = set()
        self._counted_message_order: Deque[str] = deque()

    @property
    def credit(self) -> float:
        if self._current_window_id is None:
            return 0.0
        window = self._windows[self._current_window_id]
        return float(window.entitlement) - window.replies - len(window.reservations)

    def current_window_state(self, frequency: float) -> Optional[ReplyWindowState]:
        """读取规划余额；未到齐的消息按当前频率估算完整窗口，不增加实际额度。"""
        if self._current_window_id is None:
            return None
        window = self._windows[self._current_window_id]
        remaining_message_count = MESSAGE_WINDOW_SIZE - len(window.message_ids)
        planned_entitlement = window.entitlement + Decimal(str(_clamp_frequency(frequency))) * remaining_message_count
        return ReplyWindowState(
            window_id=self._current_window_id,
            message_count=len(window.message_ids),
            planned_target=int(planned_entitlement),
            replies=window.replies,
            in_flight=len(window.reservations),
            credit=self.credit,
        )

    def add_messages(self, message_ids: Sequence[str], frequency: float) -> int:
        """每条新消息进入自己的固定窗口；重复投递不重复计数。"""
        entitlement = Decimal(str(_clamp_frequency(frequency)))
        new_count = 0
        for message_id in message_ids:
            if not self._take_uncounted_message_id(message_id):
                continue
            if self._current_window_id is None or self._windows[self._current_window_id].closed:
                self._current_window_id = 0 if self._current_window_id is None else self._current_window_id + 1
                self._windows[self._current_window_id] = _ReplyWindow()
            window = self._windows[self._current_window_id]
            window.message_ids.append(message_id)
            window.entitlement += entitlement
            self._message_windows[message_id] = self._current_window_id
            new_count += 1
        self._prune_settled_windows()
        return new_count

    def reserve_reply(self, target_message_id: str) -> Optional[ReplyReservation]:
        """发送前占名额：自然回复容忍多一次，未回复的额度不要求补足。"""
        window_id = self._message_windows.get(target_message_id, self._current_window_id)
        if window_id is None:
            return None
        window = self._windows[window_id]
        if window.entitlement <= 0:
            return None
        used = window.replies + len(window.reservations)
        limit = window.target + REPLY_OVERFLOW_TOLERANCE
        if used >= limit:
            return None
        self._next_reservation_id += 1
        reservation = ReplyReservation(window_id, target_message_id, self._next_reservation_id)
        window.reservations[reservation.reservation_id] = reservation
        return reservation

    def finish_reply(self, reservation: ReplyReservation, *, success: bool) -> None:
        """仅成功发送计一次；失败释放名额，迟到的结果仍归属于原窗口。"""
        window = self._windows[reservation.window_id]
        window.reservations.pop(reservation.reservation_id)
        if success:
            window.replies += 1
        self._prune_settled_windows()

    def record_reply(self, target_message_id: str = "") -> None:
        """兼容直接记账的调用；实际发送链路必须先 reserve，再 finish。"""
        reservation = self.reserve_reply(target_message_id)
        if reservation is None:
            raise RuntimeError("当前消息窗口没有可用的回复名额")
        self.finish_reply(reservation, success=True)

    def evaluate(
        self, probability: float, frequency: float, *, priority: bool = False,
        normal_blocked_seconds: float = 0.0,
    ) -> ReplyCreditDecision:
        """只判定 Planner 入场额度；概率软门槛与明确接话优先由外层门控处理。

        发送时再次预留名额，不能靠重复工具调用超额。
        """
        normalized_frequency = _clamp_frequency(frequency)
        required_credit = PRIORITY_CREDIT_FLOOR if priority else 1.0
        blocked_seconds = 0.0 if priority else max(0.0, normal_blocked_seconds)
        credit = self.credit
        has_credit = self._current_window_id is not None and (
            normalized_frequency >= 1.0 or credit + CREDIT_EPSILON >= required_credit
        )
        if normalized_frequency <= 0.0 or not has_credit:
            return ReplyCreditDecision(False, credit, required_credit, probability, priority, None)
        if blocked_seconds > 0.0:
            return ReplyCreditDecision(False, credit, required_credit, probability, priority, blocked_seconds)
        return ReplyCreditDecision(True, credit, required_credit, probability, priority, None)

    def _take_uncounted_message_id(self, message_id: str) -> bool:
        if not message_id or message_id in self._counted_message_ids or message_id in self._message_windows:
            return False
        self._counted_message_ids.add(message_id)
        self._counted_message_order.append(message_id)
        while len(self._counted_message_order) > MAX_COUNTED_MESSAGE_IDS:
            self._counted_message_ids.discard(self._counted_message_order.popleft())
        return True

    def _prune_settled_windows(self) -> None:
        """清理很早的闭合窗口，不追补欠额；在途回复与当前窗口始终保留。"""
        if self._current_window_id is None:
            return
        cutoff = self._current_window_id - RETAINED_SETTLED_WINDOWS
        removable = [window_id for window_id, window in self._windows.items()
                     if window_id <= cutoff and window.closed
                     and not window.reservations]
        for window_id in removable:
            window = self._windows.pop(window_id)
            for message_id in window.message_ids:
                self._message_windows.pop(message_id)


def _clamp_frequency(frequency: float) -> float:
    return min(1.0, max(0.0, frequency))
