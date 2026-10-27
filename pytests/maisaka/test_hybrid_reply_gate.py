"""混合回复门控的额度边界与概率筛选，不启动应用或模型。"""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase

import ast
import sys

from pytests.maisaka.test_dynamic_reply_gate import ReplyCreditGate


ROOT = Path(__file__).resolve().parents[2]
PROBABILITY = {"value": 0.3}
CLOCK = {"now": 0.0}


def _load_gate():
    """编译真实门控类，排除会启动配置和插件系统的模块级导入。"""
    path = ROOT / "src/maisaka/turn_trigger/gates.py"
    tree = ast.parse(path.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"TurnGateResult", "DynamicReplyTurnGate"}
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"MIN_REPLY_PROBABILITY", "LAST_REPLY_SLOT_MIN_PROBABILITY", "SOFT_GATE_MAX_DELAY_SECONDS"}
            for target in node.targets
        )
    ]
    module = ModuleType("maibot_hybrid_reply_gate_regression")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "dataclass": dataclass,
        "ReplyCreditGate": ReplyCreditGate,
        "time": SimpleNamespace(time=lambda: CLOCK["now"], monotonic=lambda: CLOCK["now"]),
        "estimate_reply_probability": lambda value: PROBABILITY["value"],
    })
    unit = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace["DynamicReplyTurnGate"]


DynamicReplyTurnGate = _load_gate()


class TestHybridReplyGate(TestCase):
    def setUp(self) -> None:
        runtime = SimpleNamespace(chat_stream=SimpleNamespace(is_group_session=True))
        self.gate = DynamicReplyTurnGate(runtime)
        self.gate._filter_external_messages = lambda messages: list(messages)
        self.gate._build_likelihood_input = lambda *args, **kwargs: None
        self.gate._is_addressed_to_bot = lambda message: message.addressed
        self.messages = [SimpleNamespace(message_id=f"m{i}", addressed=False) for i in range(10)]
        PROBABILITY["value"] = 0.3
        CLOCK["now"] = 0.0

    def _evaluate(self, *, backoff=0.0):
        return self.gate.evaluate(
            pending_messages=self.messages,
            frequency=0.4,
            idle_backoff_seconds=backoff,
        )

    def _send(self, index):
        reservation = self.gate.reserve_reply(self.messages[index].message_id)
        self.assertIsNotNone(reservation)
        self.gate.finish_reply(reservation, success=True)

    def test_low_score_with_credit_is_rechecked_and_released_within_five_seconds(self):
        self.gate.observe_messages(self.messages, 0.4)
        PROBABILITY["value"] = 0.08
        first = self._evaluate()
        self.assertEqual(first.decision, "delay")
        self.assertEqual(first.delay_seconds, 5.0)
        CLOCK["now"] = 3.0
        self.assertEqual(self._evaluate().delay_seconds, 2.0)
        CLOCK["now"] = 5.0
        self.assertTrue(self._evaluate().should_trigger)
        self.assertEqual(self.gate._gate.credit, 4.0)

    def test_last_slot_soft_wait_also_expires_without_hourly_fallback(self):
        self.gate.observe_messages(self.messages, 0.4)
        for index in range(3):
            self._send(index)
        PROBABILITY["value"] = 0.10
        self.assertEqual(self._evaluate().delay_seconds, 5.0)
        CLOCK["now"] = 5.0
        self.assertTrue(self._evaluate().should_trigger)

    def test_first_accrued_credit_is_not_the_last_planned_slot(self):
        self.gate.observe_messages(self.messages[:3], 0.4)
        state = self.gate._gate.current_window_state(0.4)
        self.assertEqual(state.planned_target, 4)
        self.assertEqual(state.remaining_target_replies, 4)
        self.assertAlmostEqual(state.credit, 1.2)
        PROBABILITY["value"] = 0.2
        self.assertTrue(self._evaluate().should_trigger)

    def test_low_probability_batch_does_not_spend_accumulated_credit(self):
        self.gate.observe_messages(self.messages, 0.4)
        PROBABILITY["value"] = 0.14
        self.assertFalse(self._evaluate().should_trigger)
        self.assertAlmostEqual(self.gate._gate.credit, 4.0)
        PROBABILITY["value"] = 0.15
        self.assertTrue(self._evaluate().should_trigger)

    def test_last_planned_slot_waits_for_a_high_probability_batch(self):
        self.gate.observe_messages(self.messages, 0.4)
        for index in range(3):
            self._send(index)
        PROBABILITY["value"] = 0.59
        self.assertFalse(self._evaluate().should_trigger)
        PROBABILITY["value"] = 0.60
        self.assertTrue(self._evaluate().should_trigger)

    def test_inflight_reservations_keep_the_last_slot_protected(self):
        self.gate.observe_messages(self.messages, 0.4)
        reservations = [self.gate.reserve_reply(message.message_id) for message in self.messages[:3]]
        self.assertFalse(self._evaluate().should_trigger)
        self.gate.finish_reply(reservations[-1], success=False)
        self.assertTrue(self._evaluate().should_trigger)
        self.assertEqual(self.gate._gate.current_window_state(0.4).remaining_target_replies, 2)

    def test_explicit_address_bypasses_soft_filter_and_backoff(self):
        self.gate.observe_messages(self.messages, 0.4)
        for index in range(3):
            self._send(index)
        self.messages[-1].addressed = True
        PROBABILITY["value"] = 0.01
        self.assertTrue(self._evaluate(backoff=45.0).should_trigger)

    def test_high_probability_does_not_create_credit(self):
        self.gate.observe_messages(self.messages[:1], 0.4)
        PROBABILITY["value"] = 0.99
        self.assertFalse(self._evaluate().should_trigger)

    def test_planned_balance_restarts_at_the_next_ten_message_window(self):
        self.gate.observe_messages(self.messages, 0.4)
        for index in range(3):
            self._send(index)
        self.gate.observe_messages([SimpleNamespace(message_id="next-window")], 0.4)
        state = self.gate._gate.current_window_state(0.4)
        self.assertEqual(state.message_count, 1)
        self.assertEqual(state.remaining_target_replies, 4)
        self.assertAlmostEqual(state.credit, 0.4)
