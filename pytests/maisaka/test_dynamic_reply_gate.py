"""十条消息回复账本的确定性回归，不启动应用或模型。"""

from pathlib import Path
from unittest import TestCase

import importlib.util
import sys


_MODULE_NAME = "maibot_reply_credit_gate_regression"
_MODULE_PATH = Path(__file__).resolve().parents[2] / "src/maisaka/turn_trigger/dynamic_gate.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ReplyCreditGate = _MODULE.ReplyCreditGate

NORMAL = 0.2
PRIORITY = 0.8
FREQUENCY = 0.4


def _send(gate, target, *, success=True):
    reservation = gate.reserve_reply(target)
    assert reservation is not None
    gate.finish_reply(reservation, success=success)
    return reservation


class TestReplyCreditGate(TestCase):
    def test_current_three_per_ten_frequency_keeps_its_natural_trigger_points(self):
        gate = ReplyCreditGate()
        replied_at = []
        for index in range(20):
            target = f"m{index}"
            gate.add_messages([target], 0.3)
            if gate.evaluate(NORMAL, 0.3).should_trigger:
                _send(gate, target)
                replied_at.append(index + 1)
        self.assertEqual(replied_at, [4, 7, 10, 14, 17, 20])

    def test_normal_messages_produce_four_successful_replies_per_ten(self):
        gate = ReplyCreditGate()
        replied_at = []
        for index in range(20):
            message_id = f"m{index}"
            gate.add_messages([message_id], FREQUENCY)
            if gate.evaluate(NORMAL, FREQUENCY).should_trigger:
                _send(gate, message_id)
                replied_at.append(index + 1)
        self.assertEqual(replied_at, [3, 5, 8, 10, 13, 15, 18, 20])

    def test_silent_planner_does_not_spend_credit_or_complete_the_window(self):
        gate = ReplyCreditGate()
        gate.add_messages(["m0", "m1", "m2"], FREQUENCY)
        self.assertTrue(gate.evaluate(NORMAL, FREQUENCY).should_trigger)
        self.assertTrue(gate.evaluate(NORMAL, FREQUENCY).should_trigger)
        self.assertAlmostEqual(gate.credit, 1.2)
        gate.add_messages([f"m{index}" for index in range(3, 10)], FREQUENCY)
        self.assertAlmostEqual(gate.credit, 4.0)
        gate.add_messages(["next-window"], FREQUENCY)
        self.assertAlmostEqual(gate.credit, 0.4)
        self.assertFalse(gate.evaluate(NORMAL, FREQUENCY).should_trigger)

    def test_fifth_natural_reply_is_tolerated_but_sixth_is_refused(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(10)], FREQUENCY)
        for index in range(5):
            _send(gate, f"m{index}")
        self.assertAlmostEqual(gate.credit, -1.0)
        self.assertIsNone(gate.reserve_reply("m5"))

    def test_concurrent_reservations_cannot_cross_the_extra_one_limit(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(10)], FREQUENCY)
        reservations = [gate.reserve_reply(f"m{index}") for index in range(5)]
        self.assertTrue(all(reservations))
        self.assertIsNone(gate.reserve_reply("m6"))
        for reservation in reservations:
            gate.finish_reply(reservation, success=True)
        self.assertIsNone(gate.reserve_reply("m7"))

    def test_delayed_send_finishes_in_its_original_window(self):
        gate = ReplyCreditGate()
        gate.add_messages(["m0", "m1", "m2"], FREQUENCY)
        delayed = gate.reserve_reply("m2")
        gate.add_messages([f"m{index}" for index in range(3, 11)], FREQUENCY)
        self.assertAlmostEqual(gate.credit, 0.4)
        gate.finish_reply(delayed, success=True)
        self.assertAlmostEqual(gate.credit, 0.4)
        self.assertEqual(gate._windows[delayed.window_id].replies, 1)

    def test_duplicate_delivery_never_creates_an_extra_window(self):
        gate = ReplyCreditGate()
        ids = [f"m{index}" for index in range(10)]
        self.assertEqual(gate.add_messages([*ids, *ids], FREQUENCY), 10)
        self.assertEqual(gate.add_messages(ids, FREQUENCY), 0)
        self.assertAlmostEqual(gate.credit, 4.0)
        self.assertEqual(len(gate._windows), 1)

    def test_priority_only_bypasses_idle_backoff_within_the_extra_one_allowance(self):
        gate = ReplyCreditGate()
        gate.add_messages(["m0", "m1", "m2"], FREQUENCY)
        blocked = gate.evaluate(NORMAL, FREQUENCY, normal_blocked_seconds=45.0)
        self.assertFalse(blocked.should_trigger)
        self.assertAlmostEqual(blocked.delay_seconds, 45.0)
        self.assertTrue(gate.evaluate(PRIORITY, FREQUENCY, priority=True, normal_blocked_seconds=45.0).should_trigger)
        _send(gate, "m2")
        self.assertTrue(gate.evaluate(PRIORITY, FREQUENCY, priority=True).should_trigger)
        _send(gate, "m1")
        self.assertFalse(gate.evaluate(PRIORITY, FREQUENCY, priority=True).should_trigger)
        self.assertIsNone(gate.reserve_reply("m0"))

    def test_high_probability_alone_does_not_borrow_credit_or_bypass_idle_backoff(self):
        gate = ReplyCreditGate()
        gate.add_messages(["m0"], FREQUENCY)
        decision = gate.evaluate(0.99, FREQUENCY)
        self.assertFalse(decision.should_trigger)
        self.assertFalse(decision.priority)
        self.assertEqual(decision.required_credit, 1.0)

        gate.add_messages(["m1", "m2"], FREQUENCY)
        blocked = gate.evaluate(0.99, FREQUENCY, normal_blocked_seconds=45.0)
        self.assertFalse(blocked.should_trigger)
        self.assertAlmostEqual(blocked.delay_seconds, 45.0)

    def test_zero_frequency_stays_silent_and_full_frequency_counts_each_success(self):
        silent = ReplyCreditGate()
        silent.add_messages([f"m{index}" for index in range(10)], 0.0)
        self.assertFalse(silent.evaluate(PRIORITY, 0.0, priority=True).should_trigger)
        self.assertIsNone(silent.reserve_reply("m0"))
        always = ReplyCreditGate()
        for index in range(10):
            target = f"m{index}"
            always.add_messages([target], 1.0)
            self.assertTrue(always.evaluate(NORMAL, 1.0).should_trigger)
            _send(always, target)
        self.assertAlmostEqual(always.credit, 0.0)

    def test_unknown_reply_anchor_still_uses_the_current_window_limit(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(10)], FREQUENCY)
        for index in range(5):
            _send(gate, f"bot-history-anchor-{index}")
        self.assertIsNone(gate.reserve_reply("another-history-anchor"))

    def test_failed_natural_send_releases_the_slot_without_counting_a_reply(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(10)], FREQUENCY)
        reservation = gate.reserve_reply("m9")
        self.assertIsNotNone(reservation)
        self.assertAlmostEqual(gate.credit, 3.0)
        gate.finish_reply(reservation, success=False)
        self.assertAlmostEqual(gate.credit, 4.0)
        _send(gate, "m9")
        self.assertAlmostEqual(gate.credit, 3.0)

    def test_silent_closed_windows_do_not_add_credit_to_the_next_window(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(20)], FREQUENCY)
        gate.add_messages(["new0"], FREQUENCY)
        self.assertAlmostEqual(gate.credit, 0.4)
        self.assertFalse(gate.evaluate(NORMAL, FREQUENCY).should_trigger)
        gate.add_messages(["new1", "new2"], FREQUENCY)
        self.assertTrue(gate.evaluate(NORMAL, FREQUENCY).should_trigger)
        _send(gate, "new2")
        self.assertAlmostEqual(gate.credit, 0.2)

    def test_unanswered_closed_windows_are_bounded_without_a_makeup_timer(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(1000)], FREQUENCY)
        self.assertEqual(len(gate._windows), _MODULE.RETAINED_SETTLED_WINDOWS)
        self.assertNotIn("m0", gate._message_windows)
        self.assertAlmostEqual(gate.credit, 4.0)

    def test_inflight_natural_reply_survives_old_window_cleanup(self):
        gate = ReplyCreditGate()
        gate.add_messages([f"m{index}" for index in range(10)], FREQUENCY)
        reservation = gate.reserve_reply("m9")
        self.assertIsNotNone(reservation)
        gate.add_messages([f"later-{index}" for index in range(210)], FREQUENCY)
        self.assertIn(reservation.window_id, gate._windows)
        gate.finish_reply(reservation, success=True)
        self.assertNotIn(reservation.window_id, gate._windows)
        self.assertEqual(len(gate._windows), _MODULE.RETAINED_SETTLED_WINDOWS)
        self.assertAlmostEqual(gate.credit, 4.0)
