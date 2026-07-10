"""Tests for finger open/closed state detection."""

from __future__ import annotations

import unittest

from vision.finger_state import FINGER_ORDER
from vision.finger_state import FingerName
from vision.finger_state import FingerState
from vision.finger_state import FingerStateDetector
from vision.finger_state import FingerStateResult


class FakePoint:
    """Simple normalized landmark point."""

    def __init__(self, x_position: float, y_position: float) -> None:
        self.x = x_position
        self.y = y_position


class FakeLandmarks:
    """Container matching MediaPipe's landmark shape."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.landmark = points


def _blank_hand() -> list[FakePoint]:
    return [FakePoint(0.5, 0.5) for _index in range(21)]


def _expected_states(state: FingerState) -> tuple[FingerStateResult, ...]:
    return tuple(
        FingerStateResult(finger=finger, state=state)
        for finger in FINGER_ORDER
    )


class FingerStateDetectorTest(unittest.TestCase):
    """Unit tests for FingerStateDetector."""

    def test_detects_open_fingers(self) -> None:
        points = _blank_hand()
        points[0] = FakePoint(0.5, 0.9)
        points[3] = FakePoint(0.35, 0.65)
        points[4] = FakePoint(0.25, 0.6)
        points[5] = FakePoint(0.45, 0.65)
        points[6] = FakePoint(0.45, 0.45)
        points[8] = FakePoint(0.45, 0.2)
        points[10] = FakePoint(0.5, 0.4)
        points[12] = FakePoint(0.5, 0.15)
        points[14] = FakePoint(0.55, 0.45)
        points[16] = FakePoint(0.55, 0.2)
        points[18] = FakePoint(0.6, 0.5)
        points[20] = FakePoint(0.6, 0.25)

        states = FingerStateDetector().detect(FakeLandmarks(points))

        self.assertEqual(states, _expected_states(FingerState.OPEN))

    def test_detects_closed_fingers(self) -> None:
        points = _blank_hand()
        points[0] = FakePoint(0.5, 0.9)
        points[3] = FakePoint(0.37, 0.65)
        points[4] = FakePoint(0.43, 0.65)
        points[5] = FakePoint(0.45, 0.65)
        points[6] = FakePoint(0.45, 0.45)
        points[8] = FakePoint(0.46, 0.6)
        points[10] = FakePoint(0.5, 0.4)
        points[12] = FakePoint(0.51, 0.58)
        points[14] = FakePoint(0.55, 0.45)
        points[16] = FakePoint(0.55, 0.6)
        points[18] = FakePoint(0.6, 0.5)
        points[20] = FakePoint(0.59, 0.62)

        states = FingerStateDetector().detect(points)

        self.assertEqual(states, _expected_states(FingerState.CLOSED))

    def test_returns_closed_states_for_missing_landmarks(self) -> None:
        states = FingerStateDetector().detect([FakePoint(0.0, 0.0)])

        self.assertEqual(states, _expected_states(FingerState.CLOSED))

    def test_preserves_finger_order(self) -> None:
        states = FingerStateDetector().detect([])

        self.assertEqual(
            [state.finger for state in states],
            [
                FingerName.THUMB,
                FingerName.INDEX,
                FingerName.MIDDLE,
                FingerName.RING,
                FingerName.PINKY,
            ],
        )


if __name__ == "__main__":
    unittest.main()
