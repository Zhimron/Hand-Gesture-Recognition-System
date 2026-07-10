"""Tests for basic gesture recognition."""

from __future__ import annotations

import unittest

from gestures.basic_gesture import BasicGestureRecognizer
from gestures.basic_gesture import GestureName
from vision.finger_state import FINGER_ORDER
from vision.finger_state import FingerName
from vision.finger_state import FingerState
from vision.finger_state import FingerStateResult


def _finger_states(
    open_fingers: set[FingerName],
) -> tuple[FingerStateResult, ...]:
    return tuple(
        FingerStateResult(
            finger=finger,
            state=(
                FingerState.OPEN
                if finger in open_fingers
                else FingerState.CLOSED
            ),
        )
        for finger in FINGER_ORDER
    )


class BasicGestureRecognizerTest(unittest.TestCase):
    """Unit tests for the fixed set of supported gestures."""

    def setUp(self) -> None:
        """Create the recognizer under test."""
        self.recognizer = BasicGestureRecognizer()

    def test_recognizes_open_palm(self) -> None:
        gesture = self.recognizer.recognize(_finger_states(set(FINGER_ORDER)))

        self.assertEqual(gesture.name, GestureName.OPEN_PALM)

    def test_recognizes_closed_fist(self) -> None:
        gesture = self.recognizer.recognize(_finger_states(set()))

        self.assertEqual(gesture.name, GestureName.CLOSED_FIST)

    def test_recognizes_peace_sign(self) -> None:
        gesture = self.recognizer.recognize(
            _finger_states({FingerName.INDEX, FingerName.MIDDLE})
        )

        self.assertEqual(gesture.name, GestureName.PEACE_SIGN)

    def test_recognizes_pointing_up(self) -> None:
        gesture = self.recognizer.recognize(
            _finger_states({FingerName.INDEX})
        )

        self.assertEqual(gesture.name, GestureName.POINTING_UP)

    def test_recognizes_thumbs_up(self) -> None:
        gesture = self.recognizer.recognize(
            _finger_states({FingerName.THUMB})
        )

        self.assertEqual(gesture.name, GestureName.THUMBS_UP)

    def test_unknown_for_unsupported_pattern(self) -> None:
        gesture = self.recognizer.recognize(
            _finger_states({FingerName.THUMB, FingerName.PINKY})
        )

        self.assertEqual(gesture.name, GestureName.UNKNOWN)

    def test_unknown_when_finger_state_is_missing(self) -> None:
        gesture = self.recognizer.recognize(
            [
                FingerStateResult(
                    finger=FingerName.INDEX,
                    state=FingerState.OPEN,
                )
            ]
        )

        self.assertEqual(gesture.name, GestureName.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
