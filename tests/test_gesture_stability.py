"""Tests for gesture stability filtering."""

from __future__ import annotations

import unittest

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName
from gestures.stability import GestureStabilityConfig
from gestures.stability import GestureStabilizer


class GestureStabilizerTest(unittest.TestCase):
    """Unit tests for real-time gesture smoothing."""

    def test_majority_vote_stabilizes_after_repeated_samples(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(window_ms=90, min_votes=2)
        )

        first = stabilizer.update(
            DetectedGesture(GestureName.OPEN_PALM, confidence=1.0),
            confidence=0.9,
            timestamp_ms=0,
        )
        second = stabilizer.update(
            DetectedGesture(GestureName.OPEN_PALM, confidence=1.0),
            confidence=0.8,
            timestamp_ms=30,
        )

        self.assertEqual(first.name, GestureName.UNKNOWN)
        self.assertEqual(second.name, GestureName.OPEN_PALM)
        self.assertAlmostEqual(second.confidence, 0.85)

    def test_single_noisy_sample_does_not_flicker_stable_gesture(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(window_ms=90, hold_ms=100, min_votes=2)
        )
        stabilizer.update(
            DetectedGesture(GestureName.CLOSED_FIST, confidence=1.0),
            confidence=0.9,
            timestamp_ms=0,
        )
        stable = stabilizer.update(
            DetectedGesture(GestureName.CLOSED_FIST, confidence=1.0),
            confidence=0.9,
            timestamp_ms=30,
        )

        noisy = stabilizer.update(
            DetectedGesture(GestureName.PEACE_SIGN, confidence=1.0),
            confidence=0.9,
            timestamp_ms=60,
        )

        self.assertEqual(stable.name, GestureName.CLOSED_FIST)
        self.assertEqual(noisy.name, GestureName.CLOSED_FIST)

    def test_low_confidence_detection_is_ignored(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(min_confidence=0.7, min_votes=1)
        )

        gesture = stabilizer.update(
            DetectedGesture(GestureName.THUMBS_UP, confidence=1.0),
            confidence=0.4,
            timestamp_ms=0,
        )

        self.assertEqual(gesture.name, GestureName.UNKNOWN)

    def test_unknown_detection_is_ignored(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(min_confidence=0.5, min_votes=1)
        )

        gesture = stabilizer.update(
            DetectedGesture(GestureName.UNKNOWN, confidence=1.0),
            confidence=1.0,
            timestamp_ms=0,
        )

        self.assertEqual(gesture.name, GestureName.UNKNOWN)

    def test_hold_expires_within_latency_budget(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(window_ms=1, hold_ms=100, min_votes=1)
        )
        stabilizer.update(
            DetectedGesture(GestureName.POINTING_UP, confidence=1.0),
            confidence=0.9,
            timestamp_ms=0,
        )

        held = stabilizer.update(
            DetectedGesture(GestureName.UNKNOWN, confidence=0.0),
            confidence=0.0,
            timestamp_ms=80,
        )
        expired = stabilizer.update(
            DetectedGesture(GestureName.UNKNOWN, confidence=0.0),
            confidence=0.0,
            timestamp_ms=101,
        )

        self.assertEqual(held.name, GestureName.POINTING_UP)
        self.assertEqual(expired.name, GestureName.UNKNOWN)

    def test_config_clamps_windows_to_100ms_latency_budget(self) -> None:
        config = GestureStabilityConfig(
            max_latency_ms=100,
            window_ms=250,
            hold_ms=250,
        )

        self.assertEqual(config.window_ms, 100)
        self.assertEqual(config.hold_ms, 100)

    def test_custom_gesture_labels_are_stabilized(self) -> None:
        stabilizer = GestureStabilizer(
            GestureStabilityConfig(min_confidence=0.5, min_votes=1)
        )

        gesture = stabilizer.update(
            DetectedGesture("Custom Wave", confidence=0.9, custom=True),
            confidence=0.9,
            timestamp_ms=0,
        )

        self.assertEqual(gesture.name, "Custom Wave")
        self.assertEqual(gesture.label, "Custom Wave")
        self.assertTrue(gesture.custom)


if __name__ == "__main__":
    unittest.main()
