"""Tests for custom gesture storage and recognition."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from gestures.basic_gesture import GestureName
from gestures.custom_gesture import CustomGestureRecognizer
from gestures.custom_gesture import CustomGestureStore
from gestures.custom_gesture import LandmarkCoordinate
from gestures.custom_gesture import extract_landmark_sample
from gestures.custom_gesture import normalize_landmark_sample
from gestures.custom_gesture import sample_distance


class FakePoint:
    """Simple point matching MediaPipe landmark attributes."""

    def __init__(
        self,
        x_position: float,
        y_position: float,
        z_position: float = 0.0,
    ) -> None:
        self.x = x_position
        self.y = y_position
        self.z = z_position


class FakeLandmarks:
    """Simple MediaPipe-like landmarks object."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.landmark = points


def _sample(offset: float = 0.0) -> tuple[LandmarkCoordinate, ...]:
    return tuple(
        LandmarkCoordinate(
            x=0.1 + offset + index * 0.01,
            y=0.2 + index * 0.005,
            z=index * 0.001,
        )
        for index in range(21)
    )


class CustomGestureTest(unittest.TestCase):
    """Unit tests for custom gesture persistence and matching."""

    def test_store_saves_named_gesture_as_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "custom_gestures.json"
            store = CustomGestureStore(path)

            saved_gesture = store.save_gesture("My Gesture", [_sample()])
            loaded_gestures = store.load()

        self.assertEqual(saved_gesture.name, "My Gesture")
        self.assertEqual(len(loaded_gestures), 1)
        self.assertEqual(loaded_gestures[0].name, "My Gesture")
        self.assertEqual(len(loaded_gestures[0].samples[0]), 21)

    def test_recognizer_matches_saved_custom_gesture(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = CustomGestureStore(Path(temp_dir) / "gestures.json")
            store.save_gesture("Spider Pose", [_sample()])
            recognizer = CustomGestureRecognizer(store=store)

            gesture = recognizer.recognize_sample(_sample())

        self.assertEqual(gesture.name, "Spider Pose")
        self.assertEqual(gesture.label, "Spider Pose")
        self.assertTrue(gesture.custom)
        self.assertGreaterEqual(gesture.confidence, 0.99)

    def test_recognizer_reload_includes_new_json_gestures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gestures.json"
            recognizer_store = CustomGestureStore(path)
            writer_store = CustomGestureStore(path)
            recognizer = CustomGestureRecognizer(store=recognizer_store)

            before_save = recognizer.recognize_sample(_sample())
            time.sleep(0.01)
            writer_store.save_gesture("Fresh Gesture", [_sample()])
            after_save = recognizer.recognize_sample(_sample())

        self.assertEqual(before_save.name, GestureName.UNKNOWN)
        self.assertEqual(after_save.name, "Fresh Gesture")

    def test_extract_landmark_sample_supports_mediapipe_shape(self) -> None:
        points = [FakePoint(0.1, 0.2, 0.3) for _index in range(21)]

        sample = extract_landmark_sample(FakeLandmarks(points))

        self.assertEqual(len(sample), 21)
        self.assertEqual(sample[0], LandmarkCoordinate(0.1, 0.2, 0.3))

    def test_normalized_distance_is_translation_invariant(self) -> None:
        first_sample = _sample(offset=0.0)
        translated_sample = _sample(offset=0.4)

        distance = sample_distance(
            normalize_landmark_sample(first_sample),
            normalize_landmark_sample(translated_sample),
        )

        self.assertAlmostEqual(distance, 0.0)

    def test_store_rejects_empty_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = CustomGestureStore(Path(temp_dir) / "gestures.json")

            with self.assertRaises(ValueError):
                store.save_gesture(" ", [_sample()])


if __name__ == "__main__":
    unittest.main()
