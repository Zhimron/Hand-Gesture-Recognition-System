"""Custom gesture storage, recording, and recognition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Sequence

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName


DEFAULT_CUSTOM_GESTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "custom_gestures.json"
)
CUSTOM_GESTURE_SCHEMA_VERSION = 1
REQUIRED_LANDMARKS = 21


@dataclass(frozen=True)
class LandmarkCoordinate:
    """Normalized hand landmark coordinate."""

    x: float
    y: float
    z: float = 0.0


LandmarkSample = tuple[LandmarkCoordinate, ...]


@dataclass(frozen=True)
class CustomGesture:
    """Saved custom gesture template."""

    name: str
    samples: tuple[LandmarkSample, ...]


class CustomGestureStore:
    """Persist custom gestures to JSON and reload them when changed."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CUSTOM_GESTURE_PATH,
    ) -> None:
        self.path = Path(path)
        self._cached_mtime_ns: int | None = None
        self._cached_gestures: tuple[CustomGesture, ...] = ()

    def load(self) -> tuple[CustomGesture, ...]:
        """Load all gestures from disk."""
        if not self.path.exists():
            self._cached_mtime_ns = None
            self._cached_gestures = ()
            return self._cached_gestures

        raw_data = json.loads(self.path.read_text(encoding="utf-8"))
        gestures = tuple(self._parse_gesture(item) for item in raw_data.get(
            "gestures",
            [],
        ))
        self._cached_mtime_ns = self.path.stat().st_mtime_ns
        self._cached_gestures = gestures
        return gestures

    def load_if_changed(self) -> tuple[CustomGesture, ...]:
        """Reload gestures only when the JSON file changed."""
        if not self.path.exists():
            if self._cached_mtime_ns is not None or self._cached_gestures:
                self._cached_mtime_ns = None
                self._cached_gestures = ()
            return self._cached_gestures

        current_mtime_ns = self.path.stat().st_mtime_ns
        if current_mtime_ns != self._cached_mtime_ns:
            return self.load()
        return self._cached_gestures

    def save_gesture(
        self,
        name: str,
        samples: Sequence[Sequence[LandmarkCoordinate]],
    ) -> CustomGesture:
        """Save a named gesture, replacing any previous gesture of that name."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Custom gesture name cannot be empty.")

        cleaned_samples = tuple(
            tuple(sample)
            for sample in samples
            if len(sample) >= REQUIRED_LANDMARKS
        )
        if not cleaned_samples:
            raise ValueError("At least one complete landmark sample is required.")

        gestures_by_name = {
            gesture.name: gesture for gesture in self.load_if_changed()
        }
        gesture = CustomGesture(
            name=normalized_name,
            samples=cleaned_samples,
        )
        gestures_by_name[normalized_name] = gesture

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CUSTOM_GESTURE_SCHEMA_VERSION,
            "gestures": [
                self._gesture_to_json(saved_gesture)
                for saved_gesture in gestures_by_name.values()
            ],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._cached_mtime_ns = self.path.stat().st_mtime_ns
        self._cached_gestures = tuple(gestures_by_name.values())
        return gesture

    @staticmethod
    def _parse_gesture(raw_gesture: dict[str, Any]) -> CustomGesture:
        return CustomGesture(
            name=str(raw_gesture.get("name", "")).strip(),
            samples=tuple(
                tuple(
                    LandmarkCoordinate(
                        x=float(raw_point.get("x", 0.0)),
                        y=float(raw_point.get("y", 0.0)),
                        z=float(raw_point.get("z", 0.0)),
                    )
                    for raw_point in raw_sample
                )
                for raw_sample in raw_gesture.get("samples", [])
            ),
        )

    @staticmethod
    def _gesture_to_json(gesture: CustomGesture) -> dict[str, Any]:
        return {
            "name": gesture.name,
            "samples": [
                [
                    {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                    }
                    for point in sample
                ]
                for sample in gesture.samples
            ],
        }


class CustomGestureRecognizer:
    """Recognize saved custom gestures from hand landmark templates."""

    def __init__(
        self,
        store: CustomGestureStore | None = None,
        max_distance: float = 0.18,
        min_confidence: float = 0.6,
    ) -> None:
        self._store = store or CustomGestureStore()
        self._max_distance = max_distance
        self._min_confidence = min_confidence

    def recognize(self, landmarks: object) -> DetectedGesture:
        """Recognize a custom gesture from raw MediaPipe landmarks."""
        return self.recognize_sample(extract_landmark_sample(landmarks))

    def recognize_sample(self, sample: LandmarkSample) -> DetectedGesture:
        """Recognize a custom gesture from an extracted landmark sample."""
        if len(sample) < REQUIRED_LANDMARKS:
            return DetectedGesture(GestureName.UNKNOWN)

        normalized_sample = normalize_landmark_sample(sample)
        best_name = ""
        best_distance: float | None = None

        for gesture in self._store.load_if_changed():
            if not gesture.name:
                continue

            for saved_sample in gesture.samples:
                if len(saved_sample) < REQUIRED_LANDMARKS:
                    continue

                distance = sample_distance(
                    normalized_sample,
                    normalize_landmark_sample(saved_sample),
                )
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_name = gesture.name

        if best_distance is None:
            return DetectedGesture(GestureName.UNKNOWN)

        confidence = max(0.0, 1.0 - (best_distance / self._max_distance))
        if confidence < self._min_confidence:
            return DetectedGesture(GestureName.UNKNOWN)

        return DetectedGesture(best_name, confidence=confidence, custom=True)


def extract_landmark_sample(landmarks: object) -> LandmarkSample:
    """Extract a serializable landmark sample from MediaPipe landmarks."""
    raw_points = getattr(landmarks, "landmark", None)
    if raw_points is None and isinstance(landmarks, list):
        raw_points = landmarks

    if raw_points is None:
        return ()

    return tuple(
        LandmarkCoordinate(
            x=float(getattr(point, "x", 0.0)),
            y=float(getattr(point, "y", 0.0)),
            z=float(getattr(point, "z", 0.0)),
        )
        for point in raw_points
    )


def normalize_landmark_sample(sample: LandmarkSample) -> LandmarkSample:
    """Normalize landmarks relative to the wrist and hand size."""
    if not sample:
        return ()

    wrist = sample[0]
    relative_points = tuple(
        LandmarkCoordinate(
            x=point.x - wrist.x,
            y=point.y - wrist.y,
            z=point.z - wrist.z,
        )
        for point in sample
    )
    scale = max(
        (
            hypot(hypot(point.x, point.y), point.z)
            for point in relative_points
        ),
        default=1.0,
    )
    if scale <= 0:
        scale = 1.0

    return tuple(
        LandmarkCoordinate(
            x=point.x / scale,
            y=point.y / scale,
            z=point.z / scale,
        )
        for point in relative_points
    )


def sample_distance(first: LandmarkSample, second: LandmarkSample) -> float:
    """Return the average normalized landmark distance between samples."""
    point_count = min(len(first), len(second))
    if point_count == 0:
        return float("inf")

    total = 0.0
    for first_point, second_point in zip(first[:point_count], second[:point_count]):
        total += hypot(
            hypot(first_point.x - second_point.x, first_point.y - second_point.y),
            first_point.z - second_point.z,
        )
    return total / point_count
