"""Gesture stability filtering for real-time recognition."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureIdentifier
from gestures.basic_gesture import GestureName


@dataclass(frozen=True)
class GestureStabilityConfig:
    """Runtime limits for gesture smoothing."""

    max_latency_ms: int = 100
    window_ms: int = 90
    hold_ms: int = 100
    min_confidence: float = 0.55
    min_votes: int = 2

    def __post_init__(self) -> None:
        """Clamp smoothing windows to the latency budget."""
        max_latency_ms = max(1, self.max_latency_ms)
        object.__setattr__(self, "max_latency_ms", max_latency_ms)
        object.__setattr__(
            self,
            "window_ms",
            max(1, min(self.window_ms, max_latency_ms)),
        )
        object.__setattr__(
            self,
            "hold_ms",
            max(0, min(self.hold_ms, max_latency_ms)),
        )
        object.__setattr__(
            self,
            "min_confidence",
            max(0.0, min(self.min_confidence, 1.0)),
        )
        object.__setattr__(self, "min_votes", max(1, self.min_votes))


@dataclass(frozen=True)
class GestureSample:
    """One recent raw gesture sample."""

    name: GestureIdentifier
    confidence: float
    timestamp_ms: int


class GestureStabilizer:
    """Smooth raw gesture predictions using recent confident votes."""

    def __init__(
        self,
        config: GestureStabilityConfig | None = None,
    ) -> None:
        self.config = config or GestureStabilityConfig()
        self._samples: deque[GestureSample] = deque()
        self._stable_gesture = DetectedGesture(GestureName.UNKNOWN)
        self._stable_timestamp_ms = 0

    def update(
        self,
        raw_gesture: DetectedGesture,
        confidence: float,
        timestamp_ms: int | None = None,
    ) -> DetectedGesture:
        """Return the currently stable gesture for a new raw sample."""
        now_ms = timestamp_ms if timestamp_ms is not None else self._now_ms()
        self._prune(now_ms)

        combined_confidence = min(raw_gesture.confidence, confidence)
        if self._is_confident(raw_gesture, combined_confidence):
            self._samples.append(
                GestureSample(
                    name=raw_gesture.name,
                    confidence=combined_confidence,
                    timestamp_ms=now_ms,
                )
            )

        candidate = self._majority_candidate()
        if candidate is not None:
            self._stable_gesture = candidate
            self._stable_timestamp_ms = now_ms
            return self._stable_gesture

        if self._should_hold(now_ms):
            return self._stable_gesture

        return DetectedGesture(GestureName.UNKNOWN)

    def reset(self) -> None:
        """Clear all smoothing history."""
        self._samples.clear()
        self._stable_gesture = DetectedGesture(GestureName.UNKNOWN)
        self._stable_timestamp_ms = 0

    def _majority_candidate(self) -> DetectedGesture | None:
        if not self._samples:
            return None

        counts = Counter(sample.name for sample in self._samples)
        top_name, top_count = counts.most_common(1)[0]
        if top_count < self.config.min_votes:
            return None

        if top_count / len(self._samples) <= 0.5:
            return None

        confidence = self._average_confidence(top_name)
        return DetectedGesture(
            name=top_name,
            confidence=confidence,
            custom=not isinstance(top_name, GestureName),
        )

    def _average_confidence(self, gesture_name: GestureIdentifier) -> float:
        matching_samples = [
            sample for sample in self._samples if sample.name == gesture_name
        ]
        if not matching_samples:
            return 0.0

        confidence = sum(sample.confidence for sample in matching_samples)
        return confidence / len(matching_samples)

    def _prune(self, now_ms: int) -> None:
        oldest_timestamp_ms = now_ms - self.config.window_ms
        while (
            self._samples
            and self._samples[0].timestamp_ms < oldest_timestamp_ms
        ):
            self._samples.popleft()

    def _should_hold(self, now_ms: int) -> bool:
        if self._stable_gesture.is_unknown:
            return False

        elapsed_ms = now_ms - self._stable_timestamp_ms
        return elapsed_ms <= self.config.hold_ms

    def _is_confident(
        self,
        raw_gesture: DetectedGesture,
        confidence: float,
    ) -> bool:
        return (
            not raw_gesture.is_unknown
            and confidence >= self.config.min_confidence
        )

    @staticmethod
    def _now_ms() -> int:
        return int(perf_counter() * 1000)
