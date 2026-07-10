"""Finger open/closed state detection from hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot


class FingerName(str, Enum):
    """Supported fingers for state detection."""

    THUMB = "Thumb"
    INDEX = "Index"
    MIDDLE = "Middle"
    RING = "Ring"
    PINKY = "Pinky"


class FingerState(str, Enum):
    """Possible finger states."""

    OPEN = "Open"
    CLOSED = "Closed"


@dataclass(frozen=True)
class FingerStateResult:
    """Detected open/closed state for one finger."""

    finger: FingerName
    state: FingerState


@dataclass(frozen=True)
class LandmarkPoint:
    """Normalized 2D landmark point."""

    x: float
    y: float


FINGER_ORDER = (
    FingerName.THUMB,
    FingerName.INDEX,
    FingerName.MIDDLE,
    FingerName.RING,
    FingerName.PINKY,
)


class FingerStateDetector:
    """Determine open or closed state for each finger."""

    _required_landmarks = 21
    _extended_ratio = 1.05
    _thumb_open_ratio = 1.15
    _finger_indices = {
        FingerName.INDEX: (6, 8),
        FingerName.MIDDLE: (10, 12),
        FingerName.RING: (14, 16),
        FingerName.PINKY: (18, 20),
    }

    def detect(self, landmarks: object) -> tuple[FingerStateResult, ...]:
        """Return open/closed states for thumb through pinky."""
        points = self._landmark_points(landmarks)
        if len(points) < self._required_landmarks:
            return self._closed_results()

        states = {
            FingerName.THUMB: self._thumb_state(points),
            FingerName.INDEX: self._finger_state(
                points,
                FingerName.INDEX,
            ),
            FingerName.MIDDLE: self._finger_state(
                points,
                FingerName.MIDDLE,
            ),
            FingerName.RING: self._finger_state(points, FingerName.RING),
            FingerName.PINKY: self._finger_state(points, FingerName.PINKY),
        }
        return tuple(
            FingerStateResult(finger=finger, state=states[finger])
            for finger in FINGER_ORDER
        )

    def _thumb_state(
        self,
        points: list[LandmarkPoint],
    ) -> FingerState:
        thumb_tip = points[4]
        thumb_ip = points[3]
        index_mcp = points[5]

        tip_distance = self._distance(thumb_tip, index_mcp)
        joint_distance = self._distance(thumb_ip, index_mcp)
        if joint_distance <= 0:
            return FingerState.CLOSED

        if tip_distance > joint_distance * self._thumb_open_ratio:
            return FingerState.OPEN
        return FingerState.CLOSED

    def _finger_state(
        self,
        points: list[LandmarkPoint],
        finger: FingerName,
    ) -> FingerState:
        pip_index, tip_index = self._finger_indices[finger]
        wrist = points[0]
        pip = points[pip_index]
        tip = points[tip_index]

        tip_distance = self._distance(tip, wrist)
        pip_distance = self._distance(pip, wrist)
        is_extended = tip_distance > pip_distance * self._extended_ratio
        points_upward = tip.y <= pip.y + 0.03

        if is_extended and points_upward:
            return FingerState.OPEN
        return FingerState.CLOSED

    @staticmethod
    def _closed_results() -> tuple[FingerStateResult, ...]:
        return tuple(
            FingerStateResult(finger=finger, state=FingerState.CLOSED)
            for finger in FINGER_ORDER
        )

    @staticmethod
    def _landmark_points(landmarks: object) -> list[LandmarkPoint]:
        raw_points = getattr(landmarks, "landmark", None)
        if raw_points is None and isinstance(landmarks, list):
            raw_points = landmarks

        if raw_points is None:
            return []

        return [
            LandmarkPoint(
                x=float(getattr(point, "x", 0.0)),
                y=float(getattr(point, "y", 0.0)),
            )
            for point in raw_points
        ]

    @staticmethod
    def _distance(first: LandmarkPoint, second: LandmarkPoint) -> float:
        return hypot(first.x - second.x, first.y - second.y)
