"""Basic gesture recognition from finger open/closed states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable
from typing import TypeAlias

from vision.finger_state import FINGER_ORDER
from vision.finger_state import FingerName
from vision.finger_state import FingerState
from vision.finger_state import FingerStateResult


class GestureName(str, Enum):
    """Supported basic gestures."""

    OPEN_PALM = "Open Palm"
    CLOSED_FIST = "Closed Fist"
    PEACE_SIGN = "Peace Sign"
    POINTING_UP = "Pointing Up"
    THUMBS_UP = "Thumbs Up"
    UNKNOWN = "Unknown"


GestureIdentifier: TypeAlias = GestureName | str


@dataclass(frozen=True)
class DetectedGesture:
    """Detected gesture label."""

    name: GestureIdentifier
    confidence: float = 0.0
    custom: bool = False

    @property
    def label(self) -> str:
        """Return the display label for this gesture."""
        if isinstance(self.name, GestureName):
            return self.name.value
        return self.name

    @property
    def is_unknown(self) -> bool:
        """Return whether this is an unknown gesture."""
        return self.name == GestureName.UNKNOWN


class BasicGestureRecognizer:
    """Recognize a small fixed set of gestures from finger states."""

    _patterns = {
        frozenset(FINGER_ORDER): GestureName.OPEN_PALM,
        frozenset(): GestureName.CLOSED_FIST,
        frozenset(
            {
                FingerName.INDEX,
                FingerName.MIDDLE,
            }
        ): GestureName.PEACE_SIGN,
        frozenset({FingerName.INDEX}): GestureName.POINTING_UP,
        frozenset({FingerName.THUMB}): GestureName.THUMBS_UP,
    }

    def recognize(
        self,
        finger_states: Iterable[FingerStateResult],
    ) -> DetectedGesture:
        """Return the gesture matching the provided finger states."""
        state_by_finger = {
            finger_state.finger: finger_state.state
            for finger_state in finger_states
        }

        if any(finger not in state_by_finger for finger in FINGER_ORDER):
            return DetectedGesture(GestureName.UNKNOWN)

        open_fingers = frozenset(
            finger
            for finger in FINGER_ORDER
            if state_by_finger[finger] == FingerState.OPEN
        )
        gesture_name = self._patterns.get(open_fingers, GestureName.UNKNOWN)
        if gesture_name == GestureName.UNKNOWN:
            return DetectedGesture(gesture_name)
        return DetectedGesture(gesture_name, confidence=1.0)
