"""Mouse control driven by recognized hand gestures."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol, Sequence

from gestures.basic_gesture import GestureName
from gestures.custom_gesture import LandmarkCoordinate
from vision.hand_detection import DetectedHand


class MouseControlError(RuntimeError):
    """Raised when mouse control cannot be initialized."""


class MouseBackend(Protocol):
    """Mouse backend operations required by the controller."""

    def screen_size(self) -> tuple[int, int]:
        """Return screen width and height in pixels."""

    def move_to(self, x_position: int, y_position: int) -> None:
        """Move the cursor to a screen position."""

    def click(self, button: str = "left") -> None:
        """Click a mouse button."""

    def double_click(self) -> None:
        """Double-click the left mouse button."""

    def mouse_down(self, button: str = "left") -> None:
        """Press and hold a mouse button."""

    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button."""

    def scroll(self, amount: int) -> None:
        """Scroll by the provided amount."""


class PyAutoGuiMouseBackend:
    """pyautogui-backed mouse backend."""

    def __init__(self) -> None:
        try:
            self._pyautogui = importlib.import_module("pyautogui")
        except ImportError as exc:
            raise MouseControlError(
                "pyautogui is required for mouse control. "
                "Install it with: pip install pyautogui"
            ) from exc

        self._pyautogui.PAUSE = 0

    def screen_size(self) -> tuple[int, int]:
        """Return screen width and height in pixels."""
        size = self._pyautogui.size()
        return int(size.width), int(size.height)

    def move_to(self, x_position: int, y_position: int) -> None:
        """Move the cursor to a screen position."""
        self._pyautogui.moveTo(x_position, y_position, duration=0)

    def click(self, button: str = "left") -> None:
        """Click a mouse button."""
        self._pyautogui.click(button=button)

    def double_click(self) -> None:
        """Double-click the left mouse button."""
        self._pyautogui.doubleClick()

    def mouse_down(self, button: str = "left") -> None:
        """Press and hold a mouse button."""
        self._pyautogui.mouseDown(button=button)

    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button."""
        self._pyautogui.mouseUp(button=button)

    def scroll(self, amount: int) -> None:
        """Scroll by the provided amount."""
        self._pyautogui.scroll(amount)


@dataclass(frozen=True)
class MouseControllerConfig:
    """Tuning values for gesture-based mouse control."""

    smoothing: float = 0.28
    input_margin: float = 0.08
    click_cooldown_ms: int = 450
    scroll_deadzone: float = 0.018
    scroll_sensitivity: float = 90.0


@dataclass(frozen=True)
class MouseControlStatus:
    """Last mouse-control action status."""

    action: str = "Idle"
    gesture: str = "Unknown"


class CursorSmoother:
    """Map normalized hand coordinates to smoothed screen coordinates."""

    def __init__(
        self,
        screen_size: tuple[int, int],
        smoothing: float,
        input_margin: float,
    ) -> None:
        self._screen_width, self._screen_height = screen_size
        self._smoothing = max(0.0, min(smoothing, 1.0))
        self._input_margin = max(0.0, min(input_margin, 0.45))
        self._last_position: tuple[float, float] | None = None

    def map(self, point: LandmarkCoordinate) -> tuple[int, int]:
        """Return a smoothed screen position for a normalized landmark."""
        x_position = self._map_axis(point.x, self._screen_width)
        y_position = self._map_axis(point.y, self._screen_height)

        if self._last_position is None:
            self._last_position = (x_position, y_position)
        else:
            last_x, last_y = self._last_position
            x_position = last_x + (x_position - last_x) * self._smoothing
            y_position = last_y + (y_position - last_y) * self._smoothing
            self._last_position = (x_position, y_position)

        return int(round(x_position)), int(round(y_position))

    def _map_axis(self, value: float, screen_length: int) -> float:
        minimum = self._input_margin
        maximum = 1.0 - self._input_margin
        normalized = (value - minimum) / (maximum - minimum)
        normalized = max(0.0, min(normalized, 1.0))
        return normalized * max(0, screen_length - 1)


class MouseGestureController:
    """Translate recognized gestures into mouse actions."""

    _move_labels = {"Pointing Up", "Move", "Cursor"}
    _left_click_labels = {"Peace Sign", "Left Click"}
    _right_click_labels = {"Thumbs Up", "Right Click"}
    _double_click_labels = {"Double Click"}
    _drag_labels = {"Closed Fist", "Drag"}
    _scroll_labels = {"Open Palm", "Scroll"}
    _scroll_up_labels = {"Scroll Up"}
    _scroll_down_labels = {"Scroll Down"}

    def __init__(
        self,
        mouse_backend: MouseBackend | None = None,
        config: MouseControllerConfig | None = None,
    ) -> None:
        self._mouse_backend = mouse_backend or PyAutoGuiMouseBackend()
        self._config = config or MouseControllerConfig()
        self._cursor = CursorSmoother(
            screen_size=self._mouse_backend.screen_size(),
            smoothing=self._config.smoothing,
            input_margin=self._config.input_margin,
        )
        self._dragging = False
        self._last_gesture = "Unknown"
        self._last_click_timestamp_ms = -self._config.click_cooldown_ms
        self._last_scroll_y: float | None = None
        self._scroll_remainder = 0.0

    def update(
        self,
        detected_hands: Sequence[DetectedHand],
        timestamp_ms: int,
    ) -> MouseControlStatus:
        """Apply mouse actions for the current recognized gesture."""
        if not detected_hands:
            self._release_drag()
            self._reset_motion_state()
            return MouseControlStatus()

        hand = detected_hands[0]
        gesture_label = hand.gesture.label
        action = "Idle"

        if self._is_drag(gesture_label):
            self._start_drag()
            action = self._move_cursor(hand.landmarks, "Drag")
        else:
            self._release_drag()

        if self._is_move(gesture_label):
            action = self._move_cursor(hand.landmarks, "Move")
        elif self._is_scroll(gesture_label):
            action = self._scroll_from_motion(hand.landmarks)
        elif self._is_scroll_up(gesture_label) and self._should_trigger(
            gesture_label,
            timestamp_ms,
        ):
            action = self._scroll_fixed(5, "Scroll Up")
            self._last_click_timestamp_ms = timestamp_ms
        elif self._is_scroll_down(gesture_label) and self._should_trigger(
            gesture_label,
            timestamp_ms,
        ):
            action = self._scroll_fixed(-5, "Scroll Down")
            self._last_click_timestamp_ms = timestamp_ms
        elif self._is_click_action(gesture_label) and self._should_trigger(
            gesture_label,
            timestamp_ms,
        ):
            action = self._trigger_click_action(gesture_label)
            if action != "Idle":
                self._last_click_timestamp_ms = timestamp_ms

        if not self._is_scroll(gesture_label):
            self._last_scroll_y = None

        self._last_gesture = gesture_label
        return MouseControlStatus(action=action, gesture=gesture_label)

    def close(self) -> None:
        """Release any held mouse state."""
        self._release_drag()

    def _move_cursor(
        self,
        landmarks: Sequence[LandmarkCoordinate],
        action: str,
    ) -> str:
        point = self._landmark(landmarks, 8)
        if point is None:
            return "Idle"

        x_position, y_position = self._cursor.map(point)
        self._mouse_backend.move_to(x_position, y_position)
        return action

    def _scroll_from_motion(
        self,
        landmarks: Sequence[LandmarkCoordinate],
    ) -> str:
        palm_y = self._palm_y(landmarks)
        if palm_y is None:
            return "Idle"

        if self._last_scroll_y is None:
            self._last_scroll_y = palm_y
            return "Scroll"

        delta_y = palm_y - self._last_scroll_y
        self._last_scroll_y = palm_y
        if abs(delta_y) < self._config.scroll_deadzone:
            return "Scroll"

        self._scroll_remainder += -delta_y * self._config.scroll_sensitivity
        scroll_amount = int(self._scroll_remainder)
        if scroll_amount == 0:
            return "Scroll"

        self._scroll_remainder -= scroll_amount
        self._mouse_backend.scroll(scroll_amount)
        return "Scroll"

    def _scroll_fixed(self, amount: int, action: str) -> str:
        self._mouse_backend.scroll(amount)
        return action

    def _trigger_click_action(self, gesture_label: str) -> str:
        if self._is_left_click(gesture_label):
            self._mouse_backend.click(button="left")
            return "Left Click"
        if self._is_right_click(gesture_label):
            self._mouse_backend.click(button="right")
            return "Right Click"
        if self._is_double_click(gesture_label):
            self._mouse_backend.double_click()
            return "Double Click"
        return "Idle"

    def _should_trigger(self, gesture_label: str, timestamp_ms: int) -> bool:
        if gesture_label == self._last_gesture:
            return False

        elapsed_ms = timestamp_ms - self._last_click_timestamp_ms
        return elapsed_ms >= self._config.click_cooldown_ms

    def _start_drag(self) -> None:
        if self._dragging:
            return

        self._mouse_backend.mouse_down(button="left")
        self._dragging = True

    def _release_drag(self) -> None:
        if not self._dragging:
            return

        self._mouse_backend.mouse_up(button="left")
        self._dragging = False

    def _reset_motion_state(self) -> None:
        self._last_gesture = "Unknown"
        self._last_scroll_y = None

    def _is_move(self, gesture_label: str) -> bool:
        return gesture_label in self._move_labels

    def _is_left_click(self, gesture_label: str) -> bool:
        return gesture_label in self._left_click_labels

    def _is_right_click(self, gesture_label: str) -> bool:
        return gesture_label in self._right_click_labels

    def _is_double_click(self, gesture_label: str) -> bool:
        return gesture_label in self._double_click_labels

    def _is_drag(self, gesture_label: str) -> bool:
        return gesture_label in self._drag_labels

    def _is_scroll(self, gesture_label: str) -> bool:
        return gesture_label in self._scroll_labels

    def _is_scroll_up(self, gesture_label: str) -> bool:
        return gesture_label in self._scroll_up_labels

    def _is_scroll_down(self, gesture_label: str) -> bool:
        return gesture_label in self._scroll_down_labels

    def _is_click_action(self, gesture_label: str) -> bool:
        return (
            self._is_left_click(gesture_label)
            or self._is_right_click(gesture_label)
            or self._is_double_click(gesture_label)
        )

    @staticmethod
    def _landmark(
        landmarks: Sequence[LandmarkCoordinate],
        index: int,
    ) -> LandmarkCoordinate | None:
        if index >= len(landmarks):
            return None
        return landmarks[index]

    @staticmethod
    def _palm_y(landmarks: Sequence[LandmarkCoordinate]) -> float | None:
        palm_indexes = (0, 5, 9, 13, 17)
        points = [
            landmarks[index]
            for index in palm_indexes
            if index < len(landmarks)
        ]
        if not points:
            return None

        return sum(point.y for point in points) / len(points)


def gesture_from_name(name: GestureName | str) -> str:
    """Return a displayable gesture name."""
    if isinstance(name, GestureName):
        return name.value
    return name
