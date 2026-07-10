"""Tests for gesture-driven mouse control."""

from __future__ import annotations

import unittest

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName
from gestures.custom_gesture import LandmarkCoordinate
from services.mouse_controller import MouseControllerConfig
from services.mouse_controller import MouseGestureController
from vision.hand_detection import DetectedHand


class FakeMouseBackend:
    """Fake mouse backend that records operations."""

    def __init__(self) -> None:
        self.operations: list[tuple[object, ...]] = []

    def screen_size(self) -> tuple[int, int]:
        """Return a fixed fake screen size."""
        return 1000, 500

    def move_to(self, x_position: int, y_position: int) -> None:
        """Record cursor movement."""
        self.operations.append(("move_to", x_position, y_position))

    def click(self, button: str = "left") -> None:
        """Record a click."""
        self.operations.append(("click", button))

    def double_click(self) -> None:
        """Record a double click."""
        self.operations.append(("double_click",))

    def mouse_down(self, button: str = "left") -> None:
        """Record a mouse down."""
        self.operations.append(("mouse_down", button))

    def mouse_up(self, button: str = "left") -> None:
        """Record a mouse up."""
        self.operations.append(("mouse_up", button))

    def scroll(self, amount: int) -> None:
        """Record scrolling."""
        self.operations.append(("scroll", amount))


def _landmarks(index_x: float = 0.5, index_y: float = 0.5) -> tuple[
    LandmarkCoordinate,
    ...,
]:
    points = [
        LandmarkCoordinate(x=0.5, y=0.5, z=0.0)
        for _index in range(21)
    ]
    points[8] = LandmarkCoordinate(x=index_x, y=index_y, z=0.0)
    return tuple(points)


def _palm_landmarks(palm_y: float) -> tuple[LandmarkCoordinate, ...]:
    points = list(_landmarks())
    for index in (0, 5, 9, 13, 17):
        points[index] = LandmarkCoordinate(x=0.5, y=palm_y, z=0.0)
    return tuple(points)


def _hand(
    gesture_name: GestureName | str,
    landmarks: tuple[LandmarkCoordinate, ...] | None = None,
) -> DetectedHand:
    return DetectedHand(
        handedness="Right",
        confidence=0.95,
        gesture=DetectedGesture(gesture_name, confidence=0.95),
        landmarks=landmarks or _landmarks(),
    )


class MouseGestureControllerTest(unittest.TestCase):
    """Unit tests for MouseGestureController."""

    def test_pointing_up_moves_cursor_with_smoothing(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(
            mouse_backend=backend,
            config=MouseControllerConfig(smoothing=0.5, input_margin=0.0),
        )

        first_status = controller.update(
            [_hand(GestureName.POINTING_UP, _landmarks(0.2, 0.2))],
            timestamp_ms=0,
        )
        second_status = controller.update(
            [_hand(GestureName.POINTING_UP, _landmarks(0.8, 0.8))],
            timestamp_ms=16,
        )

        self.assertEqual(first_status.action, "Move")
        self.assertEqual(second_status.action, "Move")
        self.assertEqual(backend.operations[0], ("move_to", 200, 100))
        self.assertEqual(backend.operations[1], ("move_to", 500, 250))

    def test_peace_sign_left_clicks_once_per_transition(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(mouse_backend=backend)

        controller.update([_hand(GestureName.PEACE_SIGN)], timestamp_ms=0)
        controller.update([_hand(GestureName.PEACE_SIGN)], timestamp_ms=16)

        self.assertEqual(backend.operations, [("click", "left")])

    def test_thumbs_up_right_clicks(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(mouse_backend=backend)

        status = controller.update([_hand(GestureName.THUMBS_UP)], timestamp_ms=0)

        self.assertEqual(status.action, "Right Click")
        self.assertEqual(backend.operations, [("click", "right")])

    def test_custom_double_click_gesture_double_clicks(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(mouse_backend=backend)

        status = controller.update([_hand("Double Click")], timestamp_ms=0)

        self.assertEqual(status.action, "Double Click")
        self.assertEqual(backend.operations, [("double_click",)])

    def test_closed_fist_drags_until_released(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(mouse_backend=backend)

        drag_status = controller.update(
            [_hand(GestureName.CLOSED_FIST)],
            timestamp_ms=0,
        )
        idle_status = controller.update([], timestamp_ms=16)

        self.assertEqual(drag_status.action, "Drag")
        self.assertEqual(idle_status.action, "Idle")
        self.assertEqual(backend.operations[0], ("mouse_down", "left"))
        self.assertEqual(backend.operations[-1], ("mouse_up", "left"))

    def test_open_palm_scrolls_from_vertical_motion(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(
            mouse_backend=backend,
            config=MouseControllerConfig(scroll_deadzone=0.0),
        )

        controller.update([_hand(GestureName.OPEN_PALM, _palm_landmarks(0.5))], 0)
        status = controller.update(
            [_hand(GestureName.OPEN_PALM, _palm_landmarks(0.47))],
            16,
        )

        self.assertEqual(status.action, "Scroll")
        self.assertIn(("scroll", 2), backend.operations)

    def test_click_cooldown_prevents_rapid_retrigger(self) -> None:
        backend = FakeMouseBackend()
        controller = MouseGestureController(mouse_backend=backend)

        controller.update([_hand(GestureName.PEACE_SIGN)], timestamp_ms=0)
        controller.update([], timestamp_ms=10)
        controller.update([_hand(GestureName.PEACE_SIGN)], timestamp_ms=100)

        self.assertEqual(backend.operations, [("click", "left")])


if __name__ == "__main__":
    unittest.main()
