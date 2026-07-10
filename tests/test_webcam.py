"""Tests for webcam discovery and camera opening."""

from __future__ import annotations

import unittest

from vision.webcam import CameraService, WebcamError


class FakeCapture:
    """Simple fake for the OpenCV VideoCapture API."""

    def __init__(
        self,
        opened: bool,
        frames: list[object | None] | None = None,
        properties: dict[int, float] | None = None,
    ) -> None:
        self.opened = opened
        self.frames = frames or []
        self.properties = properties or {}
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake capture starts opened."""
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        """Return the next fake frame, or a failed read."""
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return frame is not None, frame

    def release(self) -> None:
        """Mark the fake capture as released."""
        self.released = True

    def get(self, prop_id: int) -> float:
        """Return a configured fake property value."""
        return self.properties.get(prop_id, 0.0)


class CameraServiceTest(unittest.TestCase):
    """Unit tests for CameraService."""

    def test_detect_available_cameras_returns_readable_cameras(self) -> None:
        captures = {
            0: FakeCapture(False),
            1: FakeCapture(
                True,
                frames=[object()],
                properties={3: 1280.0, 4: 720.0, 5: 30.0},
            ),
            2: FakeCapture(True, frames=[None, None, None]),
        }

        service = CameraService(capture_factory=captures.__getitem__)

        cameras = service.detect_available_cameras(max_cameras=3)

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].index, 1)
        self.assertEqual(cameras[0].width, 1280)
        self.assertEqual(cameras[0].height, 720)
        self.assertEqual(cameras[0].reported_fps, 30.0)
        self.assertTrue(all(capture.released for capture in captures.values()))

    def test_open_camera_returns_selected_capture(self) -> None:
        capture = FakeCapture(True)
        service = CameraService(capture_factory=lambda _index: capture)

        opened_capture = service.open_camera(0)

        self.assertIs(opened_capture, capture)
        self.assertFalse(capture.released)

    def test_open_camera_raises_when_camera_cannot_open(self) -> None:
        capture = FakeCapture(False)
        service = CameraService(capture_factory=lambda _index: capture)

        with self.assertRaises(WebcamError):
            service.open_camera(0)

        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
