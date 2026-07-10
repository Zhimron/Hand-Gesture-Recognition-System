"""Tests for threaded camera, inference, and rendering orchestration."""

from __future__ import annotations

from threading import current_thread
from time import sleep
import unittest

from app.threaded_video_pipeline import ThreadedHandDetectionPipeline
from app.threaded_video_pipeline import ThreadedPipelineConfig
from vision.webcam import WebcamError


class FakeFrame:
    """Simple frame object passed through the pipeline."""

    def __init__(self, sequence_id: int) -> None:
        self.sequence_id = sequence_id
        self.flipped = False


class FakeCapture:
    """Fake capture that produces frames until the pipeline stops."""

    def __init__(self) -> None:
        self.read_count = 0
        self.released = False
        self.set_calls: list[tuple[int, int]] = []
        self.read_thread_names: list[str] = []

    def isOpened(self) -> bool:
        """Return whether the fake camera is open."""
        return True

    def read(self) -> tuple[bool, FakeFrame]:
        """Return the next fake frame."""
        self.read_thread_names.append(current_thread().name)
        self.read_count += 1
        sleep(0.001)
        return True, FakeFrame(self.read_count)

    def release(self) -> None:
        """Mark the fake camera as released."""
        self.released = True

    def get(self, _prop_id: int) -> float:
        """Return no configured capture property."""
        return 0.0

    def set(self, prop_id: int, value: int) -> bool:
        """Record capture property updates."""
        self.set_calls.append((prop_id, value))
        return True


class FailingCapture(FakeCapture):
    """Fake capture that fails on the first read."""

    def read(self) -> tuple[bool, None]:
        """Return a failed camera read."""
        self.read_thread_names.append(current_thread().name)
        return False, None


class FakeDetector:
    """Fake detector that records inference-thread usage."""

    def __init__(self) -> None:
        self.detect_thread_names: list[str] = []
        self.closed = False

    def detect_and_draw(self, frame: object) -> list[object]:
        """Record inference and return no hands."""
        self.detect_thread_names.append(current_thread().name)
        return []

    def close(self) -> None:
        """Mark the fake detector as closed."""
        self.closed = True


class FakeCv2:
    """Fake OpenCV module used by the rendering thread."""

    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, quit_after_frames: int = 3) -> None:
        self.quit_after_frames = quit_after_frames
        self.displayed_frames: list[object] = []
        self.imshow_thread_names: list[str] = []
        self.destroyed = False

    def flip(self, frame: FakeFrame, _flip_code: int, dst: FakeFrame) -> None:
        """Flip in place like OpenCV's optional destination argument."""
        dst.flipped = True

    def imshow(self, _window_name: str, frame: object) -> None:
        """Record rendered frames."""
        self.displayed_frames.append(frame)
        self.imshow_thread_names.append(current_thread().name)

    def waitKey(self, _delay_ms: int) -> int:
        """Quit after a few rendered frames."""
        if len(self.displayed_frames) >= self.quit_after_frames:
            return ord("q")
        return -1

    def destroyAllWindows(self) -> None:
        """Record window cleanup."""
        self.destroyed = True


class ThreadedHandDetectionPipelineTest(unittest.TestCase):
    """Unit tests for ThreadedHandDetectionPipeline."""

    def test_pipeline_uses_separate_workers_and_cleans_up(self) -> None:
        capture = FakeCapture()
        detector = FakeDetector()
        cv2 = FakeCv2()
        overlay_thread_names: list[str] = []

        def draw_overlay(
            frame: object,
            _processed_frame: object,
            _fps: float,
        ) -> None:
            overlay_thread_names.append(current_thread().name)
            self.assertTrue(getattr(frame, "flipped", False))

        pipeline = ThreadedHandDetectionPipeline(
            capture_factory=lambda: capture,
            detector_factory=lambda: detector,
            cv2_module=cv2,
            window_name="Test",
            frame_transform=lambda frame: _flip_frame(cv2, frame),
            draw_overlay=draw_overlay,
            config=ThreadedPipelineConfig(queue_timeout_seconds=0.01),
        )

        result = pipeline.run()

        self.assertEqual(result, 0)
        self.assertTrue(capture.released)
        self.assertTrue(detector.closed)
        self.assertTrue(cv2.destroyed)
        self.assertTrue(capture.set_calls)
        self.assertIn("camera-thread", capture.read_thread_names)
        self.assertIn("inference-thread", detector.detect_thread_names)
        self.assertIn("rendering-thread", cv2.imshow_thread_names)
        self.assertIn("rendering-thread", overlay_thread_names)
        self.assertGreaterEqual(len(cv2.displayed_frames), 3)

    def test_camera_failure_is_reported_and_resources_are_released(self) -> None:
        capture = FailingCapture()
        detector = FakeDetector()
        cv2 = FakeCv2()
        pipeline = ThreadedHandDetectionPipeline(
            capture_factory=lambda: capture,
            detector_factory=lambda: detector,
            cv2_module=cv2,
            window_name="Test",
            config=ThreadedPipelineConfig(queue_timeout_seconds=0.01),
        )

        with self.assertRaises(WebcamError):
            pipeline.run()

        self.assertTrue(capture.released)
        self.assertFalse(detector.closed)
        self.assertEqual(detector.detect_thread_names, [])
        self.assertTrue(cv2.destroyed)


def _flip_frame(cv2: FakeCv2, frame: object) -> object:
    cv2.flip(frame, 1, frame)
    return frame


if __name__ == "__main__":
    unittest.main()
