"""Tests for MediaPipe hand detection annotation behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName
from gestures.stability import GestureStabilityConfig
from gestures.stability import GestureStabilizer
from vision.finger_state import FINGER_ORDER
from vision.finger_state import FingerState
from vision.finger_state import FingerStateResult
from vision.hand_detection import DetectedHand, HandDetectionService


class FakeFlags:
    """Mutable frame flags fake."""

    def __init__(self) -> None:
        self.writeable = True


class FakeFrame:
    """Frame fake with image-like shape and flags."""

    def __init__(self) -> None:
        self.shape = (480, 640, 3)
        self.flags = FakeFlags()


class FakePoint:
    """Landmark point fake."""

    def __init__(self, x_position: float, y_position: float) -> None:
        self.x = x_position
        self.y = y_position


class FakeLandmarks:
    """Hand landmarks fake."""

    def __init__(self) -> None:
        self.landmark = [FakePoint(0.5, 0.5)]


class FakeClassification:
    """Handedness classification fake."""

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score


class FakeHandedness:
    """Handedness result fake."""

    def __init__(self, label: str, score: float) -> None:
        self.classification = [FakeClassification(label, score)]


class FakeResults:
    """MediaPipe results fake."""

    def __init__(self) -> None:
        self.multi_hand_landmarks = [FakeLandmarks()]
        self.multi_handedness = [FakeHandedness("Right", 0.91)]


class FakeTaskCategory:
    """MediaPipe Tasks category fake."""

    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score


class FakeTaskResults:
    """MediaPipe Tasks hand landmarker result fake."""

    def __init__(self) -> None:
        self.hand_landmarks = [
            [FakePoint(0.5, 0.5) for _index in range(21)]
        ]
        self.handedness = [[FakeTaskCategory("Left", 0.88)]]


class FakeHandsRunner:
    """Fake MediaPipe Hands runtime."""

    def __init__(self) -> None:
        self.processed_frames: list[object] = []
        self.closed = False

    def process(self, frame: object) -> FakeResults:
        """Record the processed frame and return one detected hand."""
        self.processed_frames.append(frame)
        return FakeResults()

    def close(self) -> None:
        """Mark the fake detector as closed."""
        self.closed = True


class FakeHandsModule:
    """Fake MediaPipe hands solution module."""

    HAND_CONNECTIONS = "hand-connections"

    def __init__(self) -> None:
        self.options: dict[str, object] | None = None
        self.runner = FakeHandsRunner()

    def Hands(self, **kwargs: object) -> FakeHandsRunner:
        """Store Hands options and return a fake runtime."""
        self.options = kwargs
        return self.runner


class FakeDrawingUtils:
    """Fake MediaPipe drawing utilities."""

    def __init__(self) -> None:
        self.draw_calls: list[tuple[object, ...]] = []

    def draw_landmarks(self, *args: object) -> None:
        """Record landmark drawing calls."""
        self.draw_calls.append(args)


class FakeDrawingStyles:
    """Fake MediaPipe drawing style helpers."""

    @staticmethod
    def get_default_hand_landmarks_style() -> str:
        """Return a fake landmark style."""
        return "landmark-style"

    @staticmethod
    def get_default_hand_connections_style() -> str:
        """Return a fake connection style."""
        return "connection-style"


class FakeSolutions:
    """Fake MediaPipe solutions namespace."""

    def __init__(self) -> None:
        self.hands = FakeHandsModule()
        self.drawing_utils = FakeDrawingUtils()
        self.drawing_styles = FakeDrawingStyles()


class FakeMediaPipe:
    """Fake MediaPipe module."""

    def __init__(self) -> None:
        self.solutions = FakeSolutions()


class FakeTaskImageFormat:
    """Fake MediaPipe ImageFormat enum."""

    SRGB = "srgb"


class FakeTaskImage:
    """Fake MediaPipe Image object."""

    def __init__(self, image_format: object, data: object) -> None:
        self.image_format = image_format
        self.data = data


class FakeBaseOptions:
    """Fake MediaPipe Tasks BaseOptions."""

    def __init__(self, model_asset_path: str) -> None:
        self.model_asset_path = model_asset_path


class FakeHandLandmarkerOptions:
    """Fake MediaPipe Tasks hand landmarker options."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeTaskRunningMode:
    """Fake MediaPipe Tasks running mode enum."""

    VIDEO = "video"


class FakeTaskHandLandmarkerRuntime:
    """Fake MediaPipe Tasks hand landmarker runtime."""

    def __init__(self) -> None:
        self.detect_calls: list[tuple[object, int]] = []
        self.closed = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> FakeTaskResults:
        """Record video detection calls and return one detected hand."""
        self.detect_calls.append((image, timestamp_ms))
        return FakeTaskResults()

    def close(self) -> None:
        """Mark the fake runtime as closed."""
        self.closed = True


class FakeTaskHandLandmarker:
    """Fake MediaPipe Tasks HandLandmarker class."""

    def __init__(self) -> None:
        self.options: FakeHandLandmarkerOptions | None = None
        self.runtime = FakeTaskHandLandmarkerRuntime()

    def create_from_options(
        self,
        options: FakeHandLandmarkerOptions,
    ) -> FakeTaskHandLandmarkerRuntime:
        """Store options and return the fake runtime."""
        self.options = options
        return self.runtime


class FakeTaskVision:
    """Fake MediaPipe Tasks vision namespace."""

    def __init__(self) -> None:
        self.RunningMode = FakeTaskRunningMode
        self.HandLandmarkerOptions = FakeHandLandmarkerOptions
        self.HandLandmarker = FakeTaskHandLandmarker()


class FakeTasks:
    """Fake MediaPipe Tasks namespace."""

    def __init__(self) -> None:
        self.BaseOptions = FakeBaseOptions
        self.vision = FakeTaskVision()


class FakeTaskMediaPipe:
    """Fake MediaPipe module that only has the Tasks API."""

    Image = FakeTaskImage
    ImageFormat = FakeTaskImageFormat

    def __init__(self) -> None:
        self.tasks = FakeTasks()


class FakeCv2:
    """Fake OpenCV module."""

    COLOR_BGR2RGB = 4
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.cvt_calls: list[tuple[object, int]] = []
        self.text_calls: list[tuple[object, str, tuple[int, int]]] = []
        self.line_calls: list[tuple[object, tuple[int, int], tuple[int, int]]] = []
        self.circle_calls: list[tuple[object, tuple[int, int]]] = []

    def cvtColor(self, frame: object, code: int) -> object:
        """Record color conversion and return the same fake frame."""
        self.cvt_calls.append((frame, code))
        return frame

    def putText(
        self,
        frame: object,
        text: str,
        origin: tuple[int, int],
        *_args: object,
    ) -> None:
        """Record text overlays."""
        self.text_calls.append((frame, text, origin))

    def line(
        self,
        frame: object,
        start: tuple[int, int],
        end: tuple[int, int],
        *_args: object,
    ) -> None:
        """Record line drawing calls."""
        self.line_calls.append((frame, start, end))

    def circle(
        self,
        frame: object,
        center: tuple[int, int],
        *_args: object,
    ) -> None:
        """Record circle drawing calls."""
        self.circle_calls.append((frame, center))


def _closed_finger_states() -> tuple[FingerStateResult, ...]:
    return tuple(
        FingerStateResult(finger=finger, state=FingerState.CLOSED)
        for finger in FINGER_ORDER
    )


def _single_vote_stabilizer() -> GestureStabilizer:
    return GestureStabilizer(GestureStabilityConfig(min_votes=1))


class HandDetectionServiceTest(unittest.TestCase):
    """Unit tests for hand detection annotation."""

    def test_detect_and_draw_returns_handedness_and_draws_annotations(
        self,
    ) -> None:
        fake_cv2 = FakeCv2()
        fake_mp = FakeMediaPipe()
        service = HandDetectionService(
            max_num_hands=5,
            cv2_module=fake_cv2,
            mediapipe_module=fake_mp,
            gesture_stabilizer_factory=_single_vote_stabilizer,
        )
        frame = FakeFrame()

        detected_hands = service.detect_and_draw(frame)

        self.assertEqual(
            detected_hands,
            [
                DetectedHand(
                    handedness="Right",
                    confidence=0.91,
                    finger_states=_closed_finger_states(),
                    gesture=DetectedGesture(
                        GestureName.CLOSED_FIST,
                        confidence=0.91,
                    ),
                )
            ],
        )
        self.assertEqual(fake_mp.solutions.hands.options["max_num_hands"], 2)
        self.assertEqual(fake_cv2.cvt_calls, [(frame, FakeCv2.COLOR_BGR2RGB)])
        self.assertEqual(len(fake_mp.solutions.drawing_utils.draw_calls), 1)
        self.assertEqual(fake_cv2.text_calls[0][1], "Right: 0.91")
        self.assertEqual(fake_cv2.text_calls[1][1], "Gesture: Closed Fist")
        self.assertEqual(fake_cv2.text_calls[2][1], "Thumb: Closed")
        self.assertTrue(frame.flags.writeable)

    def test_close_releases_mediapipe_resources(self) -> None:
        fake_mp = FakeMediaPipe()
        service = HandDetectionService(
            cv2_module=FakeCv2(),
            mediapipe_module=fake_mp,
        )

        service.close()

        self.assertTrue(fake_mp.solutions.hands.runner.closed)

    def test_tasks_api_detects_and_draws_without_solutions_api(self) -> None:
        fake_cv2 = FakeCv2()
        fake_mp = FakeTaskMediaPipe()
        frame = FakeFrame()

        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hand_landmarker.task"

            service = HandDetectionService(
                max_num_hands=5,
                cv2_module=fake_cv2,
                mediapipe_module=fake_mp,
                model_path=model_path,
                model_downloader=lambda path: path.write_bytes(b"fake"),
                gesture_stabilizer_factory=_single_vote_stabilizer,
            )
            detected_hands = service.detect_and_draw(frame)
            service.close()

        options = fake_mp.tasks.vision.HandLandmarker.options
        self.assertIsNotNone(options)
        self.assertEqual(options.kwargs["num_hands"], 2)
        self.assertEqual(
            detected_hands,
            [
                DetectedHand(
                    handedness="Left",
                    confidence=0.88,
                    finger_states=_closed_finger_states(),
                    gesture=DetectedGesture(
                        GestureName.CLOSED_FIST,
                        confidence=0.88,
                    ),
                )
            ],
        )
        self.assertEqual(len(fake_cv2.line_calls), 21)
        self.assertEqual(len(fake_cv2.circle_calls), 21)
        self.assertEqual(fake_cv2.text_calls[0][1], "Left: 0.88")
        self.assertEqual(fake_cv2.text_calls[1][1], "Gesture: Closed Fist")
        self.assertEqual(fake_cv2.text_calls[6][1], "Pinky: Closed")
        self.assertTrue(fake_mp.tasks.vision.HandLandmarker.runtime.closed)


if __name__ == "__main__":
    unittest.main()
