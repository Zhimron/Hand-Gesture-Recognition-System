"""MediaPipe Hands detection and drawing utilities."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlretrieve

from gestures.basic_gesture import BasicGestureRecognizer
from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName
from gestures.custom_gesture import CustomGestureRecognizer
from gestures.custom_gesture import LandmarkCoordinate
from gestures.custom_gesture import extract_landmark_sample
from gestures.stability import GestureStabilizer
from vision.finger_state import FingerState
from vision.finger_state import FingerStateDetector
from vision.finger_state import FingerStateResult


HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH / "hand_landmarker.task"
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


ModelDownloader = Callable[[Path], None]
GestureStabilizerFactory = Callable[[], GestureStabilizer]


@dataclass(frozen=True)
class DetectedHand:
    """Detected hand metadata displayed on the video feed."""

    handedness: str
    confidence: float
    finger_states: tuple[FingerStateResult, ...] = ()
    gesture: DetectedGesture = DetectedGesture(GestureName.UNKNOWN)
    landmarks: tuple[LandmarkCoordinate, ...] = field(
        default_factory=tuple,
        compare=False,
    )


class HandDetectionError(RuntimeError):
    """Raised when MediaPipe hand detection cannot be initialized."""


class HandDetectionService:
    """Detect hands with MediaPipe Hands and draw annotations on frames."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        cv2_module: Any | None = None,
        mediapipe_module: Any | None = None,
        model_path: str | Path | None = None,
        model_downloader: ModelDownloader | None = None,
        finger_state_detector: FingerStateDetector | None = None,
        gesture_recognizer: BasicGestureRecognizer | None = None,
        custom_gesture_recognizer: CustomGestureRecognizer | None = None,
        gesture_stabilizer_factory: GestureStabilizerFactory | None = None,
    ) -> None:
        self._max_num_hands = max(1, min(max_num_hands, 2))
        self._cv2 = cv2_module or self._load_cv2()
        self._mp = mediapipe_module or self._load_mediapipe()
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model_downloader = model_downloader or self._download_task_model
        self._finger_state_detector = (
            finger_state_detector or FingerStateDetector()
        )
        self._gesture_recognizer = (
            gesture_recognizer or BasicGestureRecognizer()
        )
        self._custom_gesture_recognizer = (
            custom_gesture_recognizer or CustomGestureRecognizer()
        )
        self._gesture_stabilizer_factory = (
            gesture_stabilizer_factory or GestureStabilizer
        )
        self._gesture_stabilizers: dict[int, GestureStabilizer] = {}
        self._last_timestamp_ms = 0
        self._use_tasks_api = not self._has_solutions_hands(self._mp)

        if self._use_tasks_api:
            self._hands = self._create_tasks_detector(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._hands_module = self._mp.solutions.hands
            self._drawing_utils = self._mp.solutions.drawing_utils
            self._drawing_styles = self._mp.solutions.drawing_styles
            self._hands = self._hands_module.Hands(
                static_image_mode=False,
                max_num_hands=self._max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def __enter__(self) -> HandDetectionService:
        """Return the initialized detector for context-manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying MediaPipe resources."""
        self.close()

    def detect_and_draw(self, frame: object) -> list[DetectedHand]:
        """Detect hands and draw landmarks, connections, and labels."""
        rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        self._set_writeable(rgb_frame, False)
        results = self._process(rgb_frame)
        self._set_writeable(rgb_frame, True)

        hand_landmarks, handednesses = self._extract_results(results)
        detected_hands: list[DetectedHand] = []

        for index, landmarks in enumerate(hand_landmarks):
            hand_info = self._extract_handedness(
                handednesses[index] if index < len(handednesses) else None
            )
            landmark_sample = extract_landmark_sample(landmarks)
            finger_states = self._finger_state_detector.detect(landmarks)
            raw_gesture = self._gesture_recognizer.recognize(finger_states)
            custom_gesture = self._custom_gesture_recognizer.recognize_sample(
                landmark_sample
            )
            if not custom_gesture.is_unknown:
                raw_gesture = custom_gesture

            gesture = self._stable_gesture(
                hand_index=index,
                raw_gesture=raw_gesture,
                confidence=hand_info.confidence if hand_info else 0.0,
            )
            detected_hand = DetectedHand(
                handedness=hand_info.handedness if hand_info else "Unknown",
                confidence=hand_info.confidence if hand_info else 0.0,
                finger_states=finger_states,
                gesture=gesture,
                landmarks=landmark_sample,
            )
            self._draw_landmarks(frame, landmarks)
            detected_hands.append(detected_hand)
            self._draw_label(frame, landmarks, detected_hand)

        return detected_hands

    def close(self) -> None:
        """Release MediaPipe detector resources."""
        close = getattr(self._hands, "close", None)
        if callable(close):
            close()

    def _create_tasks_detector(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> object:
        self._ensure_task_model()

        try:
            base_options = self._mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            )
            options = self._mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self._max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return self._mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
        except AttributeError as exc:
            raise HandDetectionError(
                "The installed MediaPipe package does not include "
                "a supported hand detection API."
            ) from exc

    def _process(self, rgb_frame: object) -> object:
        if not self._use_tasks_api:
            return self._hands.process(rgb_frame)

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        return self._hands.detect_for_video(image, self._next_timestamp_ms())

    def _extract_results(self, results: object) -> tuple[list[object], list[object]]:
        if self._use_tasks_api:
            return (
                list(getattr(results, "hand_landmarks", None) or []),
                list(getattr(results, "handedness", None) or []),
            )

        return (
            list(getattr(results, "multi_hand_landmarks", None) or []),
            list(getattr(results, "multi_handedness", None) or []),
        )

    def _draw_landmarks(self, frame: object, landmarks: object) -> None:
        if self._use_tasks_api:
            self._draw_task_landmarks(frame, landmarks)
            return

        self._drawing_utils.draw_landmarks(
            frame,
            landmarks,
            self._hands_module.HAND_CONNECTIONS,
            self._drawing_styles.get_default_hand_landmarks_style(),
            self._drawing_styles.get_default_hand_connections_style(),
        )

    def _draw_task_landmarks(self, frame: object, landmarks: object) -> None:
        height, width = self._frame_size(frame)
        points = self._landmark_points(landmarks)
        pixel_points = [
            (
                self._clamp(
                    int(float(getattr(point, "x", 0.0)) * width),
                    0,
                    max(0, width - 1),
                ),
                self._clamp(
                    int(float(getattr(point, "y", 0.0)) * height),
                    0,
                    max(0, height - 1),
                ),
            )
            for point in points
        ]

        for start_index, end_index in HAND_CONNECTIONS:
            if start_index >= len(pixel_points) or end_index >= len(pixel_points):
                continue
            self._cv2.line(
                frame,
                pixel_points[start_index],
                pixel_points[end_index],
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )

        for point in pixel_points:
            self._cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1,
                self._cv2.LINE_AA,
            )

    def _draw_label(
        self,
        frame: object,
        landmarks: object,
        detected_hand: DetectedHand,
    ) -> None:
        origin_x, origin_y = self._label_position(frame, landmarks)
        lines = [
            (
                f"{detected_hand.handedness}: "
                f"{detected_hand.confidence:.2f}",
                (255, 255, 255),
            ),
            (
                f"Gesture: {detected_hand.gesture.label}",
                (255, 255, 0),
            ),
        ]
        lines.extend(
            (
                f"{finger_state.finger.value}: {finger_state.state.value}",
                self._finger_state_color(finger_state.state),
            )
            for finger_state in detected_hand.finger_states
        )

        for index, (label, color) in enumerate(lines):
            self._cv2.putText(
                frame,
                label,
                (origin_x, origin_y + index * 22),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                self._cv2.LINE_AA,
            )

    @staticmethod
    def _extract_handedness(handedness: object | None) -> DetectedHand | None:
        if handedness is None:
            return None

        classifications = handedness
        if not isinstance(handedness, list):
            classifications = getattr(handedness, "classification", None)

        if not classifications:
            return None

        classification = classifications[0]
        label = getattr(classification, "category_name", None)
        label = label or getattr(classification, "label", "Unknown")
        confidence = float(getattr(classification, "score", 0.0))
        return DetectedHand(handedness=label, confidence=confidence)

    def _stable_gesture(
        self,
        hand_index: int,
        raw_gesture: DetectedGesture,
        confidence: float,
    ) -> DetectedGesture:
        stabilizer = self._gesture_stabilizers.get(hand_index)
        if stabilizer is None:
            stabilizer = self._gesture_stabilizer_factory()
            self._gesture_stabilizers[hand_index] = stabilizer

        return stabilizer.update(raw_gesture, confidence=confidence)

    @staticmethod
    def _label_position(frame: object, landmarks: object) -> tuple[int, int]:
        height, width = HandDetectionService._frame_size(frame)
        points = HandDetectionService._landmark_points(landmarks)

        if not points:
            return 12, 32

        wrist = points[0]
        x_position = int(float(getattr(wrist, "x", 0.0)) * width)
        y_position = int(float(getattr(wrist, "y", 0.0)) * height) - 12

        return (
            HandDetectionService._clamp(x_position, 12, max(12, width - 1)),
            HandDetectionService._clamp(y_position, 24, max(24, height - 1)),
        )

    @staticmethod
    def _finger_state_color(state: FingerState) -> tuple[int, int, int]:
        if state == FingerState.OPEN:
            return (0, 255, 0)
        return (0, 0, 255)

    @staticmethod
    def _frame_size(frame: object) -> tuple[int, int]:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return 480, 640

        return int(shape[0]), int(shape[1])

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    def _ensure_task_model(self) -> None:
        if self._model_path.exists():
            return

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._model_downloader(self._model_path)
        except (OSError, URLError) as exc:
            raise HandDetectionError(
                "Could not download the MediaPipe hand model. "
                f"Download {HAND_LANDMARKER_MODEL_URL} and save it to "
                f"{self._model_path}."
            ) from exc

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int(perf_counter() * 1000)
        self._last_timestamp_ms = max(
            timestamp_ms,
            self._last_timestamp_ms + 1,
        )
        return self._last_timestamp_ms

    @staticmethod
    def _set_writeable(frame: object, value: bool) -> None:
        flags = getattr(frame, "flags", None)
        if flags is not None and hasattr(flags, "writeable"):
            flags.writeable = value

    @staticmethod
    def _landmark_points(landmarks: object) -> list[object]:
        points = getattr(landmarks, "landmark", None)
        if points is not None:
            return list(points)

        if isinstance(landmarks, list):
            return landmarks

        return []

    @staticmethod
    def _has_solutions_hands(mediapipe_module: object) -> bool:
        solutions = getattr(mediapipe_module, "solutions", None)
        return hasattr(solutions, "hands")

    @staticmethod
    def _download_task_model(destination: Path) -> None:
        print("Downloading MediaPipe hand model...")
        urlretrieve(HAND_LANDMARKER_MODEL_URL, destination)

    @staticmethod
    def _load_cv2() -> Any:
        try:
            return importlib.import_module("cv2")
        except ImportError as exc:
            raise HandDetectionError(
                "OpenCV is required for hand detection. "
                "Install it with: pip install opencv-python"
            ) from exc

    @staticmethod
    def _load_mediapipe() -> Any:
        try:
            return importlib.import_module("mediapipe")
        except ImportError as exc:
            raise HandDetectionError(
                "MediaPipe is required for hand detection. "
                "Install it with: pip install mediapipe"
            ) from exc
