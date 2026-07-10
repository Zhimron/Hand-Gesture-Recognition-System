"""Runtime application for recording custom hand gestures."""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable, Protocol, Sequence

from app.webcam_app import InputProvider
from gestures.custom_gesture import CustomGestureStore
from gestures.custom_gesture import REQUIRED_LANDMARKS
from gestures.custom_gesture import LandmarkSample
from vision.hand_detection import DetectedHand, HandDetectionError
from vision.hand_detection import HandDetectionService
from vision.webcam import CameraInfo, CameraService, WebcamError


class GestureRecordingDetectorProtocol(Protocol):
    """Subset of the hand detector API used by gesture recording."""

    def detect_and_draw(self, frame: object) -> list[DetectedHand]:
        """Draw hand annotations and return detected hand metadata."""

    def close(self) -> None:
        """Release detector resources."""


DetectorFactory = Callable[[], GestureRecordingDetectorProtocol]


class CustomGestureRecordingApplication:
    """Record hand landmark samples and save them as a named gesture."""

    def __init__(
        self,
        camera_service: CameraService | None = None,
        detector_factory: DetectorFactory | None = None,
        gesture_store: CustomGestureStore | None = None,
        input_provider: InputProvider = input,
        cv2_module: Any | None = None,
    ) -> None:
        self._camera_service = camera_service or CameraService()
        self._detector_factory = detector_factory or HandDetectionService
        self._gesture_store = gesture_store or CustomGestureStore()
        self._input_provider = input_provider
        self._cv2 = cv2_module or self._load_cv2()

    def run(
        self,
        gesture_name: str,
        camera_index: int | None,
        max_cameras: int,
        sample_count: int,
    ) -> int:
        """Select a camera and record a custom gesture."""
        cameras = self._camera_service.detect_available_cameras(max_cameras)
        if not cameras:
            print("No available cameras were detected.")
            return 1

        self._print_cameras(cameras)
        selected_index = self._select_camera(camera_index, cameras)
        if selected_index is None:
            return 1

        return self._record_gesture(
            gesture_name=gesture_name,
            camera_index=selected_index,
            sample_count=max(1, sample_count),
        )

    def _record_gesture(
        self,
        gesture_name: str,
        camera_index: int,
        sample_count: int,
    ) -> int:
        capture = None
        detector: GestureRecordingDetectorProtocol | None = None
        samples: list[LandmarkSample] = []
        window_name = f"Record Gesture - {gesture_name}"

        try:
            detector = self._detector_factory()
            capture = self._camera_service.open_camera(camera_index)
            print(
                f"Hold '{gesture_name}' steady. "
                f"Recording {sample_count} samples. Press Q to cancel."
            )

            while len(samples) < sample_count:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(
                        "Camera feed stopped. "
                        "The camera may have disconnected."
                    )
                    return 1

                frame = self._cv2.flip(frame, 1)
                detected_hands = detector.detect_and_draw(frame)
                self._collect_sample(detected_hands, samples)
                self._draw_progress(frame, len(samples), sample_count)
                self._cv2.imshow(window_name, frame)

                key = self._cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Custom gesture recording cancelled.")
                    return 1

            self._gesture_store.save_gesture(gesture_name, samples)
            print(
                f"Saved custom gesture '{gesture_name}' to "
                f"{self._gesture_store.path}."
            )
            return 0

        except (HandDetectionError, WebcamError, ValueError) as exc:
            print(str(exc))
            return 1
        except KeyboardInterrupt:
            print("Custom gesture recording interrupted.")
            return 1
        except Exception as exc:
            print(f"Custom gesture recording stopped unexpectedly: {exc}")
            return 1
        finally:
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()
            self._cv2.destroyAllWindows()

    def _collect_sample(
        self,
        detected_hands: Sequence[DetectedHand],
        samples: list[LandmarkSample],
    ) -> None:
        if not detected_hands:
            return

        landmarks = detected_hands[0].landmarks
        if len(landmarks) >= REQUIRED_LANDMARKS:
            samples.append(landmarks)

    def _draw_progress(
        self,
        frame: object,
        collected_samples: int,
        sample_count: int,
    ) -> None:
        label = f"Recording: {collected_samples}/{sample_count}"
        self._cv2.putText(
            frame,
            label,
            (12, 32),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            self._cv2.LINE_AA,
        )

    def _select_camera(
        self,
        requested_index: int | None,
        cameras: Sequence[CameraInfo],
    ) -> int | None:
        available_indexes = {camera.index for camera in cameras}

        if requested_index is not None:
            if requested_index in available_indexes:
                return requested_index
            print(f"Camera {requested_index} is not available.")
            return None

        if len(cameras) == 1:
            return cameras[0].index

        while True:
            raw_value = self._input_provider("Select camera index: ").strip()
            try:
                selected_index = int(raw_value)
            except ValueError:
                print("Please enter a numeric camera index.")
                continue

            if selected_index in available_indexes:
                return selected_index
            print(f"Camera {selected_index} is not available.")

    @staticmethod
    def _print_cameras(cameras: Sequence[CameraInfo]) -> None:
        print("Available cameras:")
        for camera in cameras:
            size = "unknown"
            if camera.width is not None and camera.height is not None:
                size = f"{camera.width}x{camera.height}"

            fps = "unknown"
            if camera.reported_fps is not None:
                fps = f"{camera.reported_fps:.1f}"

            print(
                f"  [{camera.index}] "
                f"{camera.name} ({size}, reported FPS {fps})"
            )

    @staticmethod
    def _load_cv2() -> Any:
        try:
            return importlib.import_module("cv2")
        except ImportError as exc:
            raise WebcamError(
                "OpenCV is required for custom gesture recording. "
                "Install it with: pip install opencv-python"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the custom gesture recording parser."""
    parser = argparse.ArgumentParser(
        description="Record a named custom gesture from hand landmarks.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name for the custom gesture.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index to open. If omitted, choose from detected cameras.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=5,
        help="Number of camera indexes to scan, starting at 0.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of detected landmark samples to save.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run custom gesture recording."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        app = CustomGestureRecordingApplication()
        return app.run(
            gesture_name=args.name,
            camera_index=args.camera,
            max_cameras=args.max_cameras,
            sample_count=args.samples,
        )
    except (HandDetectionError, WebcamError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
