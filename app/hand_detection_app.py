"""Runtime application for MediaPipe hand detection on a webcam feed."""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable, Protocol, Sequence

from app.webcam_app import FpsCounter, InputProvider
from vision.hand_detection import DetectedHand, HandDetectionError
from vision.hand_detection import HandDetectionService
from vision.webcam import CameraInfo, CameraService, WebcamError


class HandDetectorProtocol(Protocol):
    """Subset of the hand detector API used by the application loop."""

    def detect_and_draw(self, frame: object) -> list[DetectedHand]:
        """Draw hand annotations and return detected hand metadata."""

    def close(self) -> None:
        """Release detector resources."""


DetectorFactory = Callable[[], HandDetectorProtocol]


class HandDetectionApplication:
    """Coordinate webcam input and real-time hand detection display."""

    def __init__(
        self,
        camera_service: CameraService | None = None,
        detector_factory: DetectorFactory | None = None,
        input_provider: InputProvider = input,
        cv2_module: Any | None = None,
    ) -> None:
        self._camera_service = camera_service or CameraService()
        self._detector_factory = detector_factory or HandDetectionService
        self._input_provider = input_provider
        self._cv2 = cv2_module or self._load_cv2()

    def run(self, camera_index: int | None, max_cameras: int) -> int:
        """Discover cameras, select one, and show hand detection."""
        cameras = self._camera_service.detect_available_cameras(max_cameras)
        if not cameras:
            print("No available cameras were detected.")
            return 1

        self._print_cameras(cameras)
        selected_index = self._select_camera(camera_index, cameras)
        if selected_index is None:
            return 1

        return self._display_feed(selected_index)

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

    def _display_feed(self, camera_index: int) -> int:
        capture = None
        detector: HandDetectorProtocol | None = None
        window_name = f"Hand Detection - Camera {camera_index}"
        fps_counter = FpsCounter()

        try:
            detector = self._detector_factory()
            capture = self._camera_service.open_camera(camera_index)
            print("Press Q to quit the hand detection preview.")

            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(
                        "Camera feed stopped. "
                        "The camera may have disconnected."
                    )
                    return 1

                frame = self._cv2.flip(frame, 1)
                detector.detect_and_draw(frame)
                fps = fps_counter.update()
                self._draw_fps(frame, fps)
                self._cv2.imshow(window_name, frame)

                key = self._cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0

        except (HandDetectionError, WebcamError) as exc:
            print(str(exc))
            return 1
        except KeyboardInterrupt:
            print("Hand detection preview interrupted.")
            return 0
        except Exception as exc:
            print(f"Hand detection preview stopped unexpectedly: {exc}")
            return 1
        finally:
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()
            self._cv2.destroyAllWindows()

    def _draw_fps(self, frame: object, fps: float) -> None:
        label = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        self._cv2.putText(
            frame,
            label,
            (12, 32),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            self._cv2.LINE_AA,
        )

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
                "OpenCV is required for hand detection. "
                "Install it with: pip install opencv-python"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the hand detection command-line parser."""
    parser = argparse.ArgumentParser(
        description="Detect and draw hands on a selected webcam feed.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real-time hand detection feature."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        app = HandDetectionApplication()
        return app.run(camera_index=args.camera, max_cameras=args.max_cameras)
    except (HandDetectionError, WebcamError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
