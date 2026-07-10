"""Runtime application for gesture-driven mouse control."""

from __future__ import annotations

import argparse
import importlib
from threading import Lock
from typing import Any, Callable, Sequence

from app.threaded_video_pipeline import HandDetectorProtocol
from app.threaded_video_pipeline import ProcessedFrame
from app.threaded_video_pipeline import ThreadedHandDetectionPipeline
from app.webcam_app import InputProvider
from services.mouse_controller import MouseControlError
from services.mouse_controller import MouseControlStatus
from services.mouse_controller import MouseGestureController
from vision.hand_detection import HandDetectionError
from vision.hand_detection import HandDetectionService
from vision.webcam import CameraInfo, CameraService, WebcamError


DetectorFactory = Callable[[], HandDetectorProtocol]


class MouseControlApplication:
    """Coordinate webcam hand detection and mouse control."""

    def __init__(
        self,
        camera_service: CameraService | None = None,
        detector_factory: DetectorFactory | None = None,
        mouse_controller: MouseGestureController | None = None,
        input_provider: InputProvider = input,
        cv2_module: Any | None = None,
    ) -> None:
        self._camera_service = camera_service or CameraService()
        self._detector_factory = detector_factory or HandDetectionService
        self._mouse_controller = mouse_controller or MouseGestureController()
        self._input_provider = input_provider
        self._cv2 = cv2_module or self._load_cv2()

    def run(self, camera_index: int | None, max_cameras: int) -> int:
        """Discover cameras, select one, and start mouse control."""
        cameras = self._camera_service.detect_available_cameras(max_cameras)
        if not cameras:
            print("No available cameras were detected.")
            return 1

        self._print_cameras(cameras)
        selected_index = self._select_camera(camera_index, cameras)
        if selected_index is None:
            return 1

        return self._display_feed(selected_index)

    def _display_feed(self, camera_index: int) -> int:
        window_name = f"Mouse Control - Camera {camera_index}"
        status_lock = Lock()
        latest_status = MouseControlStatus()

        def handle_inference(processed_frame: ProcessedFrame) -> bool:
            nonlocal latest_status

            status = self._mouse_controller.update(
                processed_frame.detected_hands,
                timestamp_ms=processed_frame.timestamp_ms,
            )
            with status_lock:
                latest_status = status
            return False

        def draw_overlay(
            frame: object,
            _processed_frame: ProcessedFrame,
            fps: float,
        ) -> None:
            with status_lock:
                status = latest_status
            self._draw_status(frame, status, fps)

        pipeline = ThreadedHandDetectionPipeline(
            capture_factory=lambda: self._camera_service.open_camera(
                camera_index,
            ),
            detector_factory=self._detector_factory,
            cv2_module=self._cv2,
            window_name=window_name,
            frame_transform=self._flip_frame,
            on_inference=handle_inference,
            draw_overlay=draw_overlay,
        )

        try:
            self._print_controls()
            return pipeline.run()
        except (HandDetectionError, MouseControlError, WebcamError) as exc:
            print(str(exc))
            return 1
        except KeyboardInterrupt:
            print("Mouse control interrupted.")
            return 0
        except Exception as exc:
            print(f"Mouse control stopped unexpectedly: {exc}")
            return 1
        finally:
            self._mouse_controller.close()

    def _draw_status(
        self,
        frame: object,
        status: MouseControlStatus,
        fps: float,
    ) -> None:
        fps_label = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        lines = (
            f"Mouse: {status.action}",
            f"Gesture: {status.gesture}",
            fps_label,
        )
        for index, line in enumerate(lines):
            self._cv2.putText(
                frame,
                line,
                (12, 32 + index * 24),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
    def _print_controls() -> None:
        print("Mouse control is active. Press Q to quit.")
        print("Pointing Up: move cursor")
        print("Peace Sign: left click")
        print("Thumbs Up: right click")
        print("Closed Fist: drag")
        print("Open Palm: scroll")
        print("Custom gesture named 'Double Click': double click")

    def _flip_frame(self, frame: object) -> object:
        try:
            self._cv2.flip(frame, 1, frame)
            return frame
        except Exception:
            return self._cv2.flip(frame, 1)

    @staticmethod
    def _load_cv2() -> Any:
        try:
            return importlib.import_module("cv2")
        except ImportError as exc:
            raise WebcamError(
                "OpenCV is required for mouse control. "
                "Install it with: pip install opencv-python"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the mouse-control command-line parser."""
    parser = argparse.ArgumentParser(
        description="Control the mouse with recognized hand gestures.",
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
    """Run gesture-driven mouse control."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        app = MouseControlApplication()
        return app.run(camera_index=args.camera, max_cameras=args.max_cameras)
    except (HandDetectionError, MouseControlError, WebcamError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
