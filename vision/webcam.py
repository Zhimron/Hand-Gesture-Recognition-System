"""Camera discovery and opening utilities for webcam initialization."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias


_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5


class VideoCaptureProtocol(Protocol):
    """Subset of OpenCV's VideoCapture API used by this feature."""

    def isOpened(self) -> bool:
        """Return whether the camera stream is open."""

    def read(self) -> tuple[bool, object | None]:
        """Read a frame from the camera."""

    def release(self) -> None:
        """Release the camera resource."""

    def get(self, prop_id: int) -> float:
        """Return a capture property value."""


CaptureFactory: TypeAlias = Callable[[int], VideoCaptureProtocol]


@dataclass(frozen=True)
class CameraInfo:
    """Metadata for a detected camera."""

    index: int
    name: str
    width: int | None = None
    height: int | None = None
    reported_fps: float | None = None


class WebcamError(RuntimeError):
    """Raised when webcam setup or streaming cannot continue."""


class CameraService:
    """Discover available webcams and open a selected camera."""

    def __init__(
        self,
        capture_factory: CaptureFactory | None = None,
        read_attempts: int = 3,
    ) -> None:
        self._capture_factory = capture_factory or self._create_capture
        self._read_attempts = max(1, read_attempts)

    def detect_available_cameras(self, max_cameras: int = 5) -> list[CameraInfo]:
        """Return cameras that can be opened and can produce a frame."""
        cameras: list[CameraInfo] = []

        for index in range(max(0, max_cameras)):
            capture: VideoCaptureProtocol | None = None
            try:
                capture = self._capture_factory(index)
                if not capture.isOpened():
                    continue

                if not self._can_read_frame(capture):
                    continue

                cameras.append(
                    CameraInfo(
                        index=index,
                        name=f"Camera {index}",
                        width=self._read_int_property(
                            capture,
                            _CAP_PROP_FRAME_WIDTH,
                        ),
                        height=self._read_int_property(
                            capture,
                            _CAP_PROP_FRAME_HEIGHT,
                        ),
                        reported_fps=self._read_float_property(
                            capture,
                            _CAP_PROP_FPS,
                        ),
                    )
                )
            except Exception:
                continue
            finally:
                if capture is not None:
                    capture.release()

        return cameras

    def open_camera(self, camera_index: int) -> VideoCaptureProtocol:
        """Open the selected camera index or raise a WebcamError."""
        capture = self._capture_factory(camera_index)
        if not capture.isOpened():
            capture.release()
            raise WebcamError(f"Unable to open camera {camera_index}.")
        return capture

    def _can_read_frame(self, capture: VideoCaptureProtocol) -> bool:
        for _ in range(self._read_attempts):
            ok, frame = capture.read()
            if ok and frame is not None:
                return True
        return False

    @staticmethod
    def _read_int_property(
        capture: VideoCaptureProtocol,
        property_id: int,
    ) -> int | None:
        value = CameraService._read_float_property(capture, property_id)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _read_float_property(
        capture: VideoCaptureProtocol,
        property_id: int,
    ) -> float | None:
        try:
            value = float(capture.get(property_id))
        except Exception:
            return None

        if value <= 0:
            return None
        return value

    @staticmethod
    def _create_capture(index: int) -> VideoCaptureProtocol:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise WebcamError(
                "OpenCV is required for camera access. "
                "Install it with: pip install opencv-python"
            ) from exc

        return cv2.VideoCapture(index)
