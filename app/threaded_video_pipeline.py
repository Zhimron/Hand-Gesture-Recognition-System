"""Threaded camera, inference, and rendering pipeline for live video apps."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Callable, Protocol

from app.webcam_app import FpsCounter
from vision.hand_detection import DetectedHand
from vision.webcam import VideoCaptureProtocol, WebcamError


_CAP_PROP_BUFFERSIZE = 38


class HandDetectorProtocol(Protocol):
    """Subset of the hand detector API required by the threaded pipeline."""

    def detect_and_draw(self, frame: object) -> list[DetectedHand]:
        """Draw hand annotations and return detected hand metadata."""

    def close(self) -> None:
        """Release detector resources."""


CaptureFactory = Callable[[], VideoCaptureProtocol]
DetectorFactory = Callable[[], HandDetectorProtocol]
FrameTransform = Callable[[object], object]
InferenceCallback = Callable[["ProcessedFrame"], bool | None]
OverlayRenderer = Callable[[object, "ProcessedFrame", float], None]


@dataclass(frozen=True)
class ThreadedPipelineConfig:
    """Tuning values for the live video pipeline."""

    frame_queue_size: int = 1
    render_queue_size: int = 1
    queue_timeout_seconds: float = 0.05
    wait_key_delay_ms: int = 1
    max_camera_fps: float = 60.0
    camera_buffer_size: int = 1


@dataclass(frozen=True)
class CapturedFrame:
    """Frame captured from the camera thread."""

    sequence_id: int
    frame: object
    captured_at: float


@dataclass(frozen=True)
class ProcessedFrame:
    """Frame annotated by the inference thread and ready to render."""

    sequence_id: int
    frame: object
    detected_hands: tuple[DetectedHand, ...]
    captured_at: float
    processed_at: float
    timestamp_ms: int


class ThreadedHandDetectionPipeline:
    """Run camera capture, hand inference, and rendering on separate threads."""

    def __init__(
        self,
        capture_factory: CaptureFactory,
        detector_factory: DetectorFactory,
        cv2_module: Any,
        window_name: str,
        frame_transform: FrameTransform | None = None,
        on_inference: InferenceCallback | None = None,
        draw_overlay: OverlayRenderer | None = None,
        config: ThreadedPipelineConfig | None = None,
        quit_key: str = "q",
    ) -> None:
        self._capture_factory = capture_factory
        self._detector_factory = detector_factory
        self._cv2 = cv2_module
        self._window_name = window_name
        self._frame_transform = frame_transform or self._identity_frame
        self._on_inference = on_inference or self._continue_after_inference
        self._draw_overlay = draw_overlay or self._skip_overlay
        self._config = config or ThreadedPipelineConfig()
        self._quit_key_code = ord(quit_key.lower()[0])
        self._stop_event = Event()
        self._error_lock = Lock()
        self._error: BaseException | None = None
        self._frame_queue: Queue[CapturedFrame] = Queue(
            maxsize=max(1, self._config.frame_queue_size),
        )
        self._render_queue: Queue[ProcessedFrame] = Queue(
            maxsize=max(1, self._config.render_queue_size),
        )

    def run(self) -> int:
        """Start all workers and block until the pipeline exits."""
        threads = [
            Thread(
                target=self._camera_worker,
                name="camera-thread",
                daemon=True,
            ),
            Thread(
                target=self._inference_worker,
                name="inference-thread",
                daemon=True,
            ),
            Thread(
                target=self._rendering_worker,
                name="rendering-thread",
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=self._config.queue_timeout_seconds)
        except KeyboardInterrupt:
            self.stop()
            raise
        finally:
            self.stop()
            for thread in threads:
                thread.join(timeout=1.0)
            self._destroy_windows()

        error = self._first_error()
        if error is not None:
            raise error

        return 0

    def stop(self) -> None:
        """Request all workers to stop."""
        self._stop_event.set()

    def _camera_worker(self) -> None:
        capture: VideoCaptureProtocol | None = None
        frame_interval = self._frame_interval()

        try:
            capture = self._capture_factory()
            self._configure_capture(capture)
            sequence_id = 0

            while not self._stop_event.is_set():
                started_at = perf_counter()
                ok, frame = capture.read()
                if self._stop_event.is_set():
                    break

                if not ok or frame is None:
                    self._fail(
                        WebcamError(
                            "Camera feed stopped. "
                            "The camera may have disconnected."
                        )
                    )
                    break

                self._put_latest(
                    self._frame_queue,
                    CapturedFrame(
                        sequence_id=sequence_id,
                        frame=frame,
                        captured_at=started_at,
                    ),
                )
                sequence_id += 1
                self._sleep_remaining_frame_interval(
                    started_at,
                    frame_interval,
                )
        except BaseException as exc:
            self._fail(exc)
        finally:
            if capture is not None:
                capture.release()

    def _inference_worker(self) -> None:
        detector: HandDetectorProtocol | None = None

        try:
            while not self._stop_event.is_set():
                try:
                    captured_frame = self._frame_queue.get(
                        timeout=self._config.queue_timeout_seconds,
                    )
                except Empty:
                    continue

                if detector is None:
                    detector = self._detector_factory()

                frame = self._frame_transform(captured_frame.frame)
                detected_hands = tuple(detector.detect_and_draw(frame))
                processed_at = perf_counter()
                processed_frame = ProcessedFrame(
                    sequence_id=captured_frame.sequence_id,
                    frame=frame,
                    detected_hands=detected_hands,
                    captured_at=captured_frame.captured_at,
                    processed_at=processed_at,
                    timestamp_ms=int(processed_at * 1000),
                )
                should_stop = bool(self._on_inference(processed_frame))
                self._put_latest(self._render_queue, processed_frame)
                if should_stop:
                    self.stop()
        except BaseException as exc:
            self._fail(exc)
        finally:
            if detector is not None:
                detector.close()

    def _rendering_worker(self) -> None:
        fps_counter = FpsCounter()

        try:
            while not self._stop_event.is_set():
                try:
                    processed_frame = self._render_queue.get(
                        timeout=self._config.queue_timeout_seconds,
                    )
                except Empty:
                    self._poll_quit_key()
                    continue

                fps = fps_counter.update()
                self._draw_overlay(processed_frame.frame, processed_frame, fps)
                self._cv2.imshow(self._window_name, processed_frame.frame)
                self._poll_quit_key()
        except BaseException as exc:
            self._fail(exc)

    def _configure_capture(self, capture: VideoCaptureProtocol) -> None:
        set_property = getattr(capture, "set", None)
        if not callable(set_property) or self._config.camera_buffer_size <= 0:
            return

        property_id = getattr(
            self._cv2,
            "CAP_PROP_BUFFERSIZE",
            _CAP_PROP_BUFFERSIZE,
        )
        try:
            set_property(property_id, self._config.camera_buffer_size)
        except Exception:
            return

    def _poll_quit_key(self) -> None:
        wait_key = getattr(self._cv2, "waitKey", None)
        if not callable(wait_key):
            return

        key = wait_key(self._config.wait_key_delay_ms) & 0xFF
        if key == self._quit_key_code:
            self.stop()

    def _destroy_windows(self) -> None:
        destroy_all = getattr(self._cv2, "destroyAllWindows", None)
        if callable(destroy_all):
            destroy_all()

    def _fail(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc
        self.stop()

    def _first_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def _frame_interval(self) -> float:
        if self._config.max_camera_fps <= 0:
            return 0.0
        return 1.0 / self._config.max_camera_fps

    def _sleep_remaining_frame_interval(
        self,
        started_at: float,
        frame_interval: float,
    ) -> None:
        if frame_interval <= 0:
            return

        while not self._stop_event.is_set():
            remaining = frame_interval - (perf_counter() - started_at)
            if remaining <= 0:
                return
            self._stop_event.wait(min(remaining, 0.005))

    @staticmethod
    def _put_latest(queue: Queue[Any], item: Any) -> None:
        try:
            queue.put_nowait(item)
            return
        except Full:
            pass

        try:
            queue.get_nowait()
        except Empty:
            pass

        try:
            queue.put_nowait(item)
        except Full:
            pass

    @staticmethod
    def _identity_frame(frame: object) -> object:
        return frame

    @staticmethod
    def _continue_after_inference(_processed_frame: ProcessedFrame) -> bool:
        return False

    @staticmethod
    def _skip_overlay(
        _frame: object,
        _processed_frame: ProcessedFrame,
        _fps: float,
    ) -> None:
        return
