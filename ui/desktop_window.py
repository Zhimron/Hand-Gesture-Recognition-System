"""PySide6 desktop interface for hand gesture recognition."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from PySide6.QtCore import QObject
from PySide6.QtCore import Qt
from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtGui import QImage
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QFrame
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from gestures.custom_gesture import CustomGestureStore
from gestures.custom_gesture import LandmarkSample
from services.keyboard_controller import KeyboardAutomationController
from services.keyboard_controller import KeyboardControlError
from services.keyboard_controller import KeyboardControlStatus
from services.mouse_controller import MouseControlError
from services.mouse_controller import MouseControlStatus
from services.mouse_controller import MouseControllerConfig
from services.mouse_controller import MouseGestureController
from vision.hand_detection import DetectedHand
from vision.hand_detection import HandDetectionError
from vision.hand_detection import HandDetectionService
from vision.webcam import CameraInfo
from vision.webcam import CameraService
from vision.webcam import WebcamError


@dataclass(frozen=True)
class DesktopSettings:
    """Runtime settings for the desktop interface."""

    max_cameras: int = 5
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_mouse_control: bool = False
    enable_keyboard_automation: bool = False
    mouse_smoothing: float = 0.28
    mouse_input_margin: float = 0.08
    mouse_click_cooldown_ms: int = 450
    mouse_scroll_sensitivity: float = 90.0
    keyboard_cooldown_ms: int = 650


@dataclass(frozen=True)
class PreviewFrame:
    """Frame and recognition metadata sent from the camera worker."""

    image: QImage
    fps: float
    gesture: str
    handedness: str
    hand_count: int
    detected_hands: tuple[DetectedHand, ...]
    mouse_status: MouseControlStatus
    keyboard_status: KeyboardControlStatus


class CameraWorker(QThread):
    """Background thread for capture, recognition, and optional automation."""

    frame_ready = Signal(object)
    error = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        camera_index: int,
        settings: DesktopSettings,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._settings = settings
        self._running = True

    def stop(self) -> None:
        """Request the worker to stop."""
        self._running = False

    def run(self) -> None:
        """Capture frames, annotate hands, and emit UI-ready packets."""
        capture = None
        detector = None
        mouse_controller = None
        try:
            cv2 = importlib.import_module("cv2")
            capture = CameraService().open_camera(self._camera_index)
            detector = HandDetectionService(
                min_detection_confidence=self._settings.min_detection_confidence,
                min_tracking_confidence=self._settings.min_tracking_confidence,
            )
            mouse_controller = self._create_mouse_controller()
            keyboard_controller = self._create_keyboard_controller()
            fps_counter = WorkerFpsCounter()

            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.error.emit(
                        "Camera feed stopped. The camera may have disconnected."
                    )
                    break

                frame = cv2.flip(frame, 1)
                detected_hands = detector.detect_and_draw(frame)
                timestamp_ms = self._timestamp_ms()
                mouse_status = self._update_mouse(
                    mouse_controller,
                    detected_hands,
                    timestamp_ms,
                )
                keyboard_status = self._update_keyboard(
                    keyboard_controller,
                    detected_hands,
                    timestamp_ms,
                )
                fps = fps_counter.update()
                qimage = self._frame_to_image(cv2, frame)
                self.frame_ready.emit(
                    PreviewFrame(
                        image=qimage,
                        fps=fps,
                        gesture=self._gesture_label(detected_hands),
                        handedness=self._handedness_label(detected_hands),
                        hand_count=len(detected_hands),
                        detected_hands=tuple(detected_hands),
                        mouse_status=mouse_status,
                        keyboard_status=keyboard_status,
                    )
                )
                self.msleep(1)

        except (
            HandDetectionError,
            ImportError,
            KeyboardControlError,
            MouseControlError,
            WebcamError,
        ) as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Desktop app stopped unexpectedly: {exc}")
        finally:
            if mouse_controller is not None:
                mouse_controller.close()
            if detector is not None:
                detector.close()
            if capture is not None:
                capture.release()
            self.stopped.emit()

    def _create_mouse_controller(self) -> MouseGestureController | None:
        if not self._settings.enable_mouse_control:
            return None

        return MouseGestureController(
            config=MouseControllerConfig(
                smoothing=self._settings.mouse_smoothing,
                input_margin=self._settings.mouse_input_margin,
                click_cooldown_ms=self._settings.mouse_click_cooldown_ms,
                scroll_sensitivity=self._settings.mouse_scroll_sensitivity,
            )
        )

    def _create_keyboard_controller(self) -> KeyboardAutomationController | None:
        if not self._settings.enable_keyboard_automation:
            return None

        return KeyboardAutomationController(
            trigger_cooldown_ms=self._settings.keyboard_cooldown_ms
        )

    @staticmethod
    def _update_mouse(
        mouse_controller: MouseGestureController | None,
        detected_hands: list[DetectedHand],
        timestamp_ms: int,
    ) -> MouseControlStatus:
        if mouse_controller is None:
            return MouseControlStatus(action="Disabled")
        return mouse_controller.update(detected_hands, timestamp_ms)

    @staticmethod
    def _update_keyboard(
        keyboard_controller: KeyboardAutomationController | None,
        detected_hands: list[DetectedHand],
        timestamp_ms: int,
    ) -> KeyboardControlStatus:
        if keyboard_controller is None:
            return KeyboardControlStatus(action="Disabled")
        return keyboard_controller.update(detected_hands, timestamp_ms)

    @staticmethod
    def _frame_to_image(cv2_module: Any, frame: object) -> QImage:
        rgb_frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return image.copy()

    @staticmethod
    def _gesture_label(detected_hands: list[DetectedHand]) -> str:
        if not detected_hands:
            return "None"
        return detected_hands[0].gesture.label

    @staticmethod
    def _handedness_label(detected_hands: list[DetectedHand]) -> str:
        if not detected_hands:
            return "None"
        return detected_hands[0].handedness

    @staticmethod
    def _timestamp_ms() -> int:
        return int(perf_counter() * 1000)


@dataclass
class WorkerFpsCounter:
    """Simple smoothed FPS counter for the capture worker."""

    smoothing: float = 0.9
    last_timestamp: float | None = None
    fps: float = 0.0

    def update(self) -> float:
        """Record one frame and return the current FPS estimate."""
        current_timestamp = perf_counter()
        if self.last_timestamp is None:
            self.last_timestamp = current_timestamp
            return self.fps

        elapsed = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        if elapsed <= 0:
            return self.fps

        instant_fps = 1.0 / elapsed
        if self.fps <= 0:
            self.fps = instant_fps
        else:
            self.fps = (
                self.smoothing * self.fps
                + (1.0 - self.smoothing) * instant_fps
            )
        return self.fps


class SettingsDialog(QDialog):
    """Dialog for desktop interface runtime settings."""

    def __init__(
        self,
        settings: DesktopSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        self._max_cameras_input = QSpinBox()
        self._max_cameras_input.setRange(1, 20)
        self._max_cameras_input.setValue(settings.max_cameras)

        self._detection_input = QDoubleSpinBox()
        self._detection_input.setRange(0.1, 0.95)
        self._detection_input.setSingleStep(0.05)
        self._detection_input.setValue(settings.min_detection_confidence)

        self._tracking_input = QDoubleSpinBox()
        self._tracking_input.setRange(0.1, 0.95)
        self._tracking_input.setSingleStep(0.05)
        self._tracking_input.setValue(settings.min_tracking_confidence)

        self._mouse_smoothing_input = QDoubleSpinBox()
        self._mouse_smoothing_input.setRange(0.05, 1.0)
        self._mouse_smoothing_input.setSingleStep(0.05)
        self._mouse_smoothing_input.setValue(settings.mouse_smoothing)

        self._mouse_margin_input = QDoubleSpinBox()
        self._mouse_margin_input.setRange(0.0, 0.4)
        self._mouse_margin_input.setSingleStep(0.01)
        self._mouse_margin_input.setValue(settings.mouse_input_margin)

        self._mouse_cooldown_input = QSpinBox()
        self._mouse_cooldown_input.setRange(0, 2000)
        self._mouse_cooldown_input.setValue(settings.mouse_click_cooldown_ms)

        self._scroll_sensitivity_input = QDoubleSpinBox()
        self._scroll_sensitivity_input.setRange(10.0, 300.0)
        self._scroll_sensitivity_input.setSingleStep(5.0)
        self._scroll_sensitivity_input.setValue(settings.mouse_scroll_sensitivity)

        self._keyboard_cooldown_input = QSpinBox()
        self._keyboard_cooldown_input.setRange(0, 3000)
        self._keyboard_cooldown_input.setValue(settings.keyboard_cooldown_ms)

        form = QFormLayout()
        form.addRow("Camera scan limit", self._max_cameras_input)
        form.addRow("Detection confidence", self._detection_input)
        form.addRow("Tracking confidence", self._tracking_input)
        form.addRow("Mouse smoothing", self._mouse_smoothing_input)
        form.addRow("Mouse input margin", self._mouse_margin_input)
        form.addRow("Mouse click cooldown ms", self._mouse_cooldown_input)
        form.addRow("Scroll sensitivity", self._scroll_sensitivity_input)
        form.addRow("Keyboard cooldown ms", self._keyboard_cooldown_input)

        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def settings(self, current_settings: DesktopSettings) -> DesktopSettings:
        """Return settings selected in the dialog."""
        return replace(
            current_settings,
            max_cameras=self._max_cameras_input.value(),
            min_detection_confidence=self._detection_input.value(),
            min_tracking_confidence=self._tracking_input.value(),
            mouse_smoothing=self._mouse_smoothing_input.value(),
            mouse_input_margin=self._mouse_margin_input.value(),
            mouse_click_cooldown_ms=self._mouse_cooldown_input.value(),
            mouse_scroll_sensitivity=self._scroll_sensitivity_input.value(),
            keyboard_cooldown_ms=self._keyboard_cooldown_input.value(),
        )


class DesktopMainWindow(QMainWindow):
    """Main PySide6 window for live preview and app controls."""

    def __init__(
        self,
        settings: DesktopSettings,
        camera_service: CameraService | None = None,
        custom_gesture_store: CustomGestureStore | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._camera_service = camera_service or CameraService()
        self._custom_gesture_store = custom_gesture_store or CustomGestureStore()
        self._worker: CameraWorker | None = None
        self._last_pixmap: QPixmap | None = None
        self._recording_name = ""
        self._recording_target = 0
        self._recorded_samples: list[LandmarkSample] = []

        self.setWindowTitle("Hand Gesture Recognition")
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_dark_theme()
        self.refresh_cameras()

    def refresh_cameras(self) -> None:
        """Refresh available cameras in the selector."""
        self._camera_combo.clear()
        cameras = self._camera_service.detect_available_cameras(
            self._settings.max_cameras
        )
        for camera in cameras:
            self._camera_combo.addItem(self._camera_label(camera), camera.index)

        if not cameras:
            self._camera_combo.addItem("No cameras detected", None)
            self._status_label.setText("No cameras detected")
            self._start_button.setEnabled(False)
        else:
            self._status_label.setText("Ready")
            self._start_button.setEnabled(True)

    def start_preview(self) -> None:
        """Start the selected camera preview."""
        if self._worker is not None and self._worker.isRunning():
            return

        camera_index = self._camera_combo.currentData()
        if camera_index is None:
            QMessageBox.warning(self, "Camera", "No camera is selected.")
            return

        self._apply_runtime_toggles()
        self._worker = CameraWorker(int(camera_index), self._settings, self)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._worker.start()

        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._camera_combo.setEnabled(False)
        self._refresh_button.setEnabled(False)
        self._settings_button.setEnabled(False)
        self._mouse_enabled_check.setEnabled(False)
        self._keyboard_enabled_check.setEnabled(False)
        self._record_button.setEnabled(True)
        self._status_label.setText("Running")

    def stop_preview(self) -> None:
        """Stop the active preview worker."""
        if self._worker is None:
            return

        self._worker.stop()
        self._worker.wait(2000)

    def open_settings(self) -> None:
        """Open the settings dialog."""
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._settings = dialog.settings(self._settings)
        self.refresh_cameras()

    def start_custom_recording(self) -> None:
        """Start recording a custom gesture from live hand landmarks."""
        if self._worker is None or not self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Custom gesture",
                "Start the camera before recording a custom gesture.",
            )
            return

        name = self._custom_name_input.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Custom gesture",
                "Enter a gesture name before recording.",
            )
            return

        self._recording_name = name
        self._recording_target = self._custom_sample_count.value()
        self._recorded_samples = []
        self._record_button.setEnabled(False)
        self._recording_status_label.setText(f"Recording {name}")
        self._recording_progress.setMaximum(self._recording_target)
        self._recording_progress.setValue(0)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the worker before closing."""
        self.stop_preview()
        event.accept()

    def resizeEvent(self, event: object) -> None:
        """Keep the preview pixmap scaled to the available area."""
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        title = QLabel("Hand Gesture Recognition")
        title.setObjectName("title")
        root_layout.addWidget(title)

        toolbar = self._build_toolbar()
        root_layout.addLayout(toolbar)

        content = QGridLayout()
        content.setSpacing(14)

        self._preview_label = QLabel("Camera preview")
        self._preview_label.setObjectName("preview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(560, 380)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        content.addWidget(self._preview_label, 0, 0)

        side_panel = self._build_side_panel()
        content.addWidget(side_panel, 0, 1)
        content.setColumnStretch(0, 4)
        content.setColumnStretch(1, 2)
        root_layout.addLayout(content, 1)
        self.setCentralWidget(root)

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(240)
        self._refresh_button = QPushButton("Refresh")
        self._start_button = QPushButton("Start")
        self._stop_button = QPushButton("Stop")
        self._settings_button = QPushButton("Settings")
        self._stop_button.setEnabled(False)

        self._refresh_button.clicked.connect(self.refresh_cameras)
        self._start_button.clicked.connect(self.start_preview)
        self._stop_button.clicked.connect(self.stop_preview)
        self._settings_button.clicked.connect(self.open_settings)

        toolbar.addWidget(QLabel("Camera"))
        toolbar.addWidget(self._camera_combo, 1)
        toolbar.addWidget(self._refresh_button)
        toolbar.addWidget(self._start_button)
        toolbar.addWidget(self._stop_button)
        toolbar.addWidget(self._settings_button)
        return toolbar

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tabs = QTabWidget()
        tabs.addTab(self._build_status_tab(), "Status")
        tabs.addTab(self._build_custom_gesture_tab(), "Custom")
        tabs.addTab(self._build_automation_tab(), "Control")
        layout.addWidget(tabs)
        return panel

    def _build_status_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self._fps_label = QLabel("FPS: --")
        self._gesture_label = QLabel("Gesture: None")
        self._handedness_label = QLabel("Hand: None")
        self._hand_count_label = QLabel("Hands: 0")
        self._status_label = QLabel("Ready")

        for label in (
            self._fps_label,
            self._gesture_label,
            self._handedness_label,
            self._hand_count_label,
            self._status_label,
        ):
            label.setObjectName("metric")
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addStretch(1)
        return tab

    def _build_custom_gesture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self._custom_name_input = QLineEdit()
        self._custom_name_input.setPlaceholderText("Gesture name")
        self._custom_sample_count = QSpinBox()
        self._custom_sample_count.setRange(5, 120)
        self._custom_sample_count.setValue(30)
        self._record_button = QPushButton("Record Gesture")
        self._record_button.setEnabled(False)
        self._record_button.clicked.connect(self.start_custom_recording)
        self._recording_progress = QProgressBar()
        self._recording_progress.setRange(0, 30)
        self._recording_progress.setValue(0)
        self._recording_status_label = QLabel("Start the camera to record.")
        self._recording_status_label.setObjectName("metric")
        self._recording_status_label.setWordWrap(True)

        layout.addWidget(QLabel("Name"))
        layout.addWidget(self._custom_name_input)
        layout.addWidget(QLabel("Samples"))
        layout.addWidget(self._custom_sample_count)
        layout.addWidget(self._record_button)
        layout.addWidget(self._recording_progress)
        layout.addWidget(self._recording_status_label)
        layout.addStretch(1)
        return tab

    def _build_automation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self._mouse_enabled_check = QCheckBox("Enable mouse control")
        self._keyboard_enabled_check = QCheckBox("Enable keyboard automation")
        self._mouse_action_label = QLabel("Mouse: Disabled")
        self._keyboard_action_label = QLabel("Keyboard: Disabled")
        self._mouse_setup_label = QLabel(
            "Mouse setup: Pointing Up moves, Peace Sign clicks, "
            "Thumbs Up right-clicks, Closed Fist drags, Open Palm scrolls."
        )
        self._keyboard_setup_label = QLabel(
            "Keyboard mappings use config/keyboard_mappings.json."
        )

        for label in (
            self._mouse_action_label,
            self._keyboard_action_label,
            self._mouse_setup_label,
            self._keyboard_setup_label,
        ):
            label.setObjectName("metric")
            label.setWordWrap(True)

        layout.addWidget(self._mouse_enabled_check)
        layout.addWidget(self._keyboard_enabled_check)
        layout.addWidget(self._mouse_action_label)
        layout.addWidget(self._keyboard_action_label)
        layout.addWidget(self._mouse_setup_label)
        layout.addWidget(self._keyboard_setup_label)
        layout.addStretch(1)
        return tab

    def _on_frame_ready(self, packet: PreviewFrame) -> None:
        self._last_pixmap = QPixmap.fromImage(packet.image)
        fps_text = f"FPS: {packet.fps:.1f}" if packet.fps > 0 else "FPS: --"
        self._fps_label.setText(fps_text)
        self._gesture_label.setText(f"Gesture: {packet.gesture}")
        self._handedness_label.setText(f"Hand: {packet.handedness}")
        self._hand_count_label.setText(f"Hands: {packet.hand_count}")
        self._mouse_action_label.setText(
            f"Mouse: {packet.mouse_status.action} ({packet.mouse_status.gesture})"
        )
        keyboard_keys = "+".join(packet.keyboard_status.keys) or "None"
        self._keyboard_action_label.setText(
            f"Keyboard: {packet.keyboard_status.action} "
            f"({packet.keyboard_status.gesture}, {keyboard_keys})"
        )
        self._collect_recording_sample(packet.detected_hands)
        self._update_preview_pixmap()

    def _collect_recording_sample(
        self,
        detected_hands: tuple[DetectedHand, ...],
    ) -> None:
        if not self._recording_name or not detected_hands:
            return

        landmarks = detected_hands[0].landmarks
        if len(landmarks) < 21:
            return

        self._recorded_samples.append(landmarks)
        self._recording_progress.setValue(len(self._recorded_samples))
        if len(self._recorded_samples) < self._recording_target:
            return

        try:
            self._custom_gesture_store.save_gesture(
                self._recording_name,
                self._recorded_samples,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Custom gesture", str(exc))
            self._recording_status_label.setText("Recording failed.")
        else:
            self._recording_status_label.setText(
                f"Saved '{self._recording_name}' to "
                f"{self._custom_gesture_store.path}"
            )
        finally:
            self._recording_name = ""
            self._recording_target = 0
            self._recorded_samples = []
            self._record_button.setEnabled(True)

    def _on_worker_error(self, message: str) -> None:
        self._status_label.setText("Stopped")
        QMessageBox.warning(self, "Preview stopped", message)

    def _on_worker_stopped(self) -> None:
        self._start_button.setEnabled(self._camera_combo.currentData() is not None)
        self._stop_button.setEnabled(False)
        self._camera_combo.setEnabled(True)
        self._refresh_button.setEnabled(True)
        self._settings_button.setEnabled(True)
        self._mouse_enabled_check.setEnabled(True)
        self._keyboard_enabled_check.setEnabled(True)
        self._record_button.setEnabled(False)
        self._status_label.setText("Stopped")
        self._worker = None

    def _update_preview_pixmap(self) -> None:
        if self._last_pixmap is None:
            return

        scaled = self._last_pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def _apply_runtime_toggles(self) -> None:
        self._settings = replace(
            self._settings,
            enable_mouse_control=self._mouse_enabled_check.isChecked(),
            enable_keyboard_automation=self._keyboard_enabled_check.isChecked(),
        )

    @staticmethod
    def _camera_label(camera: CameraInfo) -> str:
        size = "unknown"
        if camera.width is not None and camera.height is not None:
            size = f"{camera.width}x{camera.height}"
        return f"[{camera.index}] {camera.name} ({size})"

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #101318;
                color: #edf1f7;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#preview {
                background: #07090d;
                border: 1px solid #273244;
                border-radius: 8px;
                color: #687386;
            }
            QFrame#panel {
                background: #171c24;
                border: 1px solid #273244;
                border-radius: 8px;
            }
            QLabel#metric, QProgressBar {
                background: #111720;
                border: 1px solid #273244;
                border-radius: 6px;
                padding: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #273244;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #111720;
                border: 1px solid #273244;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 12px;
            }
            QTabBar::tab:selected {
                background: #1f6feb;
                border-color: #388bfd;
            }
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #1f6feb;
                border: 1px solid #388bfd;
                border-radius: 6px;
                color: #ffffff;
                min-height: 32px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #2f81f7;
            }
            QPushButton:disabled {
                background: #283142;
                border-color: #374151;
                color: #8b949e;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #111720;
                border-color: #273244;
            }
            QCheckBox {
                spacing: 8px;
                padding: 6px 0;
            }
            QProgressBar::chunk {
                background: #2f81f7;
                border-radius: 4px;
            }
            QDialog {
                background: #101318;
            }
            """
        )


def run_desktop_app(settings: DesktopSettings) -> int:
    """Run the PySide6 desktop interface."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = DesktopMainWindow(settings)
    window.show()
    return app.exec()
