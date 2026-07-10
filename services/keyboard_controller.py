"""Keyboard automation driven by recognized hand gestures."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from vision.hand_detection import DetectedHand


DEFAULT_KEYBOARD_MAPPING_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "keyboard_mappings.json"
)
DEFAULT_KEYBOARD_MAPPINGS = {
    "Thumbs Up": ("ctrl", "c"),
    "Peace": ("ctrl", "v"),
    "Peace Sign": ("ctrl", "v"),
    "Open Palm": ("esc",),
}


class KeyboardControlError(RuntimeError):
    """Raised when keyboard automation cannot be initialized."""


class KeyboardBackend(Protocol):
    """Keyboard backend operations required by the controller."""

    def press(self, key: str) -> None:
        """Press a single key."""

    def hotkey(self, keys: Sequence[str]) -> None:
        """Press a key combination."""


class PyAutoGuiKeyboardBackend:
    """pyautogui-backed keyboard backend."""

    def __init__(self) -> None:
        try:
            self._pyautogui = importlib.import_module("pyautogui")
        except ImportError as exc:
            raise KeyboardControlError(
                "pyautogui is required for keyboard automation. "
                "Install it with: pip install pyautogui"
            ) from exc

        self._pyautogui.PAUSE = 0

    def press(self, key: str) -> None:
        """Press a single key."""
        self._pyautogui.press(key)

    def hotkey(self, keys: Sequence[str]) -> None:
        """Press a key combination."""
        self._pyautogui.hotkey(*keys)


@dataclass(frozen=True)
class KeyboardAction:
    """Configured keyboard action for one gesture."""

    gesture: str
    keys: tuple[str, ...]

    @property
    def label(self) -> str:
        """Return a display label for the configured keys."""
        return "+".join(self.keys)


@dataclass(frozen=True)
class KeyboardControlStatus:
    """Last keyboard automation action status."""

    action: str = "Idle"
    gesture: str = "Unknown"
    keys: tuple[str, ...] = ()


class KeyboardMappingStore:
    """Persist gesture-to-key mappings as JSON configuration."""

    def __init__(
        self,
        path: str | Path = DEFAULT_KEYBOARD_MAPPING_PATH,
    ) -> None:
        self.path = Path(path)
        self._cached_mtime_ns: int | None = None
        self._cached_mappings: dict[str, tuple[str, ...]] = {}

    def ensure_exists(self) -> None:
        """Create the mapping file with defaults if it does not exist."""
        if self.path.exists():
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_mappings(dict(DEFAULT_KEYBOARD_MAPPINGS))

    def load(self) -> dict[str, tuple[str, ...]]:
        """Load mappings from JSON configuration."""
        self.ensure_exists()
        raw_data = json.loads(self.path.read_text(encoding="utf-8"))
        raw_mappings = raw_data.get("mappings", {})

        mappings = {
            str(gesture): tuple(
                normalize_key_name(str(key))
                for key in keys
                if str(key).strip()
            )
            for gesture, keys in raw_mappings.items()
        }
        self._cached_mappings = {
            gesture: keys for gesture, keys in mappings.items() if keys
        }
        self._cached_mtime_ns = self.path.stat().st_mtime_ns
        return dict(self._cached_mappings)

    def load_if_changed(self) -> dict[str, tuple[str, ...]]:
        """Reload mappings only when the configuration file changed."""
        self.ensure_exists()
        current_mtime_ns = self.path.stat().st_mtime_ns
        if current_mtime_ns != self._cached_mtime_ns:
            return self.load()
        return dict(self._cached_mappings)

    def save_mapping(self, gesture: str, keys: Sequence[str]) -> KeyboardAction:
        """Save or replace one gesture mapping."""
        gesture_name = gesture.strip()
        if not gesture_name:
            raise ValueError("Gesture name cannot be empty.")

        normalized_keys = tuple(
            normalize_key_name(key) for key in keys if key.strip()
        )
        if not normalized_keys:
            raise ValueError("At least one key is required.")

        mappings = self.load_if_changed()
        mappings[gesture_name] = normalized_keys
        self._write_mappings(mappings)
        return KeyboardAction(gesture=gesture_name, keys=normalized_keys)

    def _write_mappings(self, mappings: dict[str, tuple[str, ...]]) -> None:
        payload = {
            "version": 1,
            "mappings": {
                gesture: list(keys)
                for gesture, keys in sorted(mappings.items())
            },
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._cached_mappings = dict(mappings)
        self._cached_mtime_ns = self.path.stat().st_mtime_ns


class KeyboardAutomationController:
    """Trigger configured keyboard actions from recognized gestures."""

    def __init__(
        self,
        keyboard_backend: KeyboardBackend | None = None,
        mapping_store: KeyboardMappingStore | None = None,
        trigger_cooldown_ms: int = 650,
    ) -> None:
        self._keyboard_backend = keyboard_backend or PyAutoGuiKeyboardBackend()
        self._mapping_store = mapping_store or KeyboardMappingStore()
        self._trigger_cooldown_ms = max(0, trigger_cooldown_ms)
        self._last_gesture = "Unknown"
        self._last_trigger_timestamp_ms = -self._trigger_cooldown_ms

    def update(
        self,
        detected_hands: Sequence[DetectedHand],
        timestamp_ms: int,
    ) -> KeyboardControlStatus:
        """Trigger a keyboard action for the current stable gesture."""
        if not detected_hands:
            self._last_gesture = "Unknown"
            return KeyboardControlStatus()

        gesture = detected_hands[0].gesture.label
        mappings = self._mapping_store.load_if_changed()
        keys = mappings.get(gesture)
        if not keys:
            self._last_gesture = gesture
            return KeyboardControlStatus(gesture=gesture)

        if not self._should_trigger(gesture, timestamp_ms):
            self._last_gesture = gesture
            return KeyboardControlStatus(action="Cooldown", gesture=gesture, keys=keys)

        self._trigger(keys)
        self._last_gesture = gesture
        self._last_trigger_timestamp_ms = timestamp_ms
        return KeyboardControlStatus(action="Triggered", gesture=gesture, keys=keys)

    def _trigger(self, keys: Sequence[str]) -> None:
        if len(keys) == 1:
            self._keyboard_backend.press(keys[0])
            return

        self._keyboard_backend.hotkey(keys)

    def _should_trigger(self, gesture: str, timestamp_ms: int) -> bool:
        if gesture == self._last_gesture:
            return False

        elapsed_ms = timestamp_ms - self._last_trigger_timestamp_ms
        return elapsed_ms >= self._trigger_cooldown_ms


def parse_key_sequence(raw_keys: str) -> tuple[str, ...]:
    """Parse a user-provided key sequence like ctrl+c or ctrl, c."""
    normalized = raw_keys.replace(",", "+")
    return tuple(
        normalize_key_name(part)
        for part in normalized.split("+")
        if part.strip()
    )


def normalize_key_name(key: str) -> str:
    """Normalize key aliases for pyautogui."""
    normalized = key.strip().lower()
    aliases = {
        "control": "ctrl",
        "escape": "esc",
        "return": "enter",
        "del": "delete",
        "cmd": "win",
        "command": "win",
    }
    return aliases.get(normalized, normalized)
