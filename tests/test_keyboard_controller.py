"""Tests for gesture-driven keyboard automation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from gestures.basic_gesture import DetectedGesture
from gestures.basic_gesture import GestureName
from services.keyboard_controller import KeyboardAutomationController
from services.keyboard_controller import KeyboardMappingStore
from services.keyboard_controller import parse_key_sequence
from vision.hand_detection import DetectedHand


class FakeKeyboardBackend:
    """Fake keyboard backend that records operations."""

    def __init__(self) -> None:
        self.operations: list[tuple[object, ...]] = []

    def press(self, key: str) -> None:
        """Record a single-key press."""
        self.operations.append(("press", key))

    def hotkey(self, keys: tuple[str, ...]) -> None:
        """Record a key combination."""
        self.operations.append(("hotkey", tuple(keys)))


def _hand(gesture_name: GestureName | str) -> DetectedHand:
    return DetectedHand(
        handedness="Right",
        confidence=0.95,
        gesture=DetectedGesture(gesture_name, confidence=0.95),
    )


class KeyboardMappingStoreTest(unittest.TestCase):
    """Unit tests for keyboard mapping configuration."""

    def test_load_creates_default_mapping_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "keyboard_mappings.json"
            store = KeyboardMappingStore(path)

            mappings = store.load()

        self.assertEqual(mappings["Thumbs Up"], ("ctrl", "c"))
        self.assertEqual(mappings["Peace"], ("ctrl", "v"))
        self.assertEqual(mappings["Peace Sign"], ("ctrl", "v"))
        self.assertEqual(mappings["Open Palm"], ("esc",))

    def test_save_mapping_persists_custom_mapping(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "keyboard_mappings.json"
            store = KeyboardMappingStore(path)

            action = store.save_mapping("Pointing Up", ("ctrl", "shift", "p"))
            loaded = store.load()

        self.assertEqual(action.label, "ctrl+shift+p")
        self.assertEqual(loaded["Pointing Up"], ("ctrl", "shift", "p"))

    def test_parse_key_sequence_normalizes_aliases(self) -> None:
        keys = parse_key_sequence("control + shift + escape")

        self.assertEqual(keys, ("ctrl", "shift", "esc"))

    def test_load_if_changed_reloads_external_config_updates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "keyboard_mappings.json"
            reader_store = KeyboardMappingStore(path)
            writer_store = KeyboardMappingStore(path)

            before = reader_store.load_if_changed()
            time.sleep(0.01)
            writer_store.save_mapping("Custom Action", ("ctrl", "s"))
            after = reader_store.load_if_changed()

        self.assertIn("Thumbs Up", before)
        self.assertEqual(after["Custom Action"], ("ctrl", "s"))


class KeyboardAutomationControllerTest(unittest.TestCase):
    """Unit tests for KeyboardAutomationController."""

    def test_thumbs_up_triggers_ctrl_c(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
            )

            status = controller.update([_hand(GestureName.THUMBS_UP)], 0)

        self.assertEqual(status.action, "Triggered")
        self.assertEqual(status.keys, ("ctrl", "c"))
        self.assertEqual(backend.operations, [("hotkey", ("ctrl", "c"))])

    def test_open_palm_triggers_escape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
            )

            status = controller.update([_hand(GestureName.OPEN_PALM)], 0)

        self.assertEqual(status.keys, ("esc",))
        self.assertEqual(backend.operations, [("press", "esc")])

    def test_custom_mapping_triggers_configured_hotkey(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            store.save_mapping("Custom Wave", ("ctrl", "alt", "w"))
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
            )

            status = controller.update([_hand("Custom Wave")], 0)

        self.assertEqual(status.gesture, "Custom Wave")
        self.assertEqual(status.keys, ("ctrl", "alt", "w"))
        self.assertEqual(backend.operations, [("hotkey", ("ctrl", "alt", "w"))])

    def test_same_gesture_does_not_repeat_until_reset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
            )

            controller.update([_hand(GestureName.PEACE_SIGN)], 0)
            status = controller.update([_hand(GestureName.PEACE_SIGN)], 700)

        self.assertEqual(status.action, "Cooldown")
        self.assertEqual(backend.operations, [("hotkey", ("ctrl", "v"))])

    def test_no_hand_resets_transition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
                trigger_cooldown_ms=0,
            )

            controller.update([_hand(GestureName.PEACE_SIGN)], 0)
            controller.update([], 1)
            controller.update([_hand(GestureName.PEACE_SIGN)], 2)

        self.assertEqual(
            backend.operations,
            [
                ("hotkey", ("ctrl", "v")),
                ("hotkey", ("ctrl", "v")),
            ],
        )

    def test_unmapped_gesture_is_idle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = KeyboardMappingStore(Path(temp_dir) / "mappings.json")
            backend = FakeKeyboardBackend()
            controller = KeyboardAutomationController(
                keyboard_backend=backend,
                mapping_store=store,
            )

            status = controller.update([_hand("Unmapped")], 0)

        self.assertEqual(status.action, "Idle")
        self.assertEqual(status.gesture, "Unmapped")
        self.assertEqual(backend.operations, [])


if __name__ == "__main__":
    unittest.main()
