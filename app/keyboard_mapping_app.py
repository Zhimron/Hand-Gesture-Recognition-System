"""Command-line utility for keyboard gesture mappings."""

from __future__ import annotations

import argparse
from typing import Sequence

from services.keyboard_controller import KeyboardMappingStore
from services.keyboard_controller import parse_key_sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the keyboard mapping command-line parser."""
    parser = argparse.ArgumentParser(
        description="List or save gesture-to-keyboard mappings.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured gesture mappings.",
    )
    parser.add_argument(
        "--gesture",
        default=None,
        help="Gesture name to map.",
    )
    parser.add_argument(
        "--keys",
        default=None,
        help="Key sequence such as ctrl+c, ctrl+shift+s, or esc.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """List or update keyboard mappings."""
    parser = build_parser()
    args = parser.parse_args(argv)
    store = KeyboardMappingStore()

    if args.gesture and args.keys:
        action = store.save_mapping(
            gesture=args.gesture,
            keys=parse_key_sequence(args.keys),
        )
        print(f"Saved: {action.gesture} -> {action.label}")
        print(f"Config: {store.path}")
        return 0

    if args.gesture or args.keys:
        parser.error("--gesture and --keys must be provided together.")

    mappings = store.load_if_changed()
    for gesture, keys in mappings.items():
        print(f"{gesture} -> {'+'.join(keys)}")
    print(f"Config: {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
