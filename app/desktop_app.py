"""Command-line entry point for the PySide6 desktop interface."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the desktop interface command-line parser."""
    parser = argparse.ArgumentParser(
        description="Launch the desktop interface for hand gesture recognition.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=5,
        help="Initial number of camera indexes to scan, starting at 0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the PySide6 desktop interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from ui.desktop_window import DesktopSettings
        from ui.desktop_window import run_desktop_app
    except ImportError as exc:
        missing_name = getattr(exc, "name", "")
        if missing_name and missing_name.startswith("PySide6"):
            print(
                "PySide6 is required for the desktop interface. "
                "Install it with: py -3 -m pip install -r requirements.txt"
            )
            return 1
        raise

    settings = DesktopSettings(max_cameras=args.max_cameras)
    return run_desktop_app(settings)


if __name__ == "__main__":
    raise SystemExit(main())
