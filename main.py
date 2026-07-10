"""Command-line entry point for webcam and hand detection features."""

from __future__ import annotations

import sys
from typing import Sequence

from app.webcam_app import main as webcam_main


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the requested feature while preserving webcam defaults."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in {"hands", "hand-detection"}:
        from app.hand_detection_app import main as hand_detection_main

        return hand_detection_main(args[1:])

    if args and args[0] in {"record-gesture", "custom-gesture"}:
        from app.custom_gesture_app import main as custom_gesture_main

        return custom_gesture_main(args[1:])

    if args and args[0] == "webcam":
        return webcam_main(args[1:])

    return webcam_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
