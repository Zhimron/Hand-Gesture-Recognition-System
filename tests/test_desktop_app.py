"""Tests for desktop interface command parsing."""

from __future__ import annotations

import unittest

from app.desktop_app import build_parser


class DesktopAppParserTest(unittest.TestCase):
    """Unit tests for desktop app parser behavior."""

    def test_parser_accepts_max_cameras(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--max-cameras", "8"])

        self.assertEqual(args.max_cameras, 8)


if __name__ == "__main__":
    unittest.main()
