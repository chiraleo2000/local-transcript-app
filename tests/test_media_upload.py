"""Tests for multi-file Gradio upload path handling."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestNormalizeMediaPaths(unittest.TestCase):
    def test_string_and_list(self) -> None:
        from backend.media_paths import normalize_media_paths

        self.assertEqual(normalize_media_paths(None), [])
        self.assertEqual(normalize_media_paths("a.wav"), ["a.wav"])
        self.assertEqual(
            normalize_media_paths(["a.wav", "b.mp3"]),
            ["a.wav", "b.mp3"],
        )

    def test_filedata_dict_and_object(self) -> None:
        from backend.media_paths import normalize_media_paths

        self.assertEqual(
            normalize_media_paths({"path": "/tmp/x.wav", "name": "x.wav"}),
            ["/tmp/x.wav"],
        )

        class _File:
            name = "/tmp/y.wav"

        self.assertEqual(normalize_media_paths(_File()), ["/tmp/y.wav"])
        self.assertEqual(
            normalize_media_paths([{"path": "/tmp/a.wav"}, _File()]),
            ["/tmp/a.wav", "/tmp/y.wav"],
        )


class TestUiLimitsListSafe(unittest.TestCase):
    def test_ui_limits_rejects_list_without_typeerror(self) -> None:
        from backend.ui_limits import format_media_info, media_too_large_for_browser

        self.assertEqual(media_too_large_for_browser(["a.wav", "b.wav"]), (False, ""))
        self.assertEqual(format_media_info(["a.wav"]), "No file selected.")


class TestOnMediaUploadLogic(unittest.TestCase):
    def test_list_paths_pick_first_for_preview(self) -> None:
        """Mirror _on_media_upload path selection without importing Gradio app."""
        from backend.media_paths import normalize_media_paths
        from backend.ui_limits import format_media_info, media_too_large_for_browser

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "sample.wav"
            other = Path(tmp) / "other.wav"
            wav.write_bytes(b"RIFF....WAVE")
            other.write_bytes(b"RIFF....WAVE")
            paths = normalize_media_paths([str(wav), str(other)])
            self.assertEqual(len(paths), 2)
            first = paths[0]
            # Must not raise TypeError (the production bug).
            too_large, _ = media_too_large_for_browser(first)
            self.assertIsInstance(too_large, bool)
            info = format_media_info(first)
            self.assertIn("sample.wav", info)
            # Passing the raw list must not raise.
            media_too_large_for_browser(paths)  # type: ignore[arg-type]
            format_media_info(paths)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
