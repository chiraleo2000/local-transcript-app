"""Tests for browser UI memory limits."""

from backend.ui_limits import display_transcript_text, media_too_large_for_browser


def test_display_transcript_truncates_lines(monkeypatch):
    monkeypatch.setenv("UI_TRANSCRIPT_MAX_LINES", "3")
    text = "\n".join(f"line {i}" for i in range(10))
    shown = display_transcript_text(text)
    assert "Displaying last 3 lines" in shown
    assert "line 9" in shown
    assert "line 0" not in shown


def test_display_transcript_truncates(monkeypatch):
    monkeypatch.setenv("UI_TRANSCRIPT_MAX_CHARS", "50000")
    text = "x" * 80_000
    shown = display_transcript_text(text)
    assert len(shown) < len(text)
    assert "Download .txt" in shown


def test_media_too_large_by_size(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_PREVIEW_MAX_MB", "1")
    path = tmp_path / "big.wav"
    path.write_bytes(b"\0" * (2 * 1024 * 1024))
    too_large, reason = media_too_large_for_browser(str(path))
    assert too_large
    assert "MB" in reason
