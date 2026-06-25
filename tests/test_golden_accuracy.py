"""Unit tests for golden accuracy scoring (no GPU)."""

from __future__ import annotations

from tests.golden.accuracy import transcript_accuracy
from tests.golden.fixtures import active_fixture


def test_sample01_reference_matches_itself():
    expected = active_fixture("sample01").expected.read_text(encoding="utf-8")
    assert transcript_accuracy(expected, expected) >= 0.99


def test_sample01_audio_fixture_exists():
    assert active_fixture("sample01").audio.is_file()
