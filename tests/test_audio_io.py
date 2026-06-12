"""Tests for streaming audio window I/O."""

import pytest

from engines.audio_io import (
    count_audio_windows,
    iter_audio_windows_from_path,
    probe_audio_duration,
)


def test_probe_duration_small_fixture():
    path = "tests/e2e/fixtures/small.wav"
    duration = probe_audio_duration(path)
    assert duration > 0


def test_count_audio_windows():
    assert count_audio_windows(120.0, window_s=60, overlap_s=10) == 3


def test_iter_windows_from_path_is_generator():
    path = "tests/e2e/fixtures/small.wav"
    windows = iter_audio_windows_from_path(path, window_s=1, overlap_s=0)
    first = next(windows)
    offset, window = first
    assert offset == pytest.approx(0.0)
    assert len(window["raw"]) > 0
    assert window["sampling_rate"] == 16000
    count = 1
    for _ in windows:
        count += 1
    assert count == count_audio_windows(probe_audio_duration(path), 1, 0)
