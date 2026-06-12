"""Low-memory audio I/O — duration probes and slice loading without full-file decode."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_TARGET_SR = 16000


def count_audio_windows(duration_s: float, window_s: int, overlap_s: int) -> int:
    """Return how many overlapped windows cover ``duration_s`` without loading audio."""
    if duration_s <= 0 or window_s <= 0:
        return 1
    overlap_s = max(0, min(overlap_s, window_s - 1))
    step_s = max(1.0, float(window_s - overlap_s))
    count = 0
    offset_s = 0.0
    while offset_s < duration_s:
        chunk_dur = min(float(window_s), duration_s - offset_s)
        count += 1
        if offset_s + chunk_dur >= duration_s - 0.01:
            break
        offset_s += step_s
    return count


def probe_audio_duration(audio_path: str) -> float:
    """Return media duration in seconds without decoding the full waveform."""
    from backend.services.media_pipeline import audio_duration_seconds

    return audio_duration_seconds(audio_path)


def load_audio_slice(
    audio_path: str,
    offset_s: float,
    duration_s: float,
    sr: int = _TARGET_SR,
) -> dict:
    """Load only ``duration_s`` seconds starting at ``offset_s`` (16 kHz mono)."""
    import librosa
    import numpy as np

    if duration_s <= 0:
        return {"raw": np.array([], dtype=np.float32), "sampling_rate": sr}
    y, _ = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        offset=max(0.0, offset_s),
        duration=float(duration_s),
    )
    return {"raw": y, "sampling_rate": sr}


def iter_audio_windows_from_path(
    audio_path: str,
    window_s: int,
    overlap_s: int,
    sr: int = _TARGET_SR,
) -> Iterator[tuple[float, dict]]:
    """Yield overlapped ASR windows, loading one slice from disk at a time."""
    duration_s = probe_audio_duration(audio_path)
    if duration_s <= 0 or window_s <= 0:
        yield 0.0, load_audio_slice(audio_path, 0.0, max(window_s, 1), sr)
        return

    overlap_s = max(0, min(overlap_s, window_s - 1))
    step_s = max(1.0, float(window_s - overlap_s))
    total_windows = count_audio_windows(duration_s, window_s, overlap_s)
    logger.info(
        "Streaming audio windows: path=%s duration=%.1fs windows=%d window=%ds overlap=%ds",
        os.path.basename(audio_path),
        duration_s,
        total_windows,
        window_s,
        overlap_s,
    )
    offset_s = 0.0
    while offset_s < duration_s:
        chunk_dur = min(float(window_s), duration_s - offset_s)
        yield offset_s, load_audio_slice(audio_path, offset_s, chunk_dur, sr)
        if offset_s + chunk_dur >= duration_s - 0.01:
            break
        offset_s += step_s
