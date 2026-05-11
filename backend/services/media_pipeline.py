"""Local media preparation, audio enhancement, and diarization facade."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from backend.storage import AUDIO_DIR, ensure_app_dirs, safe_name


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_enhance_cache: dict[tuple, str] = {}


def normalize_media(media_path: str, job_id: str) -> str:
    """Return a local audio path; extract audio from videos with ffmpeg."""
    ensure_app_dirs()
    source = Path(media_path)
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        return media_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for video files but was not found.")

    output = AUDIO_DIR / f"{job_id}_{safe_name(source.stem)}_16k.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Video audio extraction failed: {result.stderr[-800:]}")
    return str(output)


def enhance_audio(audio_path: str) -> str:
    """Run local audio enhancement/preprocessing."""
    from engines.preprocess import preprocess_audio

    try:
        stat = os.stat(audio_path)
    except OSError:
        return preprocess_audio(audio_path)

    settings = (
        os.getenv("AUDIO_ENHANCE_TARGET_PEAK_DB", "-3.0"),
        os.getenv("AUDIO_ENHANCE_MAX_GAIN_DB", "10.0"),
        os.getenv("AUDIO_ENHANCE_NOISE_REDUCTION", "0.65"),
    )
    key = (os.path.abspath(audio_path), stat.st_mtime_ns, stat.st_size, settings)
    cached_path = _enhance_cache.get(key)
    if cached_path and os.path.isfile(cached_path):
        logger.info("Reusing cached enhanced audio: %s", cached_path)
        return cached_path

    enhanced_path = preprocess_audio(audio_path)
    if enhanced_path != audio_path and os.path.isfile(enhanced_path):
        _enhance_cache[key] = enhanced_path
    return enhanced_path


def audio_duration_seconds(audio_path: str) -> float:
    """Return media duration in seconds without moving work onto the GPU."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate and info.frames:
            return float(info.frames / info.samplerate)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.debug("soundfile duration probe skipped for %s: %s", audio_path, exc)

    try:
        import librosa

        return float(librosa.get_duration(path=audio_path))
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.warning("Could not determine audio duration for %s: %s", audio_path, exc)
        return 0.0


def diarize_audio(
    audio_path: str, min_speakers: int = 0, max_speakers: int = 0,
    diarize_kwargs: dict | None = None,
) -> list[dict]:
    """Run local speaker diarization."""
    from engines.diarization import diarize

    return diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers,
                   **(diarize_kwargs or {}))


def clear_diarization_model() -> None:
    """Unload the cached diarization model."""
    from engines import diarization

    diarization.unload_model()
