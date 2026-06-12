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
# Formats that need ffmpeg → 16 kHz WAV for reliable duration + diarization probes.
TRANSCODE_AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".wma", ".opus", ".webm", ".mp4",
}


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _transcode_to_wav(source: Path, job_id: str) -> str:
    """Convert any media to 16 kHz mono PCM WAV (fixes m4a/mp3 probe + diarization)."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for this audio format but was not found.")

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
        raise RuntimeError(f"Audio transcode failed: {result.stderr[-800:]}")
    logger.info("Media normalization: transcoded to %s", output)
    return str(output)


def normalize_media(media_path: str, job_id: str) -> str:
    """Return a local 16 kHz mono WAV path; transcode audio/video as needed."""
    ensure_app_dirs()
    source = Path(media_path)
    suffix = source.suffix.lower()

    if suffix == ".wav":
        logger.info("Media normalization: using WAV directly: %s", media_path)
        return media_path
    return _transcode_to_wav(source, job_id)


def _should_unload_vram_on_media_change() -> bool:
    return os.getenv("ASR_CLEAR_VRAM_ON_MEDIA_CHANGE", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _diarization_keep_preloaded() -> bool:
    if os.getenv("DIARIZATION_KEEP_PRELOADED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return True
    mode = os.getenv("DIARIZATION_PRELOAD_MODE", "eager").strip().lower()
    return mode in {"eager", "preload", "true", "1"}


def clear_prejob_caches() -> None:
    """Drop GPU-resident models when input media changes (unless keep-preloaded)."""
    from backend.model_registry import unload_asr_models, unload_diarization_model
    from backend.services.asr_local import clear_accelerator_cache

    keep_preloaded = os.getenv("ASR_KEEP_PRELOADED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if keep_preloaded:
        clear_accelerator_cache()
        logger.info("Pre-job cache touch (models kept preloaded).")
        return

    if _should_unload_vram_on_media_change():
        unload_asr_models()
        if not _diarization_keep_preloaded():
            unload_diarization_model()
    else:
        clear_accelerator_cache()

    logger.info(
        "Pre-job caches cleared (VRAM unload=%s).",
        _should_unload_vram_on_media_change(),
    )


def enhance_audio(audio_path: str) -> str:
    """Run local audio enhancement/preprocessing (no cross-job cache)."""
    from engines.preprocess import preprocess_audio

    enhanced_path = preprocess_audio(audio_path)
    if enhanced_path != audio_path and os.path.isfile(enhanced_path):
        logger.info("Audio enhancement complete: %s", enhanced_path)
    return enhanced_path


def _duration_ffprobe(audio_path: str) -> float | None:
    """Read duration via ffprobe (works for m4a/mp3 without librosa audioread)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def audio_duration_seconds(audio_path: str) -> float:
    """Return media duration in seconds without moving work onto the GPU."""
    duration = _duration_ffprobe(audio_path)
    if duration is not None and duration > 0:
        return duration

    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate and info.frames:
            return float(info.frames / info.samplerate)
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        logger.debug("soundfile duration probe skipped for %s: %s", audio_path, exc)

    logger.warning("Could not determine audio duration for %s", audio_path)
    return 0.0


def diarize_audio(
    audio_path: str, max_speakers: int = 0,
    audio_duration_s: float = 0.0,
    diarize_kwargs: dict | None = None,
) -> list[dict]:
    """Run local speaker diarization."""
    from engines.diarization import diarize

    kwargs = dict(diarize_kwargs or {})
    kwargs.setdefault("audio_duration_s", audio_duration_s)
    return diarize(audio_path, max_speakers=max_speakers, **kwargs)


def clear_diarization_model() -> None:
    """Unload the cached diarization model."""
    from engines import diarization

    diarization.unload_model()
