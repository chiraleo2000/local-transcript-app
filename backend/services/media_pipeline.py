"""Local media preparation, audio enhancement, and diarization facade."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from backend.storage import AUDIO_DIR, ensure_app_dirs, safe_name


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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
    from engines.preprocess import preprocess_audio

    return preprocess_audio(audio_path)


def diarize_audio(audio_path: str, min_speakers: int = 0, max_speakers: int = 0) -> list[dict]:
    from engines.diarization import diarize

    return diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
