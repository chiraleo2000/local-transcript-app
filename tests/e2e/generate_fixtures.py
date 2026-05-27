#!/usr/bin/env python3
"""Generate small/large WAV fixtures for Playwright E2E tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SMALL_TEXT = (
    "Hello, this is a short transcription test for the local transcript application."
)
LARGE_PHRASE = (
    "This is a longer audio stress test for the local transcript application."
)


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required to build E2E fixtures (install ffmpeg and retry).")
    return path


def _to_mono_16k_wav(source: Path, dest: Path) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _pad_to_duration(source: Path, dest: Path, duration_s: float) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(source),
            "-t",
            str(duration_s),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


async def _synthesize_mp3(path: Path, text: str) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "Install edge-tts to generate speech fixtures: pip install edge-tts",
        ) from exc

    voice = os.getenv("E2E_TTS_VOICE", "en-US-JennyNeural")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(path))


async def _build_speech_fixture(path: Path, text: str) -> Path:
    """Return path to a mono 16 kHz WAV clip containing spoken text."""
    tmp_mp3 = path.with_suffix(".mp3")
    tmp_wav = path.with_suffix(".tmp.wav")
    await _synthesize_mp3(tmp_mp3, text)
    _to_mono_16k_wav(tmp_mp3, tmp_wav)
    tmp_mp3.unlink(missing_ok=True)
    return tmp_wav


async def _build_fixtures(small_s: float, large_s: float) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    small = FIXTURES / "small.wav"
    large = FIXTURES / "large.wav"

    small_clip = await _build_speech_fixture(small, SMALL_TEXT)
    # Pad short clip to requested duration (repeat speech, not silence).
    _pad_to_duration(small_clip, small, max(small_s, 3.0))
    small_clip.unlink(missing_ok=True)

    large_phrase = " ".join([LARGE_PHRASE] * max(1, int(large_s // 12)))
    large_clip = await _build_speech_fixture(large, large_phrase[:500])
    _pad_to_duration(large_clip, large, large_s)
    large_clip.unlink(missing_ok=True)


def main() -> None:
    small_s = float(os.getenv("E2E_SMALL_SECONDS", "5"))
    large_s = float(os.getenv("E2E_LARGE_SECONDS", "120"))
    asyncio.run(_build_fixtures(small_s, large_s))

    import wave

    for name in ("small.wav", "large.wav"):
        path = FIXTURES / name
        with wave.open(str(path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            print(f"Wrote {path} ({duration:.1f}s @ {wav.getframerate()} Hz)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
