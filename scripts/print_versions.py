#!/usr/bin/env python3
"""Print runtime library and model versions for support logs."""

from __future__ import annotations

import os


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pylint: disable=broad-exception-caught
        try:
            mod = __import__(name.replace("-", "_").split(".")[0])
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return "not installed"


def main() -> None:
    print("=== Local Transcript App — versions ===")
    for pkg in ("torch", "transformers", "gradio", "pyannote.audio", "librosa", "accelerate", "fastapi"):
        print(f"  {pkg}: {_pkg_version(pkg)}")

    print("\n=== Model IDs (from env) ===")
    print(f"  PATHUMMA_MODEL_ID: {os.getenv('PATHUMMA_MODEL_ID', 'nectec/Pathumma-whisper-th-large-v3')}")
    print(f"  TYPHOON_MODEL_ID: {os.getenv('TYPHOON_MODEL_ID', 'typhoon-ai/typhoon-whisper-large-v3')}")
    print(f"  DIARIZATION_MODEL_ID: {os.getenv('DIARIZATION_MODEL_ID', 'pyannote/speaker-diarization-community-1')}")
    print("\n=== Quality / chunk ===")
    print(f"  ASR_QUALITY_PROFILE: {os.getenv('ASR_QUALITY_PROFILE', 'balanced')}")
    print(f"  ASR_CHUNK_LENGTH_S: {os.getenv('ASR_CHUNK_LENGTH_S', '(default)')}")
    print(f"  ASR_8GB_MAX_CHUNK_LENGTH_S: {os.getenv('ASR_8GB_MAX_CHUNK_LENGTH_S', '(default)')}")
    print(f"  ASR_MIN_CHUNKED_DURATION_S: {os.getenv('ASR_MIN_CHUNKED_DURATION_S', '(default)')}")


if __name__ == "__main__":
    main()
