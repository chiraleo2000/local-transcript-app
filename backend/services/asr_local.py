"""Local ASR service facade."""

from __future__ import annotations

import time


ENGINE_TYPHOON = "Typhoon Whisper"
ENGINE_THONBURIAN = "Thonburian Whisper"
ALL_ENGINES = [ENGINE_TYPHOON, ENGINE_THONBURIAN]

LANGUAGES = {
    "Thai": "thai",
    "English": "english",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Korean": "korean",
}


def load_model(engine_name: str) -> None:
    if engine_name == ENGINE_TYPHOON:
        from engines.typhoon_asr import load_model as load_typhoon

        load_typhoon()
        return
    if engine_name == ENGINE_THONBURIAN:
        from engines.thonburian_asr import load_model as load_thonburian

        load_thonburian()
        return
    raise ValueError(f"Unknown ASR engine: {engine_name}")


def transcribe_engine(
    engine_name: str,
    audio_path: str,
    language: str,
    diarization_segments: list[dict] | None = None,
) -> tuple[str, float]:
    whisper_language = LANGUAGES.get(language, LANGUAGES["Thai"])
    started = time.perf_counter()
    if engine_name == ENGINE_TYPHOON:
        from engines.typhoon_asr import transcribe_typhoon

        text = transcribe_typhoon(audio_path, whisper_language, diarization_segments)
    elif engine_name == ENGINE_THONBURIAN:
        from engines.thonburian_asr import transcribe_thonburian

        text = transcribe_thonburian(audio_path, whisper_language, diarization_segments)
    else:
        raise ValueError(f"Unknown ASR engine: {engine_name}")
    return text, time.perf_counter() - started
