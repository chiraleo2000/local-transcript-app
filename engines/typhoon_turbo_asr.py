"""Typhoon Whisper Turbo — faster Thai ASR with near large-v3 accuracy.

Benchmarks (lower CER is better, Typhoon ASR Benchmark, Jan 2026):
  Typhoon Whisper Large-v3: GigaSpeech2 4.69, noisy TVSpeech 6.32
  Typhoon Whisper Turbo:    GigaSpeech2 4.79, noisy TVSpeech 6.85
  Pathumma Large-v3:        GigaSpeech2 5.84, noisy TVSpeech 10.36

Turbo keeps the Whisper checkpoint interface (4 decoder layers) so it shares
the existing decode path. Large-v3 stays the quality default.
"""

from __future__ import annotations

from engines.model_cache import configured_turbo_model_id

_LABEL = "Typhoon Turbo"


def model_id() -> str:
    return configured_turbo_model_id()


def load_model() -> None:
    from engines.typhoon_asr import load_model as load_typhoon

    load_typhoon(model_id(), label=_LABEL, allow_ct2=False)


def unload_model() -> None:
    from engines.typhoon_asr import unload_model as unload_typhoon

    unload_typhoon(model_id())


def transcribe_turbo(
    audio_path: str,
    language: str = "thai",
    diarization_segments: list | None = None,
    cancel_event=None,
    window_progress=None,
    max_speakers: int = 0,
) -> str:
    """Transcribe audio with Typhoon Whisper Turbo."""
    from engines.typhoon_asr import transcribe_typhoon

    return transcribe_typhoon(
        audio_path,
        language,
        diarization_segments,
        cancel_event=cancel_event,
        window_progress=window_progress,
        max_speakers=max_speakers,
        model_id=model_id(),
        label=_LABEL,
        allow_ct2=False,
    )
