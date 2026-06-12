"""Shared Whisper generation kwargs for local ASR engines."""

from __future__ import annotations

import os

# transformers>=4.57 rejects these in model.generate() via the HF ASR pipeline.
_STRIP_FROM_GENERATE_KWARGS = frozenset({"condition_on_previous_text"})


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _strict_8gb_max_batch() -> int:
    return max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 1))


def _memory_safe_duration_batch(batch: int, audio_duration_s: float, max_8gb: int) -> int | None:
    """Apply 8 GB safety caps from recording length; None when not applicable."""
    if not _env_bool("ASR_HARD_MEMORY_SAFE", True):
        return None
    use_duration_cap = _env_bool("ASR_BATCH_DURATION_CAP", True)
    if use_duration_cap and max_8gb > 1:
        if audio_duration_s >= 120:
            return min(batch, max(1, max_8gb // 2))
        if audio_duration_s >= 60:
            return min(batch, max_8gb)
        return None
    if audio_duration_s >= 120:
        return 1
    if audio_duration_s >= 60:
        return min(batch, 1)
    return None


def _long_recording_batch_cap(batch: int, audio_duration_s: float) -> int | None:
    """Fallback duration tiers for non-windowed long-form transcription."""
    if audio_duration_s >= 3600:
        return 1
    if audio_duration_s >= 1800:
        return min(batch, 1)
    if audio_duration_s >= 900:
        return min(batch, 2)
    if audio_duration_s >= 600:
        return min(batch, 2)
    if audio_duration_s >= 300:
        return min(batch, max(1, batch // 2))
    return None


def effective_asr_batch_size(
    base_batch: int,
    audio_duration_s: float,
    *,
    windowed: bool = False,
) -> int:
    """Lower Whisper batch size for long audio to reduce peak VRAM."""
    batch = max(1, int(base_batch))
    if audio_duration_s <= 0:
        return batch
    max_8gb = _strict_8gb_max_batch()
    if windowed:
        return min(batch, max_8gb)
    safe_batch = _memory_safe_duration_batch(batch, audio_duration_s, max_8gb)
    if safe_batch is not None:
        return safe_batch
    capped = _long_recording_batch_cap(batch, audio_duration_s)
    return capped if capped is not None else batch


def whisper_generate_kwargs(language: str) -> dict:
    """Build generate_kwargs with optional hallucination guards."""
    kwargs: dict = {
        "language": language,
        "task": "transcribe",
        "num_beams": 1,
        "temperature": _env_float("ASR_TEMPERATURE", 0.0),
    }
    if _env_bool("ASR_SUPPRESS_HALLUCINATIONS", True):
        kwargs.update({
            "compression_ratio_threshold": _env_float("ASR_COMPRESSION_RATIO_THRESHOLD", 1.8),
            "logprob_threshold": _env_float("ASR_LOGPROB_THRESHOLD", -0.8),
            "no_speech_threshold": _env_float("ASR_NO_SPEECH_THRESHOLD", 0.5),
        })
    filtered = {k: v for k, v in kwargs.items() if k not in _STRIP_FROM_GENERATE_KWARGS}
    return filtered
