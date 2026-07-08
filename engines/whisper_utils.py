"""Shared Whisper generation kwargs for local ASR engines."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

# transformers>=4.57 rejects these in model.generate() via the HF ASR pipeline.
_STRIP_FROM_GENERATE_KWARGS = frozenset({"condition_on_previous_text"})

# Whisper's encoder ingests at most 30s of audio per chunk. The HF
# WhisperFeatureExtractor truncates anything longer (n_samples = 30 * sr),
# silently dropping every sample past the 30s mark of each chunk. Chunk lengths
# must never exceed this ceiling or transcription loses large spans of audio.
WHISPER_MAX_CHUNK_S = 30


def whisper_max_asr_turn_body_s() -> float:
    """Max diar turn length before ASR_TURN_PAD_S exceeds the encoder window."""
    pad = float(os.getenv("ASR_TURN_PAD_S", "0.25"))
    return max(0.5, float(WHISPER_MAX_CHUNK_S) - 2.0 * pad)


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


# Whisper's temperature-fallback schedule: decode greedily/with beams at 0.0,
# then progressively resample when the compression-ratio / logprob / no-speech
# guards reject a segment. Without a schedule the guards can only *drop* a bad
# segment; with it they trigger a better re-decode instead.
_DEFAULT_TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _parse_temperature(value: str | None):
    """Parse ASR_TEMPERATURE into a scalar or a fallback tuple."""
    if value is None or not value.strip():
        return _DEFAULT_TEMPERATURE_FALLBACK
    temps: list[float] = []
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            temps.append(float(part))
        except ValueError:
            continue
    if not temps:
        return _DEFAULT_TEMPERATURE_FALLBACK
    return temps[0] if len(temps) == 1 else tuple(temps)


def whisper_generate_kwargs(language: str) -> dict:
    """Build generate_kwargs with temperature fallback + hallucination guards."""
    kwargs: dict = {
        "language": language,
        "task": "transcribe",
        "num_beams": max(1, _env_int("ASR_NUM_BEAMS", 1)),
        "temperature": _parse_temperature(os.getenv("ASR_TEMPERATURE")),
    }
    if _env_bool("ASR_SUPPRESS_HALLUCINATIONS", True):
        # Whisper defaults; anything tighter drops genuine (quiet/tonal) speech.
        kwargs.update({
            "compression_ratio_threshold": _env_float("ASR_COMPRESSION_RATIO_THRESHOLD", 2.4),
            "logprob_threshold": _env_float("ASR_LOGPROB_THRESHOLD", -1.0),
            "no_speech_threshold": _env_float("ASR_NO_SPEECH_THRESHOLD", 0.6),
        })
    _apply_repetition_controls(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k not in _STRIP_FROM_GENERATE_KWARGS}
    return filtered


def _apply_repetition_controls(kwargs: dict) -> None:
    """Curb Whisper's phrase-loop hallucinations on long single-speaker turns.

    Both knobs are opt-in (neutral defaults) so clean, verbatim audio is
    unaffected. ``no_repeat_ngram_size`` blocks a decoded n-gram (subword-level
    for Thai) from repeating, which breaks the multi-word loops Whisper falls
    into on long turns; ``repetition_penalty`` gently discourages token reuse.
    """
    ngram = _env_int("ASR_NO_REPEAT_NGRAM_SIZE", 0)
    if ngram > 0:
        kwargs["no_repeat_ngram_size"] = ngram
    penalty = _env_float("ASR_REPETITION_PENALTY", 1.0)
    if penalty > 1.0:
        kwargs["repetition_penalty"] = penalty


_WHISPER_LOG_FILTERS_INSTALLED = False


class _WhisperPipelineLogFilter(logging.Filter):
    """Drop noisy HF Whisper pipeline warnings (logged, not warnings.warn)."""

    _DROP_PHRASES = (
        "Whisper did not predict an ending timestamp",
        "Using `chunk_length_s` is very experimental",
        "WhisperTimeStampLogitsProcessor was used during generation",
        "return_token_timestamps` is deprecated",
        "sequentially on GPU",
        "Device set to use cuda",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(phrase in message for phrase in self._DROP_PHRASES)


def prepare_whisper_generation_config(model) -> None:
    """Neutralise Whisper generation defaults that force timestamp decoding."""
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    gen_cfg.suppress_tokens = None
    gen_cfg.begin_suppress_tokens = None
    if hasattr(gen_cfg, "return_timestamps"):
        gen_cfg.return_timestamps = False


def install_whisper_pipeline_log_filters() -> None:
    """Register log filters once per process for transformers Whisper ASR noise."""
    global _WHISPER_LOG_FILTERS_INSTALLED  # pylint: disable=global-statement
    if _WHISPER_LOG_FILTERS_INSTALLED:
        return
    filt = _WhisperPipelineLogFilter()
    root = logging.getLogger()
    root.addFilter(filt)
    for logger_name in (
        "transformers",
        "transformers.pipelines",
        "transformers.pipelines.base",
        "transformers.pipelines.automatic_speech_recognition",
        "transformers.generation",
        "transformers.generation.utils",
        "transformers.models.whisper.tokenization_whisper",
        "transformers.models.whisper.feature_extraction_whisper",
    ):
        logging.getLogger(logger_name).addFilter(filt)
    if _env_bool("APP_SUPPRESS_WARNING_LOGS", True):
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except ImportError:
            pass
    _WHISPER_LOG_FILTERS_INSTALLED = True


def hf_pipeline_init_kwargs(chunk_length_fn: Callable[[], int]) -> dict:
    """Common kwargs for HuggingFace ASR pipeline construction.

    Pipeline-level ``chunk_length_s`` triggers experimental seq2seq chunking for
    every call. Long-form paths pass ``chunk_length_s`` per invoke instead; turn
    slices should run single-pass without HF internal chunking.

    ``return_timestamps`` is never set here — callers pass it per invoke so
    turn-guided diar jobs can disable chunk timestamps without fighting
    ``generation_config.return_timestamps``.
    """
    # Whisper max_target_positions is 448; leave headroom for decoder prompt tokens.
    kwargs: dict = {
        "ignore_warning": True,
        "max_new_tokens": 440,
    }
    if _env_bool("ASR_PIPELINE_INTERNAL_CHUNK", False):
        kwargs["chunk_length_s"] = chunk_length_fn()
    return kwargs


def invoke_asr_pipeline(
    pipe,
    audio_input,
    *,
    language: str,
    timestamp_mode,
    batch_size: int,
    chunk_length_s: int | None = None,
) -> dict:
    """Run the HF ASR pipeline with Whisper warning noise suppressed."""
    import warnings

    install_whisper_pipeline_log_filters()
    pipeline_input = dict(audio_input) if isinstance(audio_input, dict) else audio_input
    kwargs: dict = {
        "batch_size": batch_size,
        "generate_kwargs": whisper_generate_kwargs(language),
    }
    if timestamp_mode not in (False, None):
        kwargs["return_timestamps"] = timestamp_mode
    if chunk_length_s is not None:
        kwargs["chunk_length_s"] = chunk_length_s

    with warnings.catch_warnings():
        for pattern in (
            r".*Whisper did not predict an ending timestamp.*",
            r".*chunk_length_s.*experimental.*",
            r".*return_token_timestamps.*deprecated.*",
            r".*sequentially on GPU.*",
        ):
            warnings.filterwarnings("ignore", message=pattern)
        try:
            import torch
        except (ImportError, OSError):
            result = pipe(pipeline_input, **kwargs)
        else:
            with torch.inference_mode():
                result = pipe(pipeline_input, **kwargs)

    return patch_missing_chunk_end_timestamps(
        result,
        infer_audio_duration_s(pipeline_input),
    )


def infer_audio_duration_s(audio_input) -> float:
    """Best-effort duration from a librosa-style pipeline input dict."""
    if not isinstance(audio_input, dict):
        return 0.0
    raw = audio_input.get("raw")
    sample_rate = audio_input.get("sampling_rate") or 16000
    if raw is None or not sample_rate:
        return 0.0
    try:
        return len(raw) / float(sample_rate)
    except TypeError:
        return 0.0


def _duration_from_chunk_timestamps(chunks: list[dict]) -> float:
    """Infer audio duration from the last known chunk end timestamp."""
    duration = 0.0
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if ts and ts[0] is not None and ts[1] is not None:
            duration = max(duration, float(ts[1]))
    return duration


def _resolve_missing_chunk_end(
    start: float,
    index: int,
    chunks: list[dict],
    duration: float,
) -> float | None:
    """Best-effort end time when Whisper leaves chunk end as None."""
    end: float | None = None
    if index + 1 < len(chunks):
        nxt = chunks[index + 1].get("timestamp")
        if nxt and nxt[0] is not None:
            end = float(nxt[0])
    if end is None and duration > start:
        end = duration
    if end is None:
        end = start + 0.5
    return end


def _patch_single_chunk_end(
    chunk: dict,
    index: int,
    chunks: list[dict],
    duration: float,
) -> tuple[dict, bool]:
    """Return a chunk copy with a filled end timestamp when needed."""
    item = dict(chunk)
    ts = item.get("timestamp")
    if not ts:
        return item, False
    start, end = ts[0], ts[1]
    if start is None or end is not None:
        return item, False
    resolved = _resolve_missing_chunk_end(float(start), index, chunks, duration)
    if resolved is None:
        return item, False
    item["timestamp"] = (start, resolved)
    return item, True


def patch_missing_chunk_end_timestamps(
    result: dict,
    audio_duration_s: float = 0.0,
) -> dict:
    """Fill ``None`` chunk end timestamps so downstream repair stays stable."""
    chunks = result.get("chunks")
    if not chunks:
        return result

    duration = audio_duration_s if audio_duration_s > 0 else 0.0
    if duration <= 0:
        duration = _duration_from_chunk_timestamps(chunks)

    patched: list[dict] = []
    changed = False
    for index, chunk in enumerate(chunks):
        item, item_changed = _patch_single_chunk_end(chunk, index, chunks, duration)
        changed = changed or item_changed
        patched.append(item)

    if not changed:
        return result
    out = dict(result)
    out["chunks"] = patched
    return out
