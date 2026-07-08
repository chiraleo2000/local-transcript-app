"""Shared Whisper ASR runtime: OOM retry, long-form windowing, transcription."""

from __future__ import annotations

import gc
import logging
import os
import re
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_JOB_CANCELLED_MSG = "Job cancelled by user."
_NO_SPEECH_MSG = "(no speech detected)"
_PIPE_CHUNK_AUTO = object()


def is_cuda_oom(exc: Exception) -> bool:
    """Return True for CUDA, driver, HF, or allocator memory-pressure failures."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, OSError, AttributeError):
        pass
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "out of memory",
            "cuda out of memory",
            "cudacachingallocator",
            "handles_.at",
            "error code: out of memory",
        )
    )


def is_cuda_unknown_error(exc: Exception) -> bool:
    """Return True for driver/context failures that are not plain OOM."""
    if is_cuda_oom(exc):
        return False
    try:
        import torch

        cls = type(exc)
        if cls.__name__ in {"AcceleratorError", "CudaError"}:
            return True
        if isinstance(exc, RuntimeError) and getattr(torch, "cuda", None) is not None:
            cuda_errors = (
                getattr(torch.cuda, "CudaError", None),
                getattr(getattr(torch, "accelerator", None), "AcceleratorError", None),
            )
            if any(err is not None and isinstance(exc, err) for err in cuda_errors):
                return True
    except (ImportError, OSError, AttributeError):
        pass
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda error: unknown",
            "cudaerrorunknown",
            "cuda error",
            "cuda driver error",
            "device not ready",
            "unknown error",
            "device-side assert",
            "illegal memory access",
            "an illegal",
        )
    )


def is_cuda_recoverable(exc: Exception) -> bool:
    """Return True when a CUDA teardown/retry may succeed."""
    return is_cuda_oom(exc) or is_cuda_unknown_error(exc)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _strict_memory_mode() -> bool:
    return _env_bool("ASR_HARD_MEMORY_SAFE", True)


def _min_chunked_duration_s() -> float:
    default = 60 if _strict_memory_mode() else 300
    return max(0.0, float(_env_int("ASR_MIN_CHUNKED_DURATION_S", default)))


@dataclass(frozen=True)
class WhisperRuntime:
    """Engine-specific Whisper tuning hooks shared by Typhoon and Pathumma."""

    engine_name: str
    batch_size: Callable[[float | None], int]
    chunk_length_s: Callable[[], int]
    retry_chunk_length_s: Callable[[], int]
    long_form_min_duration_s: Callable[[], int]
    long_form_window_s: Callable[[float], int]
    long_form_overlap_s: Callable[[], int]
    clear_cuda_cache: Callable[[], None]
    max_chars_per_subchunk: Callable[[], int]
    reload_pipeline: Callable[[], Any] | None = None


def _is_windowed_asr(audio_duration_s: float, window_duration_s: float) -> bool:
    return window_duration_s > 0 and audio_duration_s > window_duration_s * 1.5


def _vram_batch_boost(batch: int, window_duration_s: float = 0.0) -> int:
    """Raise batch when free VRAM headroom allows (8 GB class with ASR resident)."""
    del window_duration_s
    try:
        from backend.vram_state import snapshot

        snap = snapshot()
        max_8gb = max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 1))
        target = min(max_8gb, max(batch, _env_int("ASR_CUDA_BATCH_SIZE", max_8gb)))
        # After Whisper-large loads on 8 GB, ~1.5–2.5 GB free is normal; 5000 MB
        # was unreachable so batch never scaled above 1.
        min_free = _env_int("ASR_BATCH_MIN_FREE_MB", 1200)
        free_mb = snap.get("free_mb", 0)
        if free_mb >= min_free:
            return target
        if free_mb >= min_free // 2 and max_8gb >= 2:
            return min(target, max(batch, 2))
    except ImportError:
        pass
    return batch


def _maybe_teardown_between_windows(runtime: WhisperRuntime, window_index: int) -> None:
    """Avoid clearing CUDA cache every window — kills GPU throughput."""
    try:
        from backend.vram_state import snapshot, teardown

        snap = snapshot()
        min_free = _env_int("ASR_WINDOW_CLEAR_MIN_FREE_MB", 1800)
        if snap.get("free_mb", 0) < min_free:
            teardown(aggressive=True)
            return
        if window_index % 4 == 0:
            gc.collect()
    except ImportError:
        runtime.clear_cuda_cache()
        gc.collect()


def _check_job_cancelled(cancel_event) -> None:
    if cancel_event and cancel_event.is_set():
        raise RuntimeError(_JOB_CANCELLED_MSG)


def _strict_8gb_batch_cap(
    batch: int,
    *,
    windowed: bool,
    window_duration_s: float,
    audio_duration_s: float,
    max_8gb: int,
) -> int:
    if windowed:
        batch = min(batch, max_8gb)
        return _vram_batch_boost(batch, window_duration_s)
    if max_8gb > 1 and _env_bool("ASR_BATCH_DURATION_CAP", True):
        if audio_duration_s >= 180 or window_duration_s >= 120:
            return min(batch, max(1, max_8gb // 2))
        if audio_duration_s >= 60 or window_duration_s >= 60:
            return min(batch, max_8gb)
        return batch
    if audio_duration_s >= 180 or window_duration_s >= 60:
        return 1
    return batch


def _pipe_batch_size(runtime: WhisperRuntime, audio_duration_s: float, window_duration_s: float) -> int:
    """Conservative batch for long / windowed audio on low-VRAM GPUs."""
    windowed = _is_windowed_asr(audio_duration_s, window_duration_s)
    sizing_dur = window_duration_s if windowed else audio_duration_s
    if windowed:
        from engines.whisper_utils import effective_asr_batch_size

        base = runtime.batch_size(None)
        batch = effective_asr_batch_size(base, sizing_dur, windowed=True)
    else:
        batch = runtime.batch_size(audio_duration_s)
    if not _strict_memory_mode():
        return batch
    max_8gb = max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 1))
    return _strict_8gb_batch_cap(
        batch,
        windowed=windowed,
        window_duration_s=window_duration_s,
        audio_duration_s=audio_duration_s,
        max_8gb=max_8gb,
    )


def _pipe_chunk_length(runtime: WhisperRuntime, audio_duration_s: float) -> int | None:
    """Return chunk length when audio is long enough; None = single-pass inference."""
    chunk_s = runtime.chunk_length_s()
    min_dur = _min_chunked_duration_s()
    if audio_duration_s >= runtime.long_form_min_duration_s():
        return chunk_s
    if audio_duration_s >= min_dur:
        return chunk_s
    return None


def _retry_halving_chunks(
    run_pipe: Callable[..., dict],
    pipe: Any,
    audio_input: Any,
    language: str,
    runtime: WhisperRuntime,
    current_chunk: int,
    floor_chunk: int,
    window_index: int,
) -> dict | None:
    while current_chunk > floor_chunk:
        next_chunk = max(floor_chunk, current_chunk // 2)
        logger.warning(
            "%s OOM at chunk=%ds; retry chunk=%ds window=%d",
            runtime.engine_name,
            current_chunk,
            next_chunk,
            window_index,
        )
        runtime.clear_cuda_cache()
        try:
            from backend import vram_state

            vram_state.teardown(aggressive=True)
            return run_pipe(pipe, audio_input, language, True, 1, next_chunk)
        except Exception as exc3:  # pylint: disable=broad-exception-caught
            if not is_cuda_recoverable(exc3):
                raise
            current_chunk = next_chunk
    return None


def _retry_pipe_on_oom(
    run_pipe: Callable[..., dict],
    pipe: Any,
    audio_input: Any,
    language: str,
    timestamp_mode: Any,
    batch_size: int,
    runtime: WhisperRuntime,
    chunk_s: int | None,
    retry_chunk_s: int,
    window_index: int = 0,
) -> dict:
    """Halve batch, then iteratively halve chunk length down to retry floor."""
    runtime.clear_cuda_cache()
    try:
        from backend import vram_state

        vram_state.teardown(aggressive=True)
    except ImportError:
        pass
    if batch_size > 1:
        halved = max(1, batch_size // 2)
        logger.warning(
            "%s CUDA OOM at batch=%d; retrying batch=%d chunk=%s.",
            runtime.engine_name,
            batch_size,
            halved,
            chunk_s,
        )
        try:
            if chunk_s is not None:
                return run_pipe(pipe, audio_input, language, timestamp_mode, halved, chunk_s)
            return run_pipe(pipe, audio_input, language, timestamp_mode, halved)
        except Exception as exc2:  # pylint: disable=broad-exception-caught
            if not is_cuda_recoverable(exc2):
                raise

    current_chunk = chunk_s if chunk_s is not None else retry_chunk_s
    floor_chunk = max(8, min(retry_chunk_s, current_chunk))
    if current_chunk is not None and current_chunk > floor_chunk:
        halved = _retry_halving_chunks(
            run_pipe, pipe, audio_input, language, runtime,
            current_chunk, floor_chunk, window_index,
        )
        if halved is not None:
            return halved

    logger.warning(
        "%s CUDA OOM persists; retrying batch=1 chunk=%ds chunk timestamps.",
        runtime.engine_name,
        retry_chunk_s,
    )
    runtime.clear_cuda_cache()
    from backend import vram_state

    vram_state.teardown(aggressive=True)
    return run_pipe(pipe, audio_input, language, True, 1, retry_chunk_s)


def _recover_cuda_context(runtime: WhisperRuntime, pipe: Any) -> Any:
    """Teardown CUDA caches and reload the pipeline when the context is corrupt."""
    from backend import vram_state

    runtime.clear_cuda_cache()
    vram_state.teardown(aggressive=True)
    vram_state.recover_cuda()
    if runtime.reload_pipeline is None:
        return pipe
    logger.warning("%s reloading pipeline after CUDA context error.", runtime.engine_name)
    return runtime.reload_pipeline()


def _retry_pipe_on_cuda_unknown(
    run_pipe: Callable[..., dict],
    pipe: Any,
    audio_input: Any,
    language: str,
    runtime: WhisperRuntime,
    retry_chunk_s: int,
) -> tuple[dict, Any]:
    """Recover a corrupted CUDA context and retry with conservative settings.

    If recovery fails again with cudaErrorUnknown, the context is dead and
    cannot be rebuilt in-process; request a process restart (a no-op that
    re-raises when CUDA_AUTO_RESTART is disabled) so a supervisor brings the
    service back with a fresh GPU context.
    """
    try:
        pipe = _recover_cuda_context(runtime, pipe)
        result = run_pipe(pipe, audio_input, language, True, 1, retry_chunk_s)
        return result, pipe
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if is_cuda_unknown_error(exc):
            from backend import vram_state

            vram_state.request_cuda_restart(
                f"{runtime.engine_name} CUDA context lost mid-transcription",
            )
        raise


def run_pipe_with_oom_retry(
    run_pipe: Callable[..., dict],
    pipe: Any,
    audio_input: Any,
    language: str,
    timestamp_mode: Any,
    batch_size: int,
    runtime: WhisperRuntime,
    audio_duration_s: float = 0.0,
    chunk_length_s: int | None | object = _PIPE_CHUNK_AUTO,
    window_index: int = 0,
) -> tuple[dict, Any]:
    """Run the HF pipeline; retry on CUDA OOM or unknown driver errors."""
    if chunk_length_s is _PIPE_CHUNK_AUTO:
        chunk_s = _pipe_chunk_length(runtime, audio_duration_s)
    else:
        chunk_s = chunk_length_s
    retry_chunk_s = runtime.retry_chunk_length_s()
    active_pipe = pipe

    def _call(batch: int, chunk: int | None, ts_mode=timestamp_mode, active=active_pipe):
        if chunk is not None:
            return run_pipe(active, audio_input, language, ts_mode, batch, chunk)
        return run_pipe(active, audio_input, language, ts_mode, batch)

    try:
        return _call(batch_size, chunk_s), active_pipe
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not is_cuda_recoverable(exc):
            raise
        if is_cuda_unknown_error(exc):
            logger.warning(
                "%s CUDA unknown error at batch=%d chunk=%s; recovering and retrying batch=1 chunk=%ds.",
                runtime.engine_name,
                batch_size,
                chunk_s,
                retry_chunk_s,
            )
            return _retry_pipe_on_cuda_unknown(
                run_pipe, active_pipe, audio_input, language, runtime, retry_chunk_s,
            )
        runtime.clear_cuda_cache()
        safe_ts = True if timestamp_mode == "word" else timestamp_mode
        if safe_ts is not timestamp_mode:
            logger.warning(
                "%s CUDA OOM with word timestamps; retrying with chunk timestamps.",
                runtime.engine_name,
            )
        result = _retry_pipe_on_oom(
            run_pipe, active_pipe, audio_input, language, safe_ts,
            batch_size, runtime, chunk_s, retry_chunk_s, window_index,
        )
        return result, active_pipe


def _post_process_window(
    window_result: dict,
    window_duration_s: float,
    max_chars_per_subchunk: int,
    offset_s: float,
) -> dict:
    from engines.timestamps import (
        normalize_window_chunks,
        offset_result_timestamps,
        subdivide_large_chunks,
    )

    normalised = normalize_window_chunks(window_result, window_duration_s)
    subdivided = subdivide_large_chunks(normalised, max_chars_per_subchunk)
    return offset_result_timestamps(subdivided, offset_s)


def _effective_long_form_window_s(
    runtime: WhisperRuntime,
    audio_duration_s: float,
    diarization_active: bool,
) -> int:
    window_s = runtime.long_form_window_s(audio_duration_s)
    if not diarization_active:
        return window_s
    fast_window = _env_int("ASR_DIAR_WINDOWED_WINDOW_S", 0)
    if fast_window > 0:
        return fast_window
    cap = max(30, _env_int("DIARIZATION_MAX_ASR_WINDOW_S", 300))
    if window_s > cap:
        logger.info(
            "Diarization active: capping ASR window from %ds to %ds for speaker alignment.",
            window_s,
            cap,
        )
        return cap
    return window_s


def run_long_form_asr_from_path(
    pipe: Any,
    audio_path: str,
    language: str,
    timestamp_mode: Any,
    runtime: WhisperRuntime,
    audio_duration_s: float,
    run_pipe: Callable[..., dict],
    cancel_event=None,
    window_progress=None,
    diarization_active: bool = False,
) -> dict:
    """Stream overlapped windows from disk — never load the full file into RAM."""
    from engines.audio_io import count_audio_windows, iter_audio_windows_from_path
    from engines.timestamps import merge_window_results
    from backend import vram_state

    window_s = _effective_long_form_window_s(runtime, audio_duration_s, diarization_active)
    overlap_s = min(runtime.long_form_overlap_s(), max(0, window_s // 4))
    total_windows = count_audio_windows(audio_duration_s, window_s, overlap_s)
    max_chars = runtime.max_chars_per_subchunk()
    logger.info(
        "%s long-form streaming: windows=%d window=%ds overlap=%ds duration=%.1fs",
        runtime.engine_name,
        total_windows,
        window_s,
        overlap_s,
        audio_duration_s,
    )
    results: list[dict] = []
    if window_progress:
        window_progress(0, total_windows)

    for index, (offset_s, window_input) in enumerate(
        iter_audio_windows_from_path(audio_path, window_s, overlap_s),
        start=1,
    ):
        _check_job_cancelled(cancel_event)
        vram_state.log_phase(f"{runtime.engine_name}_asr_window_{index}", before=True)
        window_duration_s = len(window_input["raw"]) / window_input["sampling_rate"]
        batch_size = _pipe_batch_size(runtime, audio_duration_s, window_duration_s)
        chunk_s = _pipe_chunk_length(runtime, window_duration_s)
        logger.info(
            "%s ASR window %d/%d: offset=%.1fs dur=%.1fs batch=%d chunk=%s",
            runtime.engine_name,
            index,
            total_windows,
            offset_s,
            window_duration_s,
            batch_size,
            chunk_s,
        )
        window_result, pipe = run_pipe_with_oom_retry(
            run_pipe,
            pipe,
            window_input,
            language,
            timestamp_mode,
            batch_size,
            runtime,
            audio_duration_s=audio_duration_s,
            chunk_length_s=chunk_s,
            window_index=index,
        )
        shifted = _post_process_window(window_result, window_duration_s, max_chars, offset_s)
        results.append(shifted)
        del window_input, window_result, shifted
        _maybe_teardown_between_windows(runtime, index)
        vram_state.log_phase(f"{runtime.engine_name}_asr_window_{index}", before=False)
        if window_progress:
            window_progress(index, total_windows)
    return merge_window_results(results)


def _should_stream_windows(runtime: WhisperRuntime, audio_duration_s: float) -> bool:
    """Use disk window streaming for medium+ clips to avoid full-file RAM load."""
    if audio_duration_s >= runtime.long_form_min_duration_s():
        return True
    if not _strict_memory_mode():
        return False
    min_stream_s = _env_int("ASR_8GB_STREAM_MIN_DURATION_S", 180)
    return audio_duration_s >= min_stream_s


def _turn_guided_asr_enabled(diarization_active: bool) -> bool:
    if not diarization_active:
        return False
    return _env_bool("ASR_TURN_GUIDED", True)


def _chunk_length_for_turn(dur_s: float, runtime: WhisperRuntime) -> int | None:
    from backend.asr_quality import is_accuracy_mode

    if is_accuracy_mode():
        return None
    if dur_s < 20.0:
        return None
    chunk_s = runtime.chunk_length_s()
    if dur_s <= chunk_s * 1.25:
        return None
    return min(chunk_s, max(15, int(dur_s)))


def format_turn_guided_transcript(chunks: list[dict]) -> str:
    from engines.diarization import _fmt_ts
    from engines.text_cleanup import clean_transcript_lines, clean_transcript_text

    lines: list[str] = []
    for chunk in chunks:
        text = clean_transcript_text((chunk.get("text") or "").strip())
        if not text:
            continue
        ts = chunk.get("timestamp") or (None, None)
        start, end = ts if ts else (0.0, 0.0)
        speaker = chunk.get("speaker") or "SPEAKER_01"
        lines.append(f"[{_fmt_ts(start)} → {_fmt_ts(end)}] [{speaker}]: {text}")
    if not lines:
        return _NO_SPEECH_MSG
    return clean_transcript_lines("\n".join(lines))


def _text_from_timestamped_chunks(
    chunks: list[dict],
    *,
    turn_start: float,
    turn_end: float,
    slice_offset: float,
) -> str:
    """Collect chunk text whose midpoint falls inside the turn window."""
    margin = max(0.0, _env_float("ASR_TURN_BOUNDARY_MARGIN_S", 0.08))
    keep_start = turn_start + margin
    keep_end = turn_end - margin
    parts: list[str] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        ts = chunk.get("timestamp")
        if ts and ts[0] is not None:
            abs_start = slice_offset + float(ts[0])
            abs_end = slice_offset + float(ts[1]) if ts[1] is not None else abs_start
            mid = (abs_start + abs_end) / 2.0
            if mid < keep_start or mid > keep_end:
                continue
        parts.append(text)
    return " ".join(parts).strip()


def _fallback_result_text(result: dict, chunks: list[dict]) -> str:
    text = (result.get("text") or "").strip()
    if text:
        return text
    return " ".join(
        (chunk.get("text") or "").strip()
        for chunk in chunks
        if (chunk.get("text") or "").strip()
    ).strip()


def _extract_turn_text(
    result: dict,
    *,
    turn_start: float = 0.0,
    turn_end: float = 0.0,
    slice_offset: float = 0.0,
) -> str:
    chunks = result.get("chunks") or []
    if chunks and turn_end > turn_start:
        filtered = _text_from_timestamped_chunks(
            chunks,
            turn_start=turn_start,
            turn_end=turn_end,
            slice_offset=slice_offset,
        )
        if filtered:
            return filtered
    return _fallback_result_text(result, chunks)


def _turn_audio_window(
    turn: dict,
    audio_duration_s: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return padded slice [start,end] and absolute turn bounds for word filtering."""
    from backend.asr_quality import is_accuracy_mode

    turn_start = float(turn["start"])
    turn_end = float(turn["end"])
    pad = _env_float("ASR_TURN_PAD_S", 0.20 if is_accuracy_mode() else 0.0)
    trim = _env_float("ASR_TURN_BOUNDARY_TRIM_S", 0.0)

    slice_start = max(0.0, turn_start - pad)
    slice_end = turn_end + pad
    if audio_duration_s > 0:
        slice_end = min(slice_end, audio_duration_s)

    if trim > 0 and (turn_end - turn_start) > (trim * 2 + 0.5):
        slice_start = max(0.0, turn_start - pad) + trim
        slice_end = min(slice_end, turn_end + pad) - trim
        if slice_end <= slice_start:
            slice_start = max(0.0, turn_start - pad)
            slice_end = min(turn_end + pad, audio_duration_s or turn_end + pad)

    return slice_start, slice_end, turn_start, turn_end


def _text_compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / max(1, len(zlib.compress(raw)))


def _has_repeated_ngram(text: str, n: int = 4, min_repeats: int = 3) -> bool:
    words = text.split()
    if len(words) < n * min_repeats:
        return False
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return max(Counter(ngrams).values()) >= min_repeats


def _reject_hallucinated_turn(text: str, turn_duration_s: float) -> bool:
    """Return True when turn output looks like a Whisper loop hallucination."""
    if not text.strip():
        return False
    if not _env_bool("ASR_REJECT_HALLUCINATED_TURNS", True):
        return False
    max_ratio = _env_float("ASR_TURN_MAX_COMPRESSION_RATIO", 2.0)
    max_chars_s = _env_float("ASR_TURN_MAX_CHARS_PER_S", 40.0)
    min_chars_ratio = max(80, _env_int("ASR_HALLUCINATION_MIN_CHARS", 80))
    min_chars_rate = max(120, _env_int("ASR_HALLUCINATION_RATE_MIN_CHARS", 120))
    if (
        turn_duration_s < 3.0
        and len(text) >= min_chars_ratio
        and _text_compression_ratio(text) > max_ratio
    ):
        return True
    if (
        turn_duration_s < 5.0
        and len(text) >= min_chars_rate
        and len(text) / max(0.1, turn_duration_s) > max_chars_s
    ):
        return True
    ngram_size = max(2, _env_int("ASR_HALLUCINATION_NGRAM_SIZE", 4))
    min_words_ngram = max(12, _env_int("ASR_HALLUCINATION_MIN_WORDS", 12))
    words = text.split()
    if len(words) >= min_words_ngram and _has_repeated_ngram(text, n=ngram_size):
        return True
    return False


def _run_turn_inference(
    audio_path: str,
    slice_start: float,
    slice_dur: float,
    *,
    language: str,
    ts_mode: Any,
    pipe: Any,
    run_pipe: Callable[..., dict],
    runtime: WhisperRuntime,
    window_index: int,
) -> dict:
    from engines.audio_io import load_audio_slice

    audio_input = load_audio_slice(audio_path, slice_start, slice_dur)
    try:
        out, _pipe = run_pipe_with_oom_retry(
            run_pipe,
            pipe,
            audio_input,
            language,
            ts_mode,
            1,
            runtime,
            audio_duration_s=slice_dur,
            chunk_length_s=_chunk_length_for_turn(slice_dur, runtime),
            window_index=window_index,
        )
        return out
    finally:
        del audio_input
        if slice_dur > 300:
            runtime.clear_cuda_cache()


@dataclass(frozen=True)
class _TurnDecodeParams:
    turn: dict
    index: int
    audio_path: str
    language: str
    ts_mode: Any
    pipe: Any
    run_pipe: Callable[..., dict]
    runtime: WhisperRuntime
    audio_duration_s: float
    slice_start: float
    slice_dur: float
    turn_start: float
    turn_end: float


def _retry_turn_without_hallucination(params: _TurnDecodeParams) -> str | None:
    """Re-decode a turn at temperature 0 after hallucination rejection."""
    dur = params.turn["end"] - params.turn["start"]
    saved_temp = os.environ.get("ASR_TEMPERATURE")
    saved_pad = os.environ.get("ASR_TURN_PAD_S")
    os.environ["ASR_TEMPERATURE"] = "0.0"
    retry_slice_start = params.slice_start
    retry_slice_dur = params.slice_dur
    retry_turn_start = params.turn_start
    retry_turn_end = params.turn_end
    if dur < 2.0:
        os.environ["ASR_TURN_PAD_S"] = "0.0"
        retry_slice_start, retry_slice_end, retry_turn_start, retry_turn_end = (
            _turn_audio_window(params.turn, params.audio_duration_s)
        )
        retry_slice_dur = retry_slice_end - retry_slice_start
    try:
        result = _run_turn_inference(
            params.audio_path,
            retry_slice_start,
            retry_slice_dur,
            language=params.language,
            ts_mode=params.ts_mode,
            pipe=params.pipe,
            run_pipe=params.run_pipe,
            runtime=params.runtime,
            window_index=params.index,
        )
        retry_text = _extract_turn_text(
            result,
            turn_start=retry_turn_start,
            turn_end=retry_turn_end,
            slice_offset=retry_slice_start,
        )
    finally:
        if saved_temp is None:
            os.environ.pop("ASR_TEMPERATURE", None)
        else:
            os.environ["ASR_TEMPERATURE"] = saved_temp
        if saved_pad is None:
            os.environ.pop("ASR_TURN_PAD_S", None)
        else:
            os.environ["ASR_TURN_PAD_S"] = saved_pad
    if retry_text and not _reject_hallucinated_turn(retry_text, dur):
        return retry_text
    return None


def _strip_leading_filler_bleed(text: str) -> str:
    """Drop short ASR bleed words duplicated from the prior turn boundary."""
    for prefix in ("สวย ด้วย", "ด้วย", "ครับ", "นะครับ"):
        if text.startswith(prefix + " "):
            return text[len(prefix) + 1:].strip()
        if text == prefix:
            return ""
    return text


def _trim_turn_bleed(prev: str, cur: str) -> str:
    """Drop duplicated prefix on cur when it repeats the end of prev."""
    prev_words = prev.split()
    cur_words = cur.split()
    max_words = min(len(prev_words), len(cur_words), 8)
    for size in range(max_words, 0, -1):
        if prev_words[-size:] == cur_words[:size]:
            return " ".join(cur_words[size:]).strip()
    max_chars = min(len(prev), len(cur), 48)
    for size in range(max_chars, 2, -1):
        suffix = prev[-size:]
        if cur.startswith(suffix):
            return cur[size:].strip()
    return cur


def _line_timestamp_bounds(
    result: dict,
    *,
    slice_offset: float,
    turn_start: float,
    turn_end: float,
    fallback_start: float,
    fallback_end: float,
) -> tuple[float, float]:
    """Use first/last decoded word inside the turn when word timestamps exist."""
    if not _env_bool("ASR_WORD_TIMESTAMPS_WITH_DIARIZATION", False):
        return fallback_start, fallback_end
    word_starts: list[float] = []
    word_ends: list[float] = []
    for chunk in result.get("chunks") or []:
        ts = chunk.get("timestamp")
        if not ts or ts[0] is None:
            continue
        abs_start = slice_offset + float(ts[0])
        abs_end = slice_offset + float(ts[1]) if ts[1] is not None else abs_start
        if abs_end < turn_start or abs_start > turn_end:
            continue
        word_starts.append(abs_start)
        word_ends.append(abs_end)
    if not word_starts:
        return fallback_start, fallback_end
    start = max(turn_start, min(word_starts))
    end = min(turn_end, max(word_ends))
    # Match _fmt_ts (truncate to whole seconds) for line output consistency.
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        return int(fallback_start), max(int(fallback_start) + 1, int(fallback_end))
    return start_i, end_i


def _turn_line_timestamp_bounds(
    turn: dict,
    result: dict,
    *,
    slice_offset: float,
    turn_start: float,
    turn_end: float,
) -> tuple[int, int]:
    """Display timestamps for one turn-guided line (reference parity uses diar bounds)."""
    from backend.asr_quality import is_accuracy_mode

    if _env_bool(
        "ASR_TURN_USE_DIAR_TIMESTAMPS",
        is_accuracy_mode() or _env_bool("ASR_TURN_GUIDED", True),
    ):
        start = int(turn["start"])
        end = int(turn["end"])
        if end <= start:
            end = start + max(1, int(round(turn["end"] - turn["start"])))
        return start, end
    start, end = _line_timestamp_bounds(
        result,
        slice_offset=slice_offset,
        turn_start=turn_start,
        turn_end=turn_end,
        fallback_start=turn["start"],
        fallback_end=turn["end"],
    )
    return int(start), int(end)


def _transcribe_single_turn(
    turn: dict,
    index: int,
    audio_path: str,
    language: str,
    ts_mode: Any,
    pipe: Any,
    run_pipe: Callable[..., dict],
    runtime: WhisperRuntime,
    audio_duration_s: float = 0.0,
) -> dict | None:
    from backend import vram_state

    dur = turn["end"] - turn["start"]
    if dur < _env_float("ASR_TURN_GUIDED_MIN_TURN_S", 0.4):
        return None
    slice_start, slice_end, turn_start, turn_end = _turn_audio_window(
        turn, audio_duration_s,
    )
    slice_dur = slice_end - slice_start
    if slice_dur < _env_float("ASR_TURN_GUIDED_MIN_TURN_S", 0.4):
        return None
    from engines.whisper_utils import WHISPER_MAX_CHUNK_S

    if slice_dur > WHISPER_MAX_CHUNK_S:
        logger.warning(
            "%s turn %d: slice %.1fs exceeds Whisper ceiling; clamping to %ds.",
            runtime.engine_name,
            index,
            slice_dur,
            WHISPER_MAX_CHUNK_S,
        )
        slice_dur = WHISPER_MAX_CHUNK_S
    vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=True)

    result = _run_turn_inference(
        audio_path,
        slice_start,
        slice_dur,
        language=language,
        ts_mode=ts_mode,
        pipe=pipe,
        run_pipe=run_pipe,
        runtime=runtime,
        window_index=index,
    )
    text = _extract_turn_text(
        result,
        turn_start=turn_start,
        turn_end=turn_end,
        slice_offset=slice_start,
    )
    if text and _reject_hallucinated_turn(text, dur):
        retry_min_s = _env_float("ASR_HALLUCINATION_RETRY_MIN_DURATION_S", 1.5)
        if dur < retry_min_s:
            logger.warning(
                "%s turn %d: rejected hallucinated output on %.1fs turn; skipping.",
                runtime.engine_name,
                index,
                dur,
            )
            vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=False)
            return None
        logger.warning(
            "%s turn %d: rejected hallucinated output (%d chars, %.1fs); retrying.",
            runtime.engine_name,
            index,
            len(text),
            dur,
        )
        text = _retry_turn_without_hallucination(
            _TurnDecodeParams(
                turn=turn,
                index=index,
                audio_path=audio_path,
                language=language,
                ts_mode=ts_mode,
                pipe=pipe,
                run_pipe=run_pipe,
                runtime=runtime,
                audio_duration_s=audio_duration_s,
                slice_start=slice_start,
                slice_dur=slice_dur,
                turn_start=turn_start,
                turn_end=turn_end,
            ),
        )
        if not text:
            vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=False)
            return None
    vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=False)
    if not text:
        return None
    line_start, line_end = _turn_line_timestamp_bounds(
        turn,
        result,
        slice_offset=slice_start,
        turn_start=turn_start,
        turn_end=turn_end,
    )
    return {
        "text": text,
        "timestamp": (line_start, line_end),
        "speaker": turn["speaker"],
    }


def _dedupe_adjacent_turn_bleed(chunks: list[dict]) -> list[dict]:
    """Remove duplicated words bleeding across consecutive turn slices."""
    if len(chunks) < 2:
        return chunks
    out: list[dict] = [dict(chunk) for chunk in chunks]
    for idx in range(1, len(out)):
        prev = (out[idx - 1].get("text") or "").strip()
        cur = (out[idx].get("text") or "").strip()
        if not prev or not cur:
            continue
        trimmed = _trim_turn_bleed(prev, cur)
        trimmed = _strip_leading_filler_bleed(trimmed)
        if trimmed != cur:
            out[idx]["text"] = trimmed
    return [chunk for chunk in out if (chunk.get("text") or "").strip()]


def _speaker_chunk_gap_s(prev: dict, chunk: dict) -> float:
    prev_ts = prev.get("timestamp") or (0.0, 0.0)
    cur_ts = chunk.get("timestamp") or (0.0, 0.0)
    return float(cur_ts[0] or 0.0) - float(prev_ts[1] or prev_ts[0] or 0.0)


def _combine_speaker_chunk_text(prev: dict, chunk: dict) -> None:
    prev_ts = prev.get("timestamp") or (0.0, 0.0)
    cur_ts = chunk.get("timestamp") or (0.0, 0.0)
    prev_text = (prev.get("text") or "").strip()
    cur_text = (chunk.get("text") or "").strip()
    prev["text"] = f"{prev_text} {cur_text}".strip()
    prev["timestamp"] = (prev_ts[0], cur_ts[1] if cur_ts[1] is not None else cur_ts[0])


def _merge_consecutive_speaker_chunks(chunks: list[dict]) -> list[dict]:
    """Merge adjacent turn chunks for the same speaker when diar left a short gap."""
    max_gap_s = _env_float("ASR_TURN_OUTPUT_MERGE_GAP_S", 0.0)
    if max_gap_s <= 0 or not chunks:
        return chunks
    merged: list[dict] = [dict(chunks[0])]
    for chunk in chunks[1:]:
        prev = merged[-1]
        same_speaker = (chunk.get("speaker") or "") == (prev.get("speaker") or "")
        if not same_speaker or _speaker_chunk_gap_s(prev, chunk) > max_gap_s:
            merged.append(dict(chunk))
            continue
        _combine_speaker_chunk_text(prev, chunk)
    return merged


def run_turn_guided_asr(
    audio_path: str,
    language: str,
    diarization_segments: list[dict],
    pipe: Any,
    run_pipe: Callable[..., dict],
    runtime: WhisperRuntime,
    timestamp_mode: Any,
    audio_duration_s: float,
    max_speakers: int = 0,
    cancel_event=None,
    window_progress=None,
) -> str:
    """Transcribe each diarization turn in isolation — best accuracy for dialogue."""
    from engines.diarization import prepare_asr_turns

    turns = prepare_asr_turns(diarization_segments, max_speakers)
    if not turns:
        logger.warning(
            "%s turn-guided ASR: no turns; falling back to plain transcript.",
            runtime.engine_name,
        )
        return _NO_SPEECH_MSG

    total = len(turns)
    logger.info(
        "%s turn-guided ASR: %d turn(s) over %.1fs audio.",
        runtime.engine_name,
        total,
        audio_duration_s,
    )
    ts_mode = True if timestamp_mode == "word" else timestamp_mode
    output_chunks: list[dict] = []

    for index, turn in enumerate(turns, start=1):
        _check_job_cancelled(cancel_event)
        chunk = _transcribe_single_turn(
            turn, index, audio_path, language, ts_mode, pipe, run_pipe, runtime,
            audio_duration_s=audio_duration_s,
        )
        if chunk:
            output_chunks.append(chunk)
        if window_progress:
            window_progress(index, total)

    output_chunks = _dedupe_adjacent_turn_bleed(output_chunks)
    output_chunks = _merge_consecutive_speaker_chunks(output_chunks)
    return format_turn_guided_transcript(output_chunks)


def format_asr_result(
    result: dict,
    audio_duration_s: float,
    engine_name: str,
    diarization_segments: list | None,
    format_chunks_fn: Callable[[list], str],
    log: logging.Logger,
    max_speakers: int = 0,
) -> str:
    """Repair timestamps, assign speakers, and format plain transcript text."""
    from engines.text_cleanup import clean_transcript_text
    from engines.timestamps import repair_asr_result

    result = repair_asr_result(result, audio_duration_s, engine_name, log)
    if diarization_segments:
        from engines.diarization import assign_speakers

        text = assign_speakers(
            result,
            diarization_segments,
            max_speakers=max_speakers,
            audio_duration_s=audio_duration_s,
        )
        ts_re = re.compile(r"\[\d{2}:\d{2}:\d{2} → \d{2}:\d{2}:\d{2}\] \[SPEAKER_\d+\]:")
        for line in text.splitlines():
            if "[SPEAKER_" in line and not ts_re.match(line.strip()):
                log.warning(
                    "%s diarization output missing timestamp prefix: %s",
                    engine_name,
                    line[:120],
                )
        return text
    chunks = result.get("chunks", [])
    if chunks:
        return format_chunks_fn(chunks)
    return clean_transcript_text(result.get("text", "").strip()) or _NO_SPEECH_MSG


def transcribe_whisper_audio(
    audio_path: str,
    language: str,
    diarization_segments: list | None,
    pipe: Any,
    load_audio: Callable[[str], dict],
    run_pipe: Callable[..., dict],
    runtime: WhisperRuntime,
    timestamp_mode: Any,
    format_chunks_fn: Callable[[list], str],
    cancel_event=None,
    window_progress=None,
    max_speakers: int = 0,
) -> str:
    """Shared Typhoon/Pathumma transcription entry point."""
    from engines.audio_io import probe_audio_duration

    audio_duration_s = probe_audio_duration(audio_path)
    batch_size = _pipe_batch_size(runtime, audio_duration_s, audio_duration_s)
    chunk_s = _pipe_chunk_length(runtime, audio_duration_s)
    min_chunk_dur = _min_chunked_duration_s()
    diarization_active = diarization_segments is not None
    use_turn_guided = (
        diarization_active
        and _turn_guided_asr_enabled(True)
    )
    if use_turn_guided:
        from backend.asr_performance import should_use_windowed_diar_asr

        if should_use_windowed_diar_asr(
            audio_duration_s, diarization_segments, max_speakers,
        ):
            logger.info(
                "%s using windowed ASR + turn-centric speaker assignment "
                "(audio=%.0fs, faster than per-turn passes).",
                runtime.engine_name,
                audio_duration_s,
            )
            use_turn_guided = False
    if use_turn_guided:
        return run_turn_guided_asr(
            audio_path,
            language,
            diarization_segments,
            pipe,
            run_pipe,
            runtime,
            timestamp_mode,
            audio_duration_s,
            max_speakers=max_speakers,
            cancel_event=cancel_event,
            window_progress=window_progress,
        )

    logger.info(
        "%s transcription started: audio=%.1fs language=%s diarization=%s "
        "timestamp_mode=%s batch=%d chunk=%s min_chunk_dur=%.0fs window=%ds "
        "profile=%s streaming=%s max_speakers=%d",
        runtime.engine_name,
        audio_duration_s,
        language,
        diarization_active,
        timestamp_mode,
        batch_size,
        chunk_s,
        min_chunk_dur,
        _effective_long_form_window_s(runtime, audio_duration_s, diarization_active),
        os.getenv("ASR_QUALITY_PROFILE", "balanced"),
        _should_stream_windows(runtime, audio_duration_s),
        max_speakers,
    )

    if _should_stream_windows(runtime, audio_duration_s):
        result = run_long_form_asr_from_path(
            pipe,
            audio_path,
            language,
            timestamp_mode,
            runtime,
            audio_duration_s,
            run_pipe,
            cancel_event=cancel_event,
            window_progress=window_progress,
            diarization_active=diarization_active,
        )
    else:
        if window_progress:
            window_progress(0, 1)
        _check_job_cancelled(cancel_event)
        audio_input = load_audio(audio_path)
        try:
            result, _pipe = run_pipe_with_oom_retry(
                run_pipe,
                pipe,
                audio_input,
                language,
                timestamp_mode,
                batch_size,
                runtime,
                audio_duration_s=audio_duration_s,
                chunk_length_s=chunk_s,
            )
        finally:
            del audio_input
            runtime.clear_cuda_cache()
            gc.collect()
        if window_progress:
            window_progress(1, 1)

    return format_asr_result(
        result,
        audio_duration_s,
        runtime.engine_name,
        diarization_segments,
        format_chunks_fn,
        logger,
        max_speakers=max_speakers,
    )
