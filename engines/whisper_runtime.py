"""Shared Whisper ASR runtime: OOM retry, long-form windowing, transcription."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


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


def _vram_batch_boost(batch: int) -> int:
    """Raise batch when free VRAM headroom allows (8 GB class with ASR resident)."""
    try:
        from backend.vram_state import snapshot

        snap = snapshot()
        max_8gb = max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 4))
        target = max(batch, min(max_8gb, _env_int("ASR_CUDA_BATCH_SIZE", max_8gb)))
        min_free = _env_int("ASR_BATCH_MIN_FREE_MB", 2500)
        if snap.get("free_mb", 0) >= min_free:
            return target
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
    if windowed:
        batch = min(batch, max_8gb)
        return _vram_batch_boost(batch)
    if max_8gb > 1 and _env_bool("ASR_BATCH_DURATION_CAP", True):
        if audio_duration_s >= 180 or window_duration_s >= 120:
            return min(batch, max(1, max_8gb // 2))
        if audio_duration_s >= 60 or window_duration_s >= 60:
            return min(batch, max_8gb)
        return batch
    if audio_duration_s >= 180 or window_duration_s >= 60:
        return 1
    return batch


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
    """Recover a corrupted CUDA context and retry with conservative settings."""
    pipe = _recover_cuda_context(runtime, pipe)
    result = run_pipe(pipe, audio_input, language, True, 1, retry_chunk_s)
    return result, pipe


def run_pipe_with_oom_retry(
    run_pipe: Callable[..., dict],
    pipe: Any,
    audio_input: Any,
    language: str,
    timestamp_mode: Any,
    batch_size: int,
    runtime: WhisperRuntime,
    audio_duration_s: float = 0.0,
    chunk_length_s: int | None = None,
    window_index: int = 0,
) -> tuple[dict, Any]:
    """Run the HF pipeline; retry on CUDA OOM or unknown driver errors."""
    chunk_s = chunk_length_s if chunk_length_s is not None else _pipe_chunk_length(runtime, audio_duration_s)
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
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user.")
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
        import re

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
    return clean_transcript_text(result.get("text", "").strip()) or "(no speech detected)"


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
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user.")
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
