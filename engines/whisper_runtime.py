"""Shared Whisper ASR runtime: OOM retry, long-form windowing, transcription."""

from __future__ import annotations

import gc
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_JOB_CANCELLED_MSG = "Job cancelled by user."
_NO_SPEECH_MSG = "(no speech detected)"


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
    if _strict_memory_mode() and window_duration_s >= 90:
        return batch
    try:
        from backend.vram_state import snapshot

        snap = snapshot()
        max_8gb = max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 1))
        target = min(max_8gb, max(batch, _env_int("ASR_CUDA_BATCH_SIZE", max_8gb)))
        min_free = _env_int("ASR_BATCH_MIN_FREE_MB", 5000)
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
        if window_duration_s >= 180:
            return 1
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


def _extract_turn_text(
    result: dict,
    *,
    turn_start: float = 0.0,
    turn_end: float = 0.0,
    slice_offset: float = 0.0,
    turn_dur: float = 0.0,
) -> str:
    chunks = result.get("chunks") or []
    if chunks and turn_end > turn_start:
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
        if parts:
            return " ".join(parts).strip()

    text = (result.get("text") or "").strip()
    if text:
        return text
    return " ".join(
        (chunk.get("text") or "").strip()
        for chunk in chunks
        if (chunk.get("text") or "").strip()
    ).strip()


def _turn_audio_window(
    turn: dict,
    audio_duration_s: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return padded slice [start,end] and absolute turn bounds for word filtering."""
    from backend.asr_quality import is_accuracy_mode

    turn_start = float(turn["start"])
    turn_end = float(turn["end"])
    pad = _env_float("ASR_TURN_PAD_S", 0.20 if is_accuracy_mode() else 0.0)
    trim = _env_float(
        "ASR_TURN_BOUNDARY_TRIM_S",
        0.0 if is_accuracy_mode() else 0.0,
    )

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
    from engines.audio_io import load_audio_slice
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
    vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=True)
    audio_input = load_audio_slice(audio_path, slice_start, slice_dur)
    chunk_s = _chunk_length_for_turn(slice_dur, runtime)
    try:
        result, _pipe = run_pipe_with_oom_retry(
            run_pipe,
            pipe,
            audio_input,
            language,
            ts_mode,
            1,
            runtime,
            audio_duration_s=slice_dur,
            chunk_length_s=chunk_s,
            window_index=index,
        )
    finally:
        del audio_input
        if slice_dur > 120:
            runtime.clear_cuda_cache()
            gc.collect()
    text = _extract_turn_text(
        result,
        turn_start=turn_start,
        turn_end=turn_end,
        slice_offset=slice_start,
    )
    vram_state.log_phase(f"{runtime.engine_name}_turn_{index}", before=False)
    if not text:
        return None
    return {
        "text": text,
        "timestamp": (turn["start"], turn["end"]),
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
        prev_words = prev.split()
        cur_words = cur.split()
        best_words = 0
        max_words = min(len(prev_words), len(cur_words), 8)
        for size in range(max_words, 0, -1):
            if prev_words[-size:] == cur_words[:size]:
                best_words = size
                break
        if best_words:
            trimmed = " ".join(cur_words[best_words:]).strip()
            out[idx]["text"] = trimmed
            continue
        max_chars = min(len(prev), len(cur), 48)
        for size in range(max_chars, 2, -1):
            if prev[-size:] == cur[:size]:
                out[idx]["text"] = cur[size:].strip()
                break
    return [chunk for chunk in out if (chunk.get("text") or "").strip()]


_CONTINUATION_HEADS = frozenset({
    "ด้วย", "และ", "เพราะ", "มัน", "เพจ", "ค่า", "ทริป", "หา", "ไม่", "ส่วน",
    "เยอะเลย", "ค่าเดินทางไปได้เยอะเลย", "ทริปแค่ไม่กี่วัน",
})
_SENTENCE_START_RE = re.compile(
    r"^(เรา|ผม|ฉัน|พวก|เห็น|โห|ตกลง|เฮ้ย|ถ้า|ไป|จริง|นั่น|เอา|เดี๋ยว|ระหว่าง|งั้น|แต่|กาญจนบุรี|เขา|ทะเล|ทุกคน|ทั้ง|โชค)",
)


def _looks_like_sentence_start(words: list[str]) -> bool:
    if not words:
        return True
    joined = " ".join(words[:3])
    if joined in _CONTINUATION_HEADS:
        return False
    first = words[0]
    if first in _CONTINUATION_HEADS:
        return False
    if _SENTENCE_START_RE.match(first):
        return True
    if len(first) <= 2 and len(words) > 1:
        return False
    return len(first) >= 5


def _rebalance_turn_fragments(chunks: list[dict]) -> list[dict]:
    """Move leading continuation fragments from turn N back onto turn N-1."""
    if len(chunks) < 2:
        return chunks
    out: list[dict] = [dict(chunk) for chunk in chunks]
    for idx in range(1, len(out)):
        prev_words = (out[idx - 1].get("text") or "").split()
        cur_words = (out[idx].get("text") or "").split()
        if not prev_words or not cur_words:
            continue
        moved = 0
        while cur_words and moved < 6 and not _looks_like_sentence_start(cur_words):
            prev_words.append(cur_words.pop(0))
            moved += 1
        if moved:
            out[idx - 1]["text"] = " ".join(prev_words).strip()
            out[idx]["text"] = " ".join(cur_words).strip()
    return [chunk for chunk in out if (chunk.get("text") or "").strip()]


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
    output_chunks = _rebalance_turn_fragments(output_chunks)
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
    if diarization_active and _turn_guided_asr_enabled(True):
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
