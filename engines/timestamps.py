"""ASR timestamp repair helpers for long audio outputs."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def audio_duration_from_input(audio_input: dict[str, Any]) -> float:
    """Return seconds represented by a HuggingFace ASR audio input dict."""
    raw = audio_input.get("raw")
    sample_rate = audio_input.get("sampling_rate") or 16000
    if raw is None or not sample_rate:
        return 0.0
    try:
        return float(len(raw) / sample_rate)
    except TypeError:
        return 0.0


def _timestamp_pair(chunk: dict) -> tuple[float | None, float | None]:
    timestamp = chunk.get("timestamp")
    if not timestamp or len(timestamp) < 2:
        return None, None
    return timestamp[0], timestamp[1]


def _timestamp_bounds(chunks: list[dict]) -> tuple[float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    for chunk in chunks:
        start, end = _timestamp_pair(chunk)
        if start is not None:
            starts.append(float(start))
        if end is not None:
            ends.append(float(end))
    return (min(starts) if starts else None, max(ends) if ends else None)


def _shift_timestamps(chunks: list[dict], offset_s: float) -> list[dict]:
    shifted = deepcopy(chunks)
    for chunk in shifted:
        start, end = _timestamp_pair(chunk)
        if start is None:
            continue
        new_start = max(0.0, float(start) - offset_s)
        new_end = max(new_start, float(end) - offset_s) if end is not None else None
        timestamp_type = type(chunk.get("timestamp"))
        chunk["timestamp"] = timestamp_type((new_start, new_end))
    return shifted


def _is_pathological_full_text_chunk(
    chunk: dict,
    full_text_len: int,
    audio_duration_s: float,
) -> bool:
    text = chunk.get("text", "").strip()
    if len(text) < 1000 or full_text_len < 1000:
        return False
    if len(text) < full_text_len * 0.70:
        return False

    start, end = _timestamp_pair(chunk)
    if start is None or end is None or end <= start:
        return True

    duration = float(end) - float(start)
    max_chars_per_second = _env_float("ASR_MAX_CHUNK_CHARS_PER_SECOND", 35.0)
    looks_too_dense = len(text) / max(duration, 0.001) > max_chars_per_second
    starts_late = float(start) > _env_float("ASR_SUSPICIOUS_FIRST_TS_S", 60.0)
    too_short_for_full_audio = audio_duration_s > 0 and duration < audio_duration_s * 0.50
    return looks_too_dense and (starts_late or too_short_for_full_audio)


def _repair_pathological_full_text_chunk(
    result: dict,
    audio_duration_s: float,
    engine_name: str,
    logger: logging.Logger,
) -> dict | None:
    chunks = result.get("chunks") or []
    full_text = result.get("text", "").strip()
    if not chunks or not full_text:
        return None

    for chunk in chunks:
        if _is_pathological_full_text_chunk(chunk, len(full_text), audio_duration_s):
            repaired = dict(result)
            repaired["chunks"] = [{
                "text": full_text,
                "timestamp": (0.0, audio_duration_s or None),
            }]
            start, end = _timestamp_pair(chunk)
            logger.warning(
                "%s ASR returned a pathological full-text chunk "
                "(chars=%d, ts=%s-%s). Re-mapping full transcript across %.1fs.",
                engine_name,
                len(full_text),
                start,
                end,
                audio_duration_s,
            )
            return repaired
    return None


def repair_asr_result(
    result: dict,
    audio_duration_s: float,
    engine_name: str,
    logger: logging.Logger,
) -> dict:
    """Repair impossible timestamp output from long-form ASR chunking."""
    chunks = result.get("chunks") or []
    first_start, last_end = _timestamp_bounds(chunks)
    logger.info(
        "%s ASR result: text_chars=%d chunks=%d audio=%.1fs first_ts=%s last_ts=%s",
        engine_name,
        len(result.get("text", "")),
        len(chunks),
        audio_duration_s,
        first_start,
        last_end,
    )

    repaired = _repair_pathological_full_text_chunk(
        result, audio_duration_s, engine_name, logger
    )
    if repaired is not None:
        return repaired

    if not chunks or first_start is None or last_end is None or audio_duration_s <= 0:
        return result

    tolerance_s = _env_float("ASR_TIMESTAMP_AUDIO_END_TOLERANCE_S", 30.0)
    suspicious_start_s = _env_float("ASR_SUSPICIOUS_FIRST_TS_S", 60.0)
    if first_start >= suspicious_start_s and last_end > audio_duration_s + tolerance_s:
        repaired = dict(result)
        repaired["chunks"] = _shift_timestamps(chunks, first_start)
        logger.warning(
            "%s ASR timestamps exceeded audio duration "
            "(first=%.1fs last=%.1fs audio=%.1fs); shifted by %.1fs.",
            engine_name,
            first_start,
            last_end,
            audio_duration_s,
            first_start,
        )
        return repaired

    if first_start >= suspicious_start_s:
        logger.warning(
            "%s ASR first timestamp is %.1fs into a %.1fs file. "
            "Keeping timestamps because they do not exceed audio duration.",
            engine_name,
            first_start,
            audio_duration_s,
        )
    return result
