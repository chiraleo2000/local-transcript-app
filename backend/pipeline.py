"""Core backend transcription pipeline."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.asr_local import (
    ALL_ENGINES,
    ENGINE_THONBURIAN,
    ENGINE_TYPHOON,
    asr_worker_count,
    clear_accelerator_cache,
    should_clear_model_between_engines,
    should_clear_models_after_job,
    strict_memory_mode_active,
    transcribe_engine,
    unload_model,
)
from backend.services.media_pipeline import (
    audio_duration_seconds,
    clear_diarization_model,
    diarize_audio,
    enhance_audio,
    normalize_media,
)
from backend.storage import new_job_id, now_iso, save_job_manifest, save_transcript


logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    """Raise RuntimeError if a cancellation has been requested."""
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user.")


def _selected_engines(selected_engines: list[str]) -> list[str]:
    selected = [engine for engine in selected_engines if engine in ALL_ENGINES]
    return selected or list(ALL_ENGINES)


def _speaker_bounds(diarization: bool, min_speakers: int, max_speakers: int) -> tuple[int, int]:
    if not diarization:
        return 0, 0
    n_min = max(1, int(min_speakers))
    n_max = max(n_min, int(max_speakers))
    return n_min, n_max


def _run_diarization(
    process_path: str, enabled: bool, n_min: int, n_max: int,
    diarize_kwargs: dict | None = None,
) -> list[dict] | None:
    if not enabled:
        return None
    try:
        return diarize_audio(process_path, n_min, n_max, diarize_kwargs=diarize_kwargs)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Diarization failed: %s", exc, exc_info=True)
        return None
    finally:
        clear_diarization_model()
        clear_accelerator_cache()


def _success_result(job_id: str, engine: str, text: str, elapsed: float) -> dict:
    transcript_path = save_transcript(job_id, engine, text)
    return {
        "text": text,
        "elapsed": elapsed,
        "download_path": transcript_path,
        "note": "",
    }


def _error_result(engine: str, exc: Exception) -> dict:
    if _is_cuda_oom(exc):
        logger.error("%s failed with CUDA OOM after memory-safe retries: %s", engine, exc)
    else:
        logger.error("%s failed: %s", engine, exc, exc_info=True)
    return {
        "text": f"ERROR: {exc}",
        "elapsed": 0.0,
        "download_path": None,
        "note": "",
    }


def _is_cuda_oom(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, OSError, AttributeError):
        pass
    return "CUDA out of memory" in str(exc)


def _should_fallback_to_thonburian(engine: str, exc: Exception) -> bool:
    return engine == ENGINE_TYPHOON and strict_memory_mode_active() and _is_cuda_oom(exc)


def _run_one_asr_engine(
    job_id: str,
    engine: str,
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
) -> dict:
    try:
        text, elapsed = transcribe_engine(engine, process_path, language, diar_segments)
        return _success_result(job_id, engine, text, elapsed)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if _should_fallback_to_thonburian(engine, exc):
            logger.warning(
                "Typhoon hit CUDA OOM on an 8 GB-class GPU; falling back to Thonburian."
            )
            _unload_asr_engine(engine)
            try:
                text, elapsed = transcribe_engine(
                    ENGINE_THONBURIAN,
                    process_path,
                    language,
                    diar_segments,
                )
                result = _success_result(job_id, engine, text, elapsed)
                result["note"] = "Typhoon recovered by falling back to Thonburian."
                return result
            except Exception as fallback_exc:  # pylint: disable=broad-exception-caught
                return _error_result(ENGINE_THONBURIAN, fallback_exc)
        return _error_result(engine, exc)


def _unload_asr_engine(engine: str) -> None:
    try:
        unload_model(engine)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("%s unload skipped: %s", engine, exc)
    clear_accelerator_cache()


def _run_asr_sequential(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    clear_between = should_clear_model_between_engines()
    for engine in selected:
        _check_cancel(cancel_event)
        results[engine] = _run_one_asr_engine(job_id, engine, process_path, language, diar_segments)
        if clear_between:
            _unload_asr_engine(engine)
    return results


def _run_asr_parallel(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    workers: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(transcribe_engine, engine, process_path, language, diar_segments): engine
            for engine in selected
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                text, elapsed = future.result()
                results[engine] = _success_result(job_id, engine, text, elapsed)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results[engine] = _error_result(engine, exc)
    return results


def _run_asr_engines(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, dict], int]:
    workers = asr_worker_count(len(selected))
    logger.info("Running %d ASR engine(s) with %d worker(s).", len(selected), workers)
    if workers == 1:
        results = _run_asr_sequential(job_id, selected, process_path, language, diar_segments, cancel_event)
        return results, workers

    results = _run_asr_parallel(job_id, selected, process_path, language, diar_segments, workers)
    return results, workers


def _performance_target_seconds(audio_duration_s: float) -> float:
    if audio_duration_s <= 0:
        return 0.0
    if audio_duration_s < 9 * 60:
        return 180.0
    return audio_duration_s / 3.0


def _clear_asr_models(selected: list[str]) -> None:
    if not should_clear_models_after_job():
        return
    for engine in selected:
        try:
            unload_model(engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("%s unload skipped: %s", engine, exc)
    clear_accelerator_cache()


def run_transcription_job(
    media_path: str,
    selected_engines: list[str],
    language: str,
    diarization: bool,
    min_speakers: int,
    max_speakers: int,
    enhance: bool,
    local_correction: bool = False,
    diarize_kwargs: dict | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run the full local transcript pipeline and persist outputs."""
    job_started = time.perf_counter()
    job_id = new_job_id()
    process_path = normalize_media(media_path, job_id)
    _check_cancel(cancel_event)
    if enhance:
        process_path = enhance_audio(process_path)
    _check_cancel(cancel_event)
    audio_duration_s = audio_duration_seconds(process_path)

    selected = _selected_engines(selected_engines)
    n_min, n_max = _speaker_bounds(diarization, min_speakers, max_speakers)
    diar_segments = _run_diarization(process_path, diarization, n_min, n_max, diarize_kwargs=diarize_kwargs)
    _check_cancel(cancel_event)
    results, workers = _run_asr_engines(job_id, selected, process_path, language, diar_segments, cancel_event)
    _clear_asr_models(selected)

    total_elapsed_s = time.perf_counter() - job_started
    target_elapsed_s = _performance_target_seconds(audio_duration_s)
    target_met = bool(target_elapsed_s and total_elapsed_s <= target_elapsed_s)
    if target_elapsed_s:
        logger.info(
            "Job %s performance: audio=%.2fs elapsed=%.2fs target=%.2fs met=%s.",
            job_id,
            audio_duration_s,
            total_elapsed_s,
            target_elapsed_s,
            target_met,
        )

    manifest = {
        "job_id": job_id,
        "created_at": now_iso(),
        "source_path": media_path,
        "processed_path": process_path,
        "selected_engines": selected,
        "language": language,
        "diarization": diarization,
        "min_speakers": n_min,
        "max_speakers": n_max,
        "enhance": enhance,
        "local_correction": local_correction,
        "asr_workers": workers,
        "audio_duration_s": audio_duration_s,
        "total_elapsed_s": total_elapsed_s,
        "target_elapsed_s": target_elapsed_s,
        "target_met": target_met,
        "results": results,
    }
    manifest_path = save_job_manifest(job_id, manifest)
    return {
        "job_id": job_id,
        "manifest_path": manifest_path,
        "audio_duration_s": audio_duration_s,
        "total_elapsed_s": total_elapsed_s,
        "target_elapsed_s": target_elapsed_s,
        "target_met": target_met,
        "results": results,
    }
