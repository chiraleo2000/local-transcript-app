"""Core backend transcription pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.asr_local import (
    ALL_ENGINES,
    asr_worker_count,
    clear_accelerator_cache,
    default_asr_engines,
    load_model,
    should_clear_model_between_engines,
    should_clear_models_after_job,
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

try:
    from backend.progress import JobProgress
except ImportError:  # pragma: no cover
    JobProgress = None  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)


def _check_cancel(cancel_event: threading.Event | None) -> None:
    """Raise RuntimeError if a cancellation has been requested."""
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user.")


def _cleanup_cancelled_job(temp_files: list[str]) -> None:
    """Unload all models and delete temp files for a cancelled job.

    Unloading is mandatory: a broad-except in ASR engine code may have already
    swallowed the cancel RuntimeError, leaving model weights pinned in VRAM.
    Without an explicit unload, the next job starts with a fragmented VRAM
    state and frequently hits OOM.
    """
    # Unload ASR models first — releases the largest VRAM consumers.
    for engine in ALL_ENGINES:
        try:
            unload_model(engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Cancel cleanup: ASR model unload failed for %s: %s", engine, exc)
    # Unload diarization pipeline.
    try:
        clear_diarization_model()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("Cancel cleanup: diarization model unload failed: %s", exc)
    # Delete temporary audio files created for this job.
    tmp_root = tempfile.gettempdir()
    for path in temp_files:
        try:
            if not os.path.isfile(path):
                continue
            parent = os.path.dirname(os.path.abspath(path))
            # preprocess_audio writes into mkdtemp(prefix="asr_preprocess_") dirs
            if os.path.basename(parent).startswith("asr_preprocess_") and \
                    os.path.commonpath([parent, tmp_root]) == tmp_root:
                shutil.rmtree(parent, ignore_errors=True)
                logger.info("Cancel cleanup: removed preprocess dir %s", parent)
            else:
                os.remove(path)
                logger.info("Cancel cleanup: removed temp audio %s", path)
        except OSError as exc:
            logger.debug("Cancel cleanup failed for %s: %s", path, exc)
    clear_accelerator_cache()


def _selected_engines(selected_engines: list[str] | str) -> list[str]:
    if isinstance(selected_engines, str):
        candidates = [selected_engines]
    else:
        candidates = list(selected_engines or [])
    selected = [engine for engine in candidates if engine in ALL_ENGINES]
    return (selected or default_asr_engines())[:1]


def _speaker_limit(diarization: bool, max_speakers: int) -> int:
    if not diarization:
        return 0
    return max(1, int(max_speakers))


def _run_diarization(
    process_path: str, enabled: bool, max_speakers: int,
    diarize_kwargs: dict | None = None,
) -> list[dict] | None:
    if not enabled:
        logger.info("Diarization disabled for this job.")
        return None
    try:
        logger.info(
            "Diarization starting: max_speakers=%d overrides=%s",
            max_speakers,
            diarize_kwargs or {},
        )
        segments = diarize_audio(process_path, max_speakers, diarize_kwargs=diarize_kwargs)
        logger.info("Diarization finished: segments=%d", len(segments))
        return segments
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Diarization failed: %s", exc)
        return None
    finally:
        clear_diarization_model()
        clear_accelerator_cache()


def _success_result(job_id: str, engine: str, text: str, elapsed: float) -> dict:
    transcript_path = save_transcript(job_id, engine, text)
    logger.info(
        "Job %s ASR success: engine=%s elapsed=%.2fs chars=%d transcript=%s",
        job_id,
        engine,
        elapsed,
        len(text),
        transcript_path,
    )
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
        import torch  # pylint: disable=import-outside-toplevel

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, OSError, AttributeError):
        pass
    return "CUDA out of memory" in str(exc)


def _run_one_asr_engine(
    job_id: str,
    engine: str,
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
    window_progress=None,
) -> dict:
    try:
        text, elapsed = transcribe_engine(
            engine,
            process_path,
            language,
            diar_segments,
            cancel_event=cancel_event,
            window_progress=window_progress,
        )
        return _success_result(job_id, engine, text, elapsed)
    except RuntimeError as exc:
        # Re-raise cancellation so run_transcription_job's cleanup handler fires.
        # A broad except further up the call chain must not swallow this.
        if "cancelled" in str(exc).lower():
            raise
        return _error_result(engine, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _error_result(engine, exc)


def _unload_asr_engine(engine: str) -> None:
    try:
        unload_model(engine)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug("%s unload skipped: %s", engine, exc)
    clear_accelerator_cache()


def _prepare_selected_asr_model(selected: list[str]) -> None:
    if not selected:
        return
    selected_engine = selected[0]
    for engine in ALL_ENGINES:
        if engine != selected_engine:
            _unload_asr_engine(engine)
    logger.info("Ensuring selected ASR model is loaded: %s", selected_engine)
    load_model(selected_engine)
    clear_accelerator_cache()


def _run_asr_sequential(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
    window_progress=None,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    clear_between = should_clear_model_between_engines()
    for engine in selected:
        _check_cancel(cancel_event)
        results[engine] = _run_one_asr_engine(
            job_id,
            engine,
            process_path,
            language,
            diar_segments,
            cancel_event,
            window_progress=window_progress,
        )
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
    window_progress=None,
) -> tuple[dict[str, dict], int]:
    workers = asr_worker_count(len(selected))
    logger.info("Running %d ASR engine(s) with %d worker(s).", len(selected), workers)
    if workers == 1:
        results = _run_asr_sequential(
            job_id,
            selected,
            process_path,
            language,
            diar_segments,
            cancel_event,
            window_progress=window_progress,
        )
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
    max_speakers: int,
    enhance: bool,
    local_correction: bool = False,
    diarize_kwargs: dict | None = None,
    cancel_event: threading.Event | None = None,
    progress: JobProgress | None = None,
) -> dict:
    """Run the full local transcript pipeline and persist outputs."""

    def _phase(phase: str, message: str, percent: float) -> None:
        if progress is not None:
            progress.set_phase(phase, message, percent)

    job_started = time.perf_counter()
    _phase("starting", "Starting transcription job\u2026", 2)
    job_id = new_job_id()
    if progress is not None:
        progress.set_job_id(job_id)
    logger.info(
        "Job %s started: source=%s engines=%s language=%s diarization=%s enhance=%s",
        job_id,
        media_path,
        selected_engines,
        language,
        diarization,
        enhance,
    )
    _temp_files: list[str] = []
    try:
        _phase("normalize", "Normalizing media\u2026", 8)
        process_path = normalize_media(media_path, job_id)
        if process_path != media_path:
            _temp_files.append(process_path)
        logger.info("Job %s normalized media path: %s", job_id, process_path)
        _check_cancel(cancel_event)
        if enhance:
            _phase("enhance", "Enhancing audio\u2026", 18)
            enhanced_path = enhance_audio(process_path)
            if enhanced_path != process_path:
                _temp_files.append(enhanced_path)
            process_path = enhanced_path
            logger.info("Job %s enhanced audio path: %s", job_id, process_path)
        _check_cancel(cancel_event)
        audio_duration_s = audio_duration_seconds(process_path)
        if progress is not None:
            progress.set_audio_duration(audio_duration_s)
        logger.info("Job %s audio duration: %.2fs", job_id, audio_duration_s)
        _phase("prepare", "Preparing models\u2026", 28)

        selected = _selected_engines(selected_engines)
        speaker_limit = _speaker_limit(diarization, max_speakers)
        logger.info("Job %s selected engines after policy: %s", job_id, selected)
        _prepare_selected_asr_model(selected)
        _phase("asr_prepare", "Loading ASR model on GPU\u2026", 40)
        diar_segments = None
        if diarization:
            _phase("diarize", "Running speaker diarization\u2026", 32)
            diar_segments = _run_diarization(
                process_path, diarization, speaker_limit, diarize_kwargs=diarize_kwargs
            )
        _check_cancel(cancel_event)

        def _window_progress(current: int, total: int) -> None:
            if progress is not None:
                progress.set_asr_window(current, total)

        _phase("asr", "Transcribing on GPU\u2026", 45)
        results, workers = _run_asr_engines(
            job_id,
            selected,
            process_path,
            language,
            diar_segments,
            cancel_event,
            window_progress=_window_progress,
        )
        _phase("finalize", "Saving transcript\u2026", 96)
        _clear_asr_models(selected)
    except RuntimeError as exc:
        if "cancelled" in str(exc).lower():
            logger.info("Job %s cancelled; cleaning up GPU and temp files.", job_id)
            _cleanup_cancelled_job(_temp_files)
            raise
        raise

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
        "max_speakers": speaker_limit,
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
    if progress is not None:
        progress.finish("Transcription complete.")
    return {
        "job_id": job_id,
        "manifest_path": manifest_path,
        "audio_duration_s": audio_duration_s,
        "total_elapsed_s": total_elapsed_s,
        "target_elapsed_s": target_elapsed_s,
        "target_met": target_met,
        "results": results,
    }
