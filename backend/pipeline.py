"""Core backend transcription pipeline."""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from backend.services.asr_local import (
    ALL_ENGINES,
    asr_worker_count,
    clear_accelerator_cache,
    default_asr_engines,
    load_model,
    model_is_loaded,
    models_resident_on_gpu,
    should_clear_model_between_engines,
    should_clear_models_after_job,
    should_unload_asr_for_diarization,
    should_unload_on_cancel,
    should_warm_start_gpu_job,
    transcribe_engine,
    unload_model,
)
from backend.services.media_pipeline import (
    audio_duration_seconds,
    clear_diarization_model,
    diarize_audio,
    enhance_audio,
    enhance_audio_for_asr,
    normalize_media,
    stage_audio_for_inference,
)
from backend.storage import (
    copy_input_file,
    new_job_id,
    now_iso,
    save_transcript,
    write_job_record,
)

try:
    from backend.progress import JobProgress
except ImportError:  # pragma: no cover
    JobProgress = None  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)

_active_jobs_lock = threading.Lock()
_active_job_count = 0


@dataclass
class JobMeta:
    tab_id: str = ""
    display_name: str = ""
    source_filename: str = ""
    output_name: str | None = None
    client_ip: str = ""
    user_id: int = 0
    username: str = ""


@dataclass
class TranscriptionStageContext:
    job_id: str
    media_path: str
    selected_engines: list[str]
    language: str
    diarization: bool
    max_speakers: int
    enhance: bool
    diarize_kwargs: dict | None
    cancel_event: threading.Event | None
    progress: JobProgress | None
    phase: Callable[[str, str, float], None]
    meta: JobMeta
    manifest_sync: Callable[[dict], None] | None = None


def _max_concurrent_jobs() -> int:
    try:
        return max(1, int(os.getenv("UI_MAX_CONCURRENT_JOBS", "1")))
    except ValueError:
        return 1


# Fixed at import — restart the process after changing UI_MAX_CONCURRENT_JOBS.
_job_semaphore = threading.Semaphore(_max_concurrent_jobs())


def register_job_started() -> None:
    global _active_job_count
    with _active_jobs_lock:
        _active_job_count += 1


def register_job_finished() -> None:
    global _active_job_count
    with _active_jobs_lock:
        _active_job_count = max(0, _active_job_count - 1)


def active_job_count() -> int:
    with _active_jobs_lock:
        return _active_job_count


def _phase_teardown(label: str, *, aggressive: bool = False) -> None:
    """Clear transient tensors between pipeline phases without unloading resident models."""
    from backend import vram_state

    vram_state.log_phase(label, before=False)
    if models_resident_on_gpu() and not aggressive:
        vram_state.teardown(aggressive=False)
    else:
        clear_accelerator_cache()
        if aggressive:
            gc.collect()


def _check_cancel(cancel_event: threading.Event | None) -> None:
    """Raise RuntimeError if a cancellation has been requested."""
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user.")


def unload_all_pipeline_models() -> None:
    """Unload all ASR engines and diarization; clear CUDA cache."""
    from backend.model_registry import unload_all_models

    unload_all_models()


def _cleanup_cancelled_job(temp_files: list[str]) -> None:
    """Delete temp files for a cancelled job; keep ASR models loaded unless configured."""
    if should_unload_on_cancel():
        unload_all_pipeline_models()
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
    _phase_teardown("cancel_cleanup", aggressive=should_unload_on_cancel())


def _selected_engines(
    selected_engines: list[str] | str,
    language: str = "Thai",
) -> list[str]:
    from backend.services.asr_local import resolve_asr_engines

    if isinstance(selected_engines, str):
        candidates = [selected_engines]
    else:
        candidates = list(selected_engines or [])
    if not candidates:
        candidates = default_asr_engines()
    return resolve_asr_engines(language, candidates)


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
            "Diarization starting: max_speakers=%d duration=%.1fs overrides=%s",
            max_speakers,
            audio_duration_seconds(process_path),
            diarize_kwargs or {},
        )
        segments = diarize_audio(
            process_path,
            max_speakers,
            audio_duration_s=audio_duration_seconds(process_path),
            diarize_kwargs=diarize_kwargs,
        )
        logger.info("Diarization finished: segments=%d", len(segments))
        return segments
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Diarization failed: %s", exc)
        return None
    finally:
        _phase_teardown("diarize", aggressive=True)
        from engines.diarization import release_after_job

        release_after_job()
        clear_accelerator_cache()
        gc.collect()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_enhance(enhance: bool, diarization: bool) -> bool:
    if enhance:
        return True
    return diarization and _env_bool("AUDIO_ENHANCE_WHEN_DIARIZATION", False)


def _enhance_asr_only() -> bool:
    return _env_bool("AUDIO_ENHANCE_ASR_ONLY", False)


def _success_result(
    job_id: str,
    engine: str,
    text: str,
    elapsed: float,
    output_name: str | None = None,
) -> dict:
    transcript_path = save_transcript(job_id, engine, text, output_name=output_name)
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
    from engines.whisper_runtime import is_cuda_oom

    return is_cuda_oom(exc)


def _run_one_asr_engine(
    job_id: str,
    engine: str,
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
    window_progress=None,
    max_speakers: int = 0,
    output_name: str | None = None,
) -> dict:
    try:
        text, elapsed = transcribe_engine(
            engine,
            process_path,
            language,
            diar_segments,
            cancel_event=cancel_event,
            window_progress=window_progress,
            max_speakers=max_speakers,
        )
        return _success_result(job_id, engine, text, elapsed, output_name=output_name)
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


def _asr_keep_preloaded() -> bool:
    return os.getenv("ASR_KEEP_PRELOADED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prepare_selected_asr_model(selected: list[str], *, after_diarization: bool = False) -> None:
    if not selected:
        return
    if after_diarization:
        try:
            from backend import vram_state

            fast = os.getenv("ASR_FAST_MODE", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
            vram_state.teardown(aggressive=not fast)
        except ImportError:
            clear_accelerator_cache()
    selected_engine = selected[0]
    cuda_ok = True
    try:
        from backend.vram_state import ensure_cuda_healthy_or_restart

        # Fail-fast + self-heal: a dead CUDA context (host GPU reset / WSL2
        # contention) cannot be rebuilt in-process, so this restarts the
        # service to get a fresh context instead of limping into a doomed load.
        cuda_ok = ensure_cuda_healthy_or_restart(f"{selected_engine} ASR load")
    except ImportError:
        pass
    if (
        cuda_ok
        and model_is_loaded(selected_engine)
        and all(
            engine == selected_engine or not model_is_loaded(engine)
            for engine in ALL_ENGINES
        )
    ):
        logger.info("ASR model already resident: %s (skip reload)", selected_engine)
        return
    from backend.services.asr_local import switch_asr_engine

    logger.info("Loading ASR model for job: %s", selected_engine)
    switch_asr_engine(selected_engine)
    if not models_resident_on_gpu():
        clear_accelerator_cache()


def _run_asr_sequential(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    cancel_event: threading.Event | None = None,
    window_progress=None,
    max_speakers: int = 0,
    output_name: str | None = None,
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
            max_speakers=max_speakers,
            output_name=output_name,
        )
        if clear_between:
            _unload_asr_engine(engine)
        else:
            _phase_teardown(f"asr_{engine}", aggressive=False)
    return results


def _run_asr_parallel(
    job_id: str,
    selected: list[str],
    process_path: str,
    language: str,
    diar_segments: list[dict] | None,
    workers: int,
    cancel_event: threading.Event | None = None,
    window_progress=None,
    max_speakers: int = 0,
    output_name: str | None = None,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one_asr_engine,
                job_id,
                engine,
                process_path,
                language,
                diar_segments,
                cancel_event,
                window_progress,
                max_speakers,
                output_name,
            ): engine
            for engine in selected
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                results[engine] = future.result()
            except RuntimeError as exc:
                if "cancelled" in str(exc).lower():
                    raise
                results[engine] = _error_result(engine, exc)
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
    max_speakers: int = 0,
    output_name: str | None = None,
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
            max_speakers=max_speakers,
            output_name=output_name,
        )
        return results, workers

    results = _run_asr_parallel(
        job_id,
        selected,
        process_path,
        language,
        diar_segments,
        workers,
        cancel_event,
        window_progress=window_progress,
        max_speakers=max_speakers,
        output_name=output_name,
    )
    return results, workers


def _performance_target_seconds(audio_duration_s: float) -> float:
    from backend.asr_performance import performance_target_seconds

    return performance_target_seconds(audio_duration_s)


def _clear_asr_models(selected: list[str]) -> None:
    if not should_clear_models_after_job():
        return
    for engine in selected:
        try:
            unload_model(engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("%s unload skipped: %s", engine, exc)
    clear_accelerator_cache()


def _apply_enhancement(
    process_path: str,
    enhance: bool,
    progress: JobProgress | None,
    temp_files: list[str],
    *,
    asr_only: bool = False,
) -> str:
    if not enhance:
        return process_path
    if progress is not None:
        progress.set_phase("enhance", "Enhancing audio\u2026", 18)
    enhance_fn = enhance_audio_for_asr if asr_only else enhance_audio
    enhanced_path = enhance_fn(process_path)
    if enhanced_path != process_path:
        temp_files.append(enhanced_path)
    return enhanced_path


def _unload_asr_for_diarization() -> None:
    if not should_unload_asr_for_diarization():
        logger.info("ASR stays loaded on GPU (diarization runs on CPU).")
        return
    logger.info("Staging ASR off GPU so speaker diarization can use CUDA.")
    for engine in ALL_ENGINES:
        try:
            unload_model(engine)
        except ValueError:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError, OSError):
        pass
    clear_accelerator_cache()


def _maybe_diarize(
    process_path: str,
    diarization: bool,
    speaker_limit: int,
    diarize_kwargs: dict | None,
    progress: JobProgress | None,
) -> list[dict] | None:
    if not diarization:
        return None
    _unload_asr_for_diarization()
    try:
        from backend import vram_state

        vram_state.ensure_cuda_healthy_or_restart("diarization")
        vram_state.teardown(aggressive=False)
    except ImportError:
        pass
    if progress is not None:
        progress.set_phase("diarize", "Running speaker diarization\u2026", 32)
    segments = _run_diarization(process_path, diarization, speaker_limit, diarize_kwargs=diarize_kwargs)
    if segments is not None:
        logger.info("Unloading diarization model before ASR to free GPU VRAM.")
        try:
            from backend.model_registry import unload_diarization_model
            from backend import vram_state

            unload_diarization_model()
            vram_state.teardown(aggressive=True)
            vram_state.apply_cuda_memory_fraction()
        except ImportError:
            clear_accelerator_cache()
    return segments


def _prepare_job_enhance_flags(
    ctx: TranscriptionStageContext,
    speaker_limit: int,
) -> tuple[bool, bool]:
    """Resolve enhance / adaptive-enhance flags and apply duration profiles."""
    enhance = _resolve_enhance(ctx.enhance, ctx.diarization)
    adaptive_enhance = _env_bool("AUDIO_ENHANCE_ADAPTIVE", False) and ctx.diarization
    from backend.asr_quality import (
        apply_enhance_profile,
        apply_two_speaker_overrides,
        enhance_asr_only_enabled,
    )

    if not (enhance or adaptive_enhance):
        return enhance, adaptive_enhance
    probe_duration_s = audio_duration_seconds(ctx.media_path)
    apply_enhance_profile(probe_duration_s, speaker_limit)
    apply_two_speaker_overrides(speaker_limit)
    if adaptive_enhance and not enhance:
        enhance = _resolve_enhance(False, ctx.diarization) or enhance_asr_only_enabled()
    return enhance, adaptive_enhance


def _normalize_and_stage_audio(
    ctx: TranscriptionStageContext,
    *,
    enhance: bool,
    adaptive_enhance: bool,
    temp_files: list[str],
) -> tuple[str, float]:
    """Normalize, optionally enhance, and stage audio for inference."""
    from backend.asr_quality import enhance_asr_only_enabled

    ctx.phase("normalize", "Normalizing media\u2026", 8)
    process_path = normalize_media(ctx.media_path, ctx.job_id)
    if process_path != ctx.media_path:
        temp_files.append(process_path)
    logger.info("Job %s normalized media path: %s", ctx.job_id, process_path)
    _check_cancel(ctx.cancel_event)

    audio_duration_s = audio_duration_seconds(process_path)
    if enhance or adaptive_enhance:
        from backend.asr_quality import apply_short_audio_asr_overrides

        apply_short_audio_asr_overrides(audio_duration_s)

    enhance_before_diar = enhance and not enhance_asr_only_enabled()
    process_path = _apply_enhancement(
        process_path, enhance_before_diar, ctx.progress, temp_files,
    )
    if enhance_before_diar:
        logger.info("Job %s enhanced audio path: %s", ctx.job_id, process_path)
    _check_cancel(ctx.cancel_event)
    staged = stage_audio_for_inference(process_path, ctx.job_id)
    if staged != process_path:
        temp_files.append(staged)
        process_path = staged
    if ctx.progress is not None:
        ctx.progress.set_audio_duration(audio_duration_s)
    logger.info("Job %s audio duration: %.2fs", ctx.job_id, audio_duration_s)
    return process_path, audio_duration_s


def _run_transcription_asr_phase(
    ctx: TranscriptionStageContext,
    process_path: str,
    selected: list[str],
    speaker_limit: int,
    diar_segments: list[dict] | None,
    temp_files: list[str],
) -> tuple[dict[str, dict], int]:
    """Prepare ASR models and transcribe all selected engines."""
    from backend.asr_quality import enhance_asr_only_enabled

    # ASR-only enhance must still honor the UI checkbox (ctx.enhance).
    if enhance_asr_only_enabled() and ctx.diarization and ctx.enhance:
        asr_path = _apply_enhancement(
            process_path, True, ctx.progress, temp_files, asr_only=True,
        )
        if asr_path != process_path:
            process_path = asr_path
            logger.info("Job %s ASR-only enhanced path: %s", ctx.job_id, process_path)

    if ctx.progress is not None:
        ctx.progress.set_phase("prepare", "Preparing ASR model\u2026", 40)
    _prepare_selected_asr_model(selected, after_diarization=diar_segments is not None)
    if ctx.progress is not None:
        ctx.progress.set_phase("asr_prepare", "Loading ASR model on GPU\u2026", 45)

    def _window_progress(current: int, total: int) -> None:
        if ctx.progress is not None:
            ctx.progress.set_asr_window(current, total)

    ctx.phase("asr", "Transcribing on GPU\u2026", 45)
    results, workers = _run_asr_engines(
        ctx.job_id,
        selected,
        process_path,
        ctx.language,
        diar_segments,
        ctx.cancel_event,
        window_progress=_window_progress,
        max_speakers=speaker_limit,
        output_name=ctx.meta.output_name,
    )
    return results, workers, process_path


def _execute_transcription_stages(
    ctx: TranscriptionStageContext,
) -> tuple[str, list[str], int, dict, int, float, list[str], list[dict] | None]:
    """Normalize, diarize, transcribe; return job artifacts or raise on cancel."""
    if ctx.progress is not None:
        ctx.progress.set_job_id(ctx.job_id)
    logger.info(
        "Job %s started: source=%s engines=%s language=%s diarization=%s",
        ctx.job_id,
        ctx.media_path,
        ctx.selected_engines,
        ctx.language,
        ctx.diarization,
    )
    try:
        from backend import vram_state

        # Warm start keeps preloaded ASR/diar weights; staging still unloads ASR
        # before CUDA diarization inside the pipeline when needed.
        vram_state.prepare_exclusive_gpu_job(
            unload_models=not should_warm_start_gpu_job(),
        )
    except ImportError:
        pass
    if ctx.manifest_sync:
        ctx.manifest_sync({
            "status": "running",
            "tab_id": ctx.meta.tab_id,
            "display_name": ctx.meta.display_name,
            "source_filename": ctx.meta.source_filename,
            "source_path": ctx.media_path,
            "client_ip": ctx.meta.client_ip,
            "user_id": ctx.meta.user_id,
            "username": ctx.meta.username,
        })
    temp_files: list[str] = []
    if ctx.meta.source_filename:
        archived = copy_input_file(ctx.media_path, ctx.job_id, ctx.meta.source_filename)
        if archived:
            logger.info("Job %s archived upload: %s", ctx.job_id, archived)

    speaker_limit = _speaker_limit(ctx.diarization, ctx.max_speakers)
    enhance, adaptive_enhance = _prepare_job_enhance_flags(ctx, speaker_limit)
    # Keep resolved enhance on the context so later phases (incl. ASR-only)
    # follow the same checkbox / WHEN_DIARIZATION decision.
    ctx.enhance = enhance
    process_path, audio_duration_s = _normalize_and_stage_audio(
        ctx,
        enhance=enhance,
        adaptive_enhance=adaptive_enhance,
        temp_files=temp_files,
    )
    from backend.asr_performance import apply_performance_policy

    apply_performance_policy(audio_duration_s, diarization=ctx.diarization)
    selected = _selected_engines(ctx.selected_engines, ctx.language)
    logger.info("Job %s selected engines after policy: %s", ctx.job_id, selected)

    diar_segments = _maybe_diarize(
        process_path, ctx.diarization, speaker_limit, ctx.diarize_kwargs, ctx.progress,
    )
    _check_cancel(ctx.cancel_event)

    results, workers, process_path = _run_transcription_asr_phase(
        ctx, process_path, selected, speaker_limit, diar_segments, temp_files,
    )
    ctx.phase("finalize", "Saving transcript\u2026", 96)
    _clear_asr_models(selected)
    return (
        process_path, selected, speaker_limit, results, workers,
        audio_duration_s, temp_files, diar_segments,
    )


def _acquire_gpu_job_slot(
    cancel_event: threading.Event | None,
    progress: JobProgress | None,
) -> None:
    """Wait for a GPU pipeline slot; update queue status and honor cancel."""
    while True:
        _check_cancel(cancel_event)
        if _job_semaphore.acquire(blocking=False):
            return
        if progress is not None:
            ahead = max(0, active_job_count() - _max_concurrent_jobs())
            slots = _max_concurrent_jobs()
            progress.set_phase(
                "queued",
                (
                    f"Queued — {ahead} job(s) ahead "
                    f"({slots} GPU slot{'s' if slots != 1 else ''})…"
                ),
                1.0,
            )
        time.sleep(0.4)


def run_transcription_job(
    media_path: str,
    selected_engines: list[str],
    language: str,
    diarization: bool,
    max_speakers: int,
    enhance: bool,
    *,
    diarize_kwargs: dict | None = None,
    cancel_event: threading.Event | None = None,
    progress: JobProgress | None = None,
    meta: JobMeta | None = None,
    job_id: str | None = None,
) -> dict:
    """Run the full local transcript pipeline and persist outputs."""
    register_job_started()
    try:
        _acquire_gpu_job_slot(cancel_event, progress)
        try:
            return _run_transcription_job_impl(
                media_path,
                selected_engines,
                language,
                diarization,
                max_speakers,
                enhance,
                diarize_kwargs,
                cancel_event,
                progress,
                meta or JobMeta(),
                job_id=job_id,
            )
        finally:
            _job_semaphore.release()
    finally:
        register_job_finished()


def _run_transcription_job_impl(
    media_path: str,
    selected_engines: list[str],
    language: str,
    diarization: bool,
    max_speakers: int,
    enhance: bool,
    diarize_kwargs: dict | None,
    cancel_event: threading.Event | None,
    progress: JobProgress | None,
    meta: JobMeta,
    job_id: str | None = None,
) -> dict:
    """Inner job runner — limited by ``_job_semaphore`` (default 2 concurrent tabs)."""
    job_id = (job_id or "").strip() or new_job_id()
    last_manifest_sync = 0.0

    def _manifest_sync(patch: dict, *, force: bool = False) -> None:
        nonlocal last_manifest_sync
        now = time.time()
        if not force and now - last_manifest_sync < 2.0:
            return
        last_manifest_sync = now
        write_job_record(job_id, patch)

    def _phase(phase: str, message: str, percent: float) -> None:
        if progress is not None:
            progress.set_phase(phase, message, percent)
            snap = progress.snapshot()
            _manifest_sync({
                "progress": {
                    "phase": snap["phase"],
                    "message": snap["message"],
                    "percent": snap["percent"],
                    "elapsed_s": snap["elapsed_s"],
                },
            })

    job_started = time.perf_counter()
    _phase("starting", "Starting transcription job\u2026", 2)
    temp_files: list[str] = []
    try:
        (
            process_path,
            selected,
            speaker_limit,
            results,
            workers,
            audio_duration_s,
            temp_files,
            diar_segments,
        ) = _execute_transcription_stages(
            TranscriptionStageContext(
                job_id=job_id,
                media_path=media_path,
                selected_engines=selected_engines,
                language=language,
                diarization=diarization,
                max_speakers=max_speakers,
                enhance=enhance,
                diarize_kwargs=diarize_kwargs,
                cancel_event=cancel_event,
                progress=progress,
                phase=_phase,
                meta=meta,
                manifest_sync=_manifest_sync,
            ),
        )
    except RuntimeError as exc:
        if "cancelled" in str(exc).lower():
            logger.info("Job %s cancelled; cleaning up GPU and temp files.", job_id)
            _manifest_sync({"status": "cancelled", "error": str(exc)}, force=True)
            _cleanup_cancelled_job(temp_files)
            raise
        _manifest_sync({"status": "failed", "error": str(exc)}, force=True)
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
        "status": "completed",
        "tab_id": meta.tab_id,
        "display_name": meta.display_name,
        "source_filename": meta.source_filename,
        "source_path": media_path,
        "processed_path": process_path,
        "selected_engines": selected,
        "language": language,
        "diarization": diarization,
        "max_speakers": speaker_limit,
        "enhance": _resolve_enhance(enhance, diarization),
        "asr_workers": workers,
        "audio_duration_s": audio_duration_s,
        "total_elapsed_s": total_elapsed_s,
        "target_elapsed_s": target_elapsed_s,
        "target_met": target_met,
        "client_ip": meta.client_ip,
        "user_id": meta.user_id,
        "username": meta.username,
        "results": results,
    }
    manifest_path = write_job_record(job_id, manifest)
    if progress is not None:
        progress.finish("Transcription complete.")
    _phase_teardown("job_complete", aggressive=False)
    return {
        "job_id": job_id,
        "manifest_path": manifest_path,
        "audio_duration_s": audio_duration_s,
        "total_elapsed_s": total_elapsed_s,
        "target_elapsed_s": target_elapsed_s,
        "target_met": target_met,
        "results": results,
        # Raw speaker turns (post merge/cap) — lets callers score speaker
        # count and boundary fidelity at the diarization layer, where they
        # are defined, instead of on merged transcript lines.
        "diarization_segments": diar_segments,
    }
