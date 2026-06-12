"""Local ASR service facade."""

# pylint: disable=duplicate-code,import-outside-toplevel

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager


logger = logging.getLogger(__name__)


ENGINE_TYPHOON = "Typhoon Whisper"
ENGINE_PATHUMMA = "Pathumma Whisper"
_LEGACY_ENGINE_NAMES: dict[str, str] = {}
ALL_ENGINES = [ENGINE_PATHUMMA, ENGINE_TYPHOON]
FAST_8GB_ENGINES = [ENGINE_PATHUMMA]

LANGUAGES = {
    "Thai": "thai",
    "English": "english",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Korean": "korean",
}


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
        logger.warning("Invalid %s=%r; using %d.", name, value, default)
        return default


def _cuda_vram_mb() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except (ImportError, RuntimeError, OSError, AttributeError):
        return 0
    return 0


def _is_8gb_class_cuda() -> bool:
    """Return True for CUDA GPUs that need one-model-at-a-time ASR."""
    vram_mb = _cuda_vram_mb()
    if not vram_mb:
        return False
    class_limit_mb = _env_int("ASR_8GB_CLASS_MAX_MB", 9000)
    return vram_mb <= class_limit_mb


def strict_memory_mode_active() -> bool:
    """Return whether hard 8 GB VRAM protection is active."""
    return _env_bool("ASR_HARD_MEMORY_SAFE", True) and _is_8gb_class_cuda()


def requires_sequential_gpu_models() -> bool:
    """8 GB GPUs cannot keep ASR + diarization resident on CUDA together."""
    return strict_memory_mode_active()


def should_clear_model_between_engines() -> bool:
    """Return whether a model should be unloaded immediately after each engine.

    Only applies in sequential mode. Parallel mode keeps both models in VRAM
    for the duration of both transcriptions.
    """
    return _env_bool("ASR_CLEAR_VRAM_BETWEEN_ENGINES", False)


def default_asr_engines() -> list[str]:
    """Return one default ASR engine for startup preload and UI selection."""
    configured = os.getenv("ASR_DEFAULT_ENGINES", "").strip()
    if configured:
        names = [part.strip() for part in configured.split(",") if part.strip()]
        selected = [
            _LEGACY_ENGINE_NAMES.get(engine, engine)
            for engine in names
            if _LEGACY_ENGINE_NAMES.get(engine, engine) in ALL_ENGINES
        ]
        if selected:
            return selected[:1]
        logger.warning(
            "ASR_DEFAULT_ENGINES=%r has no known engines; using policy default.", configured
        )
    if strict_memory_mode_active():
        return FAST_8GB_ENGINES.copy()
    return [ALL_ENGINES[0]]


def clear_accelerator_cache() -> None:
    """Release unused Python and CUDA memory back to the driver."""
    from backend import vram_state

    vram_state.teardown(aggressive=True)


def max_concurrent_asr_inference() -> int:
    """How many CUDA ASR inferences may run at once (shared Whisper pipeline)."""
    return max(1, _env_int("ASR_MAX_CONCURRENT_INFERENCE", 1))


_inference_semaphore = threading.Semaphore(max_concurrent_asr_inference())


@contextmanager
def asr_inference_slot():
    """Serialize CUDA ASR when multiple job threads share one pipeline."""
    _inference_semaphore.acquire()
    try:
        yield
    finally:
        _inference_semaphore.release()


def should_unload_on_cancel() -> bool:
    """Unload models on cancel (off by default — models stay ready for the next job)."""
    return _env_bool("ASR_UNLOAD_ON_CANCEL", False)


def switch_asr_engine(selected_engine: str) -> None:
    """Unload other engines and load the selected one (UI engine change only)."""
    selected_engine = _LEGACY_ENGINE_NAMES.get(selected_engine, selected_engine)
    if selected_engine not in ALL_ENGINES:
        raise ValueError(f"Unknown ASR engine: {selected_engine}")
    for engine in ALL_ENGINES:
        if engine != selected_engine:
            try:
                unload_model(engine)
            except ValueError:
                pass
    if not model_is_loaded(selected_engine):
        load_model(selected_engine)
    logger.info("ASR engine active: %s (others unloaded).", selected_engine)


def asr_worker_count(selected_count: int) -> int:
    """Return a safe worker count for selected ASR engines.

    ASR_PARALLEL_MODE=parallel forces both engines at once.
    ASR_PARALLEL_MODE=memory_safe forces sequential execution.
    ASR_PARALLEL_MODE=auto runs in parallel only when VRAM is comfortably above
    ASR_PARALLEL_MIN_VRAM_MB, avoiding Windows shared-memory spill on 8 GB cards.
    """
    worker_count = 1
    if selected_count > 1:
        mode = os.getenv("ASR_PARALLEL_MODE", "auto").strip().lower()
        if strict_memory_mode_active() and not _env_bool("ASR_ALLOW_8GB_PARALLEL", False):
            logger.info(
                "ASR parallelism limited to 1 worker by strict low-VRAM mode. "
                "Set ASR_ALLOW_8GB_PARALLEL=true only on machines with enough free VRAM."
            )
        elif mode in {"parallel", "force", "true", "1"}:
            worker_count = selected_count
        elif mode not in {"memory_safe", "safe", "sequential", "false", "0"}:
            min_parallel_mb = _env_int("ASR_PARALLEL_MIN_VRAM_MB", 12 * 1024)
            vram_mb = _cuda_vram_mb()
            if not vram_mb:
                logger.info("ASR parallelism limited to 1 worker (CUDA VRAM not detected).")
            elif vram_mb < min_parallel_mb:
                logger.info(
                    "ASR parallelism limited to 1 worker (%d MB VRAM < %d MB threshold).",
                    vram_mb,
                    min_parallel_mb,
                )
            else:
                worker_count = selected_count
    return worker_count


def should_clear_models_after_job() -> bool:
    """Return whether ASR models should be unloaded after each job."""
    if models_resident_on_gpu():
        return False
    return _env_bool("ASR_CLEAR_VRAM_AFTER_JOB", False)


def diarization_inference_uses_cuda() -> bool:
    """True when pyannote diarization will run on CUDA (not CPU)."""
    try:
        import torch

        from engines.diarization import _select_diarization_device

        return str(_select_diarization_device(torch)).startswith("cuda")
    except (ImportError, RuntimeError, OSError, AttributeError):
        return False


def models_resident_on_gpu() -> bool:
    """Keep ASR on GPU between jobs when preloaded (independent of diarization device)."""
    if not _env_bool("ASR_KEEP_PRELOADED", False):
        return False
    if _env_bool("ASR_CLEAR_VRAM_AFTER_JOB", False):
        return False
    return True


def should_unload_asr_for_diarization() -> bool:
    """Never unload ASR for diarization when GPU co-resident policy is enabled."""
    if _env_bool("DIARIZATION_GPU_CO_RESIDENT", True):
        return False
    if not diarization_inference_uses_cuda():
        return False
    if models_resident_on_gpu():
        return _env_bool("ASR_UNLOAD_FOR_DIARIZATION", False)
    return _env_bool("ASR_UNLOAD_FOR_DIARIZATION", True)


def model_is_loaded(engine_name: str) -> bool:
    """Return True when an ASR engine pipeline is already cached in memory."""
    engine_name = _LEGACY_ENGINE_NAMES.get(engine_name, engine_name)
    if engine_name == ENGINE_TYPHOON:
        from engines import typhoon_asr

        return bool(typhoon_asr._pipeline_cache)  # noqa: SLF001
    if engine_name == ENGINE_PATHUMMA:
        from engines import pathumma_asr

        return bool(pathumma_asr._pipeline_cache)  # noqa: SLF001
    return False


def load_model(engine_name: str) -> None:
    """Load one ASR engine by display name."""
    engine_name = _LEGACY_ENGINE_NAMES.get(engine_name, engine_name)
    if engine_name == ENGINE_TYPHOON:
        from engines.typhoon_asr import load_model as load_typhoon

        load_typhoon()
        return
    if engine_name == ENGINE_PATHUMMA:
        from engines.pathumma_asr import load_model as load_pathumma

        load_pathumma()
        return
    raise ValueError(f"Unknown ASR engine: {engine_name}")


def unload_model(engine_name: str) -> None:
    """Unload one ASR engine by display name and clear accelerator cache."""
    engine_name = _LEGACY_ENGINE_NAMES.get(engine_name, engine_name)
    if engine_name == ENGINE_TYPHOON:
        from engines.typhoon_asr import unload_model as unload_typhoon

        unload_typhoon()
    elif engine_name == ENGINE_PATHUMMA:
        from engines.pathumma_asr import unload_model as unload_pathumma

        unload_pathumma()
    else:
        raise ValueError(f"Unknown ASR engine: {engine_name}")
    clear_accelerator_cache()


def transcribe_engine(
    engine_name: str,
    audio_path: str,
    language: str,
    diarization_segments: list[dict] | None = None,
    cancel_event=None,
    window_progress=None,
    max_speakers: int = 0,
) -> tuple[str, float]:
    """Run one ASR engine and return transcript text plus elapsed seconds."""
    engine_name = _LEGACY_ENGINE_NAMES.get(engine_name, engine_name)
    whisper_language = LANGUAGES.get(language, LANGUAGES["Thai"])
    started = time.perf_counter()
    logger.info(
        "ASR engine starting: engine=%s language=%s diarization_segments=%d",
        engine_name,
        whisper_language,
        len(diarization_segments or []),
    )
    with asr_inference_slot():
        if engine_name == ENGINE_TYPHOON:
            from engines.typhoon_asr import transcribe_typhoon

            text = transcribe_typhoon(
                audio_path, whisper_language, diarization_segments,
                cancel_event=cancel_event, window_progress=window_progress,
                max_speakers=max_speakers,
            )
        elif engine_name == ENGINE_PATHUMMA:
            from engines.pathumma_asr import transcribe_pathumma

            text = transcribe_pathumma(
                audio_path, whisper_language, diarization_segments,
                cancel_event=cancel_event, window_progress=window_progress,
                max_speakers=max_speakers,
            )
        else:
            raise ValueError(f"Unknown ASR engine: {engine_name}")
    elapsed = time.perf_counter() - started
    logger.info(
        "ASR engine finished: engine=%s elapsed=%.2fs transcript_chars=%d",
        engine_name,
        elapsed,
        len(text),
    )
    return text, elapsed
