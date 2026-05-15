"""Local ASR service facade."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import gc
import logging
import os
import time


logger = logging.getLogger(__name__)


ENGINE_TYPHOON = "Typhoon Whisper"
ENGINE_PATHUMMA = "Pathumma Whisper"
_LEGACY_ENGINE_NAMES: dict[str, str] = {}
ALL_ENGINES = [ENGINE_TYPHOON, ENGINE_PATHUMMA]
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


def should_clear_model_between_engines() -> bool:
    """Return whether a model should be unloaded immediately after each engine.

    Only applies in sequential mode. Parallel mode keeps both models in VRAM
    for the duration of both transcriptions.
    """
    return _env_bool("ASR_CLEAR_VRAM_BETWEEN_ENGINES", False)


def default_asr_engines() -> list[str]:
    """Return UI/default engine selection for the current memory policy."""
    configured = os.getenv("ASR_DEFAULT_ENGINES", "").strip()
    if configured:
        names = [part.strip() for part in configured.split(",") if part.strip()]
        selected = [
            _LEGACY_ENGINE_NAMES.get(engine, engine)
            for engine in names
            if _LEGACY_ENGINE_NAMES.get(engine, engine) in ALL_ENGINES
        ]
        if selected:
            return selected
        logger.warning(
            "ASR_DEFAULT_ENGINES=%r has no known engines; using policy default.", configured
        )
    if strict_memory_mode_active():
        return FAST_8GB_ENGINES.copy()
    return ALL_ENGINES.copy()


def clear_accelerator_cache() -> None:
    """Release unused Python and CUDA memory back to the driver."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("CUDA cache cleanup skipped: %s", exc)


def asr_worker_count(selected_count: int) -> int:
    """Return a safe worker count for selected ASR engines.

    ASR_PARALLEL_MODE=parallel forces both engines at once.
    ASR_PARALLEL_MODE=memory_safe forces sequential execution.
    ASR_PARALLEL_MODE=auto runs in parallel only when VRAM is comfortably above
    ASR_PARALLEL_MIN_VRAM_MB, avoiding Windows shared-memory spill on 8 GB cards.
    """
    if selected_count <= 1:
        return 1

    mode = os.getenv("ASR_PARALLEL_MODE", "auto").strip().lower()

    if mode in {"parallel", "force", "true", "1"}:
        return selected_count
    if mode in {"memory_safe", "safe", "sequential", "false", "0"}:
        return 1

    min_parallel_mb = _env_int("ASR_PARALLEL_MIN_VRAM_MB", 12 * 1024)
    vram_mb = _cuda_vram_mb()
    if not vram_mb:
        logger.info("ASR parallelism limited to 1 worker (CUDA VRAM not detected).")
        return 1
    if vram_mb and vram_mb < min_parallel_mb:
        logger.info(
            "ASR parallelism limited to 1 worker (%d MB VRAM < %d MB threshold).",
            vram_mb,
            min_parallel_mb,
        )
        return 1
    return selected_count


def should_clear_models_after_job() -> bool:
    """Return whether ASR models should be unloaded after each job."""
    return _env_bool("ASR_CLEAR_VRAM_AFTER_JOB", False)


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
    if engine_name == ENGINE_TYPHOON:
        from engines.typhoon_asr import transcribe_typhoon

        text = transcribe_typhoon(audio_path, whisper_language, diarization_segments)
    elif engine_name == ENGINE_PATHUMMA:
        from engines.pathumma_asr import transcribe_pathumma

        text = transcribe_pathumma(audio_path, whisper_language, diarization_segments)
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
