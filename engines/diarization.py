"""Shared speaker diarization using pyannote/speaker-diarization-community-1.

Defaults to the September 2025 community-1 pipeline (pyannote.audio 4.x), which
reports 30-50% lower DER than the legacy 3.1 pipeline on most benchmarks and
adds an `exclusive_speaker_diarization` output designed for clean alignment
with transcription timestamps. Override via DIARIZATION_MODEL_ID.
"""

# pylint: disable=import-outside-toplevel

import logging
import os
import re
import shutil
import subprocess
import threading
import wave
import warnings

# Suppress torchcodec DLL probe warnings emitted when torch imports its video
# codec library and none of the bundled FFmpeg shared libs can be loaded.
# These are purely informational — our pipeline uses librosa, not torchcodec.
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec.*|.*libtorchcodec.*|.*FFmpeg.*version.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"[\s\S]*torchcodec[\s\S]*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32.*|.*TF32.*",
)

logger = logging.getLogger(__name__)

_CUDA_INTERNAL_ASSERT = "internal assert"
_CUDA_CACHING_ALLOCATOR = "cachingallocator"

MODEL_ID = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")

_pipeline_cache: list = []
_tracked_device: str | None = None
_last_inference_device: str | None = None
_diarization_lock = threading.Lock()

_NO_SPEECH = "(no speech detected)"


def _fmt_ts(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        logger.warning(
            "ffmpeg not found in PATH — audio preprocessing may fail. "
            "Install ffmpeg: https://ffmpeg.org/download.html"
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %.4f.", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_pipeline_params() -> dict | None:
    """Build pyannote pipeline instantiate() params from environment variables.

    Tuning guide (only applied when value >= 0; otherwise the model's own
    training-tuned defaults are kept — recommended for community-1):
      DIARIZATION_SEGMENTATION_THRESHOLD  — activity threshold; lower = catches
          quieter / shorter speaker turns (pyannote default ~0.5).
      DIARIZATION_MIN_DURATION_OFF        — minimum silence gap (s) to split
          a speaker turn; smaller = more splits (pyannote default 0.0).
      DIARIZATION_CLUSTERING_THRESHOLD    — speaker embedding distance; lower =
          more distinct speakers separated (pyannote default ~0.70; community-1
          VBx AHC-init distance, default 0.6).
      DIARIZATION_MIN_CLUSTER_SIZE        — minimum segments to form a cluster;
          lower = rare/short speakers still detected (pyannote default 12;
          ignored by community-1 VBx).
      DIARIZATION_VBX_FA                  — community-1 VBx acoustic scaling;
          HIGHER keeps more distinct speakers (model default 0.07 merges
          aggressively — brief speakers vanish in meetings).
      DIARIZATION_VBX_FB                  — community-1 VBx speaker-prior
          scaling (model default 0.8).
    """
    seg_threshold   = _env_float("DIARIZATION_SEGMENTATION_THRESHOLD", -1.0)
    seg_min_dur_off = _env_float("DIARIZATION_MIN_DURATION_OFF",        -1.0)
    clust_threshold = _env_float("DIARIZATION_CLUSTERING_THRESHOLD",    -1.0)
    clust_min_size  = _env_int(  "DIARIZATION_MIN_CLUSTER_SIZE",        -1)
    vbx_fa          = _env_float("DIARIZATION_VBX_FA",                  -1.0)
    vbx_fb          = _env_float("DIARIZATION_VBX_FB",                  -1.0)

    params: dict = {}

    seg_params: dict = {}
    if seg_threshold >= 0:
        seg_params["threshold"] = seg_threshold
    if seg_min_dur_off >= 0:
        seg_params["min_duration_off"] = seg_min_dur_off
    if seg_params:
        params["segmentation"] = seg_params

    clust_params: dict = {}
    if clust_threshold >= 0:
        clust_params["threshold"] = clust_threshold
    if clust_min_size >= 0:
        clust_params["min_cluster_size"] = clust_min_size
    if vbx_fa >= 0:
        clust_params["Fa"] = vbx_fa
    if vbx_fb >= 0:
        clust_params["Fb"] = vbx_fb
    if clust_params:
        params["clustering"] = clust_params

    return params if params else None


def _supported_pipeline_params(pipe) -> dict[str, set[str]]:
    supported: dict[str, set[str]] = {}
    for source in ("parameters", "default_parameters"):
        try:
            values = getattr(pipe, source)(instantiated=True) if source == "parameters" else getattr(pipe, source)()
        except (AttributeError, TypeError, RuntimeError, ValueError):
            continue
        for section, section_values in (values or {}).items():
            if isinstance(section_values, dict):
                supported.setdefault(section, set()).update(section_values)
    return supported


def _filter_pipeline_params(pipe, params: dict | None) -> dict | None:
    if not params:
        return None
    supported = _supported_pipeline_params(pipe)
    if not supported:
        return params
    filtered: dict = {}
    skipped: list[str] = []
    for section, section_values in params.items():
        allowed = supported.get(section, set())
        if not isinstance(section_values, dict) or not allowed:
            skipped.append(section)
            continue
        kept = {key: value for key, value in section_values.items() if key in allowed}
        skipped.extend(f"{section}.{key}" for key in section_values if key not in allowed)
        if kept:
            filtered[section] = kept
    if skipped:
        logger.info("Diarization params ignored by this pyannote pipeline: %s", skipped)
    return filtered or None


def _cuda_vram_mb(torch_module) -> int:
    if not torch_module.cuda.is_available():
        return 0
    try:
        total = torch_module.cuda.get_device_properties(0).total_memory
        return int(total // (1024 * 1024))
    except (RuntimeError, OSError, AttributeError):
        return 0


def _cuda_free_mb(torch_module) -> int:
    if not torch_module.cuda.is_available():
        return 0
    try:
        free_bytes, _total = torch_module.cuda.mem_get_info()
        return int(free_bytes // (1024 * 1024))
    except (RuntimeError, OSError, AttributeError):
        return 0


def _requires_cuda_process_restart(msg: str) -> bool:
    """Errors that corrupt the CUDA context; in-process retry makes things worse."""
    lowered = msg.lower()
    return any(
        token in lowered
        for token in (
            "device not ready",
            _CUDA_CACHING_ALLOCATOR,
            _CUDA_INTERNAL_ASSERT,
            "cuda error: unknown",
            "cudaerrorunknown",
        )
    )


def _is_recoverable_cuda_error(msg: str) -> bool:
    lowered = msg.lower()
    return any(
        token in lowered
        for token in (
            "out of memory",
            "device not ready",
            _CUDA_CACHING_ALLOCATOR,
            _CUDA_INTERNAL_ASSERT,
            "cuda error",
        )
    )


def _recover_cuda_after_failure(torch_module) -> None:
    """Best-effort CUDA cleanup after a failed device move."""
    import gc

    gc.collect()
    if not torch_module.cuda.is_available():
        return
    try:
        torch_module.cuda.synchronize()
    except RuntimeError:
        pass
    try:
        torch_module.cuda.empty_cache()
        torch_module.cuda.ipc_collect()
    except (RuntimeError, AttributeError):
        pass


def _device_for_preload(torch_module):
    """Place pyannote on CUDA when allowed; never CPU-cache when CUDA is required."""
    preload = os.getenv("DIARIZATION_PRELOAD_DEVICE", "cuda").strip().lower()
    if preload not in {"cuda", "gpu"} or not torch_module.cuda.is_available():
        if _diarization_cuda_required():
            _raise_if_cuda_required("preload (CUDA unavailable)")
        return torch_module.device("cpu")
    allow_low_vram_cuda = _env_bool("DIARIZATION_ALLOW_8GB_CUDA", False)
    if _diarization_cuda_required() or allow_low_vram_cuda:
        return torch_module.device("cuda")
    vram_mb = _cuda_vram_mb(torch_module)
    low_vram_limit_mb = _env_int("ASR_8GB_CLASS_MAX_MB", 9000)
    if 0 < vram_mb <= low_vram_limit_mb and not allow_low_vram_cuda:
        return torch_module.device("cpu")
    return torch_module.device("cuda")


def _diarization_stays_on_gpu() -> bool:
    preload = os.getenv("DIARIZATION_PRELOAD_DEVICE", "cuda").strip().lower()
    return preload in {"cuda", "gpu"} and _keep_diarization_preloaded()


def _keep_diarization_preloaded() -> bool:
    if _env_bool("DIARIZATION_KEEP_PRELOADED", False):
        return True
    mode = os.getenv("DIARIZATION_PRELOAD_MODE", "eager").strip().lower()
    return mode in {"eager", "preload", "true", "1"}


def last_inference_device() -> str | None:
    """Last device used for diarization inference (survives post-job teardown)."""
    return _last_inference_device


def _diarization_cuda_required() -> bool:
    """When true, refuse silent CPU fallback (golden / explicit CUDA-only runs)."""
    if _env_bool("DIARIZATION_REQUIRE_CUDA", False):
        return True
    return os.getenv("GOLDEN_REQUIRE_GPU", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _raise_if_cuda_required(context: str) -> None:
    if _diarization_cuda_required():
        raise RuntimeError(
            f"CUDA diarization required but unavailable ({context}). "
            "Free GPU VRAM, set DIARIZATION_ALLOW_8GB_CUDA=true, or lower "
            "DIARIZATION_CUDA_MIN_FREE_MB."
        )


def _set_tracked_device(device, *, inference: bool = False) -> None:
    global _tracked_device, _last_inference_device
    target_str = str(device)
    _tracked_device = target_str
    if inference:
        _last_inference_device = target_str


def _record_inference_device() -> None:
    """Sticky snapshot of the device used for the latest diarization pass."""
    if _tracked_device:
        _set_tracked_device(_tracked_device, inference=True)


def _move_pipeline_to_device(pipe, device, reason: str) -> None:
    """Move pyannote pipeline and record device (pipe.parameters() is hyperparams, not tensors)."""
    import torch

    target = torch.device(device) if not isinstance(device, torch.device) else device
    target_str = str(target)
    if _tracked_device == target_str:
        return
    previous = _tracked_device or "unknown"
    try:
        pipe.to(target)
    except RuntimeError as exc:
        if target.type != "cpu":
            if _diarization_cuda_required():
                raise RuntimeError(
                    f"Diarization move to {target_str} failed: {exc}"
                ) from exc
            logger.warning(
                "Diarization move to %s failed (%s); falling back to CPU.",
                target_str,
                exc,
            )
            _recover_cuda_after_failure(torch)
            target = torch.device("cpu")
            target_str = "cpu"
            pipe.to(target)
        else:
            raise
    _set_tracked_device(target_str)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Diarization pipeline moved %s -> %s (%s).", previous, target_str, reason)


def _move_pipeline_to_inference_device(pipe) -> None:
    """Move cached pipeline to the inference device (CUDA when VRAM allows, else CPU)."""
    import torch

    try:
        from backend import vram_state

        vram_state.teardown(aggressive=False)
    except ImportError:
        pass
    device = _select_diarization_device(torch)
    if device.type == "cuda":
        try:
            from backend.vram_state import cuda_device_healthy

            if not cuda_device_healthy():
                if _diarization_cuda_required():
                    _raise_if_cuda_required("CUDA unhealthy after ASR unload")
                logger.warning(
                    "CUDA unhealthy after ASR; diarization inference on CPU."
                )
                device = torch.device("cpu")
        except ImportError:
            pass
    elif _diarization_cuda_required():
        _raise_if_cuda_required("device selection chose CPU")
    _move_pipeline_to_device(pipe, device, "inference")


def _move_pipeline_to_preload_device(pipe) -> None:
    """Return pipeline to the startup preload device (CPU when ASR owns the GPU)."""
    import torch

    _move_pipeline_to_device(pipe, _device_for_preload(torch), "preload")


def release_after_job() -> None:
    """After a job: keep weights when preloaded; move to preload device or unload otherwise."""
    if not _pipeline_cache:
        return
    if _keep_diarization_preloaded():
        if _diarization_stays_on_gpu() and _tracked_device == "cuda":
            return
        if _tracked_device == "cuda":
            _move_pipeline_to_preload_device(_pipeline_cache[0])
            return
        return
    unload_model()


def _asr_model_loaded() -> bool:
    try:
        from backend.services.asr_local import ALL_ENGINES, model_is_loaded

        return any(model_is_loaded(engine) for engine in ALL_ENGINES)
    except ImportError:
        return False


def _diarization_run_min_free_mb() -> int:
    """Minimum free VRAM to keep diarization on CUDA during inference."""
    if os.getenv("DIARIZATION_CUDA_RUN_MIN_FREE_MB", "").strip():
        return _env_int("DIARIZATION_CUDA_RUN_MIN_FREE_MB", 1024)
    low_vram_limit_mb = _env_int("ASR_8GB_CLASS_MAX_MB", 9000)
    vram_mb = 0
    try:
        import torch

        vram_mb = _cuda_vram_mb(torch)
    except ImportError:
        pass
    if 0 < vram_mb <= low_vram_limit_mb:
        return 1024
    return _env_int("DIARIZATION_CUDA_MIN_FREE_MB", 3072)


def _multi_pass_diarization_active() -> bool:
    try:
        from engines.diarization_sampling import multi_sample_sweep_enabled

        return multi_sample_sweep_enabled()
    except ImportError:
        return False


def _cuda_path_flags(
    *,
    free_mb: int,
    min_free_mb: int,
    vram_mb: int,
    min_cuda_vram_mb: int,
    low_vram_cuda: bool,
    gpu_co_resident: bool,
) -> tuple[bool, bool, bool, bool]:
    free_ok = free_mb >= min_free_mb
    asr_loaded = _asr_model_loaded()
    co_resident_ok = gpu_co_resident and asr_loaded and free_ok
    if co_resident_ok and low_vram_cuda and _multi_pass_diarization_active():
        logger.info(
            "Diarization multi-pass on %d MB GPU requires ASR staged off CUDA; "
            "not using co-resident CUDA.",
            vram_mb,
        )
        co_resident_ok = False
    staging_ok = (
        low_vram_cuda
        and not asr_loaded
        and free_ok
        and vram_mb >= min_cuda_vram_mb
    )
    standard_ok = vram_mb >= min_cuda_vram_mb and free_ok
    large_gpu_ok = vram_mb >= 12 * 1024 and free_ok
    return staging_ok, co_resident_ok, standard_ok, large_gpu_ok


def _explicit_cuda_eligible(
    requested: str,
    *,
    vram_mb: int,
    free_mb: int,
    low_vram_cuda: bool,
    allow_low_vram_cuda: bool,
    gpu_co_resident: bool,
    asr_loaded: bool,
    staging_ok: bool,
    co_resident_ok: bool,
    standard_ok: bool,
) -> bool:
    if low_vram_cuda and not allow_low_vram_cuda and not gpu_co_resident and asr_loaded:
        logger.info(
            "Diarization uses CPU on %d MB CUDA GPU so ASR can keep GPU VRAM.",
            vram_mb,
        )
        return False
    if staging_ok or co_resident_ok or standard_ok:
        return True
    if _diarization_cuda_required():
        _raise_if_cuda_required(
            f"DIARIZATION_DEVICE={requested} total={vram_mb}MB free={free_mb}MB"
        )
    logger.warning(
        "DIARIZATION_DEVICE=%s unavailable (total=%d MB free=%d MB); using CPU.",
        requested,
        vram_mb,
        free_mb,
    )
    return False


def _auto_cuda_eligible(
    staging_ok: bool,
    co_resident_ok: bool,
    standard_ok: bool,
    large_gpu_ok: bool,
    free_mb: int,
) -> bool:
    if staging_ok:
        logger.info(
            "Diarization inference on CUDA (free=%d MB, ASR staged off GPU).",
            free_mb,
        )
        return True
    if co_resident_ok:
        logger.info(
            "Diarization inference on CUDA (free=%d MB, ASR co-resident).",
            free_mb,
        )
        return True
    if large_gpu_ok or standard_ok:
        return True
    logger.info("GPU diarization unavailable; diarization on CPU, ASR unchanged.")
    return False


def _diarization_cuda_eligible(
    requested: str,
    *,
    vram_mb: int,
    free_mb: int,
    min_cuda_vram_mb: int,
    min_free_mb: int,
    low_vram_cuda: bool,
    allow_low_vram_cuda: bool,
    gpu_co_resident: bool,
) -> bool:
    staging_ok, co_resident_ok, standard_ok, large_gpu_ok = _cuda_path_flags(
        free_mb=free_mb,
        min_free_mb=min_free_mb,
        vram_mb=vram_mb,
        min_cuda_vram_mb=min_cuda_vram_mb,
        low_vram_cuda=low_vram_cuda,
        gpu_co_resident=gpu_co_resident,
    )
    if requested in {"cuda", "gpu"}:
        return _explicit_cuda_eligible(
            requested,
            vram_mb=vram_mb,
            free_mb=free_mb,
            low_vram_cuda=low_vram_cuda,
            allow_low_vram_cuda=allow_low_vram_cuda,
            gpu_co_resident=gpu_co_resident,
            asr_loaded=_asr_model_loaded(),
            staging_ok=staging_ok,
            co_resident_ok=co_resident_ok,
            standard_ok=standard_ok,
        )
    return _auto_cuda_eligible(
        staging_ok, co_resident_ok, standard_ok, large_gpu_ok, free_mb,
    )


def _select_diarization_device(torch_module):
    """Select inference device; auto promotes to CUDA when free VRAM allows ASR co-resident."""
    requested = os.getenv("DIARIZATION_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested not in {"cuda", "gpu", "auto"} or not torch_module.cuda.is_available():
        return torch_module.device("cpu")

    vram_mb = _cuda_vram_mb(torch_module)
    free_mb = _cuda_free_mb(torch_module)
    low_vram_limit_mb = _env_int("ASR_8GB_CLASS_MAX_MB", 9000)
    use_cuda = _diarization_cuda_eligible(
        requested,
        vram_mb=vram_mb,
        free_mb=free_mb,
        min_cuda_vram_mb=_env_int("DIARIZATION_CUDA_MIN_VRAM_MB", 6000),
        min_free_mb=_env_int("DIARIZATION_CUDA_MIN_FREE_MB", 3072),
        low_vram_cuda=0 < vram_mb <= low_vram_limit_mb,
        allow_low_vram_cuda=_env_bool("DIARIZATION_ALLOW_8GB_CUDA", False),
        gpu_co_resident=_env_bool("DIARIZATION_GPU_CO_RESIDENT", False),
    )
    if not use_cuda and _diarization_cuda_required():
        _raise_if_cuda_required(
            f"free_mb={free_mb} vram_mb={vram_mb} allow_8gb={_env_bool('DIARIZATION_ALLOW_8GB_CUDA', False)}"
        )
    return torch_module.device("cuda" if use_cuda else "cpu")


def load_offline_pyannote_pipeline(model_id: str | None = None):
    """Load pyannote pipeline from the local Hugging Face cache only (no hub download)."""
    from pyannote.audio import Pipeline

    from engines.model_cache import (
        _sync_hub_constants,
        offline_cache_error_message,
        require_cached_pipeline,
        resolve_pretrained_checkpoint,
    )

    pipeline_id = model_id or MODEL_ID
    hf_token = os.getenv("HF_TOKEN")
    _sync_hub_constants()
    require_cached_pipeline(pipeline_id, logger)
    checkpoint = resolve_pretrained_checkpoint(pipeline_id)
    logger.info(
        "Offline pyannote pipeline load (%s) from %s",
        pipeline_id,
        checkpoint,
    )
    try:
        load_kwargs: dict = {}
        if hf_token:
            load_kwargs["token"] = hf_token
        pipeline = Pipeline.from_pretrained(checkpoint, **load_kwargs)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{pipeline_id}' from local cache: {exc}. "
            f"{offline_cache_error_message(pipeline_id)}"
        ) from exc
    if pipeline is None:
        raise RuntimeError(f"Failed to load pyannote pipeline '{pipeline_id}'")
    return pipeline


def _get_diarization_pipeline():
    """Lazy-load the configured pyannote diarization pipeline."""
    if _pipeline_cache:
        return _pipeline_cache[0]

    import torch

    _check_ffmpeg()

    pipeline = load_offline_pyannote_pipeline(MODEL_ID)
    device = _device_for_preload(torch)
    pipeline = pipeline.to(device)
    _set_tracked_device(device)

    # Apply tunable segmentation / clustering hyperparameters.
    custom_params = _filter_pipeline_params(pipeline, _build_pipeline_params())
    if custom_params:
        try:
            pipeline.instantiate(custom_params)
            logger.info("Diarization hyperparameters applied: %s", custom_params)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Could not apply custom diarization params: %s", exc)

    _pipeline_cache.append(pipeline)
    logger.info("Pyannote diarization pipeline ready on %s.", device)
    return _pipeline_cache[0]


def load_model() -> None:
    """Pre-load the pyannote diarization pipeline. Safe to call multiple times."""
    _get_diarization_pipeline()
    logger.info("Pyannote diarization model pre-loaded.")


def unload_model() -> None:
    """Unload the cached pyannote pipeline and release CUDA cache."""
    global _tracked_device
    _pipeline_cache.clear()
    _tracked_device = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
    except (ImportError, RuntimeError, OSError, AttributeError):
        pass
    logger.info("Pyannote diarization model cache cleared.")


def _prepare_audio_for_pyannote(audio_path: str, preprocess_sr: int | None = None) -> dict:
    """Load audio as an in-memory 16 kHz mono waveform for pyannote.

    Pipeline:
      1. FFmpeg — convert to mono WAV at DIARIZATION_PREPROCESS_SR (default
         44 100 Hz) with a speech-optimised bandpass filter.  Using a higher
         intermediate rate preserves more spectral detail for the noise-
         reduction stage before the final 16 kHz downsample.
      2. Optional noisereduce — non-stationary spectral gating at the
         intermediate rate (DIARIZATION_NOISE_REDUCTION; 0.0 = skip).
      3. librosa — resample + load as 16 kHz mono tensor for pyannote.

    Docker uses a minimal torchcodec stub because transformers probes the
    package at import time. Passing file paths to pyannote would make pyannote
    call torchcodec's AudioDecoder for metadata, which fails with that stub.
    The supported pyannote workaround is to pass a preloaded audio dictionary.
    """
    import torch

    # community-1 ingests 16 kHz mono per its model card. Using 16 kHz directly
    # skips a needless resample. Noise reduction defaults to OFF because
    # aggressive spectral gating degrades speaker embedding quality; the
    # ffmpeg loudness normalization below handles level/clipping issues.
    effective_sr = (
        preprocess_sr if preprocess_sr is not None else _env_int("DIARIZATION_PREPROCESS_SR", 16000)
    )
    noise_reduction = _env_float("DIARIZATION_NOISE_REDUCTION", 0.0)
    source_path, tmp_files = _build_diarization_source(
        audio_path, effective_sr, noise_reduction
    )

    try:
        waveform, sample_rate = _load_diarization_waveform(source_path)
    finally:
        _cleanup_tmp_files(tmp_files)

    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    duration = tensor.shape[-1] / 16000 if tensor.shape[-1] else 0.0
    logger.info(
        "Prepared diarization audio: %.1fs at %d Hz (intermediate=%d Hz, nr=%.2f)",
        duration,
        sample_rate,
        effective_sr,
        noise_reduction,
    )
    return {"waveform": tensor, "sample_rate": 16000}


def _build_diarization_source(
    audio_path: str,
    preprocess_sr: int,
    noise_reduction: float,
) -> tuple[str, list[str]]:
    """Return a pyannote-ready source path plus temporary files to remove."""
    ffmpeg = shutil.which("ffmpeg")
    source_path = audio_path
    tmp_files: list[str] = []
    already_ready = (
        _is_16k_mono_wav(audio_path)
        and preprocess_sr == 16000
        and noise_reduction <= 0.0
    )

    if ffmpeg and not already_ready:
        source_path = _run_ffmpeg_diarization_prep(ffmpeg, audio_path, preprocess_sr)
        if source_path != audio_path:
            tmp_files.append(source_path)
    elif not ffmpeg and not already_ready:
        logger.warning("ffmpeg not found - loading original audio with librosa.")

    if noise_reduction > 0.0 and source_path != audio_path:
        source_path = _apply_diarization_noise_reduction(
            source_path,
            os.path.splitext(audio_path)[0],
            preprocess_sr,
            noise_reduction,
            tmp_files,
        )
    return source_path, tmp_files


def _run_ffmpeg_diarization_prep(
    ffmpeg: str,
    audio_path: str,
    preprocess_sr: int,
) -> str:
    """Run FFmpeg bandpass/loudness normalization for diarization."""
    inter_path = os.path.splitext(audio_path)[0] + "_diarize_inter.wav"
    # Audio filter chain optimised for WeSpeaker speaker embeddings
    # (pyannote community-1 backbone, trained on VoxCeleb at 16 kHz):
    #   highpass=80 Hz  — removes sub-80 Hz rumble/HVAC; no speaker-ID info below this.
    #   lowpass=8000 Hz — 8 kHz is the Nyquist limit of 16 kHz audio; upper freqs
    #                     add no embedding information and only increase noise.
    #   acompressor     — gentle 3:1 downward compression starting at -20 dBFS;
    #                     equalises loud/quiet speakers so embeddings cluster tighter.
    #   loudnorm        — EBU R128 -23 LUFS integrated, LRA=7 (tight range) ensures
    #                     consistent per-speaker loudness and prevents clipping artefacts
    #                     that corrupt similarity scores.
    diarize_filters = (
        "highpass=f=80,"
        "lowpass=f=8000,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
        "loudnorm=I=-23:TP=-1.0:LRA=7"
    )
    cmd = [
        ffmpeg, "-y", "-i", audio_path,
        "-af", diarize_filters,
        "-ar", str(preprocess_sr),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-vn",
        inter_path,
    ]
    logger.info(
        "Preparing diarization audio with ffmpeg: sr=%d filters=%s",
        preprocess_sr,
        diarize_filters,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode == 0:
        return inter_path
    logger.warning(
        "ffmpeg diarization prep failed (%s); loading original with librosa.",
        result.stderr[-300:],
    )
    return audio_path


def _apply_diarization_noise_reduction(
    source_path: str,
    base_path: str,
    preprocess_sr: int,
    noise_reduction: float,
    tmp_files: list[str],
) -> str:
    """Apply optional noisereduce pass to the diarization source file."""
    try:
        import librosa
        import noisereduce as nr
        import soundfile as sf

        y_inter, sr_inter = librosa.load(source_path, sr=preprocess_sr, mono=True)
        y_reduced = nr.reduce_noise(
            y=y_inter,
            sr=sr_inter,
            stationary=False,
            prop_decrease=noise_reduction,
            n_fft=2048,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )
        nr_path = base_path + "_diarize_nr.wav"
        sf.write(nr_path, y_reduced, sr_inter, subtype="PCM_16")
        tmp_files.append(nr_path)
        logger.info(
            "Diarization noise reduction applied (prop=%.2f) at %d Hz.",
            noise_reduction,
            sr_inter,
        )
        return nr_path
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("Diarization noise reduction skipped: %s", exc)
        return source_path


def _load_diarization_waveform(source_path: str):
    """Load the final diarization waveform as 16 kHz mono."""
    import librosa

    return librosa.load(source_path, sr=16000, mono=True)


def _cleanup_tmp_files(tmp_files: list[str]) -> None:
    """Remove temporary diarization audio files."""
    for tmp in tmp_files:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _is_16k_mono_wav(audio_path: str) -> bool:
    if os.path.splitext(audio_path)[1].lower() != ".wav":
        return False
    try:
        with wave.open(audio_path, "rb") as wav_file:
            return wav_file.getframerate() == 16000 and wav_file.getnchannels() == 1
    except (wave.Error, OSError, EOFError):
        return False


def _override_params(
    seg_threshold: float | None,
    seg_min_duration_off: float | None,
    clust_threshold: float | None,
    clust_min_size: int | None,
    vbx_fa: float | None = None,
    vbx_fb: float | None = None,
) -> dict | None:
    """Build pyannote instantiate() params from explicit per-call overrides."""
    params: dict = {}
    seg: dict = {}
    if seg_threshold is not None:
        seg["threshold"] = float(seg_threshold)
    if seg_min_duration_off is not None:
        seg["min_duration_off"] = float(seg_min_duration_off)
    if seg:
        params["segmentation"] = seg
    clust: dict = {}
    if clust_threshold is not None:
        clust["threshold"] = float(clust_threshold)
    if clust_min_size is not None:
        clust["min_cluster_size"] = int(clust_min_size)
    if vbx_fa is not None:
        clust["Fa"] = float(vbx_fa)
    if vbx_fb is not None:
        clust["Fb"] = float(vbx_fb)
    if clust:
        params["clustering"] = clust
    return params if params else None


def _adaptive_pipeline_params(
    audio_duration_s: float,
    max_speakers: int,
) -> dict | None:
    """Return community-1 params tuned for short / multi-speaker clips.

    community-1 defaults (min_cluster_size=12) need many embedding windows per
    speaker; short clips with 2–3 speakers often collapse to one cluster without
    these adjustments.
    """
    if audio_duration_s <= 0 or audio_duration_s >= 90.0:
        return None

    seg: dict = {"threshold": 0.42}
    clust: dict = {"min_cluster_size": 3, "threshold": 0.58}
    if max_speakers >= 3:
        clust["threshold"] = 0.55
    if audio_duration_s < 30.0:
        seg["min_duration_off"] = 0.05
    return {"segmentation": seg, "clustering": clust}


def _retry_pipeline_params(base_params: dict | None) -> dict:
    """Looser clustering for a second pass when only one speaker was detected."""
    clust = dict((base_params or {}).get("clustering") or {})
    if _is_vbx_pipeline():
        # VBx merge pressure lives in Fa, not the AHC-init threshold.
        clust["Fa"] = min(0.40, float(clust.get("Fa", 0.07)) + 0.13)
        clust["threshold"] = max(0.45, float(clust.get("threshold", 0.60)) - 0.05)
    else:
        threshold = float(clust.get("threshold", 0.58))
        clust["threshold"] = max(0.35, threshold - 0.08)
        clust["min_cluster_size"] = 2
    seg = dict((base_params or {}).get("segmentation") or {})
    return {"segmentation": seg, "clustering": clust}


def _instantiate_pipeline_params(pipe, params: dict | None, label: str) -> None:
    """Apply instantiate() params when supported by the loaded pipeline."""
    filtered = _filter_pipeline_params(pipe, params)
    if not filtered:
        return
    try:
        pipe.instantiate(filtered)
        logger.info("Diarization %s params applied: %s", label, filtered)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("Could not apply diarization %s params: %s", label, exc)


def _overcluster_extra(max_speakers: int) -> int:
    """Extra clusters to request beyond the cap on meeting-scale VBx jobs.

    community-1's constrained VBx assignment starves most clusters on long
    meetings (measured: 6 populated of ~13 on a 90-min 11-speaker file), while
    a forced KMeans partition at cap+extra splits crowded clusters cleanly.
    The same-voice duplicates this creates are reunified afterwards by
    _merge_similar_speaker_clusters, and _enforce_max_speakers re-applies the
    user's cap. Requires the user's Max Speakers to approximate the real
    attendee count, which the UI guidance states for meetings.
    """
    configured = _env_int("DIARIZATION_OVERCLUSTER_EXTRA", -1)
    from backend.asr_quality import is_accuracy_mode

    if not (is_accuracy_mode() and _is_vbx_pipeline()):
        if configured >= 0:
            return configured
        return 0
    if max_speakers >= 11:
        if configured >= 0:
            return configured
        return 7
    if max_speakers >= 6:
        return 3
    if max_speakers >= 3:
        return 1
    return 0


def _build_diarize_kwargs(
    num_speakers: int,
    max_speakers: int,
    audio_duration_s: float = 0.0,
    min_speakers_hint: int = 0,
) -> dict:
    """Construct pyannote pipeline call kwargs from speaker count hints.

    ``max_speakers`` is a strict UPPER bound. We do NOT synthesise a
    ``min_speakers`` floor from the slider value: forcing a floor makes pyannote
    fabricate phantom speakers on monologue / 2-person recordings. A floor is
    applied only when an explicit ``min_speakers_hint`` is supplied (e.g. a
    re-diarization retry that already detected a collapse), and it is always
    clamped to the upper bound.

    Meeting-scale VBx jobs overcluster on purpose (see _overcluster_extra);
    centroid merging + the max-speaker cap bring the count back down.
    """
    del audio_duration_s
    # Meeting-scale VBx: request hint+extra even with an exact speaker count.
    # community-1 otherwise under-fills (measured 8/11 on 309 before centroid
    # merge collapsed the rest). _enforce_max_speakers clamps afterward.
    if num_speakers > 0:
        extra = _overcluster_extra(max_speakers if max_speakers > 0 else num_speakers)
        if extra > 0 and num_speakers >= 6:
            return {"num_speakers": num_speakers + extra}
        return {"num_speakers": num_speakers}
    if min_speakers_hint <= 0 and max_speakers > 0:
        extra = _overcluster_extra(max_speakers)
        if extra > 0:
            return {"num_speakers": max_speakers + extra}
    kwargs: dict = {}
    if max_speakers > 0:
        kwargs["max_speakers"] = max_speakers
    if min_speakers_hint > 0:
        kwargs["min_speakers"] = (
            min(min_speakers_hint, max_speakers) if max_speakers > 0 else min_speakers_hint
        )
    return kwargs


def _merge_diarization_params(base: dict | None, overlay: dict | None) -> dict | None:
    """Deep-merge instantiate() param dicts (segmentation / clustering)."""
    if not base:
        return overlay
    if not overlay:
        return base
    merged: dict = {}
    for key in ("segmentation", "clustering"):
        part = dict((base or {}).get(key) or {})
        part.update((overlay or {}).get(key) or {})
        if part:
            merged[key] = part
    return merged or None


def _max_speaker_cap_params(max_speakers: int, audio_duration_s: float) -> dict:
    """Tighter community-1 clustering to avoid phantom extra speakers."""
    from backend.asr_quality import is_accuracy_mode, is_high_quality_profile

    if is_accuracy_mode():
        return _accuracy_mode_params(max_speakers, audio_duration_s)

    # Higher clustering threshold => merge embeddings => fewer speakers.
    threshold = 0.58 + min(4, max(0, max_speakers - 2)) * 0.05
    if is_high_quality_profile() and not is_accuracy_mode():
        threshold += 0.04
    threshold = min(0.80, max(0.55, threshold))
    min_cluster = max(4, min(12, 16 - max_speakers * 2))
    seg: dict = {"threshold": 0.44, "min_duration_off": 0.12}
    if 0 < audio_duration_s < 45.0:
        seg["min_duration_off"] = 0.08
    return {
        "segmentation": seg,
        "clustering": {"threshold": threshold, "min_cluster_size": min_cluster},
    }


def _is_vbx_pipeline() -> bool:
    """community-1 clusters with VBx (threshold/Fa/Fb), not agglomerative."""
    return "community" in MODEL_ID.lower()


def _accuracy_mode_params(max_speakers: int, audio_duration_s: float) -> dict:
    """Sensitive segmentation/clustering for maximum speaker separation."""
    del audio_duration_s
    if _is_vbx_pipeline():
        # community-1 VBx: default Fa=0.07 merges rapid dialogue and brief
        # meeting speakers. Tier Fa/threshold by expected attendee count.
        if max_speakers >= 11:
            return {
                "segmentation": {"min_duration_off": 0.03},
                "clustering": {"threshold": 0.56, "Fa": 0.32, "Fb": 0.8},
            }
        if max_speakers >= 6:
            return {
                "segmentation": {"min_duration_off": 0.04},
                "clustering": {"threshold": 0.60, "Fa": 0.22, "Fb": 0.8},
            }
        if max_speakers >= 3:
            if max_speakers <= 5:
                return {
                    "segmentation": {"min_duration_off": 0.03},
                    "clustering": {"threshold": 0.54, "Fa": 0.22, "Fb": 0.8},
                }
            return {
                "segmentation": {"min_duration_off": 0.04},
                "clustering": {"threshold": 0.55, "Fa": 0.18, "Fb": 0.8},
            }
        if max_speakers == 2:
            return {
                "segmentation": {"min_duration_off": 0.08},
                "clustering": {"threshold": 0.42, "Fa": 0.15, "Fb": 0.8},
            }
    if max_speakers == 2:
        clust_threshold = 0.42
        min_off = 0.03
    elif max_speakers >= 3:
        clust_threshold = 0.44
        min_off = 0.04
    else:
        clust_threshold = 0.48
        min_off = 0.05
    return {
        "segmentation": {"min_duration_off": min_off},
        "clustering": {"threshold": clust_threshold, "min_cluster_size": 2},
    }


def _long_audio_adaptive_params(max_speakers: int, audio_duration_s: float) -> dict | None:
    """Meeting-length files (>= 180 s) use sensitive params in accuracy mode."""
    from backend.asr_quality import is_accuracy_mode

    if not is_accuracy_mode() or audio_duration_s < 180.0:
        return None
    return _accuracy_mode_params(max_speakers, audio_duration_s)


def _speaker_durations(segments: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        spk = seg["speaker"]
        totals[spk] = totals.get(spk, 0.0) + max(0.0, seg["end"] - seg["start"])
    return totals


def _dominant_speaker_share(segments: list[dict]) -> tuple[float, int]:
    """Return (max speaker time share, unique speaker count)."""
    totals = _speaker_durations(segments)
    if not totals:
        return 0.0, 0
    total = sum(totals.values())
    if total <= 0:
        return 0.0, len(totals)
    return max(totals.values()) / total, len(totals)


def _nearest_kept_speaker(
    seg: dict, segments: list[dict], keep: set[str],
) -> str:
    """Pick kept speaker label closest in time to seg."""
    mid = (seg["start"] + seg["end"]) / 2.0
    best = next(iter(keep))
    best_dist = float("inf")
    for other in segments:
        if other["speaker"] not in keep:
            continue
        o_mid = (other["start"] + other["end"]) / 2.0
        dist = abs(o_mid - mid)
        if dist < best_dist:
            best_dist = dist
            best = other["speaker"]
    return best


def _best_overlap_speaker(seg: dict, segments: list[dict], keep: set[str]) -> str | None:
    best_spk = None
    best_ov = 0.0
    for other in segments:
        if other["speaker"] not in keep:
            continue
        overlap = max(0.0, min(seg["end"], other["end"]) - max(seg["start"], other["start"]))
        if overlap > best_ov:
            best_ov = overlap
            best_spk = other["speaker"]
    return best_spk


def _enforce_max_speakers(segments: list[dict], max_speakers: int) -> list[dict]:
    """Reassign minor speakers so unique labels never exceed max_speakers."""
    if max_speakers <= 0 or not segments:
        return segments
    unique = sorted({s["speaker"] for s in segments})
    if len(unique) <= max_speakers:
        return segments

    durations = _speaker_durations(segments)
    keep = set(sorted(durations.keys(), key=lambda k: durations[k], reverse=True)[:max_speakers])
    logger.info(
        "Capping diarization speakers %d -> %d (dropping %s)",
        len(unique),
        max_speakers,
        sorted(set(unique) - keep),
    )

    reassigned: list[dict] = []
    for seg in segments:
        spk = seg["speaker"]
        if spk in keep:
            reassigned.append(seg)
            continue
        best_spk = _best_overlap_speaker(seg, segments, keep)
        if best_spk is None:
            best_spk = _nearest_kept_speaker(seg, segments, keep)
        reassigned.append({**seg, "speaker": best_spk})
    return sorted(reassigned, key=lambda s: s["start"])


def _merge_adjacent_same_speaker(
    segments: list[dict], max_gap_s: float = 0.45,
) -> list[dict]:
    """Merge consecutive turns for the same speaker when the gap is tiny."""
    if len(segments) < 2:
        return segments
    merged: list[dict] = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= max_gap_s:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def _chunk_turn_slices(
    turn: dict,
    max_turn_s: float,
    min_turn_s: float,
) -> list[dict]:
    slices: list[dict] = []
    start = turn["start"]
    end = turn["end"]
    while start < end - 0.01:
        chunk_end = min(end, start + max_turn_s)
        if chunk_end - start >= min_turn_s:
            slices.append({
                "start": start,
                "end": chunk_end,
                "speaker": turn["speaker"],
            })
        start = chunk_end
    return slices


def _split_turns_for_asr(
    turns: list[dict],
    raw_segments: list[dict],
    max_turn_s: float,
    min_turn_s: float,
) -> list[dict]:
    """Split merged speaker turns so each ASR pass stays within max_turn_s."""
    del raw_segments
    out: list[dict] = []
    for turn in turns:
        out.extend(_chunk_turn_slices(turn, max_turn_s, min_turn_s))
    return out


def prepare_asr_turns(
    diarization_segments: list[dict],
    max_speakers: int,
) -> list[dict]:
    """Build diarization turns sized for per-turn Whisper passes."""
    if not diarization_segments:
        return []
    from backend.asr_quality import is_accuracy_mode

    configured_gap = os.getenv("ASR_TURN_GUIDED_MERGE_GAP_S", "").strip()
    if configured_gap:
        try:
            merge_gap = float(configured_gap)
        except ValueError:
            merge_gap = 0.0 if is_accuracy_mode() else _env_float("ASR_TURN_GUIDED_MERGE_GAP_S", 1.0)
    elif is_accuracy_mode():
        merge_gap = 0.0
    else:
        merge_gap = _env_float("ASR_TURN_GUIDED_MERGE_GAP_S", 1.0)
    max_turn_s = _env_float("ASR_TURN_GUIDED_MAX_TURN_S", 45.0)
    min_turn_s = _env_float("ASR_TURN_GUIDED_MIN_TURN_S", 0.4)
    from engines.whisper_utils import whisper_max_asr_turn_body_s

    max_turn_s = min(max_turn_s, whisper_max_asr_turn_body_s())

    segments = sorted(
        (dict(seg) for seg in diarization_segments),
        key=lambda seg: seg["start"],
    )
    if max_speakers > 0:
        segments = _enforce_max_speakers(segments, max_speakers)
    merged = _merge_adjacent_same_speaker(segments, max_gap_s=merge_gap)
    turns = _split_turns_for_asr(merged, segments, max_turn_s, min_turn_s)
    logger.info(
        "Prepared %d ASR turn(s) from %d diarization segment(s) "
        "(merge_gap=%.1fs max_turn=%.0fs).",
        len(turns),
        len(diarization_segments),
        merge_gap,
        max_turn_s,
    )
    return turns


def _postprocess_merge_gap_s() -> float:
    """Gap for merging adjacent same-speaker diar fragments after pyannote."""
    from backend.asr_quality import is_accuracy_mode

    if _env_bool("DIARIZATION_LOCK_PARAMS", False):
        for key in (
            "DIARIZATION_TRANSCRIPT_MERGE_GAP_S",
            "DIARIZATION_ASSIGN_TURN_MERGE_GAP_S",
            "ASR_TURN_GUIDED_MERGE_GAP_S",
        ):
            raw = os.getenv(key, "").strip()
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    pass
    return 0.35 if is_accuracy_mode() else 0.45


def _postprocess_diarization_segments(
    segments: list[dict], max_speakers: int,
) -> list[dict]:
    """Stabilize pyannote output: merge fragments, cap speaker count."""
    from backend.asr_quality import is_accuracy_mode

    if not segments:
        return segments
    min_frag = 0.3 if is_accuracy_mode() else 0.4
    merge_gap = _postprocess_merge_gap_s()
    out = _merge_short_segments(list(segments), min_duration_s=min_frag)
    out = _merge_adjacent_same_speaker(out, max_gap_s=merge_gap)
    out = _enforce_max_speakers(out, max_speakers)
    out = _merge_adjacent_same_speaker(out, max_gap_s=merge_gap)
    return out


# (labels, centroid matrix) of the most recent pyannote pass. Consumed once by
# _merge_similar_speaker_clusters right after the MAIN full pass; sub-passes
# (tune windows, span refines) overwrite it but their label sets never match
# the global segment labels at merge time, and the consumer clears it.
_last_pass_embeddings: list = []
_embedding_rescue_cache: tuple[list[str], object] | None = None


def _segments_from_diarization(diarization) -> list[dict]:
    """Extract sorted segment list from a pyannote diarization result."""
    annotation = (
        getattr(diarization, "exclusive_speaker_diarization", None)
        or getattr(diarization, "speaker_diarization", diarization)
    )
    if annotation is not diarization and getattr(
        diarization, "exclusive_speaker_diarization", None
    ) is not None:
        logger.info("Using community-1 exclusive_speaker_diarization for ASR alignment.")

    _stash_speaker_embeddings(diarization, annotation)

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    segments.sort(key=lambda s: s["start"])
    return _merge_short_segments(segments)


def _stash_speaker_embeddings(diarization, annotation) -> None:
    """Remember per-speaker centroids of this pass for cluster reunification."""
    global _embedding_rescue_cache
    _last_pass_embeddings.clear()
    embeddings = getattr(diarization, "speaker_embeddings", None)
    if embeddings is None:
        return
    try:
        labels = [str(label) for label in annotation.labels()]
        if len(labels) == getattr(embeddings, "shape", (0,))[0]:
            _last_pass_embeddings.append((labels, embeddings))
            _embedding_rescue_cache = (labels, embeddings)
    except (AttributeError, TypeError, ValueError):
        pass


def _centroid_merge_threshold(max_speakers: int = 0) -> float:
    """Cosine similarity above which two clusters are the same voice (0 = off).

    Defaults on only for meeting-scale jobs (max_speakers >= 6) in accuracy
    mode, where higher VBx Fa is in play and one voice can land in two
    clusters. Small-group runs keep the model's own clustering untouched
    unless DIARIZATION_CENTROID_MERGE_THRESHOLD is set explicitly.
    """
    configured = _env_float("DIARIZATION_CENTROID_MERGE_THRESHOLD", -1.0)
    if configured >= 0:
        return configured
    from backend.asr_quality import is_accuracy_mode

    # 0.72 sits in the measured gap between same-voice duplicate centroids
    # (>= 0.74 across passes on the 309 fixture) and distinct-voice pairs
    # (<= 0.69). Applies to unit-normalized WeSpeaker centroid cosines.
    return 0.72 if (is_accuracy_mode() and max_speakers >= 6) else 0.0


def _union_find_root(parent: dict[str, str], label: str) -> str:
    while parent[label] != label:
        parent[label] = parent[parent[label]]
        label = parent[label]
    return label


def _should_merge_centroid_pair(
    labels: list[str],
    similarity,
    valid,
    durations: dict[str, float],
    threshold: float,
    parent: dict[str, str],
    i: int,
    i2: int,
) -> bool:
    if not (valid[i] and valid[i2]):
        return False
    if similarity[i, i2] < threshold:
        return False
    shorter = min(durations.get(labels[i], 0.0), durations.get(labels[i2], 0.0))
    sim = float(similarity[i, i2])
    # Ambiguous band: brief voices must not collapse into a dominant cluster.
    if shorter < 15.0 and 0.72 <= sim < 0.80:
        return False
    root_big = _union_find_root(parent, labels[i])
    root_small = _union_find_root(parent, labels[i2])
    return root_big != root_small


def _collect_centroid_merge_pairs(
    labels: list[str],
    similarity,
    valid,
    durations: dict[str, float],
    threshold: float,
) -> tuple[list[tuple[str, str, float]], dict[str, str]]:
    """Return merge pairs and the union-find parent map."""
    order = sorted(range(len(labels)), key=lambda i: -durations.get(labels[i], 0.0))
    parent = {label: label for label in labels}
    merged_pairs: list[tuple[str, str, float]] = []
    for pos_a, i in enumerate(order):
        for i2 in order[pos_a + 1:]:
            if not _should_merge_centroid_pair(
                labels, similarity, valid, durations, threshold, parent, i, i2,
            ):
                continue
            root_big = _union_find_root(parent, labels[i])
            root_small = _union_find_root(parent, labels[i2])
            parent[root_small] = root_big
            merged_pairs.append((labels[i2], labels[i], float(similarity[i, i2])))
    return merged_pairs, parent


def _merge_similar_speaker_clusters(
    segments: list[dict], max_speakers: int = 0,
) -> list[dict]:
    """Reunify clusters whose speaker centroids are the same voice.

    VBx can split one speaker across two clusters when their acoustics drift
    (speaker moves relative to the mic mid-meeting). Merging by centroid
    cosine similarity fixes those splits without touching genuinely distinct
    voices. Uses the centroids stashed by the immediately preceding pass and
    only acts when its labels exactly match the segment labels.
    """
    threshold = _centroid_merge_threshold(max_speakers)
    if threshold <= 0 or not segments or not _last_pass_embeddings:
        _last_pass_embeddings.clear()
        return segments
    labels, embeddings = _last_pass_embeddings[0]
    _last_pass_embeddings.clear()
    seg_labels = sorted({seg["speaker"] for seg in segments})
    if sorted(labels) != seg_labels or len(labels) < 2:
        return segments

    import numpy as np

    matrix = np.asarray(embeddings, dtype=float)
    # Rows with non-finite values (silent clusters) must never merge.
    valid = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    similarity = unit @ unit.T
    durations = _speaker_durations(segments)
    merged_pairs, parent = _collect_centroid_merge_pairs(
        labels, similarity, valid, durations, threshold,
    )

    if not merged_pairs:
        return segments
    for src, dst, sim in merged_pairs:
        logger.info(
            "Centroid merge: %s -> %s (cosine=%.3f >= %.2f, same voice).",
            src, dst, sim, threshold,
        )
    return [
        {**seg, "speaker": _union_find_root(parent, seg["speaker"])}
        for seg in segments
    ]


def _settle_cuda_transient(torch_module) -> None:
    """Brief pause + cache clear for WSL2 ``device not ready`` transients."""
    import time

    _recover_cuda_after_failure(torch_module)
    time.sleep(5.0)
    try:
        torch_module.cuda.synchronize()
        warm = torch_module.zeros(1, device="cuda")
        del warm
        torch_module.cuda.synchronize()
    except RuntimeError:
        pass


def _request_diarization_cuda_restart(exc: RuntimeError, msg: str) -> None:
    """Exit the process when allocator corruption cannot be recovered in-process."""
    if _CUDA_CACHING_ALLOCATOR not in msg and _CUDA_INTERNAL_ASSERT not in msg:
        return
    try:
        from backend import vram_state

        vram_state.request_cuda_restart(f"Diarization CUDA: {exc}")
    except ImportError:
        pass


def _retry_diarization_on_cuda(_pipe, audio_input: dict, kwargs: dict, exc: RuntimeError):
    """Unload/reload pipeline and retry diarization on CUDA after a recoverable error."""
    import torch

    logger.warning(
        "Diarization CUDA error (%s); recovering and retrying on CUDA.",
        exc,
    )
    _recover_cuda_after_failure(torch)
    try:
        from backend import vram_state

        vram_state.teardown(aggressive=True)
    except ImportError:
        pass
    try:
        unload_model()
    except (RuntimeError, AttributeError):
        pass
    reloaded_pipe = _get_diarization_pipeline()
    _move_pipeline_to_inference_device(reloaded_pipe)
    try:
        result = _call_pyannote(reloaded_pipe, audio_input, kwargs)
    except RuntimeError as retry_exc:
        try:
            from backend import vram_state

            if _is_recoverable_cuda_error(str(retry_exc).lower()):
                vram_state.request_cuda_restart(
                    f"CUDA diarization after retry: {retry_exc}",
                )
        except ImportError:
            pass
        raise RuntimeError(
            f"CUDA diarization failed after retry: {retry_exc}",
        ) from retry_exc
    try:
        from backend import vram_state

        unload_model()
        vram_state.teardown(aggressive=True)
        vram_state.ensure_cuda_healthy_or_restart("post-diarization CUDA retry")
    except (ImportError, RuntimeError, AttributeError):
        pass
    return result


def _retry_diarization_on_cpu(pipe, audio_input: dict, kwargs: dict, exc: RuntimeError):
    """Fall back to CPU diarization after CUDA failure when allowed."""
    import torch

    logger.warning(
        "Diarization CUDA error (%s); recovering and retrying on CPU.",
        exc,
    )
    _recover_cuda_after_failure(torch)
    try:
        from backend import vram_state

        vram_state.teardown(aggressive=True)
    except ImportError:
        pass
    _move_pipeline_to_cpu(pipe)
    return _call_pyannote(pipe, audio_input, kwargs)


def _retry_after_transient_cuda(pipe, audio_input: dict, kwargs: dict, exc: RuntimeError):
    """Settle the CUDA context and retry once after a device-not-ready error."""
    import torch

    msg = str(exc).lower()
    if _CUDA_CACHING_ALLOCATOR in msg or _CUDA_INTERNAL_ASSERT in msg:
        _request_diarization_cuda_restart(exc, msg)
        raise
    logger.warning(
        "Transient CUDA before diarization (%s); settling and retrying once.",
        exc,
    )
    _settle_cuda_transient(torch)
    return _call_pyannote(pipe, audio_input, kwargs)


def _recover_from_diarization_cuda_error(pipe, audio_input: dict, kwargs: dict, exc: RuntimeError):
    """Dispatch CUDA diarization recovery: CUDA retry, hard fail, or CPU fallback."""
    if _diarization_cuda_required() and _tracked_device == "cuda":
        return _retry_diarization_on_cuda(pipe, audio_input, kwargs, exc)
    if _diarization_cuda_required():
        raise RuntimeError(
            f"CUDA diarization failed and CPU fallback is disabled: {exc}"
        ) from exc
    return _retry_diarization_on_cpu(pipe, audio_input, kwargs, exc)


def _run_pyannote(pipe, audio_input: dict, kwargs: dict):
    """Call pyannote pipeline; use ProgressHook when available."""
    _move_pipeline_to_cpu_if_cuda_memory_low(pipe)
    try:
        return _call_pyannote(pipe, audio_input, kwargs)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if not _is_recoverable_cuda_error(msg):
            raise
        if _requires_cuda_process_restart(msg):
            try:
                return _retry_after_transient_cuda(pipe, audio_input, kwargs, exc)
            except RuntimeError as settle_exc:
                if not _is_recoverable_cuda_error(str(settle_exc).lower()):
                    raise
                exc = settle_exc
        return _recover_from_diarization_cuda_error(pipe, audio_input, kwargs, exc)


def _call_pyannote(pipe, audio_input: dict, kwargs: dict):
    """Call pyannote with optional progress hook."""
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            return pipe(audio_input, hook=hook, **kwargs)
    except ImportError:
        return pipe(audio_input, **kwargs)


def _move_pipeline_to_cpu(pipe) -> None:
    import torch

    _move_pipeline_to_device(pipe, torch.device("cpu"), "low-vram-fallback")
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError):
            pass


def _move_pipeline_to_cpu_if_cuda_memory_low(pipe) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        # Model weights already on CUDA reduce reported free VRAM; do not downgrade.
        if _tracked_device == "cuda":
            return
        min_free_mb = _diarization_run_min_free_mb()
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mb = int(free_bytes // (1024 * 1024))
        if free_mb >= min_free_mb:
            return
        if _diarization_cuda_required():
            _raise_if_cuda_required(
                f"only {free_mb}MB free CUDA (< {min_free_mb}MB) before load"
            )
        logger.warning(
            "Diarization sees only %d MB free CUDA memory (< %d MB); using CPU.",
            free_mb,
            min_free_mb,
        )
        _move_pipeline_to_cpu(pipe)
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug("Diarization CUDA free-memory probe skipped: %s", exc)


def _is_transition_fragment(
    seg: dict, prev_seg: dict, next_seg: dict, min_duration_s: float,
) -> bool:
    """Return True when seg is a short mis-attributed boundary fragment."""
    if seg["end"] - seg["start"] >= min_duration_s:
        return False
    gap_left = seg["start"] - prev_seg["end"]
    gap_right = next_seg["start"] - seg["end"]
    return (
        prev_seg["speaker"] != seg["speaker"]
        and next_seg["speaker"] != seg["speaker"]
        and gap_left <= 0.1
        and gap_right <= 0.1
    )


def _absorb_fragment(merged: list[dict], seg: dict, next_seg_ref: list[dict]) -> None:
    """Absorb a transition fragment into the longer of its two neighbours."""
    prev = merged[-1]
    nxt = next_seg_ref[0]
    left_dur = prev["end"] - prev["start"]
    right_dur = nxt["end"] - nxt["start"]
    if left_dur >= right_dur:
        merged[-1] = {**prev, "end": seg["end"]}
    else:
        next_seg_ref[0] = {**nxt, "start": seg["start"]}


def _merge_short_segments(
    segments: list[dict], min_duration_s: float = 0.4,
) -> list[dict]:
    """Merge isolated short boundary fragments into the longer adjacent turn.

    Pyannote can produce sub-0.4s segments at speaker-change boundaries that
    are mis-attributed noise.  A segment is merged only when its duration is
    below the threshold AND it has a different speaker on both sides AND both
    neighbouring gaps are <= 0.1s (i.e. it is a genuine transition fragment).
    """
    if len(segments) < 3:
        return segments
    result = list(segments)
    changed = True
    while changed:
        changed = False
        merged: list[dict] = []
        i = 0
        while i < len(result):
            seg = result[i]
            prev = merged[-1] if merged else None
            nxt_ref = [result[i + 1]] if i < len(result) - 1 else None
            if prev is not None and nxt_ref is not None and _is_transition_fragment(
                seg, prev, nxt_ref[0], min_duration_s
            ):
                _absorb_fragment(merged, seg, nxt_ref)
                result[i + 1] = nxt_ref[0]
                changed = True
            else:
                merged.append(seg)
            i += 1
        result = merged
    return result


_CHUNK_LABEL_RE = re.compile(r"^(.+)_S(\d+)$")


def _parse_chunk_speaker_label(speaker: str) -> tuple[int, str] | None:
    match = _CHUNK_LABEL_RE.match(speaker)
    if not match:
        return None
    return int(match.group(2)), match.group(1)


def _overlap_between_chunk_labels(
    segments: list[dict],
    chunk_a: int,
    label_a: str,
    chunk_b: int,
    label_b: str,
) -> float:
    total = 0.0
    for seg_a in segments:
        parsed_a = _parse_chunk_speaker_label(seg_a["speaker"])
        if not parsed_a or parsed_a != (chunk_a, label_a):
            continue
        for seg_b in segments:
            parsed_b = _parse_chunk_speaker_label(seg_b["speaker"])
            if not parsed_b or parsed_b != (chunk_b, label_b):
                continue
            total += max(
                0.0,
                min(seg_a["end"], seg_b["end"]) - max(seg_a["start"], seg_b["start"]),
            )
    return total


def _parse_segments_for_alignment(segments: list[dict]) -> list[dict]:
    parsed: list[dict] = []
    for seg in segments:
        chunk_label = _parse_chunk_speaker_label(seg["speaker"])
        if chunk_label is None:
            parsed.append({**seg, "chunk": -1, "local": seg["speaker"]})
        else:
            parsed.append({**seg, "chunk": chunk_label[0], "local": chunk_label[1]})
    return parsed


def _best_prev_chunk_global(
    parsed: list[dict],
    chunk: int,
    local: str,
    prev_chunk: int,
    canonical: dict[tuple[int, str], str],
) -> tuple[str | None, float]:
    best_global = None
    best_overlap = 0.0
    for (prev_c, prev_local), global_id in canonical.items():
        if prev_c != prev_chunk:
            continue
        overlap = _overlap_between_chunk_labels(
            parsed, chunk, local, prev_chunk, prev_local,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_global = global_id
    return best_global, best_overlap


def _build_chunk_canonical_map(
    parsed: list[dict],
    chunk_ids: list[int],
    min_overlap_s: float,
) -> dict[tuple[int, str], str]:
    canonical: dict[tuple[int, str], str] = {}
    next_global = 0
    for chunk in chunk_ids:
        locals_here = sorted({item["local"] for item in parsed if item["chunk"] == chunk})
        for local in locals_here:
            key = (chunk, local)
            if chunk == chunk_ids[0]:
                canonical[key] = f"G{next_global}"
                next_global += 1
                continue
            best_global, best_overlap = _best_prev_chunk_global(
                parsed, chunk, local, chunk - 1, canonical,
            )
            if best_global is not None and best_overlap >= min_overlap_s:
                canonical[key] = best_global
            else:
                canonical[key] = f"G{next_global}"
                next_global += 1
    return canonical


def _apply_canonical_speaker_map(
    parsed: list[dict],
    canonical: dict[tuple[int, str], str],
) -> list[dict]:
    aligned: list[dict] = []
    for seg in parsed:
        if seg["chunk"] < 0:
            aligned.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
            })
            continue
        global_id = canonical.get((seg["chunk"], seg["local"]), seg["local"])
        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": global_id,
        })
    aligned.sort(key=lambda item: item["start"])
    return aligned


def _align_segmented_speakers(
    segments: list[dict],
    min_overlap_s: float | None = None,
) -> list[dict]:
    """Map per-chunk pyannote labels to consistent globals using overlap agreement."""
    if not segments:
        return segments
    if min_overlap_s is None:
        from backend.asr_quality import is_accuracy_mode

        default = 1.5 if is_accuracy_mode() else 0.5
        min_overlap_s = _env_float("DIARIZATION_CHUNK_ALIGN_MIN_OVERLAP_S", default)

    parsed = _parse_segments_for_alignment(segments)
    chunk_ids = sorted({item["chunk"] for item in parsed if item["chunk"] >= 0})
    if not chunk_ids:
        return segments

    canonical = _build_chunk_canonical_map(parsed, chunk_ids, min_overlap_s)
    aligned = _apply_canonical_speaker_map(parsed, canonical)
    logger.info(
        "Aligned segmented speakers: %d chunks -> %d global label(s).",
        len(chunk_ids),
        len({s["speaker"] for s in aligned}),
    )
    return aligned


def _max_segment_duration(segments: list[dict]) -> float:
    if not segments:
        return 0.0
    return max(seg["end"] - seg["start"] for seg in segments)


def _mega_turn_retry_params(base_params: dict | None) -> dict:
    """Looser segmentation/clustering for under-segmented long turns."""
    retry = _retry_pipeline_params(base_params)
    clustering = dict(retry.get("clustering") or {})
    segmentation = dict(retry.get("segmentation") or {})
    if _is_vbx_pipeline():
        clustering["Fa"] = min(0.40, float(clustering.get("Fa", 0.20)) + 0.05)
        return {"segmentation": segmentation, "clustering": clustering}
    clustering["threshold"] = max(
        0.32,
        float(clustering.get("threshold", 0.48)) - 0.04,
    )
    segmentation["threshold"] = max(
        0.32,
        float(segmentation.get("threshold", 0.40)) - 0.02,
    )
    clustering["min_cluster_size"] = 2
    return {"segmentation": segmentation, "clustering": clustering}


def _diarize_audio_span(
    pipe,
    audio_path: str,
    start_s: float,
    end_s: float,
    kwargs: dict,
) -> list[dict]:
    """Run one pyannote pass on a sub-range of the source file."""
    import subprocess
    import tempfile

    duration_s = max(0.01, end_s - start_s)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_s), "-t", str(duration_s),
            "-i", audio_path,
            "-ac", "1", "-ar", "16000", tmp_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        audio_input = _prepare_audio_for_pyannote(tmp_path)
        try:
            segments = _execute_pyannote_pass(pipe, audio_input, kwargs)
        finally:
            del audio_input
            from backend import vram_state
            vram_state.teardown(aggressive=False)
        return [
            {
                "start": seg["start"] + start_s,
                "end": seg["end"] + start_s,
                "speaker": seg["speaker"],
            }
            for seg in segments
        ]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _global_speaker_order(segments: list[dict]) -> list[str]:
    """Unique speakers in first-chronological-appearance order."""
    order: list[str] = []
    for seg in sorted(segments, key=lambda item: item["start"]):
        spk = str(seg.get("speaker") or "")
        if spk and spk not in order:
            order.append(spk)
    return order


def _reload_diarization_pipeline_for_refine(pipe):
    """Tear down VRAM and reload the diarization pipeline; return fallback on failure."""
    try:
        from backend import vram_state
        import torch

        vram_state.teardown(aggressive=True)
        unload_model()
        _settle_cuda_transient(torch)
        reloaded = _get_diarization_pipeline()
        _move_pipeline_to_inference_device(reloaded)
        return reloaded
    except (ImportError, RuntimeError, AttributeError):
        return pipe


def _speakers_in_appearance_order(segments: list[dict]) -> list[str]:
    order: list[str] = []
    for seg in segments:
        spk = str(seg.get("speaker") or "")
        if spk and spk not in order:
            order.append(spk)
    return order


def _global_order_for_span(
    span: dict,
    span_idx: int,
    context_segments: list[dict] | None,
) -> list[str]:
    global_order = _global_speaker_order(context_segments or [])
    anchor = str(span.get("speaker") or "")
    if anchor and anchor not in global_order:
        global_order.append(anchor)
    if not global_order:
        global_order = [anchor or f"REFINE{span_idx}_A"]
    return global_order


def _sub_to_global_speaker_mapping(
    sub_order: list[str],
    global_order: list[str],
    anchor: str,
) -> dict[str, str]:
    try:
        start_idx = global_order.index(anchor) if anchor in global_order else 0
    except ValueError:
        start_idx = 0
    return {
        sub_spk: global_order[(start_idx + offset) % len(global_order)]
        for offset, sub_spk in enumerate(sub_order)
    }


def _segment_overlaps_speaker_in_span(
    seg: dict,
    speaker: str,
    span: dict,
    context_segments: list[dict] | None,
) -> bool:
    for other in context_segments or []:
        if str(other.get("speaker") or "") != speaker:
            continue
        if other["end"] <= span["start"] + 0.01 or other["start"] >= span["end"] - 0.01:
            continue
        if other["end"] <= seg["start"] + 0.01 or other["start"] >= seg["end"] - 0.01:
            continue
        return True
    return False


def _rebrand_span_replacement(
    replacement: list[dict],
    span: dict,
    span_idx: int,
    context_segments: list[dict] | None = None,
) -> list[dict]:
    """Map sub-pass speaker labels onto the global timeline.

    Sub-passes return unrelated SPEAKER_00.. labels. The old dominant-voice
    heuristic mis-assigned minority voices (e.g. sample01 SPEAKER_03 speech
    labelled SPEAKER_04). Chronological mapping assigns sub-voices in time
    order to the global speaker cycle starting at the span's speaker.
    """
    if not replacement:
        return replacement

    sub_order = _speakers_in_appearance_order(replacement)
    if len(sub_order) <= 1:
        return [{**seg, "speaker": span["speaker"]} for seg in replacement]

    anchor = str(span.get("speaker") or "")
    global_order = _global_order_for_span(span, span_idx, context_segments)
    mapping = _sub_to_global_speaker_mapping(sub_order, global_order, anchor)

    rebranded: list[dict] = []
    for seg in replacement:
        target = mapping.get(str(seg.get("speaker") or ""), span["speaker"])
        if _segment_overlaps_speaker_in_span(seg, target, span, context_segments):
            target = f"REFINE{span_idx}_{seg['speaker']}"
        rebranded.append({**seg, "speaker": target})
    return rebranded


def _replace_interval_segments(
    segments: list[dict],
    interval: dict,
    replacements: list[dict],
) -> list[dict]:
    """Swap segments overlapping interval with refined replacements."""
    kept = [
        seg for seg in segments
        if seg["end"] <= interval["start"] + 0.01 or seg["start"] >= interval["end"] - 0.01
    ]
    merged = kept + list(replacements)
    merged.sort(key=lambda item: item["start"])
    return merged


def _build_span_refine_kwargs(
    kwargs: dict,
    span_dur: float,
    max_speakers: int,
    retry_s: float,
) -> dict:
    span_kwargs = dict(kwargs)
    if span_kwargs.get("num_speakers") and _env_bool("DIARIZATION_EXACT_NUM_SPEAKERS", False):
        # Local sub-pass: allow pyannote to split voices inside one global label.
        span_kwargs = {k: v for k, v in span_kwargs.items() if k != "num_speakers"}
        if max_speakers > 0:
            span_kwargs["max_speakers"] = max(4, max_speakers)
    if span_dur >= retry_s * 0.5:
        span_kwargs.setdefault("min_speakers", 2)
    return span_kwargs


def _diarize_span_maybe_on_cpu(
    pipe,
    audio_path: str,
    span: dict,
    span_kwargs: dict,
    *,
    cpu_subpass: bool,
) -> list[dict]:
    if cpu_subpass:
        import torch

        _move_pipeline_to_device(pipe, torch.device("cpu"), "short-span-refine")
    try:
        return _diarize_audio_span(
            pipe, audio_path, span["start"], span["end"], span_kwargs,
        )
    finally:
        if cpu_subpass:
            try:
                _move_pipeline_to_inference_device(pipe)
            except RuntimeError:
                pass


def _should_force_span_split(
    replacement: list[dict],
    span_dur: float,
    retry_s: float,
    new_mega: float,
    new_score: float,
    old_score: float,
    audio_duration_s: float,
    *,
    cpu_subpass: bool,
) -> bool:
    sub_speakers = len({str(s.get("speaker") or "") for s in replacement})
    short_exact = (
        _env_bool("DIARIZATION_EXACT_NUM_SPEAKERS", False)
        and audio_duration_s > 0
        and audio_duration_s < _env_float("DIARIZATION_MEGA_TURN_SHORT_AUDIO_S", 600.0)
    )
    force_split = (
        _env_bool("DIARIZATION_EXACT_NUM_SPEAKERS", False)
        and span_dur >= retry_s * 0.85
        and len(replacement) >= 2
        and sub_speakers >= 2
        and new_mega < span_dur * 0.72
        and new_score >= old_score - 0.01
    )
    if (
        cpu_subpass
        and short_exact
        and len(replacement) >= 2
        and sub_speakers >= 2
        and _env_bool("DIARIZATION_FORCE_SHORT_REFINE_SPLIT", True)
    ):
        force_split = True
    return force_split


def _reject_span_refine(
    old_score: float,
    new_score: float,
    new_mega: float,
    span_dur: float,
    force_split: bool,
    refined: list[dict],
) -> tuple[list[dict], bool, float] | None:
    if new_score <= old_score and not force_split:
        return refined, False, _max_segment_duration(refined)
    if not force_split and new_score <= old_score and new_mega >= span_dur * 0.75:
        return refined, False, new_mega
    return None


def _try_refine_long_span(
    pipe,
    audio_path: str,
    span: dict,
    span_idx: int,
    refined: list[dict],
    kwargs: dict,
    audio_duration_s: float,
    max_speakers: int,
    *,
    cpu_subpass: bool = False,
) -> tuple[list[dict], bool, float]:
    """Re-diarize one long span; return (segments, improved, longest_after)."""
    from engines.diarization_sampling import score_segments

    span_dur = span["end"] - span["start"]
    logger.info(
        "Re-diarizing long span %.1fs-%.1fs (%.1fs) for better speaker splits%s.",
        span["start"],
        span["end"],
        span_dur,
        " on CPU" if cpu_subpass else "",
    )
    retry_s = _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 120.0)
    span_kwargs = _build_span_refine_kwargs(kwargs, span_dur, max_speakers, retry_s)
    replacement = _diarize_span_maybe_on_cpu(
        pipe, audio_path, span, span_kwargs, cpu_subpass=cpu_subpass,
    )
    if not replacement:
        return refined, False, _max_segment_duration(refined)
    replacement = _rebrand_span_replacement(replacement, span, span_idx, refined)
    candidate = _replace_interval_segments(refined, span, replacement)
    old_score = score_segments(refined, audio_duration_s, max_speakers)
    new_score = score_segments(candidate, audio_duration_s, max_speakers)
    new_mega = _max_segment_duration(candidate)
    force_split = _should_force_span_split(
        replacement, span_dur, retry_s, new_mega, new_score, old_score,
        audio_duration_s, cpu_subpass=cpu_subpass,
    )
    rejected = _reject_span_refine(
        old_score, new_score, new_mega, span_dur, force_split, refined,
    )
    if rejected is not None:
        return rejected
    logger.info(
        "Long-span refine accepted: score %.3f -> %.3f, longest=%.1fs.",
        old_score,
        new_score,
        new_mega,
    )
    return candidate, True, new_mega


def _mega_turn_refine_threshold_s(mega_turn_threshold_s: float | None) -> float:
    if mega_turn_threshold_s is not None:
        return mega_turn_threshold_s
    return _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 120.0)


def _should_skip_mega_turn_refine(
    segments: list[dict],
    max_speakers: int,
    audio_duration_s: float,
    *,
    allow_short_audio: bool,
) -> bool:
    from backend.asr_quality import is_accuracy_mode

    if max_speakers < 2 or not segments:
        return True
    if not is_accuracy_mode() and not _refine_after_segmented_enabled():
        return True
    if max(0, _env_int("DIARIZATION_MEGA_TURN_MAX_REFINES", 10)) == 0:
        return True
    min_audio_s = _env_float("DIARIZATION_MEGA_TURN_MIN_AUDIO_S", 600.0)
    return (
        not allow_short_audio
        and min_audio_s > 0
        and audio_duration_s < min_audio_s
    )


def _teardown_vram_between_refines() -> None:
    try:
        from backend import vram_state

        vram_state.teardown(aggressive=True)
    except ImportError:
        pass


def _same_speaker_runs(
    segments: list[dict],
    *,
    max_gap_s: float = 0.5,
) -> list[dict]:
    """Merge consecutive same-label diar segments into contiguous speaker runs."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda seg: seg["start"])
    runs: list[dict] = []
    run_start = float(ordered[0]["start"])
    run_end = float(ordered[0]["end"])
    run_speaker = str(ordered[0].get("speaker") or "")
    for seg in ordered[1:]:
        gap = float(seg["start"]) - run_end
        speaker = str(seg.get("speaker") or "")
        if speaker == run_speaker and gap <= max_gap_s:
            run_end = max(run_end, float(seg["end"]))
        else:
            runs.append({"start": run_start, "end": run_end, "speaker": run_speaker})
            run_start = float(seg["start"])
            run_end = float(seg["end"])
            run_speaker = speaker
    runs.append({"start": run_start, "end": run_end, "speaker": run_speaker})
    return runs


def _refine_long_diarization_spans(
    pipe,
    audio_path: str,
    segments: list[dict],
    kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
    audio_duration_s: float,
    *,
    mega_turn_threshold_s: float | None = None,
    allow_short_audio: bool = False,
) -> list[dict]:
    """Re-diarize spans where one speaker turn is implausibly long."""
    if _should_skip_mega_turn_refine(
        segments, max_speakers, audio_duration_s, allow_short_audio=allow_short_audio,
    ):
        return segments

    max_refines = max(0, _env_int("DIARIZATION_MEGA_TURN_MAX_REFINES", 10))
    threshold_s = _mega_turn_refine_threshold_s(mega_turn_threshold_s)
    run_gap = min(0.5, max(0.0, _postprocess_merge_gap_s()))
    if allow_short_audio:
        run_gap = max(
            run_gap,
            _env_float("ASR_TURN_GUIDED_MERGE_GAP_S", 0.35),
            _env_float("ASR_TURN_OUTPUT_MERGE_GAP_S", 0.0),
        )
    speaker_runs = _same_speaker_runs(segments, max_gap_s=run_gap)
    long_spans = sorted(
        (
            run for run in speaker_runs
            if run["end"] - run["start"] >= threshold_s
        ),
        key=lambda run: run["end"] - run["start"],
        reverse=True,
    )
    if not long_spans:
        return segments

    if allow_short_audio:
        # Reuse the warm pipeline — unload/reload between passes corrupts CUDA on 8 GB.
        active_pipe = pipe
    else:
        active_pipe = _reload_diarization_pipeline_for_refine(pipe)
    retry_params = _mega_turn_retry_params(adaptive_params)
    _instantiate_pipeline_params(active_pipe, retry_params, "mega-turn-retry")
    refined = list(segments)
    improved = False

    for index, span in enumerate(long_spans):
        if index >= max_refines:
            logger.info(
                "Mega-turn refine cap reached (%d); skipping %d remaining span(s).",
                max_refines,
                len(long_spans) - index,
            )
            break
        refined, accepted, _longest = _try_refine_long_span(
            active_pipe,
            audio_path,
            span,
            index,
            refined,
            kwargs,
            audio_duration_s,
            max_speakers,
            cpu_subpass=allow_short_audio
            and _env_bool("DIARIZATION_SHORT_AUDIO_MEGA_REFINE_CPU", True),
        )
        improved = improved or accepted
        if index + 1 < min(len(long_spans), max_refines):
            _teardown_vram_between_refines()

    if improved and adaptive_params:
        _instantiate_pipeline_params(active_pipe, adaptive_params, "adaptive-restore")
    return refined


def _dominant_speaker_label(
    durations: dict[str, float],
    *,
    min_share: float = 0.35,
) -> str | None:
    total_dur = sum(durations.values()) or 1.0
    dominant_label: str | None = None
    dominant_share = 0.0
    for spk, dur in durations.items():
        share = dur / total_dur
        if share > dominant_share:
            dominant_share = share
            dominant_label = spk
    if dominant_label is None or dominant_share < min_share:
        return None
    return dominant_label


def _best_brief_alt_speaker(
    similarity,
    label_idx: dict[str, int],
    dominant_label: str,
    durations: dict[str, float],
    *,
    brief_threshold_s: float = 30.0,
) -> str | None:
    dom_idx = label_idx.get(dominant_label)
    if dom_idx is None:
        return None
    best_alt = None
    best_sim = -1.0
    for alt_label, alt_idx in label_idx.items():
        if alt_label == dominant_label:
            continue
        if durations.get(alt_label, 0.0) >= brief_threshold_s:
            continue
        sim = float(similarity[dom_idx, alt_idx])
        if 0.68 <= sim < 0.80 and sim > best_sim:
            best_sim = sim
            best_alt = alt_label
    return best_alt


def _rescue_brief_dominant_segments(
    segments: list[dict],
    dominant_label: str,
    label_idx: dict[str, int],
    similarity,
    durations: dict[str, float],
) -> tuple[list[dict], int]:
    rescued: list[dict] = []
    next_id = 0
    max_span_s = _env_float("DIARIZATION_BRIEF_RESCUE_MAX_SPAN_S", 8.0)
    for seg in segments:
        if seg["speaker"] != dominant_label:
            rescued.append(seg)
            continue
        dur = seg["end"] - seg["start"]
        if dur > max_span_s:
            rescued.append(seg)
            continue
        best_alt = _best_brief_alt_speaker(
            similarity, label_idx, dominant_label, durations,
        )
        if best_alt is None:
            rescued.append(seg)
            continue
        new_label = f"BRIEF_{next_id}_{best_alt}"
        next_id += 1
        rescued.append({**seg, "speaker": new_label})
        logger.info(
            "Brief-speaker rescue: split %.1fs segment from %s (alt=%s).",
            dur,
            dominant_label,
            best_alt,
        )
    return rescued, next_id


def _recover_brief_speakers(
    segments: list[dict],
    max_speakers: int,
) -> list[dict]:
    """Split brief-voice clusters incorrectly absorbed into a dominant speaker."""
    from backend.asr_quality import is_accuracy_mode

    if not is_accuracy_mode() or max_speakers < 4 or not segments:
        return segments
    if _embedding_rescue_cache is None:
        return segments

    unique = len({s["speaker"] for s in segments})
    if unique >= max_speakers:
        return segments

    labels, embeddings = _embedding_rescue_cache
    import numpy as np

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or len(labels) < 2:
        return segments
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    similarity = unit @ unit.T
    label_idx = {label: i for i, label in enumerate(labels)}
    durations = _speaker_durations(segments)
    dominant_label = _dominant_speaker_label(durations)
    if dominant_label is None:
        return segments

    rescued, rescued_count = _rescue_brief_dominant_segments(
        segments, dominant_label, label_idx, similarity, durations,
    )
    if rescued_count == 0:
        return segments
    return _merge_similar_speaker_clusters(rescued, max_speakers)


def _intro_recovery_span_s() -> float:
    return max(60.0, _env_float("DIARIZATION_INTRO_RECOVERY_SPAN_S", 240.0))


def _intro_recovery_params() -> dict:
    """Aggressive VBx params for the meeting intro / roll-call window."""
    return {
        "segmentation": {"min_duration_off": 0.03},
        "clustering": {"threshold": 0.55, "Fa": 0.32, "Fb": 0.8},
    }


def _intro_recovery_end_s(audio_duration_s: float) -> float | None:
    intro_end = min(_intro_recovery_span_s(), audio_duration_s)
    if intro_end < 60.0 or audio_duration_s < intro_end + 30.0:
        return None
    return intro_end


def _intro_window_looks_dominant(
    intro_window: list[dict],
    intro_durations: dict[str, float],
) -> bool:
    total_intro = sum(intro_durations.values()) or 1.0
    dominant_share = (
        max(intro_durations.values()) / total_intro if intro_durations else 0.0
    )
    max_intro_seg = _max_segment_duration(intro_window) if intro_window else 0.0
    mega_t = _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 45.0)
    return dominant_share >= 0.38 or max_intro_seg >= mega_t * 0.85


def _should_skip_intro_recovery(
    segments: list[dict],
    max_speakers: int,
    intro_end: float,
) -> bool:
    from backend.asr_quality import is_accuracy_mode

    if not _env_bool("DIARIZATION_INTRO_RECOVERY", True):
        return True
    if not is_accuracy_mode() or max_speakers < 8 or not segments:
        return True
    unique = len({s["speaker"] for s in segments})
    intro_window = [seg for seg in segments if seg["start"] < intro_end - 0.01]
    intro_durations = _speaker_durations(intro_window)
    intro_dominant = _intro_window_looks_dominant(intro_window, intro_durations)
    return unique >= max_speakers and not intro_dominant


def _intro_recovery_kwargs(kwargs: dict, max_speakers: int) -> dict:
    intro_kwargs = dict(kwargs)
    extra = max(5, _overcluster_extra(max_speakers))
    if max_speakers > 0:
        intro_kwargs["num_speakers"] = max_speakers + extra
        intro_kwargs.pop("max_speakers", None)
    return intro_kwargs


def _diarize_intro_window(
    pipe,
    audio_path: str,
    intro_end: float,
    intro_kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
) -> list[dict]:
    active_pipe = _reload_diarization_pipeline_for_refine(pipe)
    _instantiate_pipeline_params(active_pipe, _intro_recovery_params(), "intro-recovery")
    try:
        intro_segments = _diarize_audio_span(
            active_pipe, audio_path, 0.0, intro_end, intro_kwargs,
        )
        return _merge_similar_speaker_clusters(intro_segments, max_speakers)
    finally:
        if adaptive_params:
            _instantiate_pipeline_params(active_pipe, adaptive_params, "adaptive-restore")


def _build_intro_recovery_candidate(
    segments: list[dict],
    intro_segments: list[dict],
    intro_end: float,
    max_speakers: int,
) -> list[dict]:
    rebranded = [
        {**seg, "speaker": f"INTRO_{idx}_{seg['speaker']}"}
        for idx, seg in enumerate(intro_segments)
    ]
    interval = {"start": 0.0, "end": intro_end}
    candidate = _replace_interval_segments(segments, interval, rebranded)
    return _merge_similar_speaker_clusters(candidate, max_speakers)


def _accept_intro_recovery_candidate(
    segments: list[dict],
    candidate: list[dict],
    intro_segments: list[dict],
    intro_end: float,
    unique: int,
    max_speakers: int,
    audio_duration_s: float,
) -> list[dict] | None:
    from engines.diarization_sampling import score_segments

    intro_unique = len({s["speaker"] for s in intro_segments})
    main_intro = [seg for seg in segments if seg["start"] < intro_end - 0.01]
    main_intro_unique = len({s["speaker"] for s in main_intro}) if main_intro else 0
    if intro_unique <= main_intro_unique:
        logger.info(
            "Intro recovery did not increase intro-window speakers (%d -> %d).",
            main_intro_unique,
            intro_unique,
        )
        return None

    old_score = score_segments(segments, audio_duration_s, max_speakers)
    new_score = score_segments(candidate, audio_duration_s, max_speakers)
    new_unique = len({s["speaker"] for s in candidate})
    if new_score > old_score + 0.01 or new_unique > unique:
        logger.info(
            "Intro recovery accepted: speakers %d->%d score %.3f->%.3f.",
            unique,
            new_unique,
            old_score,
            new_score,
        )
        return candidate

    logger.info("Intro recovery did not improve score; keeping first pass.")
    return None


def _recover_intro_speakers(
    pipe,
    audio_path: str,
    segments: list[dict],
    kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
    audio_duration_s: float,
) -> list[dict]:
    """Re-diarize the intro window when large meetings miss brief voices."""
    intro_end = _intro_recovery_end_s(audio_duration_s)
    if intro_end is None or _should_skip_intro_recovery(segments, max_speakers, intro_end):
        return segments

    unique = len({s["speaker"] for s in segments})
    logger.info(
        "Intro speaker recovery: %d/%d speakers detected; re-diarizing 0-%.0fs.",
        unique,
        max_speakers,
        intro_end,
    )

    intro_kwargs = _intro_recovery_kwargs(kwargs, max_speakers)
    intro_segments = _diarize_intro_window(
        pipe, audio_path, intro_end, intro_kwargs, adaptive_params, max_speakers,
    )
    if not intro_segments:
        return segments

    candidate = _build_intro_recovery_candidate(
        segments, intro_segments, intro_end, max_speakers,
    )
    accepted = _accept_intro_recovery_candidate(
        segments, candidate, intro_segments, intro_end, unique, max_speakers, audio_duration_s,
    )
    return accepted if accepted is not None else segments


def _remap_speakers(segments: list[dict]) -> dict:
    """Remap pyannote labels to SPEAKER_01..N by first chronological appearance."""
    first_seen: dict[str, float] = {}
    for segment in sorted(segments, key=lambda item: item["start"]):
        raw = segment["speaker"]
        if raw not in first_seen:
            first_seen[raw] = float(segment["start"])
    unique_raw = sorted(first_seen, key=lambda spk: (first_seen[spk], spk))
    speaker_map = {spk: f"SPEAKER_{i + 1:02d}" for i, spk in enumerate(unique_raw)}
    for segment in segments:
        segment["speaker"] = speaker_map[segment["speaker"]]
    return speaker_map


def _slice_audio_input(audio_input: dict, start_s: float, end_s: float) -> dict:
    """Slice a preloaded waveform dict for tune-window multi-sample."""
    if "waveform" in audio_input:
        tensor = audio_input["waveform"]
        sr = int(audio_input.get("sample_rate", 16000))
        start_i = max(0, int(start_s * sr))
        end_i = min(int(tensor.shape[-1]), int(end_s * sr))
        if end_i <= start_i:
            end_i = min(int(tensor.shape[-1]), start_i + sr)
        return {"waveform": tensor[..., start_i:end_i], "sample_rate": sr}

    waveform = audio_input["raw"]
    sr = int(audio_input["sampling_rate"])
    start_i = max(0, int(start_s * sr))
    end_i = min(len(waveform), int(end_s * sr))
    if end_i <= start_i:
        end_i = min(len(waveform), start_i + sr)
    return {"raw": waveform[start_i:end_i], "sampling_rate": sr}


def _execute_pyannote_pass(pipe, audio_input: dict, kwargs: dict) -> list[dict]:
    """Run one pyannote pass and return raw segments."""
    diarization = _run_pyannote(pipe, audio_input, kwargs)
    return _segments_from_diarization(diarization)


def _effective_diarization_segment_s(audio_duration_s: float) -> int:
    """Scale segment length up on very long files to reduce pyannote pass count."""
    from backend.asr_quality import is_accuracy_mode

    base = max(300, _env_int("DIARIZATION_SEGMENT_S", 600))
    if is_accuracy_mode() and audio_duration_s < 3600:
        base = min(base, 360)
    if not _env_bool("DIARIZATION_ADAPTIVE_SEGMENT_S", True):
        return base
    if audio_duration_s >= 10800:
        return max(base, 900) if not is_accuracy_mode() else min(base, 420)
    if audio_duration_s >= 7200:
        return max(base, 600) if not is_accuracy_mode() else min(base, 360)
    if audio_duration_s >= 3600:
        return min(base, 300)
    return base


def _count_diarization_segments(
    audio_duration_s: float,
    segment_s: int,
    overlap_s: int,
) -> int:
    """Return how many ffmpeg/pyannote segments a long file will need."""
    if audio_duration_s <= 0:
        return 0
    count = 0
    offset_s = 0.0
    while offset_s < audio_duration_s - 0.01:
        remaining = audio_duration_s - offset_s
        chunk_dur = min(float(segment_s), remaining)
        count += 1
        if remaining <= segment_s + 0.01:
            break
        offset_s += chunk_dur - overlap_s
    return count


def _refine_after_segmented_enabled() -> bool:
    """Mega-turn refinement re-runs pyannote on long spans after segmented pass."""
    from backend.asr_quality import is_accuracy_mode

    default = is_accuracy_mode()
    return _env_bool("DIARIZATION_REFINE_AFTER_SEGMENTED", default)


def _tune_segmented_diarization_params(
    pipe,
    audio_path: str,
    kwargs: dict,
    audio_duration_s: float,
    max_speakers: int,
    adaptive_params: dict | None,
    segment_s: int,
) -> dict | None:
    """Run multi-sample on a short window to pick hyperparameters for long files."""
    from engines.diarization_sampling import (
        multi_sample_sweep_enabled,
        select_best_diarization_params,
    )

    if not multi_sample_sweep_enabled() or max_speakers < 2:
        return adaptive_params

    from engines.diarization_sampling import tune_window_bounds

    bounds = tune_window_bounds(audio_duration_s)
    if bounds:
        tune_start, tune_end = bounds
        tune_dur = tune_end - tune_start
    else:
        tune_start = min(max(60.0, audio_duration_s * 0.08), max(0.0, audio_duration_s - 120.0))
        tune_dur = min(float(segment_s), 180.0, max(60.0, audio_duration_s - tune_start))
        tune_end = tune_start + tune_dur
    logger.info(
        "Tuning diarization params on sample window %.1fs-%.1fs (%.1fs) before segmented pass.",
        tune_start,
        tune_end,
        tune_dur,
    )

    def run_tune() -> list[dict]:
        return _diarize_audio_span(
            pipe, audio_path, tune_start, tune_start + tune_dur, kwargs,
        )

    tuned_params, winner, score = select_best_diarization_params(
        lambda params, label: _instantiate_pipeline_params(pipe, params, label),
        run_tune,
        tune_dur,
        max_speakers,
        adaptive_params,
        param_filter=lambda params: _filter_pipeline_params(pipe, params),
    )
    if tuned_params:
        _instantiate_pipeline_params(pipe, tuned_params, f"segmented-tune:{winner}")
        logger.info(
            "Segmented diarization using tuned params %s (sample score=%.3f).",
            winner,
            score,
        )
    return tuned_params or adaptive_params


def _diarize_segmented(
    pipe,
    audio_path: str,
    kwargs: dict,
    audio_duration_s: float,
    segment_s: int | None = None,
    adaptive_params: dict | None = None,
    max_speakers: int = 0,
) -> list[dict]:
    """Run pyannote on ffmpeg segments for very long files to limit RAM spikes."""
    import subprocess
    import tempfile

    if segment_s is None:
        segment_s = _effective_diarization_segment_s(audio_duration_s)
    overlap_s = max(0, _env_int("DIARIZATION_SEGMENT_OVERLAP_S", 60))
    total_segments = _count_diarization_segments(audio_duration_s, segment_s, overlap_s)
    logger.info(
        "Segmented diarization: duration=%.1fs segment=%ds overlap=%ds passes=%d",
        audio_duration_s,
        segment_s,
        overlap_s,
        total_segments,
    )
    adaptive_params = _tune_segmented_diarization_params(
        pipe, audio_path, kwargs, audio_duration_s, max_speakers, adaptive_params, segment_s,
    )
    merged: list[dict] = []
    offset_s = 0.0
    segment_idx = 0
    while offset_s < audio_duration_s - 0.01:
        remaining = audio_duration_s - offset_s
        chunk_dur = min(float(segment_s), remaining)
        segment_idx += 1
        logger.info(
            "Diarization segment %d/%d: offset=%.1fs duration=%.1fs",
            segment_idx,
            total_segments,
            offset_s,
            chunk_dur,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(offset_s), "-t", str(chunk_dur),
                "-i", audio_path,
                "-ac", "1",
                "-ar", str(max(16000, _env_int("DIARIZATION_PREPROCESS_SR", 16000))),
                tmp_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            audio_input = _prepare_audio_for_pyannote(tmp_path)
            try:
                try:
                    segments = _execute_pyannote_pass(pipe, audio_input, kwargs)
                except RuntimeError as exc:
                    if not _is_recoverable_cuda_error(str(exc)):
                        raise
                    logger.warning(
                        "Segmented diarization segment %d failed (%s); reloading pipeline.",
                        segment_idx,
                        exc,
                    )
                    import torch
                    from backend import vram_state

                    vram_state.teardown(aggressive=True)
                    _recover_cuda_after_failure(torch)
                    unload_model()
                    pipe = _get_diarization_pipeline()
                    _move_pipeline_to_inference_device(pipe)
                    segments = _execute_pyannote_pass(pipe, audio_input, kwargs)
            finally:
                del audio_input
                from backend import vram_state

                vram_state.teardown(aggressive=True)
                import torch

                _recover_cuda_after_failure(torch)
            for seg in segments:
                merged.append({
                    "start": seg["start"] + offset_s,
                    "end": seg["end"] + offset_s,
                    "speaker": f"{seg['speaker']}_S{segment_idx}",
                })
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if remaining <= segment_s + 0.01:
            break
        offset_s += chunk_dur - overlap_s
    merged = _align_segmented_speakers(merged)
    logger.info(
        "Segmented diarization complete: raw_segments=%d aligned_speakers=%d",
        len(merged),
        len({s["speaker"] for s in merged}),
    )
    return merged


def _diarization_ui_override(
    seg_threshold: float | None,
    seg_min_duration_off: float | None,
    clust_threshold: float | None,
    clust_min_size: int | None,
    vbx_fa: float | None = None,
    vbx_fb: float | None = None,
) -> bool:
    return any(
        value is not None
        for value in (
            seg_threshold, seg_min_duration_off, clust_threshold, clust_min_size,
            vbx_fa, vbx_fb,
        )
    )


def _configure_diarization_pipeline(
    pipe,
    ui_override: bool,
    seg_threshold: float | None,
    seg_min_duration_off: float | None,
    clust_threshold: float | None,
    clust_min_size: int | None,
    max_speakers: int,
    audio_duration_s: float,
    vbx_fa: float | None = None,
    vbx_fb: float | None = None,
) -> dict | None:
    if ui_override:
        _instantiate_pipeline_params(
            pipe,
            _override_params(
                seg_threshold,
                seg_min_duration_off,
                clust_threshold,
                clust_min_size,
                vbx_fa,
                vbx_fb,
            ),
            "UI override",
        )
        return None

    from backend.asr_quality import is_accuracy_mode

    if _env_bool("DIARIZATION_LOCK_PARAMS", False):
        return None

    adaptive_params: dict | None
    if is_accuracy_mode():
        adaptive_params = _long_audio_adaptive_params(max_speakers, audio_duration_s)
        if not adaptive_params:
            adaptive_params = _accuracy_mode_params(max_speakers, audio_duration_s)
    else:
        adaptive_params = _adaptive_pipeline_params(audio_duration_s, max_speakers)
        if max_speakers > 0:
            cap_params = _max_speaker_cap_params(max_speakers, audio_duration_s)
            adaptive_params = _merge_diarization_params(cap_params, adaptive_params)
    if adaptive_params:
        _instantiate_pipeline_params(pipe, adaptive_params, "adaptive")
    return adaptive_params


def _run_waveform_diarization_on_input(
    pipe,
    audio_input: dict,
    kwargs: dict,
    audio_duration_s: float,
    max_speakers: int,
    adaptive_params: dict | None,
    ui_override: bool,
    num_speakers: int,
    use_multi_sample: bool,
) -> list[dict]:
    from engines.diarization_sampling import (
        run_multi_sample_diarization,
        select_best_diarization_params,
        tune_window_bounds,
    )

    param_filter = lambda params: _filter_pipeline_params(pipe, params)
    tune_bounds = tune_window_bounds(audio_duration_s) if use_multi_sample else None
    if use_multi_sample and tune_bounds:
        tune_start, tune_end = tune_bounds
        tune_dur = tune_end - tune_start
        tune_input = _slice_audio_input(audio_input, tune_start, tune_end)
        tuned_params, winner, score = select_best_diarization_params(
            lambda params, label: _instantiate_pipeline_params(pipe, params, label),
            lambda: _execute_pyannote_pass(pipe, tune_input, kwargs),
            tune_dur,
            max_speakers,
            adaptive_params,
            param_filter=param_filter,
        )
        if tuned_params:
            _instantiate_pipeline_params(pipe, tuned_params, f"tune-window:{winner}")
        logger.info(
            "Long-audio tune window %.1fs-%.1fs selected %s (score=%.4f); running full pass.",
            tune_start,
            tune_end,
            winner,
            score,
        )
        segments = _merge_similar_speaker_clusters(
            _execute_pyannote_pass(pipe, audio_input, kwargs), max_speakers,
        )
    elif use_multi_sample:
        segments, winner, score = run_multi_sample_diarization(
            lambda params, label: _instantiate_pipeline_params(pipe, params, label),
            lambda: _execute_pyannote_pass(pipe, audio_input, kwargs),
            audio_duration_s,
            max_speakers,
            adaptive_params,
            param_filter=param_filter,
        )
        logger.info("Multi-sample diarization selected config %s (score=%.4f).", winner, score)
        segments = _merge_similar_speaker_clusters(segments, max_speakers)
    else:
        segments = _merge_similar_speaker_clusters(
            _execute_pyannote_pass(pipe, audio_input, kwargs), max_speakers,
        )
    return _maybe_retry_single_speaker(
        pipe, audio_input, kwargs, segments, adaptive_params,
        ui_override, num_speakers, max_speakers, audio_duration_s,
    )


def _run_waveform_diarization(
    pipe,
    audio_path: str,
    kwargs: dict,
    audio_duration_s: float,
    max_speakers: int,
    adaptive_params: dict | None,
    ui_override: bool,
    num_speakers: int,
) -> list[dict]:
    from engines.diarization_sampling import (
        multi_sample_preprocess_srs,
        multi_sample_sweep_enabled,
        score_segments,
    )
    from backend import vram_state

    use_multi_sample = multi_sample_sweep_enabled() and not ui_override
    preprocess_srs = multi_sample_preprocess_srs() if use_multi_sample else []
    if not preprocess_srs:
        preprocess_srs = [_env_int("DIARIZATION_PREPROCESS_SR", 16000)]

    from engines.diarization_sampling import (
        multi_sample_max_total_tries,
        multi_sample_max_tries,
    )

    logger.info(
        "Running diarization duration=%.1fs kwargs=%s multi_sample=%s preprocess_srs=%s "
        "budget=%d total (%d per SR) ...",
        audio_duration_s,
        kwargs or "(auto)",
        use_multi_sample,
        preprocess_srs,
        multi_sample_max_total_tries() if use_multi_sample else 0,
        multi_sample_max_tries() if use_multi_sample else 0,
    )

    best_segments: list[dict] = []
    best_score = -1.0
    best_label = "none"

    for sr in preprocess_srs:
        audio_input = _prepare_audio_for_pyannote(audio_path, preprocess_sr=sr)
        try:
            segments = _run_waveform_diarization_on_input(
                pipe,
                audio_input,
                kwargs,
                audio_duration_s,
                max_speakers,
                adaptive_params,
                ui_override,
                num_speakers,
                use_multi_sample,
            )
            if len(preprocess_srs) > 1:
                score = score_segments(segments, audio_duration_s, max_speakers)
                n_spk = len({s["speaker"] for s in segments})
                logger.info(
                    "Diarization preprocess SR=%d Hz: segments=%d speakers=%d score=%.4f",
                    sr,
                    len(segments),
                    n_spk,
                    score,
                )
                if score > best_score:
                    best_score = score
                    best_segments = segments
                    best_label = f"sr={sr}"
            else:
                best_segments = segments
        finally:
            del audio_input
            vram_state.teardown(aggressive=False)

    if len(preprocess_srs) > 1:
        logger.info(
            "Multi-SR diarization winner: %s (score=%.4f, segments=%d)",
            best_label,
            best_score,
            len(best_segments),
        )

    return best_segments


def _finalize_diarization_segments(
    segments: list[dict],
    num_speakers: int,
    max_speakers: int,
    *,
    postprocess: bool = True,
) -> list[dict]:
    unique_raw = sorted({s["speaker"] for s in segments})
    logger.info(
        "Pyannote returned %d segments, %d unique speaker(s): %s",
        len(segments), len(unique_raw), unique_raw,
    )
    if not segments:
        logger.warning("Diarization produced no segments; transcript will have no speaker labels.")
        return segments
    if len(unique_raw) == 1 and num_speakers > 1:
        logger.warning(
            "Pyannote detected only 1 speaker despite exact speaker hint (num=%d).",
            num_speakers,
        )
    elif len(unique_raw) == 1 and max_speakers > 1:
        logger.warning(
            "Pyannote detected only 1 speaker but max_speakers=%d.",
            max_speakers,
        )
    if postprocess:
        segments = _postprocess_diarization_segments(segments, max_speakers)
    speaker_map = _remap_speakers(segments)
    logger.info("Diarization complete: %d segments, speaker map: %s", len(segments), speaker_map)
    return segments


def _maybe_refine_short_audio_spans(
    pipe,
    audio_path: str,
    segments: list[dict],
    kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
    audio_duration_s: float,
) -> list[dict]:
    """CPU mega-turn refine after recovery — targets merged same-speaker runs."""
    if max(0, _env_int("DIARIZATION_MEGA_TURN_MAX_REFINES", 10)) == 0:
        return segments
    if not _env_bool("DIARIZATION_SHORT_AUDIO_MEGA_REFINE", False):
        return segments
    short_max_s = _env_float("DIARIZATION_MEGA_TURN_SHORT_AUDIO_S", 600.0)
    if short_max_s > 0 and audio_duration_s >= short_max_s:
        return segments
    mega_threshold = min(
        30.0,
        max(18.0, _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 45.0) * 0.85),
    )
    return _refine_long_diarization_spans(
        pipe,
        audio_path,
        segments,
        kwargs,
        adaptive_params,
        max_speakers,
        audio_duration_s,
        mega_turn_threshold_s=mega_threshold,
        allow_short_audio=True,
    )


def diarize(
    audio_path: str, num_speakers: int = 0,
    max_speakers: int = 0,
    audio_duration_s: float = 0.0,
    min_speakers_hint: int = 0,
    seg_threshold: float | None = None,
    seg_min_duration_off: float | None = None,
    clust_threshold: float | None = None,
    clust_min_size: int | None = None,
    vbx_fa: float | None = None,
    vbx_fb: float | None = None,
) -> list[dict]:
    """Run speaker diarization on audio file.

    Returns list of {"start": float, "end": float, "speaker": str}
    sorted by start time.
    """
    from backend import vram_state

    vram_state.log_phase("diarize", before=True)
    with _diarization_lock:
        pipe = _get_diarization_pipeline()
        _move_pipeline_to_inference_device(pipe)

        ui_override = _diarization_ui_override(
            seg_threshold, seg_min_duration_off, clust_threshold, clust_min_size,
            vbx_fa, vbx_fb,
        )
        adaptive_params = _configure_diarization_pipeline(
            pipe,
            ui_override,
            seg_threshold,
            seg_min_duration_off,
            clust_threshold,
            clust_min_size,
            max_speakers,
            audio_duration_s,
            vbx_fa,
            vbx_fb,
        )

        kwargs = _build_diarize_kwargs(
            num_speakers, max_speakers, audio_duration_s, min_speakers_hint,
        )
        # Single-pass global clustering up to 2 h: segmented mode re-clusters
        # each chunk independently and chains labels through overlap zones,
        # which fragments/merges speakers on multi-speaker meetings. One pass
        # over 2 h of 16 kHz mono fits comfortably in RAM/VRAM.
        segment_min_s = max(0, _env_int("DIARIZATION_SEGMENT_LONG_AUDIO_MIN_S", 7200))
        use_segmented = (
            audio_duration_s > segment_min_s
            and _env_bool("DIARIZATION_SEGMENT_LONG_AUDIO", True)
        )

        if use_segmented:
            logger.info(
                "Using segmented diarization for long audio (%.1fs > %ds).",
                audio_duration_s,
                segment_min_s,
            )
            segments = _diarize_segmented(
                pipe, audio_path, kwargs, audio_duration_s,
                adaptive_params=adaptive_params,
                max_speakers=max_speakers,
            )
            if _refine_after_segmented_enabled():
                segments = _refine_long_diarization_spans(
                    pipe, audio_path, segments, kwargs, adaptive_params,
                    max_speakers, audio_duration_s,
                )
            else:
                logger.info(
                    "Skipping mega-turn diarization refine after segmented pass "
                    "(DIARIZATION_REFINE_AFTER_SEGMENTED=false).",
                )
            segments = _maybe_retry_dominant_diarization(
                pipe,
                audio_path,
                segments,
                kwargs,
                adaptive_params,
                max_speakers,
                audio_duration_s,
                min_speakers_hint,
            )
        else:
            segments = _run_waveform_diarization(
                pipe, audio_path, kwargs, audio_duration_s, max_speakers,
                adaptive_params, ui_override, num_speakers,
            )

        segments = _recover_intro_speakers(
            pipe, audio_path, segments, kwargs, adaptive_params,
            max_speakers, audio_duration_s,
        )
        segments = _recover_brief_speakers(segments, max_speakers)
        segments = _postprocess_diarization_segments(segments, max_speakers)
        segments = _maybe_refine_short_audio_spans(
            pipe,
            audio_path,
            segments,
            kwargs,
            adaptive_params,
            max_speakers,
            audio_duration_s,
        )
        segments = _finalize_diarization_segments(
            segments, num_speakers, max_speakers, postprocess=False,
        )
        _record_inference_device()
        vram_state.log_phase("diarize", before=False)
        return segments


def _maybe_retry_single_speaker(
    pipe,
    audio_input: dict,
    kwargs: dict,
    segments: list[dict],
    adaptive_params: dict | None,
    ui_override: bool,
    num_speakers: int,
    max_speakers: int,
    audio_duration_s: float,
) -> list[dict]:
    """Second pass with looser clustering when only one speaker was detected."""
    from backend.asr_quality import is_accuracy_mode

    unique_raw = sorted({s["speaker"] for s in segments})
    skip_for_override = ui_override and not is_accuracy_mode()
    exact = _env_bool("DIARIZATION_EXACT_NUM_SPEAKERS", False)
    if (
        skip_for_override
        or (exact and num_speakers > 0)
        or len(unique_raw) != 1
        or max_speakers < 2
        or audio_duration_s < 5.0
    ):
        return segments

    retry_params = _retry_pipeline_params(adaptive_params)
    _instantiate_pipeline_params(pipe, retry_params, "retry")
    logger.info("Diarization retry: only 1 speaker detected; re-running with looser clustering.")
    retry_segments = _postprocess_diarization_segments(
        _execute_pyannote_pass(pipe, audio_input, kwargs),
        max_speakers,
    )
    retry_unique = sorted({s["speaker"] for s in retry_segments})
    if len(retry_unique) > len(unique_raw):
        if max_speakers > 0 and len(retry_unique) > max_speakers:
            logger.info(
                "Diarization retry found %d speakers (> max %d); keeping first pass.",
                len(retry_unique),
                max_speakers,
            )
            return segments
        logger.info(
            "Diarization retry improved speaker count: 1 -> %d (%s)",
            len(retry_unique),
            retry_unique,
        )
        return retry_segments
    logger.info("Diarization retry did not increase speaker count; keeping first pass.")
    return segments


def _maybe_retry_dominant_diarization(
    pipe,
    audio_path: str,
    segments: list[dict],
    kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
    audio_duration_s: float,
    min_speakers_hint: int = 0,
) -> list[dict]:
    """Re-run mega-turn refine when one speaker owns most of the timeline."""
    from engines.diarization_sampling import score_segments

    if max_speakers < 2 or not segments or audio_duration_s < 30.0:
        return segments

    dominance, unique = _dominant_speaker_share(segments)
    min_expected = min_speakers_hint if min_speakers_hint > 0 else 2
    ratio_threshold = _env_float("DIARIZATION_DOMINANCE_RETRY_RATIO", 0.82)
    needs_retry = (
        unique < min_expected
        or (unique >= 2 and dominance > ratio_threshold)
    )
    if not needs_retry:
        return segments

    logger.warning(
        "Diarization dominance retry: share=%.1f%% speakers=%d (threshold=%.0f%%, min=%d).",
        dominance * 100.0,
        unique,
        ratio_threshold * 100.0,
        min_expected,
    )
    aggressive_threshold = min(
        30.0,
        _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 45.0) * 0.65,
    )
    refined = _refine_long_diarization_spans(
        pipe,
        audio_path,
        segments,
        kwargs,
        adaptive_params,
        max_speakers,
        audio_duration_s,
        mega_turn_threshold_s=aggressive_threshold,
        allow_short_audio=True,
    )
    new_dominance, new_unique = _dominant_speaker_share(refined)
    old_score = score_segments(segments, audio_duration_s, max_speakers)
    new_score = score_segments(refined, audio_duration_s, max_speakers)
    improved = (
        new_unique > unique
        or new_dominance < dominance - 0.03
        or new_score > old_score + 0.02
    )
    if improved:
        logger.info(
            "Dominance retry accepted: speakers %d->%d share %.1f%%->%.1f%% score %.3f->%.3f.",
            unique,
            new_unique,
            dominance * 100.0,
            new_dominance * 100.0,
            old_score,
            new_score,
        )
        return refined
    logger.info("Dominance retry did not improve diarization; keeping first pass.")
    return segments


def _find_speaker(start: float | None, end: float | None, segments: list[dict]) -> str:
    """Return the speaker with greatest time overlap in [start, end].

    Handles None end time (last segment in a chunk window) by using a
    short lookahead window, and falls back to nearest-midpoint matching
    when no segment overlaps the chunk at all.
    """
    if start is None or not segments:
        return segments[0]["speaker"] if segments else "SPEAKER"

    # When end is missing, use a tiny lookahead so overlap math still works
    eff_end = end if (end is not None and end > start) else start + 0.02

    best_speaker = segments[0]["speaker"]
    best_overlap = 0.0

    for seg in segments:
        overlap = max(0.0, min(eff_end, seg["end"]) - max(start, seg["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    # No overlap (gap between speaker turns): pick segment whose midpoint is
    # closest to the chunk midpoint instead of silently falling back.
    if best_overlap < 1e-9:
        mid = (start + eff_end) / 2
        best_speaker = min(
            segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - mid)
        )["speaker"]

    return best_speaker


def _dedup_repetitions(text: str) -> str:
    """Collapse Whisper hallucination loops in speaker-turn text."""
    from engines.text_cleanup import clean_transcript_text

    return clean_transcript_text(text)


def _flush_speaker_group(
    lines: list[str], speaker: str | None, words: list[str],
    start: float | None, end: float | None,
    diarization_segments: list[dict] | None = None,
) -> None:
    """Append one completed speaker turn to lines."""
    if not words or speaker is None:
        return
    text = _dedup_repetitions(" ".join(words))
    if not text:
        return
    if start is None and diarization_segments:
        for seg in diarization_segments:
            if seg["speaker"] == speaker:
                start = seg["start"]
                end = seg["end"]
                break
        if start is None:
            start, end = _segment_bounds_for_speaker(speaker, diarization_segments)
    ts_prefix = (
        f"[{_fmt_ts(start)} → {_fmt_ts(end)}] " if start is not None else ""
    )
    lines.append(f"{ts_prefix}[{speaker}]: {text}")


def _segment_bounds_for_speaker(speaker: str, segments: list[dict]) -> tuple[float | None, float | None]:
    """Return earliest start / latest end for one speaker label."""
    starts = [seg["start"] for seg in segments if seg["speaker"] == speaker]
    ends = [seg["end"] for seg in segments if seg["speaker"] == speaker]
    if not starts:
        return None, None
    return min(starts), max(ends)


def _estimate_chunk_ts(
    chunk_idx: int, total_chunks: int, total_dur: float,
) -> tuple[float, float]:
    """Estimate (start, end) for a chunk when timestamps are unavailable."""
    c_start = (chunk_idx / total_chunks) * total_dur
    c_end = ((chunk_idx + 1) / total_chunks) * total_dur
    return c_start, c_end


def _chunk_ts_out_of_bounds(
    start: float | None, end: float | None, total_dur: float,
) -> bool:
    if total_dur <= 0:
        return False
    tolerance_s = _env_float("ASR_TIMESTAMP_AUDIO_END_TOLERANCE_S", 30.0)
    if start is None:
        return True
    if start < 0 or start > total_dur + tolerance_s:
        return True
    if end is not None and (end < start or end > total_dur + tolerance_s):
        return True
    return False


def _bounded_chunk_ts(
    start: float | None,
    end: float | None,
    chunk_idx: int,
    total_chunks: int,
    total_dur: float,
) -> tuple[float | None, float | None, bool]:
    if total_dur <= 0:
        return start, end, False
    if _chunk_ts_out_of_bounds(start, end, total_dur):
        est_start, est_end = _estimate_chunk_ts(chunk_idx, total_chunks, total_dur)
        return est_start, est_end, True
    bounded_start = max(0.0, min(float(start), total_dur)) if start is not None else None
    if end is None:
        bounded_end = None
    else:
        bounded_end = max(bounded_start or 0.0, min(float(end), total_dur))
    return bounded_start, bounded_end, False


def _chunk_ts_for_assignment(
    chunk: dict,
    chunk_idx: int,
    all_ts_none: bool,
    total_chunks: int,
    total_dur: float,
) -> tuple[float | None, float | None, bool]:
    ts = chunk.get("timestamp")
    if ts is not None and not _ts_is_none(ts):
        c_start, c_end = ts[0], ts[1]
    else:
        c_start, c_end = None, None

    if (all_ts_none or c_start is None) and total_dur > 0 and total_chunks > 0:
        c_start, c_end = _estimate_chunk_ts(chunk_idx, total_chunks, total_dur)

    if total_dur <= 0 or total_chunks <= 0:
        return c_start, c_end, False
    return _bounded_chunk_ts(c_start, c_end, chunk_idx, total_chunks, total_dur)


def _ts_is_none(ts) -> bool:
    """Return True when a Whisper timestamp value is effectively absent.

    Handles both the plain ``None`` case and the ``(None, None)`` / ``(None, end)``
    tuple that some transformers versions emit when return_timestamps is not
    fully honoured.
    """
    if ts is None:
        return True
    if isinstance(ts, (tuple, list)) and len(ts) == 2 and ts[0] is None:
        return True
    return False


def _format_plain(chunks: list[dict]) -> str:
    """Format Whisper chunks as plain timestamped lines with no speaker labels.

    Used as fallback when diarization_segments is empty.
    """
    from engines.text_cleanup import clean_transcript_text

    lines = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        text = clean_transcript_text(text)
        if not text:
            continue
        ts = chunk.get("timestamp")
        c_start = ts[0] if (ts and ts[0] is not None) else None
        c_end = ts[1] if (ts and len(ts) > 1 and ts[1] is not None) else None
        if c_start is not None:
            lines.append(f"[{_fmt_ts(c_start)} \u2192 {_fmt_ts(c_end)}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines) if lines else _NO_SPEECH


def _overlapping_speaker_turns(
    start: float | None, end: float | None, segments: list[dict],
) -> list[dict]:
    """Return diarization turns overlapping one ASR chunk, clipped to chunk bounds."""
    if start is None or end is None or end <= start:
        return []
    turns = []
    for seg in segments:
        overlap_start = max(start, seg["start"])
        overlap_end = min(end, seg["end"])
        if overlap_end - overlap_start > 0.05:
            turns.append({
                "start": overlap_start,
                "end": overlap_end,
                "speaker": seg["speaker"],
            })
    return turns


def _significant_turns_for_split(
    turns: list[dict],
    span: float,
) -> list[dict]:
    min_rel = _env_float("DIARIZATION_SPLIT_MIN_TURN_FRACTION", 0.03)
    min_abs_s = _env_float("DIARIZATION_SPLIT_MIN_TURN_S", 1.5)
    significant = [
        turn for turn in turns
        if (turn["end"] - turn["start"]) >= min_abs_s
        and (turn["end"] - turn["start"]) / span >= min_rel
    ]
    return significant if len(significant) > 1 else []


def _next_unit_cursor(
    idx: int,
    turns: list[dict],
    units_len: int,
    cursor: int,
    span_start: float,
    duration: float,
) -> int:
    if idx == len(turns) - 1:
        return units_len
    fraction = (turns[idx]["end"] - span_start) / duration
    return min(units_len, max(cursor + 1, round(fraction * units_len)))


def _unit_slice_for_turn(
    units: list[str],
    cursor: int,
    next_cursor: int,
    use_chars: bool,
) -> str:
    if use_chars:
        return "".join(units[cursor:next_cursor]).strip()
    return " ".join(units[cursor:next_cursor]).strip()


def _split_text_across_turns(
    text: str, turns: list[dict], max_speakers: int = 0,
) -> list[tuple[float, float, str, str]]:
    """Split a long chunk across overlapping speakers (words or Thai characters)."""
    words = text.split()
    use_chars = len(words) <= 1 and len(text.strip()) > 1
    units: list[str] = list(text.strip()) if use_chars else words
    if len(turns) <= 1 or len(units) < len(turns):
        return []
    unique_turn_speakers = {t["speaker"] for t in turns}
    if max_speakers > 0 and len(unique_turn_speakers) > max_speakers:
        return []
    span = max(0.001, turns[-1]["end"] - turns[0]["start"])
    significant = _significant_turns_for_split(turns, span)
    if not significant:
        return []
    turns = significant

    span_start = turns[0]["start"]
    duration = max(0.001, turns[-1]["end"] - span_start)
    pieces: list[tuple[float, float, str, str]] = []
    cursor = 0

    for idx, turn in enumerate(turns):
        next_cursor = _next_unit_cursor(
            idx, turns, len(units), cursor, span_start, duration,
        )
        piece = _unit_slice_for_turn(units, cursor, next_cursor, use_chars)
        if piece:
            pieces.append((turn["start"], turn["end"], piece, turn["speaker"]))
        cursor = next_cursor
        if cursor >= len(units):
            break

    return pieces


def _chunk_assignment_pieces(
    chunk: dict,
    chunk_idx: int,
    all_ts_none: bool,
    total_chunks: int,
    total_dur: float,
    diarization_segments: list[dict],
    max_speakers: int,
) -> tuple[list[tuple[float, float, str, str]], int, int]:
    """Return split pieces, fixed-timestamp count, and next chunk index."""
    text = chunk.get("text", "").strip()
    if not text:
        return [], 0, chunk_idx
    c_start, c_end, fixed = _chunk_ts_for_assignment(
        chunk, chunk_idx, all_ts_none, total_chunks, total_dur,
    )
    turns = _overlapping_speaker_turns(c_start, c_end, diarization_segments)
    split_pieces = _split_text_across_turns(text, turns, max_speakers)
    if split_pieces:
        return split_pieces, fixed, chunk_idx + 1
    speaker = _find_speaker(c_start, c_end, diarization_segments)
    return [(c_start, c_end, text, speaker)], fixed, chunk_idx + 1


def _iter_chunks(
    chunks: list[dict],
    diarization_segments: list[dict],
    all_ts_none: bool,
    total_chunks: int,
    total_dur: float,
    max_speakers: int = 0,
):
    """Yield (c_start, c_end, text, speaker) for each non-empty chunk."""
    chunk_idx = 0
    fixed_timestamps = 0
    for chunk in chunks:
        pieces, fixed, chunk_idx = _chunk_assignment_pieces(
            chunk,
            chunk_idx,
            all_ts_none,
            total_chunks,
            total_dur,
            diarization_segments,
            max_speakers,
        )
        fixed_timestamps += fixed
        yield from pieces

    if fixed_timestamps:
        logger.warning(
            "assign_speakers repaired %d ASR chunk timestamp(s) outside audio duration %.1fs.",
            fixed_timestamps,
            total_dur,
        )


_LINE_TS_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2}) → (\d{2}):(\d{2}):(\d{2})\] \[(\S+)\]: (.*)$"
)


def _parse_hms(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return 0.0
    return hours * 3600 + minutes * 60 + seconds


def _transcript_merge_gap_s() -> float:
    try:
        return max(0.0, float(os.getenv("DIARIZATION_TRANSCRIPT_MERGE_GAP_S", "1.25")))
    except ValueError:
        return 1.25


def _assign_turn_merge_gap_s() -> float:
    """Merge adjacent diarization turns before ASR alignment (wider than line merge)."""
    try:
        return max(0.0, float(os.getenv("DIARIZATION_ASSIGN_TURN_MERGE_GAP_S", "2.0")))
    except ValueError:
        return 8.0


def _novel_turn_text(previous: str, new: str) -> str:
    """Return only text not already covered by the previous same-speaker turn."""
    previous = previous.strip()
    new = new.strip()
    if not new:
        return ""
    if not previous:
        return new
    if new == previous:
        return ""
    if new.startswith(previous):
        return new[len(previous):].strip()
    prev_words = previous.split()
    new_words = new.split()
    if len(new_words) >= len(prev_words) and new_words[: len(prev_words)] == prev_words:
        return " ".join(new_words[len(prev_words) :]).strip()
    if len(prev_words) >= len(new_words) and prev_words[: len(new_words)] == new_words:
        return ""
    return new


def _join_turn_texts(previous: str, new: str) -> str:
    novel = _novel_turn_text(previous, new)
    if not novel:
        return previous
    return f"{previous} {novel}".strip()


def _merge_transcript_lines(lines: list[str], max_gap_s: float | None = None) -> list[str]:
    """Merge consecutive lines from the same speaker into one continuous turn."""
    if max_gap_s is None:
        max_gap_s = _transcript_merge_gap_s()
    if len(lines) < 2:
        return lines
    merged: list[str] = []
    for line in lines:
        match = _LINE_TS_RE.match(line.strip())
        if not match:
            merged.append(line)
            continue
        if not merged:
            merged.append(line)
            continue
        prev_match = _LINE_TS_RE.match(merged[-1].strip())
        if not prev_match:
            merged.append(line)
            continue
        p_start_h, p_start_m, p_start_s, p_end_h, p_end_m, p_end_s, p_speaker, p_text = (
            prev_match.groups()
        )
        s_start_h, s_start_m, s_start_s, s_end_h, s_end_m, s_end_s, s_speaker, s_text = (
            match.groups()
        )
        if p_speaker != s_speaker:
            merged.append(line)
            continue
        gap = _parse_hms(f"{s_start_h}:{s_start_m}:{s_start_s}") - _parse_hms(
            f"{p_end_h}:{p_end_m}:{p_end_s}"
        )
        if gap > max_gap_s:
            merged.append(line)
            continue
        merged[-1] = (
            f"[{p_start_h}:{p_start_m}:{p_start_s} → {s_end_h}:{s_end_m}:{s_end_s}] "
            f"[{p_speaker}]: {_join_turn_texts(p_text.strip(), s_text.strip())}"
        )
    return merged


def _text_unit_count(text: str) -> int:
    """Count assignable units: whitespace tokens, or characters for unsegmented Thai."""
    words = text.split()
    if len(words) > 1:
        return len(words)
    return len(text.strip())


def _slice_text_units(text: str, i0: int, i1: int) -> str:
    """Return text slice for unit range [i0, i1)."""
    words = text.split()
    if len(words) > 1:
        i0 = max(0, min(len(words), i0))
        i1 = max(i0, min(len(words), i1))
        return " ".join(words[i0:i1]).strip()
    chars = list(text.strip())
    i0 = max(0, min(len(chars), i0))
    i1 = max(i0, min(len(chars), i1))
    return "".join(chars[i0:i1]).strip()


def _slice_text_for_interval(
    text: str,
    chunk_start: float,
    chunk_end: float,
    interval_start: float,
    interval_end: float,
) -> str:
    """Return the portion of *text* that falls inside one time interval."""
    unit_count = _text_unit_count(text)
    if unit_count == 0:
        return ""
    chunk_dur = max(0.001, chunk_end - chunk_start)
    start_frac = max(0.0, min(1.0, (interval_start - chunk_start) / chunk_dur))
    end_frac = max(start_frac, min(1.0, (interval_end - chunk_start) / chunk_dur))
    if end_frac <= start_frac:
        return ""
    i0 = int(round(start_frac * unit_count))
    i1 = int(round(end_frac * unit_count))
    i0 = max(0, min(unit_count - 1, i0))
    i1 = max(i0 + 1, min(unit_count, i1))
    return _slice_text_units(text, i0, i1)


def _build_asr_timeline(chunks: list[dict], total_dur: float) -> list[dict]:
    """Normalise non-empty ASR chunks to absolute ``{start, end, text}`` records."""
    non_empty = [c for c in chunks if c.get("text", "").strip()]
    total_chunks = len(non_empty)
    if total_chunks == 0:
        return []

    all_ts_none = all(_ts_is_none(c.get("timestamp")) for c in non_empty)
    timeline: list[dict] = []
    for chunk_idx, chunk in enumerate(non_empty):
        text = chunk.get("text", "").strip()
        c_start, c_end, _ = _chunk_ts_for_assignment(
            chunk,
            chunk_idx,
            all_ts_none,
            total_chunks,
            total_dur,
        )
        if c_start is not None and c_end is None:
            est_dur = max(0.5, len(text) / 12.0)
            c_end = min(total_dur, c_start + est_dur) if total_dur > 0 else c_start + est_dur
        if c_start is None and total_dur > 0:
            c_start, c_end = _estimate_chunk_ts(chunk_idx, total_chunks, total_dur)
        if c_start is not None and c_end is not None and c_end > c_start:
            timeline.append({"start": float(c_start), "end": float(c_end), "text": text})
    return timeline


def _word_indices_for_overlap(
    cs: float,
    ce: float,
    overlap_start: float,
    overlap_end: float,
    unit_count: int,
) -> tuple[int, int]:
    chunk_dur = max(0.001, ce - cs)
    start_frac = max(0.0, min(1.0, (overlap_start - cs) / chunk_dur))
    end_frac = max(start_frac, min(1.0, (overlap_end - cs) / chunk_dur))
    i0 = int(round(start_frac * unit_count))
    i1 = int(round(end_frac * unit_count))
    i0 = max(0, min(unit_count - 1, i0))
    i1 = max(i0 + 1, min(unit_count, i1))
    return i0, i1


def _exclusive_units_for_overlap(
    text: str,
    consumed: dict[int, set[int]],
    ti: int,
    i0: int,
    i1: int,
) -> str | None:
    bucket = consumed.setdefault(ti, set())
    taken = [idx for idx in range(i0, i1) if idx not in bucket]
    if not taken:
        return None
    bucket.update(taken)
    return _slice_text_units(text, taken[0], taken[-1] + 1)


def _exclusive_words_for_overlap(
    text: str,
    consumed: dict[int, set[int]],
    ti: int,
    i0: int,
    i1: int,
) -> str | None:
    return _exclusive_units_for_overlap(text, consumed, ti, i0, i1)


def _collect_text_for_interval(
    timeline: list[dict],
    iv_start: float,
    iv_end: float,
    consumed: dict[int, set[int]] | None = None,
) -> str:
    """Collect ASR text overlapping one diarization turn (exclusive unit assignment)."""
    text, _, _ = _collect_text_and_bounds_for_interval(
        timeline, iv_start, iv_end, consumed,
    )
    return text


def _chunk_overlap_slice(
    item: dict,
    ti: int,
    iv_start: float,
    iv_end: float,
    min_overlap_s: float,
    consumed: dict[int, set[int]] | None,
) -> tuple[tuple[float, str] | None, float | None, float | None]:
    """Return (text part, bound_start, bound_end) for one timeline chunk."""
    cs, ce, text = item["start"], item["end"], item["text"]
    overlap_start = max(cs, iv_start)
    overlap_end = min(ce, iv_end)
    if overlap_end - overlap_start < min_overlap_s:
        return None, None, None
    unit_count = _text_unit_count(text)
    if unit_count == 0:
        return None, None, None
    i0, i1 = _word_indices_for_overlap(cs, ce, overlap_start, overlap_end, unit_count)
    if consumed is None:
        sliced = _slice_text_for_interval(text, cs, ce, overlap_start, overlap_end)
        if not sliced:
            return None, None, None
        return (overlap_start, sliced), overlap_start, overlap_end
    sliced = _exclusive_words_for_overlap(text, consumed, ti, i0, i1)
    if not sliced:
        return None, None, None
    word_start = cs + (i0 / max(1, unit_count)) * (ce - cs)
    word_end = cs + (i1 / max(1, unit_count)) * (ce - cs)
    return (overlap_start, sliced), word_start, word_end


def _collect_text_and_bounds_for_interval(
    timeline: list[dict],
    iv_start: float,
    iv_end: float,
    consumed: dict[int, set[int]] | None = None,
) -> tuple[str, float | None, float | None]:
    """Collect ASR text and timestamp bounds from overlapping timeline chunks."""
    min_overlap_s = _env_float("DIARIZATION_MIN_OVERLAP_S", 0.04)
    parts: list[tuple[float, str]] = []
    bound_start: float | None = None
    bound_end: float | None = None
    for ti, item in enumerate(timeline):
        part, chunk_start, chunk_end = _chunk_overlap_slice(
            item, ti, iv_start, iv_end, min_overlap_s, consumed,
        )
        if part is None:
            continue
        parts.append(part)
        if chunk_start is not None:
            bound_start = chunk_start if bound_start is None else min(bound_start, chunk_start)
        if chunk_end is not None:
            bound_end = chunk_end if bound_end is None else max(bound_end, chunk_end)
    parts.sort(key=lambda pair: pair[0])
    text_out = " ".join(fragment for _, fragment in parts).strip()
    return text_out, bound_start, bound_end


def _turns_for_transcript(
    diarization_segments: list[dict], max_speakers: int,
) -> list[dict]:
    """Prepare diarization turns for transcript output."""
    from backend.asr_quality import is_accuracy_mode

    segments = [dict(seg) for seg in diarization_segments]
    if max_speakers > 0:
        segments = _enforce_max_speakers(segments, max_speakers)
    if is_accuracy_mode():
        merge_gap = 0.0
    else:
        merge_gap = max(_transcript_merge_gap_s(), _assign_turn_merge_gap_s())
    return _merge_adjacent_same_speaker(segments, max_gap_s=merge_gap)


def _line_start_seconds(line: str) -> float:
    match = _LINE_TS_RE.match(line.strip())
    if not match:
        return 0.0
    h, m, s, _, _, _, _, _ = match.groups()
    return _parse_hms(f"{h}:{m}:{s}")


def _previous_same_speaker_line_text(lines: list[str], speaker: str) -> str:
    """Return body text from the most recent output line for this speaker."""
    for line in reversed(lines):
        match = _LINE_TS_RE.match(line.strip())
        if match and match.group(7) == speaker:
            return match.group(8).strip()
    return ""


def _append_turn_line(
    lines: list[str],
    speaker: str,
    start: float,
    end: float,
    text: str,
) -> None:
    text = _dedup_repetitions(text)
    text = _novel_turn_text(_previous_same_speaker_line_text(lines, speaker), text)
    if not text:
        return
    lines.append(
        f"[{_fmt_ts(start)} → {_fmt_ts(end)}] [{speaker}]: {text}"
    )


def _timeline_chunk_covered(cs: float, ce: float, turns: list[dict]) -> bool:
    return any(
        max(0.0, min(ce, turn["end"]) - max(cs, turn["start"])) >= 0.08
        for turn in turns
    )


def _orphan_text_for_chunk(item: dict, consumed: dict[int, set[int]], ti: int) -> str:
    text = item["text"].strip()
    unit_count = _text_unit_count(text)
    assigned = consumed.get(ti, set())
    remaining = [idx for idx in range(unit_count) if idx not in assigned]
    if not remaining:
        return ""
    return _slice_text_units(text, remaining[0], remaining[-1] + 1)


def _uncovered_intervals(
    turns: list[dict],
    total_dur: float,
    min_gap_s: float = 0.25,
) -> list[tuple[float, float]]:
    """Return timeline spans with no diarization turn coverage."""
    if total_dur <= 0:
        return []
    ordered = sorted(turns, key=lambda turn: turn["start"])
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for turn in ordered:
        if turn["start"] - cursor >= min_gap_s:
            gaps.append((cursor, turn["start"]))
        cursor = max(cursor, turn["end"])
    if total_dur - cursor >= min_gap_s:
        gaps.append((cursor, total_dur))
    return gaps


def _append_orphan_timeline_lines(
    lines: list[str],
    timeline: list[dict],
    turns: list[dict],
    diarization_segments: list[dict],
    consumed: dict[int, set[int]],
    total_dur: float,
) -> None:
    for gap_start, gap_end in _uncovered_intervals(turns, total_dur):
        text = _collect_text_for_interval(timeline, gap_start, gap_end, consumed)
        if not text:
            continue
        speaker = _find_speaker(gap_start, gap_end, diarization_segments)
        _append_turn_line(lines, speaker, gap_start, gap_end, text)

    for ti, item in enumerate(timeline):
        cs, ce = item["start"], item["end"]
        unit_count = _text_unit_count(item["text"])
        assigned = consumed.get(ti, set())
        if unit_count and len(assigned) >= unit_count:
            continue
        if _timeline_chunk_covered(cs, ce, turns) and len(assigned) >= unit_count:
            continue
        speaker = _find_speaker(cs, ce, diarization_segments)
        _append_turn_line(
            lines,
            speaker,
            cs,
            ce,
            _orphan_text_for_chunk(item, consumed, ti),
        )
    lines.sort(key=_line_start_seconds)


def _assign_speakers_by_turns(
    result: dict,
    diarization_segments: list[dict],
    max_speakers: int,
    audio_duration_s: float = 0.0,
) -> list[str]:
    """Assign ASR text to each pyannote turn (not one label per ASR chunk)."""
    diar_end = diarization_segments[-1]["end"] if diarization_segments else 0.0
    total_dur = max(audio_duration_s, diar_end) if audio_duration_s > 0 else diar_end
    timeline = _build_asr_timeline(result.get("chunks") or [], total_dur)
    turns = _turns_for_transcript(diarization_segments, max_speakers)

    lines: list[str] = []
    consumed: dict[int, set[int]] = {}
    for turn in turns:
        text, ts_start, ts_end = _collect_text_and_bounds_for_interval(
            timeline, turn["start"], turn["end"], consumed,
        )
        line_start = ts_start if ts_start is not None else turn["start"]
        line_end = ts_end if ts_end is not None else turn["end"]
        _append_turn_line(
            lines,
            turn["speaker"],
            line_start,
            line_end,
            text,
        )

    if timeline and turns:
        _append_orphan_timeline_lines(
            lines, timeline, turns, diarization_segments, consumed, total_dur,
        )

    return _merge_transcript_lines(lines)


def _speaker_char_balance(lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        match = _LINE_TS_RE.match(line.strip())
        if not match:
            continue
        speaker = match.group(7)
        body = (match.group(8) or "").strip()
        counts[speaker] = counts.get(speaker, 0) + len(body)
    return counts


def _assignment_is_imbalanced(
    lines: list[str],
    diarization_segments: list[dict],
    max_speakers: int,
) -> bool:
    """True when transcript text sits on one label but diarization found several."""
    diar_speakers = {seg["speaker"] for seg in diarization_segments}
    if max_speakers < 2 or len(diar_speakers) < 2:
        return False
    counts = _speaker_char_balance(lines)
    if len(counts) < 2:
        return True
    total = sum(counts.values())
    if total <= 0:
        return False
    dominance = max(counts.values()) / total
    threshold = _env_float("DIARIZATION_ASSIGN_IMBALANCE_RATIO", 0.65)
    return dominance > threshold


def _assignment_underuses_speakers(
    lines: list[str],
    diarization_segments: list[dict],
) -> bool:
    """True when diarization found several voices but text sits on too few labels."""
    diar_speakers = {seg["speaker"] for seg in diarization_segments}
    if len(diar_speakers) < 3:
        return False
    counts = _speaker_char_balance(lines)
    min_chars = max(20, _env_int("DIARIZATION_ASSIGN_MIN_SPEAKER_CHARS", 20))
    active = sum(1 for count in counts.values() if count >= min_chars)
    ratio = active / len(diar_speakers)
    threshold = _env_float("DIARIZATION_ASSIGN_SPEAKER_USE_RATIO", 0.70)
    return ratio < threshold


def _should_use_chunk_assignment(
    lines: list[str],
    diarization_segments: list[dict],
    max_speakers: int,
) -> bool:
    return (
        _assignment_is_imbalanced(lines, diarization_segments, max_speakers)
        or _assignment_underuses_speakers(lines, diarization_segments)
    )


def _assign_speakers_by_chunks(
    result: dict,
    diarization_segments: list[dict],
    max_speakers: int,
    audio_duration_s: float,
) -> list[str]:
    """Chunk-centric assignment — splits long ASR spans across overlapping turns."""
    chunks = result.get("chunks") or []
    non_empty = [c for c in chunks if c.get("text", "").strip()]
    diar_end = diarization_segments[-1]["end"] if diarization_segments else 0.0
    total_dur = max(audio_duration_s, diar_end) if audio_duration_s > 0 else diar_end
    all_ts_none = all(_ts_is_none(c.get("timestamp")) for c in non_empty)
    lines: list[str] = []
    for c_start, c_end, text, speaker in _iter_chunks(
        non_empty,
        diarization_segments,
        all_ts_none,
        len(non_empty),
        total_dur,
        max_speakers,
    ):
        _append_turn_line(lines, speaker, c_start, c_end, text)
    return _merge_transcript_lines(lines)


def assign_speakers(
    result: dict,
    diarization_segments: list[dict],
    max_speakers: int = 0,
    audio_duration_s: float = 0.0,
) -> str:
    """Align Whisper output chunks with speaker segments.

    Uses diarization turns as the primary axis so a single long ASR chunk
    (common with 30–180 s Whisper chunking) is split across speakers instead
    of being assigned to whichever speaker dominates the chunk window.

    Args:
        result: Whisper pipeline output dict with "text" and "chunks" keys.
        diarization_segments: Output of diarize().
    Returns:
        Formatted transcript string with [SPEAKER_XX]: labels.
    """
    chunks = result.get("chunks", [])
    if not chunks:
        return result.get("text", "").strip() or _NO_SPEECH

    if not diarization_segments:
        return _format_plain(chunks)

    non_empty = [c for c in chunks if c.get("text", "").strip()]
    diar_end = diarization_segments[-1]["end"]
    total_dur = max(audio_duration_s, diar_end) if audio_duration_s > 0 else diar_end
    all_ts_none = all(_ts_is_none(c.get("timestamp")) for c in non_empty)

    if max_speakers <= 0:
        max_speakers = len({s["speaker"] for s in diarization_segments})

    logger.info(
        "assign_speakers: total_chunks=%d  all_ts_none=%s  audio_dur=%.1fs  "
        "diar_end=%.1fs  timeline_dur=%.1fs  diar_segments=%d  speakers=%s  mode=turn_centric",
        len(non_empty),
        all_ts_none,
        audio_duration_s,
        diar_end,
        total_dur,
        len(diarization_segments),
        sorted({s["speaker"] for s in diarization_segments}),
    )

    lines = _assign_speakers_by_turns(
        result, diarization_segments, max_speakers, audio_duration_s,
    )
    if _should_use_chunk_assignment(lines, diarization_segments, max_speakers):
        logger.warning(
            "Turn-centric speaker assignment poor (speakers=%s); "
            "retrying chunk-centric split.",
            _speaker_char_balance(lines),
        )
        lines = _assign_speakers_by_chunks(
            result, diarization_segments, max_speakers, audio_duration_s,
        )
    logger.info("assign_speakers complete: output_lines=%d", len(lines))
    from engines.text_cleanup import clean_transcript_lines

    body = "\n".join(lines) if lines else _NO_SPEECH
    return clean_transcript_lines(body) if lines else _NO_SPEECH
