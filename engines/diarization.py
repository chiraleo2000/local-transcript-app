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

MODEL_ID = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")

_pipeline_cache: list = []
_tracked_device: str | None = None
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
          more distinct speakers separated (pyannote default ~0.70).
      DIARIZATION_MIN_CLUSTER_SIZE        — minimum segments to form a cluster;
          lower = rare/short speakers still detected (pyannote default 12).
    """
    seg_threshold   = _env_float("DIARIZATION_SEGMENTATION_THRESHOLD", -1.0)
    seg_min_dur_off = _env_float("DIARIZATION_MIN_DURATION_OFF",        -1.0)
    clust_threshold = _env_float("DIARIZATION_CLUSTERING_THRESHOLD",    -1.0)
    clust_min_size  = _env_int(  "DIARIZATION_MIN_CLUSTER_SIZE",        -1)

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
        logger.debug("Diarization params ignored by this pyannote pipeline: %s", skipped)
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
    """Load diarization on CPU on 8 GB GPUs so ASR can own CUDA VRAM."""
    preload = os.getenv("DIARIZATION_PRELOAD_DEVICE", "cuda").strip().lower()
    if preload not in {"cuda", "gpu"} or not torch_module.cuda.is_available():
        return torch_module.device("cpu")
    try:
        from backend.services.asr_local import requires_sequential_gpu_models

        if requires_sequential_gpu_models():
            return torch_module.device("cpu")
    except ImportError:
        pass
    vram_mb = _cuda_vram_mb(torch_module)
    low_vram_limit_mb = _env_int("ASR_8GB_CLASS_MAX_MB", 9000)
    allow_low_vram_cuda = _env_bool("DIARIZATION_ALLOW_8GB_CUDA", False)
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


def _set_tracked_device(device) -> None:
    global _tracked_device
    _tracked_device = str(device)


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
                logger.warning(
                    "CUDA unhealthy after ASR; diarization inference on CPU."
                )
                device = torch.device("cpu")
        except ImportError:
            pass
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
    free_ok = free_mb >= min_free_mb
    asr_loaded = _asr_model_loaded()
    co_resident_ok = gpu_co_resident and asr_loaded and free_ok
    staging_ok = (
        low_vram_cuda
        and not asr_loaded
        and free_ok
        and vram_mb >= min_cuda_vram_mb
    )
    standard_ok = vram_mb >= min_cuda_vram_mb and free_ok
    large_gpu_ok = vram_mb >= 12 * 1024 and free_ok

    if requested in {"cuda", "gpu"}:
        if low_vram_cuda and not allow_low_vram_cuda and not gpu_co_resident and asr_loaded:
            logger.info(
                "Diarization uses CPU on %d MB CUDA GPU so ASR can keep GPU VRAM.",
                vram_mb,
            )
            return False
        if staging_ok or co_resident_ok or standard_ok:
            return True
        logger.warning(
            "DIARIZATION_DEVICE=%s unavailable (total=%d MB free=%d MB); using CPU.",
            requested,
            vram_mb,
            free_mb,
        )
        return False

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
    return torch_module.device("cuda" if use_cuda else "cpu")


def _get_diarization_pipeline():
    """Lazy-load the configured pyannote diarization pipeline."""
    if _pipeline_cache:
        return _pipeline_cache[0]

    import torch
    from pyannote.audio import Pipeline

    _check_ffmpeg()

    hf_token = os.getenv("HF_TOKEN")
    logger.info("Loading pyannote speaker diarization pipeline (%s)...", MODEL_ID)

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
    if pipeline is None:
        raise RuntimeError(f"Failed to load pyannote pipeline '{MODEL_ID}'")
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


def _build_diarize_kwargs(
    num_speakers: int,
    max_speakers: int,
    audio_duration_s: float = 0.0,
) -> dict:
    """Construct pyannote pipeline call kwargs from speaker count hints."""
    del audio_duration_s
    if num_speakers > 0:
        return {"num_speakers": num_speakers}
    kwargs: dict = {}
    if max_speakers > 0:
        kwargs["max_speakers"] = max_speakers
    if max_speakers >= 2:
        from backend.asr_quality import is_accuracy_mode

        if is_accuracy_mode():
            kwargs["min_speakers"] = min(2, max_speakers)
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


def _accuracy_mode_params(max_speakers: int, audio_duration_s: float) -> dict:
    """Sensitive segmentation/clustering for maximum speaker separation."""
    del audio_duration_s
    if max_speakers == 2:
        seg_threshold = 0.38
        clust_threshold = 0.44
    elif max_speakers >= 3:
        seg_threshold = 0.38
        clust_threshold = 0.46
    else:
        seg_threshold = 0.40
        clust_threshold = 0.48
    min_cluster = 2
    return {
        "segmentation": {"threshold": seg_threshold, "min_duration_off": 0.03},
        "clustering": {"threshold": clust_threshold, "min_cluster_size": min_cluster},
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


def _postprocess_diarization_segments(
    segments: list[dict], max_speakers: int,
) -> list[dict]:
    """Stabilize pyannote output: merge fragments, cap speaker count."""
    from backend.asr_quality import is_accuracy_mode

    if not segments:
        return segments
    min_frag = 0.3 if is_accuracy_mode() else 0.4
    merge_gap = 0.35 if is_accuracy_mode() else 0.45
    out = _merge_short_segments(list(segments), min_duration_s=min_frag)
    out = _merge_adjacent_same_speaker(out, max_gap_s=merge_gap)
    out = _enforce_max_speakers(out, max_speakers)
    out = _merge_adjacent_same_speaker(out, max_gap_s=merge_gap)
    return out


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

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    segments.sort(key=lambda s: s["start"])
    return _merge_short_segments(segments)


def _run_pyannote(pipe, audio_input: dict, kwargs: dict):
    """Call pyannote pipeline; use ProgressHook when available."""
    _move_pipeline_to_cpu_if_cuda_memory_low(pipe)
    return _call_pyannote(pipe, audio_input, kwargs)


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
        min_free_mb = _diarization_run_min_free_mb()
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mb = int(free_bytes // (1024 * 1024))
        if free_mb >= min_free_mb:
            return
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


def _refine_long_diarization_spans(
    pipe,
    audio_path: str,
    segments: list[dict],
    kwargs: dict,
    adaptive_params: dict | None,
    max_speakers: int,
    audio_duration_s: float,
) -> list[dict]:
    """Re-diarize spans where one speaker turn is implausibly long."""
    from backend.asr_quality import is_accuracy_mode
    from engines.diarization_sampling import score_segments

    if not is_accuracy_mode() or max_speakers < 2 or not segments:
        return segments

    threshold_s = _env_float("DIARIZATION_MEGA_TURN_RETRY_S", 120.0)
    long_spans = [
        seg for seg in segments
        if seg["end"] - seg["start"] >= threshold_s
    ]
    if not long_spans:
        return segments

    retry_params = _mega_turn_retry_params(adaptive_params)
    _instantiate_pipeline_params(pipe, retry_params, "mega-turn-retry")
    refined = list(segments)
    improved = False
    max_refines = max(0, _env_int("DIARIZATION_MEGA_TURN_MAX_REFINES", 10))

    for index, span in enumerate(long_spans):
        if index >= max_refines:
            logger.info(
                "Mega-turn refine cap reached (%d); skipping %d remaining span(s).",
                max_refines,
                len(long_spans) - index,
            )
            break
        span_dur = span["end"] - span["start"]
        logger.info(
            "Re-diarizing long span %.1fs-%.1fs (%.1fs) for better speaker splits.",
            span["start"],
            span["end"],
            span_dur,
        )
        replacement = _diarize_audio_span(
            pipe, audio_path, span["start"], span["end"], kwargs,
        )
        if not replacement:
            continue
        candidate = _replace_interval_segments(refined, span, replacement)
        old_score = score_segments(refined, audio_duration_s, max_speakers)
        new_score = score_segments(candidate, audio_duration_s, max_speakers)
        new_mega = _max_segment_duration(candidate)
        if new_score > old_score or new_mega < span_dur * 0.75:
            refined = candidate
            improved = True
            logger.info(
                "Long-span refine accepted: score %.3f -> %.3f, longest=%.1fs.",
                old_score,
                new_score,
                new_mega,
            )

    if improved and adaptive_params:
        _instantiate_pipeline_params(pipe, adaptive_params, "adaptive-restore")
    return refined


def _remap_speakers(segments: list[dict]) -> dict:
    """Remap pyannote 0-indexed labels to 1-indexed in-place; return map."""
    unique_raw = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"SPEAKER_{i + 1:02d}" for i, spk in enumerate(unique_raw)}
    for s in segments:
        s["speaker"] = speaker_map[s["speaker"]]
    return speaker_map


def _slice_audio_input(audio_input: dict, start_s: float, end_s: float) -> dict:
    """Slice a preloaded waveform dict for tune-window multi-sample."""
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
    if is_accuracy_mode():
        base = min(base, 360)
    if not _env_bool("DIARIZATION_ADAPTIVE_SEGMENT_S", True):
        return base
    if audio_duration_s >= 10800:
        return max(base, 900) if not is_accuracy_mode() else min(base, 420)
    if audio_duration_s >= 7200:
        return max(base, 600) if not is_accuracy_mode() else min(base, 360)
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
) -> bool:
    return any(
        value is not None
        for value in (seg_threshold, seg_min_duration_off, clust_threshold, clust_min_size)
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
) -> dict | None:
    if ui_override:
        _instantiate_pipeline_params(
            pipe,
            _override_params(
                seg_threshold,
                seg_min_duration_off,
                clust_threshold,
                clust_min_size,
            ),
            "UI override",
        )
        return None

    from backend.asr_quality import is_accuracy_mode

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
        )
        if tuned_params:
            _instantiate_pipeline_params(pipe, tuned_params, f"tune-window:{winner}")
        logger.info(
            "Long-audio tune window selected %s (score=%.3f); running full pass.",
            winner,
            score,
        )
        segments = _execute_pyannote_pass(pipe, audio_input, kwargs)
    elif use_multi_sample:
        segments, winner, score = run_multi_sample_diarization(
            lambda params, label: _instantiate_pipeline_params(pipe, params, label),
            lambda: _execute_pyannote_pass(pipe, audio_input, kwargs),
            audio_duration_s,
            max_speakers,
            adaptive_params,
        )
        logger.info("Multi-sample diarization selected config %s (score=%.3f).", winner, score)
    else:
        segments = _execute_pyannote_pass(pipe, audio_input, kwargs)
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
                    "Diarization preprocess SR=%d Hz: segments=%d speakers=%d score=%.3f",
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
            "Multi-SR diarization winner: %s (score=%.3f, segments=%d)",
            best_label,
            best_score,
            len(best_segments),
        )

    return _refine_long_diarization_spans(
        pipe, audio_path, best_segments, kwargs, adaptive_params,
        max_speakers, audio_duration_s,
    )


def _finalize_diarization_segments(
    segments: list[dict],
    num_speakers: int,
    max_speakers: int,
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
    segments = _postprocess_diarization_segments(segments, max_speakers)
    speaker_map = _remap_speakers(segments)
    logger.info("Diarization complete: %d segments, speaker map: %s", len(segments), speaker_map)
    return segments


def diarize(
    audio_path: str, num_speakers: int = 0,
    max_speakers: int = 0,
    audio_duration_s: float = 0.0,
    seg_threshold: float | None = None,
    seg_min_duration_off: float | None = None,
    clust_threshold: float | None = None,
    clust_min_size: int | None = None,
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
        )

        kwargs = _build_diarize_kwargs(num_speakers, max_speakers, audio_duration_s)
        use_segmented = audio_duration_s > 3600 and _env_bool("DIARIZATION_SEGMENT_LONG_AUDIO", True)

        if use_segmented:
            logger.info(
                "Using segmented diarization for long audio (%.1fs > 3600s).",
                audio_duration_s,
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
        else:
            segments = _run_waveform_diarization(
                pipe, audio_path, kwargs, audio_duration_s, max_speakers,
                adaptive_params, ui_override, num_speakers,
            )

        segments = _finalize_diarization_segments(segments, num_speakers, max_speakers)
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
    if (
        skip_for_override
        or num_speakers > 0
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


def _split_text_across_turns(
    text: str, turns: list[dict], max_speakers: int = 0,
) -> list[tuple[float, float, str, str]]:
    """Split a long chunk's words proportionally across overlapping speakers."""
    words = text.split()
    if len(turns) <= 1 or len(words) < len(turns):
        return []
    unique_turn_speakers = {t["speaker"] for t in turns}
    if max_speakers > 0 and len(unique_turn_speakers) > max_speakers:
        return []
    span = max(0.001, turns[-1]["end"] - turns[0]["start"])
    min_rel = _env_float("DIARIZATION_SPLIT_MIN_TURN_FRACTION", 0.03)
    min_abs_s = _env_float("DIARIZATION_SPLIT_MIN_TURN_S", 2.0)
    significant = [
        t for t in turns
        if (t["end"] - t["start"]) >= min_abs_s
        and (t["end"] - t["start"]) / span >= min_rel
    ]
    if len(significant) <= 1:
        return []
    turns = significant

    start = turns[0]["start"]
    end = turns[-1]["end"]
    duration = max(0.001, end - start)
    pieces = []
    cursor = 0

    for idx, turn in enumerate(turns):
        if idx == len(turns) - 1:
            next_cursor = len(words)
        else:
            fraction = (turn["end"] - start) / duration
            next_cursor = round(fraction * len(words))
            next_cursor = min(len(words), max(cursor + 1, next_cursor))
        piece = " ".join(words[cursor:next_cursor]).strip()
        if piece:
            pieces.append((turn["start"], turn["end"], piece, turn["speaker"]))
        cursor = next_cursor
        if cursor >= len(words):
            break

    return pieces


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
        text = chunk.get("text", "").strip()
        if not text:
            continue
        c_start, c_end, fixed = _chunk_ts_for_assignment(
            chunk, chunk_idx, all_ts_none, total_chunks, total_dur,
        )
        fixed_timestamps += int(fixed)

        turns = _overlapping_speaker_turns(c_start, c_end, diarization_segments)
        split_pieces = _split_text_across_turns(text, turns, max_speakers)
        if split_pieces:
            chunk_idx += 1
            yield from split_pieces
            continue

        speaker = _find_speaker(c_start, c_end, diarization_segments)
        chunk_idx += 1
        yield c_start, c_end, text, speaker

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
        return max(0.0, float(os.getenv("DIARIZATION_ASSIGN_TURN_MERGE_GAP_S", "8.0")))
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
    min_overlap_s = _env_float("DIARIZATION_MIN_OVERLAP_S", 0.08)
    parts: list[tuple[float, str]] = []
    for ti, item in enumerate(timeline):
        cs, ce, text = item["start"], item["end"], item["text"]
        overlap_start = max(cs, iv_start)
        overlap_end = min(ce, iv_end)
        if overlap_end - overlap_start < min_overlap_s:
            continue
        unit_count = _text_unit_count(text)
        if unit_count == 0:
            continue
        i0, i1 = _word_indices_for_overlap(cs, ce, overlap_start, overlap_end, unit_count)
        if consumed is None:
            sliced = _slice_text_for_interval(text, cs, ce, overlap_start, overlap_end)
            if sliced:
                parts.append((overlap_start, sliced))
            continue
        sliced = _exclusive_words_for_overlap(text, consumed, ti, i0, i1)
        if sliced:
            parts.append((overlap_start, sliced))
    parts.sort(key=lambda pair: pair[0])
    return " ".join(fragment for _, fragment in parts).strip()


def _turns_for_transcript(
    diarization_segments: list[dict], max_speakers: int,
) -> list[dict]:
    """Prepare diarization turns for transcript output."""
    segments = [dict(seg) for seg in diarization_segments]
    if max_speakers > 0:
        segments = _enforce_max_speakers(segments, max_speakers)
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
        text = _collect_text_for_interval(
            timeline, turn["start"], turn["end"], consumed,
        )
        _append_turn_line(
            lines,
            turn["speaker"],
            turn["start"],
            turn["end"],
            text,
        )

    if timeline and turns:
        _append_orphan_timeline_lines(
            lines, timeline, turns, diarization_segments, consumed, total_dur,
        )

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
    logger.info("assign_speakers complete: output_lines=%d", len(lines))
    from engines.text_cleanup import clean_transcript_lines

    body = "\n".join(lines) if lines else _NO_SPEECH
    return clean_transcript_lines(body) if lines else _NO_SPEECH
