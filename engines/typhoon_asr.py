"""Typhoon Whisper Large v3 — Thai ASR via OpenVINO."""

# pylint: disable=import-outside-toplevel

import logging
import os

logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "typhoon-ai/typhoon-whisper-large-v3"

_pipeline_cache: list = []


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f.", name, value, default)
        return default


def _cuda_vram_mb() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except (ImportError, RuntimeError, OSError, AttributeError):
        return 0
    return 0


def _strict_8gb_mode() -> bool:
    vram_mb = _cuda_vram_mb()
    if not vram_mb:
        return False
    return _env_bool("ASR_HARD_MEMORY_SAFE", True) and vram_mb <= _env_int(
        "ASR_8GB_CLASS_MAX_MB", 9000,
    )


def _configure_torch_runtime() -> None:
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available() and _strict_8gb_mode():
            fraction = min(1.0, max(0.5, _env_float("ASR_CUDA_MEMORY_FRACTION", 0.90)))
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            logger.info("Typhoon CUDA memory fraction capped at %.2f.", fraction)
    except (ImportError, RuntimeError, OSError, AttributeError):
        pass


def _clear_cuda_cache() -> None:
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


def _asr_batch_size() -> int:
    default = _env_int("ASR_8GB_BATCH_SIZE", 1) if _strict_8gb_mode() else 4
    batch_size = max(1, _env_int("ASR_CUDA_BATCH_SIZE", default))
    if _strict_8gb_mode():
        max_batch_size = max(1, _env_int("ASR_8GB_MAX_BATCH_SIZE", 1))
        if batch_size > max_batch_size:
            logger.warning(
                "ASR_CUDA_BATCH_SIZE=%d capped to %d by strict 8 GB mode.",
                batch_size,
                max_batch_size,
            )
            return max_batch_size
    return batch_size


def _chunk_length_s() -> int:
    default = _env_int("ASR_8GB_CHUNK_LENGTH_S", 20) if _strict_8gb_mode() else 30
    chunk_length_s = max(10, _env_int("ASR_CHUNK_LENGTH_S", default))
    if _strict_8gb_mode():
        max_chunk_s = max(10, _env_int("ASR_8GB_MAX_CHUNK_LENGTH_S", 20))
        if chunk_length_s > max_chunk_s:
            logger.info(
                "ASR_CHUNK_LENGTH_S=%d capped to %d by strict 8 GB mode.",
                chunk_length_s,
                max_chunk_s,
            )
            return max_chunk_s
    return chunk_length_s


def _retry_chunk_length_s() -> int:
    return max(10, _env_int("ASR_8GB_RETRY_CHUNK_LENGTH_S", 10))


def _timestamp_mode(diarization_segments: list | None):
    if diarization_segments and _env_bool("ASR_WORD_TIMESTAMPS_WITH_DIARIZATION", True):
        if _strict_8gb_mode() and not _env_bool("TYPHOON_WORD_TIMESTAMPS_ON_8GB", True):
            logger.info("Typhoon strict 8 GB mode uses chunk timestamps to avoid CUDA OOM.")
            return True
        return "word"
    return True


def _model_load_kwargs(hf_token: str | None, dtype) -> dict:
    kwargs = {
        "dtype": dtype,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    attention = os.getenv("ASR_ATTENTION_IMPLEMENTATION", "sdpa").strip()
    if attention:
        kwargs["attn_implementation"] = attention
    return kwargs


def _is_cuda_oom(exc: Exception) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, OSError, AttributeError):
        pass
    return "CUDA out of memory" in str(exc)


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_cuda_pipeline(hf_token: str | None):
    """Build Typhoon pipeline on NVIDIA CUDA (float16)."""
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
    from transformers import pipeline as hf_pipeline
    from transformers.models.whisper.processing_whisper import WhisperProcessor

    _configure_torch_runtime()
    logger.info("Using CUDA (float16) backend for Typhoon Whisper.")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        **_model_load_kwargs(hf_token, torch.float16),
    )
    # Fix meta tensors left over from sharded checkpoint loading
    meta_params = [
        (n, p) for n, p in model.named_parameters()
        if p.device.type == "meta"
    ]
    for name, param in meta_params:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], torch.nn.Parameter(
            torch.zeros(param.shape, dtype=param.dtype),
        ))
    model.tie_weights()
    model = model.to("cuda")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, token=hf_token)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        feature_extractor=processor.feature_extractor,  # pylint: disable=no-member
        chunk_length_s=_chunk_length_s(),
        return_timestamps=True,
        max_new_tokens=445,
    )


def _load_cpu_pipeline(hf_token: str | None):
    """CPU/float32 fallback when OpenVINO export is unavailable."""
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
    from transformers import pipeline as hf_pipeline
    from transformers.models.whisper.processing_whisper import WhisperProcessor

    logger.info("Using CPU (float32) fallback pipeline for Typhoon Whisper.")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        **_model_load_kwargs(hf_token, torch.float32),
    )
    processor = WhisperProcessor.from_pretrained(MODEL_ID, token=hf_token)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        feature_extractor=processor.feature_extractor,  # pylint: disable=no-member
        chunk_length_s=_chunk_length_s(),
        return_timestamps=True,
        max_new_tokens=445,
    )


def _load_ov_pipeline(device: str, hf_token: str | None):
    """Build Typhoon pipeline via OpenVINO IR; falls back to CPU on export failure."""
    from transformers.models.whisper.processing_whisper import WhisperProcessor
    from transformers import pipeline as hf_pipeline

    cache_dir = os.getenv("OV_CACHE_DIR", "./ov_cache")
    export_dir = os.path.join(cache_dir, "typhoon")
    ir_path = os.path.join(export_dir, "openvino_encoder_model.xml")

    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        if os.path.isdir(export_dir) and os.path.isfile(ir_path):
            logger.info("Loading Typhoon from cached OpenVINO IR: %s", export_dir)
            model = OVModelForSpeechSeq2Seq.from_pretrained(export_dir, device=device, compile=True)
            processor = WhisperProcessor.from_pretrained(export_dir)
        else:
            logger.info("Exporting Typhoon to OpenVINO IR (first run, may take several minutes)...")
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID, export=True, device=device, compile=True, token=hf_token,
            )
            processor = WhisperProcessor.from_pretrained(MODEL_ID, token=hf_token)
            model.save_pretrained(export_dir)
            processor.save_pretrained(export_dir)
            logger.info("Typhoon OpenVINO IR saved to %s", export_dir)
        return hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,  # pylint: disable=no-member
            feature_extractor=processor.feature_extractor,  # pylint: disable=no-member
            chunk_length_s=_chunk_length_s(),
            return_timestamps=True,
            max_new_tokens=445,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("OpenVINO export/load failed (%s); falling back to CPU pipeline.", exc)
        return _load_cpu_pipeline(hf_token)


def _format_chunks(chunks):
    """Format timestamped chunks into readable lines."""
    lines = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        c_start, c_end = (ts if ts else (None, None))
        text = chunk.get("text", "").strip()
        if not text:
            continue
        if c_start is not None:
            lines.append(f"[{_fmt_ts(c_start)} \u2192 {_fmt_ts(c_end)}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines) if lines else "(no speech detected)"


def _get_pipeline():
    """Lazy-load the Typhoon Whisper pipeline (CUDA or OpenVINO)."""
    if _pipeline_cache:
        return _pipeline_cache[0]

    from engines.hardware import detect_hardware

    hw = detect_hardware()
    device = hw["selected_device"]
    hf_token = os.getenv("HF_TOKEN")

    logger.info("Loading Typhoon Whisper (%s) on device=%s ...", MODEL_ID, device)
    if hw["backend"] == "cuda":
        pipe = _load_cuda_pipeline(hf_token)
    else:
        pipe = _load_ov_pipeline(device, hf_token)
    _pipeline_cache.append(pipe)
    logger.info("Typhoon Whisper pipeline ready on %s.", device)
    return _pipeline_cache[0]


def load_model():
    """Pre-load the Typhoon Whisper model. Safe to call multiple times."""
    _get_pipeline()
    logger.info("Typhoon Whisper model pre-loaded.")


def unload_model():
    """Unload Typhoon Whisper from process memory and clear CUDA cache."""
    _pipeline_cache.clear()
    _clear_cuda_cache()
    logger.info("Typhoon Whisper model cache cleared.")


def _load_audio(audio_path: str):
    """Load audio as numpy array at 16 kHz mono.

    Returns a dict compatible with the HuggingFace ASR pipeline,
    bypassing torchcodec which requires FFmpeg DLLs on Windows.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return {"raw": y, "sampling_rate": sr}


def _run_pipe(
    pipe, audio_input, language: str, timestamp_mode, batch_size: int, chunk_length_s=None
):
    pipeline_input = dict(audio_input) if isinstance(audio_input, dict) else audio_input
    kwargs = {
        "batch_size": batch_size,
        "generate_kwargs": {"language": language, "task": "transcribe", "num_beams": 1},
        "return_timestamps": timestamp_mode,
    }
    if chunk_length_s is not None:
        kwargs["chunk_length_s"] = chunk_length_s
    try:
        import torch
    except (ImportError, OSError):
        return pipe(pipeline_input, **kwargs)
    with torch.inference_mode():
        return pipe(pipeline_input, **kwargs)


def transcribe_typhoon(
    audio_path: str, language: str = "thai",
    diarization_segments: list | None = None,
) -> str:
    """Transcribe audio using Typhoon Whisper Large v3.

    If diarization_segments is provided (pre-computed by the caller),
    each Whisper chunk is labelled with the overlapping speaker.
    """
    pipe = _get_pipeline()
    audio_input = _load_audio(audio_path)
    from engines.timestamps import audio_duration_from_input, repair_asr_result

    audio_duration_s = audio_duration_from_input(audio_input)
    timestamp_mode = _timestamp_mode(diarization_segments)
    logger.info(
        "Typhoon transcription started: audio=%.1fs language=%s diarization=%s "
        "timestamp_mode=%s batch=%d chunk=%ds",
        audio_duration_s,
        language,
        bool(diarization_segments),
        timestamp_mode,
        _asr_batch_size(),
        _chunk_length_s(),
    )
    try:
        result = _run_pipe(pipe, audio_input, language, timestamp_mode, _asr_batch_size())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not (_strict_8gb_mode() and _is_cuda_oom(exc)):
            raise
        retry_chunk_s = _retry_chunk_length_s()
        logger.warning(
            "Typhoon CUDA OOM in strict 8 GB mode; retrying with batch=1, "
            "chunk=%ds and chunk timestamps.",
            retry_chunk_s,
        )
        _clear_cuda_cache()
        result = _run_pipe(pipe, audio_input, language, True, 1, retry_chunk_s)
    result = repair_asr_result(result, audio_duration_s, "Typhoon", logger)

    if diarization_segments:
        from engines.diarization import assign_speakers
        return assign_speakers(result, diarization_segments)

    chunks = result.get("chunks", [])
    if chunks:
        return _format_chunks(chunks)

    return result.get("text", "").strip() or "(no speech detected)"
