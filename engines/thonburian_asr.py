"""Thonburian Whisper Large v3 — Thai ASR via OpenVINO."""

import logging
import os

logger = logging.getLogger(__name__)

MODEL_ID = "biodatlab/distill-whisper-th-large-v3"

_pipeline_cache: list = []


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_cuda_pipeline(hf_token: str | None):
    """Build Thonburian pipeline on NVIDIA CUDA (float16)."""
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
    from transformers import pipeline as hf_pipeline
    from transformers.models.whisper.processing_whisper import WhisperProcessor

    logger.info("Using CUDA (float16) backend for Thonburian Whisper.")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        use_safetensors=True,
        token=hf_token,
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
        chunk_length_s=30,
        return_timestamps=True,
        max_new_tokens=445,
    )


def _load_cpu_pipeline(hf_token: str | None):
    """CPU/float32 fallback when OpenVINO export is unavailable."""
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
    from transformers import pipeline as hf_pipeline
    from transformers.models.whisper.processing_whisper import WhisperProcessor

    logger.info("Using CPU (float32) fallback pipeline for Thonburian Whisper.")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, dtype=torch.float32, use_safetensors=True, token=hf_token,
    )
    processor = WhisperProcessor.from_pretrained(MODEL_ID, token=hf_token)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,  # pylint: disable=no-member
        feature_extractor=processor.feature_extractor,  # pylint: disable=no-member
        chunk_length_s=30,
        return_timestamps=True,
        max_new_tokens=445,
    )


def _load_ov_pipeline(device: str, hf_token: str | None):
    """Build Thonburian pipeline via OpenVINO IR; falls back to CPU on export failure."""
    from transformers.models.whisper.processing_whisper import WhisperProcessor
    from transformers import pipeline as hf_pipeline

    cache_dir = os.getenv("OV_CACHE_DIR", "./ov_cache")
    export_dir = os.path.join(cache_dir, "thonburian")
    ir_path = os.path.join(export_dir, "openvino_encoder_model.xml")

    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        if os.path.isdir(export_dir) and os.path.isfile(ir_path):
            logger.info("Loading Thonburian from cached OpenVINO IR: %s", export_dir)
            model = OVModelForSpeechSeq2Seq.from_pretrained(export_dir, device=device, compile=True)
            processor = WhisperProcessor.from_pretrained(export_dir)
        else:
            logger.info("Exporting Thonburian to OpenVINO IR (first run, may take several minutes)...")
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID, export=True, device=device, compile=True, token=hf_token,
            )
            processor = WhisperProcessor.from_pretrained(MODEL_ID, token=hf_token)
            model.save_pretrained(export_dir)
            processor.save_pretrained(export_dir)
            logger.info("Thonburian OpenVINO IR saved to %s", export_dir)
        return hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,  # pylint: disable=no-member
            feature_extractor=processor.feature_extractor,  # pylint: disable=no-member
            chunk_length_s=30,
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
    """Lazy-load the Thonburian Whisper pipeline (CUDA or OpenVINO)."""
    if _pipeline_cache:
        return _pipeline_cache[0]

    from engines.hardware import detect_hardware

    hw = detect_hardware()
    device = hw["selected_device"]
    hf_token = os.getenv("HF_TOKEN")

    logger.info("Loading Thonburian Whisper (%s) on device=%s ...", MODEL_ID, device)
    if hw["backend"] == "cuda":
        pipe = _load_cuda_pipeline(hf_token)
    else:
        pipe = _load_ov_pipeline(device, hf_token)
    _pipeline_cache.append(pipe)
    logger.info("Thonburian Whisper pipeline ready on %s.", device)
    return _pipeline_cache[0]


def load_model():
    """Pre-load the Thonburian Whisper model. Safe to call multiple times."""
    _get_pipeline()
    logger.info("Thonburian Whisper model pre-loaded.")


def _load_audio(audio_path: str):
    """Load audio as numpy array at 16 kHz mono.

    Returns a dict compatible with the HuggingFace ASR pipeline,
    bypassing torchcodec which requires FFmpeg DLLs on Windows.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return {"raw": y, "sampling_rate": sr}


def transcribe_thonburian(
    audio_path: str, language: str = "thai",
    diarization_segments: list | None = None,
) -> str:
    """Transcribe audio using Thonburian Whisper Large v3.

    If diarization_segments is provided (pre-computed by the caller),
    each Whisper chunk is labelled with the overlapping speaker.
    """
    pipe = _get_pipeline()
    audio_input = _load_audio(audio_path)
    result = pipe(
        audio_input,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=True,
    )
    logger.debug(
        "Thonburian result: text_len=%d chunks=%d first_ts=%s",
        len(result.get("text", "")),
        len(result.get("chunks", [])),
        result.get("chunks", [{}])[0].get("timestamp") if result.get("chunks") else "N/A",
    )

    if diarization_segments:
        from engines.diarization import assign_speakers
        return assign_speakers(result, diarization_segments)

    chunks = result.get("chunks", [])
    if chunks:
        return _format_chunks(chunks)

    return result.get("text", "").strip() or "(no speech detected)"
