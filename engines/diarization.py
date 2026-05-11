"""Shared speaker diarization using pyannote/speaker-diarization-3.1."""

# pylint: disable=import-outside-toplevel

import logging
import os
import re
import shutil
import subprocess
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

MODEL_ID = os.getenv("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-3.1")

_pipeline_cache: list = []

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


def _build_pipeline_params() -> dict | None:
    """Build pyannote pipeline instantiate() params from environment variables.

    Tuning guide:
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


def _cuda_vram_mb(torch_module) -> int:
    if not torch_module.cuda.is_available():
        return 0
    try:
        total = torch_module.cuda.get_device_properties(0).total_memory
        return int(total // (1024 * 1024))
    except (RuntimeError, OSError, AttributeError):
        return 0


def _select_diarization_device(torch_module):
    """Keep diarization off 8 GB GPUs unless a larger GPU is available."""
    requested = os.getenv("DIARIZATION_DEVICE", "cpu").strip().lower()
    min_cuda_vram_mb = _env_int("DIARIZATION_CUDA_MIN_VRAM_MB", 12288)
    vram_mb = _cuda_vram_mb(torch_module)

    if requested in {"cuda", "gpu"}:
        if vram_mb >= min_cuda_vram_mb:
            return torch_module.device("cuda")
        logger.warning(
            "DIARIZATION_DEVICE=%s ignored: %d MB VRAM is below %d MB. Using CPU.",
            requested,
            vram_mb,
            min_cuda_vram_mb,
        )
        return torch_module.device("cpu")

    if requested == "auto" and vram_mb >= min_cuda_vram_mb:
        return torch_module.device("cuda")

    return torch_module.device("cpu")


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
    device = _select_diarization_device(torch)
    pipeline = pipeline.to(device)

    # Apply tunable segmentation / clustering hyperparameters.
    custom_params = _build_pipeline_params()
    if custom_params:
        try:
            pipeline.instantiate(custom_params)
            logger.info("Diarization hyperparameters applied: %s", custom_params)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not apply custom diarization params: %s", exc)

    _pipeline_cache.append(pipeline)
    logger.info("Pyannote diarization pipeline ready on %s.", device)
    return _pipeline_cache[0]


def unload_model() -> None:
    """Unload the cached pyannote pipeline and release CUDA cache."""
    _pipeline_cache.clear()
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


def _prepare_audio_for_pyannote(audio_path: str) -> dict:
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

    preprocess_sr = _env_int("DIARIZATION_PREPROCESS_SR", 44100)
    noise_reduction = _env_float("DIARIZATION_NOISE_REDUCTION", 0.60)
    source_path, tmp_files = _build_diarization_source(
        audio_path, preprocess_sr, noise_reduction
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
        preprocess_sr,
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
    diarize_filters = "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11"
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


def _build_diarize_kwargs(num_speakers: int, min_speakers: int, max_speakers: int) -> dict:
    """Construct pyannote pipeline call kwargs from speaker count hints."""
    if num_speakers > 0:
        return {"num_speakers": num_speakers}
    # Force exact count when min == max to avoid pyannote collapsing to 1 speaker.
    if min_speakers > 0 and max_speakers > 0 and min_speakers == max_speakers:
        return {"num_speakers": min_speakers}
    kwargs: dict = {}
    if min_speakers > 0:
        kwargs["min_speakers"] = min_speakers
    if max_speakers > 0:
        kwargs["max_speakers"] = max_speakers
    return kwargs


def _run_pyannote(pipe, audio_input: dict, kwargs: dict):
    """Call pyannote pipeline; use ProgressHook when available."""
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            return pipe(audio_input, hook=hook, **kwargs)
    except (ImportError, TypeError):
        return pipe(audio_input, **kwargs)


def _remap_speakers(segments: list[dict]) -> dict:
    """Remap pyannote 0-indexed labels to 1-indexed in-place; return map."""
    unique_raw = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"SPEAKER_{i + 1:02d}" for i, spk in enumerate(unique_raw)}
    for s in segments:
        s["speaker"] = speaker_map[s["speaker"]]
    return speaker_map


def diarize(
    audio_path: str, num_speakers: int = 0,
    min_speakers: int = 0, max_speakers: int = 0,
    seg_threshold: float | None = None,
    seg_min_duration_off: float | None = None,
    clust_threshold: float | None = None,
    clust_min_size: int | None = None,
) -> list[dict]:
    """Run speaker diarization on audio file.

    Returns list of {"start": float, "end": float, "speaker": str}
    sorted by start time.
    """
    pipe = _get_diarization_pipeline()

    # Apply per-call UI overrides (re-instantiate pipeline params for this run).
    override = _override_params(
        seg_threshold,
        seg_min_duration_off,
        clust_threshold,
        clust_min_size,
    )
    if override:
        try:
            pipe.instantiate(override)
            logger.info("Diarization params overridden for this run: %s", override)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not apply diarization param overrides: %s", exc)

    kwargs = _build_diarize_kwargs(num_speakers, min_speakers, max_speakers)
    audio_input = _prepare_audio_for_pyannote(audio_path)

    logger.info("Running diarization on preloaded waveform kwargs=%s ...", kwargs or "(auto)")
    diarization = _run_pyannote(pipe, audio_input, kwargs)

    # pyannote 3.x returns Annotation directly; some community wrappers wrap it.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start":   float(turn.start),
            "end":     float(turn.end),
            "speaker": str(speaker),
        })
    segments.sort(key=lambda s: s["start"])

    unique_raw = sorted({s["speaker"] for s in segments})
    logger.info(
        "Pyannote returned %d segments, %d unique speaker(s): %s",
        len(segments), len(unique_raw), unique_raw,
    )

    if not segments:
        logger.warning("Diarization produced no segments; transcript will have no speaker labels.")
        return segments

    if len(unique_raw) == 1 and (min_speakers > 1 or num_speakers > 1):
        logger.warning(
            "Pyannote detected only 1 speaker despite hint (min=%d, num=%d). "
            "The audio may be too short, too noisy, or the speakers sound alike.",
            min_speakers, num_speakers,
        )

    speaker_map = _remap_speakers(segments)
    logger.info("Diarization complete: %d segments, speaker map: %s", len(segments), speaker_map)
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
    """Collapse Whisper hallucination loops: 3+ consecutive identical phrases → 1.

    Uses regex replacement for each phrase length 8..1.  Works on
    space-separated tokens (Thai ASR output already has spaces).
    """
    for n in range(8, 0, -1):
        # Build pattern: <phrase of n tokens> repeated 3+ times
        inner = r"(?:\S+[ \t]+)" * (n - 1) + r"\S+"
        pattern = rf"({inner})(?:[ \t]+\1){{2,}}"
        text = re.sub(pattern, r"\1", text)
    return text.strip()


def _flush_speaker_group(
    lines: list[str], speaker: str | None, words: list[str],
    start: float | None, end: float | None,
) -> None:
    """Append one completed speaker turn to lines."""
    if not words or speaker is None:
        return
    text = _dedup_repetitions(" ".join(words))
    if not text:
        return
    ts_prefix = (
        f"[{_fmt_ts(start)} → {_fmt_ts(end)}] " if start is not None else ""
    )
    lines.append(f"{ts_prefix}[{speaker}]: {text}")


def _estimate_chunk_ts(
    chunk_idx: int, total_chunks: int, total_dur: float,
) -> tuple[float, float]:
    """Estimate (start, end) for a chunk when timestamps are unavailable."""
    c_start = (chunk_idx / total_chunks) * total_dur
    c_end = ((chunk_idx + 1) / total_chunks) * total_dur
    return c_start, c_end


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
    lines = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
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


def _split_text_across_turns(text: str, turns: list[dict]) -> list[tuple[float, float, str, str]]:
    """Split a long chunk's words proportionally across overlapping speakers."""
    words = text.split()
    if len(turns) <= 1 or len(words) < len(turns):
        return []

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
):
    """Yield (c_start, c_end, text, speaker) for each non-empty chunk."""
    chunk_idx = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        ts = chunk.get("timestamp")
        if ts is not None and not _ts_is_none(ts):
            c_start, c_end = ts[0], ts[1]
        else:
            c_start, c_end = None, None

        # When ALL timestamps are None OR this individual chunk is missing
        # timestamps, fall back to positional estimation so the chunk is still
        # mapped to the correct time-window and therefore the correct speaker.
        if (all_ts_none or c_start is None) and total_dur > 0 and total_chunks > 0:
            c_start, c_end = _estimate_chunk_ts(chunk_idx, total_chunks, total_dur)

        turns = _overlapping_speaker_turns(c_start, c_end, diarization_segments)
        split_pieces = _split_text_across_turns(text, turns)
        if split_pieces:
            chunk_idx += 1
            yield from split_pieces
            continue

        speaker = _find_speaker(c_start, c_end, diarization_segments)
        chunk_idx += 1
        yield c_start, c_end, text, speaker


def assign_speakers(result: dict, diarization_segments: list[dict]) -> str:
    """Align Whisper output chunks with speaker segments.

    Args:
        result: Whisper pipeline output dict with "text" and "chunks" keys.
        diarization_segments: Output of diarize().
    Returns:
        Formatted transcript string with [SPEAKER_XX]: labels.
    """
    chunks = result.get("chunks", [])
    if not chunks:
        return result.get("text", "").strip() or _NO_SPEECH

    # When no diarization segments are available (e.g. diarize() returned an
    # empty list), fall back to plain timestamped format without speaker labels.
    if not diarization_segments:
        return _format_plain(chunks)

    non_empty = [c for c in chunks if c.get("text", "").strip()]
    total_chunks = len(non_empty)

    # Detect if ALL chunks lack timestamps.
    all_ts_none = all(_ts_is_none(c.get("timestamp")) for c in non_empty)

    # Use the later of the last diarization segment end OR the last chunk
    # timestamp end, so position-estimation covers the full audio duration.
    diar_end = diarization_segments[-1]["end"] if diarization_segments else 0.0
    chunk_ts_end = 0.0
    for c in non_empty:
        ts = c.get("timestamp")
        if ts and not _ts_is_none(ts) and ts[1] is not None:
            chunk_ts_end = max(chunk_ts_end, ts[1])
    total_dur = max(diar_end, chunk_ts_end)

    logger.info(
        "assign_speakers: total_chunks=%d  all_ts_none=%s  total_dur=%.1fs  "
        "diar_segments=%d  speakers=%s",
        total_chunks, all_ts_none, total_dur,
        len(diarization_segments),
        sorted({s["speaker"] for s in diarization_segments}),
    )

    lines: list[str] = []
    current_speaker: str | None = None
    current_words: list[str] = []
    group_start: float | None = None
    group_end: float | None = None

    for c_start, c_end, text, speaker in _iter_chunks(
        chunks, diarization_segments, all_ts_none, total_chunks, total_dur,
    ):
        if speaker == current_speaker:
            current_words.append(text)
            if c_end is not None:
                group_end = c_end
            continue

        _flush_speaker_group(lines, current_speaker, current_words, group_start, group_end)
        current_speaker = speaker
        current_words = [text]
        group_start = c_start
        group_end = c_end

    _flush_speaker_group(lines, current_speaker, current_words, group_start, group_end)
    logger.info("assign_speakers complete: output_lines=%d", len(lines))
    return "\n".join(lines) if lines else _NO_SPEECH

