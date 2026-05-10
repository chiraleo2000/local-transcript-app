"""Shared speaker diarization using pyannote/speaker-diarization-community-1."""

import logging
import os
import re
import shutil
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
    message=r".*TensorFloat-32.*|.*TF32.*",
)

logger = logging.getLogger(__name__)

MODEL_ID = "pyannote/speaker-diarization-community-1"

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


def _get_diarization_pipeline():
    """Lazy-load the pyannote community diarization pipeline."""
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = pipeline.to(device)
    _pipeline_cache.append(pipeline)
    logger.info("Pyannote diarization pipeline ready on %s.", device)
    return _pipeline_cache[0]


def _prepare_audio_for_pyannote(audio_path: str) -> str:
    """Ensure audio is a 16 kHz mono WAV file suitable for pyannote.

    Returns the path to use (original if already valid WAV, or a temp file).
    Pyannote is most reliable when given a file path rather than a waveform
    tensor, so we always return a path.
    """
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Can't convert; return original and hope for the best
        logger.warning("ffmpeg not found — passing audio as-is to pyannote.")
        return audio_path

    # Write to a named temp WAV in the same directory
    base = os.path.splitext(audio_path)[0]
    out_path = base + "_diarize_tmp.wav"

    cmd = [
        ffmpeg, "-y", "-i", audio_path,
        "-ar", "16000",      # 16 kHz
        "-ac", "1",          # mono
        "-sample_fmt", "s16",# 16-bit PCM
        "-vn",               # drop video
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        logger.warning(
            "ffmpeg audio conversion failed (%s); using original file.", result.stderr[-300:]
        )
        return audio_path
    return out_path


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


def _run_pyannote(pipe, wav_path: str, kwargs: dict):
    """Call pyannote pipeline; use ProgressHook when available."""
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            return pipe(wav_path, hook=hook, **kwargs)
    except (ImportError, TypeError):
        return pipe(wav_path, **kwargs)


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
) -> list[dict]:
    """Run speaker diarization on audio file.

    Returns list of {"start": float, "end": float, "speaker": str}
    sorted by start time.
    """
    pipe = _get_diarization_pipeline()
    kwargs = _build_diarize_kwargs(num_speakers, min_speakers, max_speakers)
    wav_path = _prepare_audio_for_pyannote(audio_path)
    tmp_created = wav_path != audio_path

    logger.info("Running diarization on %s  kwargs=%s ...", wav_path, kwargs or "(auto)")

    try:
        diarization = _run_pyannote(pipe, wav_path, kwargs)
    finally:
        if tmp_created and os.path.isfile(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

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
        logger.warning("Diarization produced no segments — transcript will have no speaker labels.")
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

    logger.debug(
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
    return "\n".join(lines) if lines else _NO_SPEECH

