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


def diarize(
    audio_path: str, num_speakers: int = 0,
    min_speakers: int = 0, max_speakers: int = 0,
) -> list[dict]:
    """Run speaker diarization on audio file.

    Returns list of {"start": float, "end": float, "speaker": str}
    sorted by start time.
    """
    import librosa
    import torch
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipe = _get_diarization_pipeline()

    kwargs = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    elif min_speakers > 0 or max_speakers > 0:
        if min_speakers > 0:
            kwargs["min_speakers"] = min_speakers
        if max_speakers > 0:
            kwargs["max_speakers"] = max_speakers

    # Load audio via librosa to bypass torchcodec requirement in torchaudio 2.11+
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, samples)
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    logger.info(
        "Running diarization on %s (num_speakers=%s)...", audio_path, num_speakers or "auto",
    )

    with ProgressHook() as hook:
        diarization = pipe(audio_input, hook=hook, **kwargs)

    # pyannote/speaker-diarization-community-1 returns a DiarizeOutput dataclass;
    # the actual Annotation is on .speaker_diarization (has .itertracks).
    # Older pipelines return an Annotation directly — handle both.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start":   turn.start,
            "end":     turn.end,
            "speaker": speaker,
        })

    segments.sort(key=lambda s: s["start"])

    # Remap pyannote labels (SPEAKER_00, SPEAKER_01 …) to 1-based (SPEAKER_01, SPEAKER_02 …)
    unique_speakers = sorted({s["speaker"] for s in segments})
    speaker_map = {spk: f"SPEAKER_{i + 1:02d}" for i, spk in enumerate(unique_speakers)}
    for s in segments:
        s["speaker"] = speaker_map[s["speaker"]]

    logger.info("Diarization complete: %d segments, speakers: %s",
                len(segments), sorted({s["speaker"] for s in segments}))
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
        if all_ts_none and total_dur > 0 and total_chunks > 0:
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

    # When return_timestamps=True is not honoured (transformers regression),
    # ALL chunks arrive with timestamp=None or (None,None).  Detect this
    # upfront and fall back to position-based estimation so speaker
    # attribution still works.
    non_empty = [c for c in chunks if c.get("text", "").strip()]
    total_chunks = len(non_empty)
    all_ts_none = all(_ts_is_none(c.get("timestamp")) for c in non_empty)
    total_dur = diarization_segments[-1]["end"] if diarization_segments else 0.0

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
