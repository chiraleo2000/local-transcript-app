"""Transcript accuracy scoring against golden reference files."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_LINE_PREFIX_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2})\s*→\s*(?P<end>\d{2}:\d{2}:\d{2})\]\s*"
    r"(?:\[(?P<speaker>SPEAKER_\d+|เสียง[^\]]*)\]:\s*)?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\u0E00-\u0E7F]+", re.UNICODE)
_SPEAKER_RE = re.compile(r"\[(SPEAKER_\d+)\]:")
_THAI_FILLERS_RE = re.compile(
    r"(เอ่อ+|อ่า+|อืม+|เออ+|ฮะ|นะครับ|ครับผม|ครับๆ|ฮัลโหล)"
)
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUMBER_WORDS = {
    "หนึ่ง": "1",
    "สอง": "2",
    "สาม": "3",
    "สี่": "4",
    "ห้า": "5",
    "นึง": "1",
}
_TS_START_TOLERANCE_S = 3.0
_TS_END_TOLERANCE_S = 5.0


def _line_body(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    text = _LINE_PREFIX_RE.sub("", text)
    text = re.sub(r"^\([^)]*\)\s*", "", text)
    return text.strip()


def _is_non_dialogue_line(line: str) -> bool:
    body = _line_body(line)
    if not body:
        return True
    if body.startswith("(") and body.endswith(")"):
        return True
    if "เสียงสัญญาณ" in body or "เสียงเรียก" in body:
        return True
    return False


def normalize_transcript_text(text: str) -> str:
    """Normalize a single utterance for fuzzy comparison."""
    try:
        from engines.text_cleanup import fix_common_thai_asr_variants

        text = fix_common_thai_asr_variants(text)
    except ImportError:
        pass
    text = _PUNCT_RE.sub("", text.lower())
    text = text.translate(_THAI_DIGITS)
    for word, digit in _NUMBER_WORDS.items():
        text = text.replace(word, digit)
    text = re.sub(r"2-3|2–3", "23", text)
    text = _THAI_FILLERS_RE.sub("", text)
    return text


def normalize_transcript_corpus(text: str) -> str:
    parts = [
        normalize_transcript_text(_line_body(line))
        for line in text.splitlines()
        if not _is_non_dialogue_line(line)
    ]
    return "".join(part for part in parts if part)


def _parsed_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    for line in text.splitlines():
        if _is_non_dialogue_line(line):
            continue
        match = _LINE_PREFIX_RE.match(line.strip())
        if not match:
            continue
        body = _line_body(line)
        if not body:
            continue
        start = _ts_to_seconds(match.group("start"))
        end = _ts_to_seconds(match.group("end"))
        segments.append({
            "start": start,
            "end": end,
            "speaker": (match.group("speaker") or "").upper(),
            "text": normalize_transcript_text(body),
        })
    return segments


def _ts_to_seconds(ts: str) -> float:
    hours, minutes, seconds = (int(part) for part in ts.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _normalized_lines(text: str) -> list[str]:
    return [
        normalize_transcript_text(_line_body(line))
        for line in text.splitlines()
        if not _is_non_dialogue_line(line) and _line_body(line).strip()
    ]


def _overlap(a: dict, b: dict) -> float:
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    return max(0.0, end - start)


def _timestamp_score(expected: dict, actual: dict) -> float:
    start_delta = abs(expected["start"] - actual["start"])
    end_delta = abs(expected["end"] - actual["end"])
    start_score = max(0.0, 1.0 - start_delta / _TS_START_TOLERANCE_S)
    end_score = max(0.0, 1.0 - end_delta / _TS_END_TOLERANCE_S)
    return (start_score + end_score) / 2.0


def _speaker_score(expected: dict, actual: dict) -> float:
    exp_speaker = (expected.get("speaker") or "").upper()
    act_speaker = (actual.get("speaker") or "").upper()
    if not exp_speaker:
        return 1.0
    if not act_speaker:
        return 0.0
    return 1.0 if exp_speaker == act_speaker else 0.0


def strict_segment_accuracy(expected: dict, actual: dict) -> float:
    """Score one utterance on text, speaker label, and timestamp alignment."""
    if not expected["text"]:
        return 1.0
    text_score = SequenceMatcher(None, expected["text"], actual["text"]).ratio()
    return (
        text_score * 0.55
        + _speaker_score(expected, actual) * 0.25
        + _timestamp_score(expected, actual) * 0.20
    )


def strict_transcript_accuracy(expected: str, actual: str) -> float:
    """Golden score: each reference turn must match text, speaker, and timestamps."""
    exp_segments = _parsed_segments(expected)
    act_segments = _parsed_segments(actual)
    if not exp_segments:
        return 1.0 if not act_segments else 0.0
    if not act_segments:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []
    for exp in exp_segments:
        if not exp["text"]:
            continue
        best = 0.0
        best_overlap = 0.0
        for act in act_segments:
            overlap = _overlap(exp, act)
            if overlap <= 0:
                continue
            segment_score = strict_segment_accuracy(exp, act)
            if overlap > best_overlap or (overlap == best_overlap and segment_score > best):
                best = segment_score
                best_overlap = overlap
        duration = max(0.1, exp["end"] - exp["start"])
        scores.append(best if best_overlap > 0 else 0.0)
        weights.append(duration)

    if not scores:
        return 0.0
    line_ratio = min(len(act_segments), len(exp_segments)) / max(
        1, max(len(act_segments), len(exp_segments))
    )
    weighted = sum(s * w for s, w in zip(scores, weights)) / (sum(weights) or 1.0)
    return weighted * (0.85 + 0.15 * line_ratio)


def time_aligned_accuracy(expected: str, actual: str) -> float:
    """Score by matching utterances on timestamp overlap (text only)."""
    exp_segments = _parsed_segments(expected)
    act_segments = _parsed_segments(actual)
    if not exp_segments:
        return 1.0 if not act_segments else 0.0
    if not act_segments:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []
    for exp in exp_segments:
        if not exp["text"]:
            continue
        best = 0.0
        best_overlap = 0.0
        for act in act_segments:
            ov = _overlap(exp, act)
            if ov <= 0:
                continue
            ratio = SequenceMatcher(None, exp["text"], act["text"]).ratio()
            if ov > best_overlap or (ov == best_overlap and ratio > best):
                best = ratio
                best_overlap = ov
        if best_overlap > 0:
            scores.append(best)
            weights.append(exp["end"] - exp["start"])
        else:
            scores.append(0.0)
            weights.append(exp["end"] - exp["start"])

    if not scores:
        return 0.0
    total_weight = sum(weights) or 1.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def line_best_match_accuracy(expected: str, actual: str) -> float:
    exp_lines = _normalized_lines(expected)
    act_lines = _normalized_lines(actual)
    if not exp_lines:
        return 1.0 if not act_lines else 0.0
    if not act_lines:
        return 0.0
    scores = [
        max(SequenceMatcher(None, exp, act).ratio() for act in act_lines)
        for exp in exp_lines
    ]
    return sum(scores) / len(scores)


def speaker_sequence_score(expected: str, actual: str) -> float:
    """Fraction of lines where speaker label matches golden order."""
    exp_spks = [seg["speaker"] for seg in _parsed_segments(expected) if seg.get("speaker")]
    act_spks = [seg["speaker"] for seg in _parsed_segments(actual) if seg.get("speaker")]
    if not exp_spks:
        return 1.0 if not act_spks else 0.0
    if not act_spks:
        return 0.0
    if len(exp_spks) == len(act_spks):
        matches = sum(1 for e, a in zip(exp_spks, act_spks) if e == a)
        return matches / len(exp_spks)
    scores = [
        max(
            1.0 if exp_spk == act_spk else 0.0
            for act_spk in act_spks
        )
        for exp_spk in exp_spks
    ]
    return sum(scores) / len(exp_spks)


def timestamp_alignment_score(expected: str, actual: str) -> float:
    """Average timestamp alignment score across matched reference turns."""
    exp_segments = _parsed_segments(expected)
    act_segments = _parsed_segments(actual)
    if not exp_segments:
        return 1.0 if not act_segments else 0.0
    if not act_segments:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []
    for exp in exp_segments:
        if not exp["text"]:
            continue
        best = 0.0
        best_overlap = 0.0
        for act in act_segments:
            overlap = _overlap(exp, act)
            if overlap <= 0:
                continue
            ts_score = _timestamp_score(exp, act)
            if overlap > best_overlap or (overlap == best_overlap and ts_score > best):
                best = ts_score
                best_overlap = overlap
        duration = max(0.1, exp["end"] - exp["start"])
        scores.append(best if best_overlap > 0 else 0.0)
        weights.append(duration)

    if not scores:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / (sum(weights) or 1.0)


def content_accuracy(expected: str, actual: str) -> float:
    """Text-only accuracy (speaker/timestamp ignored)."""
    exp_norm = normalize_transcript_corpus(expected)
    act_norm = normalize_transcript_corpus(actual)
    line_score = line_best_match_accuracy(expected, actual)
    time_score = time_aligned_accuracy(expected, actual)
    corpus_score = SequenceMatcher(None, exp_norm, act_norm).ratio()
    return max(line_score, time_score, corpus_score)


def transcript_accuracy(expected: str, actual: str) -> float:
    """Golden accuracy: text match with speaker labels on each line."""
    text_score = content_accuracy(expected, actual)
    strict_score = strict_transcript_accuracy(expected, actual)
    speaker_score = speaker_sequence_score(expected, actual)

    if text_score >= 0.95 and speaker_score >= 0.99:
        return text_score
    if text_score >= 0.93 and speaker_score >= 0.99:
        return text_score * 0.98
    return min(text_score, strict_score, (text_score * 0.7) + (speaker_score * 0.3))


def count_speaker_lines(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = _SPEAKER_RE.search(line)
        if match:
            spk = match.group(1)
            counts[spk] = counts.get(spk, 0) + 1
    return counts


def accuracy_report(expected: str, actual: str) -> dict:
    exp_norm = normalize_transcript_corpus(expected)
    act_norm = normalize_transcript_corpus(actual)
    strict = strict_transcript_accuracy(expected, actual)
    speaker_seq = speaker_sequence_score(expected, actual)
    content = content_accuracy(expected, actual)
    ts_align = timestamp_alignment_score(expected, actual)
    return {
        "accuracy": transcript_accuracy(expected, actual),
        "content_accuracy": content,
        "timestamp_accuracy": ts_align,
        "strict_accuracy": strict,
        "speaker_sequence": speaker_seq,
        "line_best_match": line_best_match_accuracy(expected, actual),
        "time_aligned": time_aligned_accuracy(expected, actual),
        "corpus_similarity": SequenceMatcher(None, exp_norm, act_norm).ratio(),
        "expected_chars": len(exp_norm),
        "actual_chars": len(act_norm),
        "coverage_ratio": len(act_norm) / max(1, len(exp_norm)),
        "expected_lines": sum(
            1 for ln in expected.splitlines()
            if ln.strip() and not _is_non_dialogue_line(ln)
        ),
        "actual_lines": sum(
            1 for ln in actual.splitlines()
            if ln.strip() and not _is_non_dialogue_line(ln)
        ),
        "expected_speakers": count_speaker_lines(expected),
        "actual_speakers": count_speaker_lines(actual),
    }
