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


def _swap_speakers(text: str) -> str:
    return (
        text.replace("SPEAKER_01", "SPEAKER__TMP")
        .replace("SPEAKER_02", "SPEAKER_01")
        .replace("SPEAKER__TMP", "SPEAKER_02")
    )


def _overlap(a: dict, b: dict) -> float:
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    return max(0.0, end - start)


def time_aligned_accuracy(expected: str, actual: str) -> float:
    """Score by matching utterances on timestamp overlap."""
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


def transcript_accuracy(expected: str, actual: str) -> float:
    """Return dialogue accuracy in [0, 1]."""
    line_score = line_best_match_accuracy(expected, actual)
    time_score = time_aligned_accuracy(expected, actual)
    corpus_scores = [
        SequenceMatcher(
            None,
            normalize_transcript_corpus(expected),
            normalize_transcript_corpus(actual),
        ).ratio(),
        SequenceMatcher(
            None,
            normalize_transcript_corpus(expected),
            normalize_transcript_corpus(_swap_speakers(actual)),
        ).ratio(),
    ]
    return max(line_score, time_score, max(corpus_scores))


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
    return {
        "accuracy": transcript_accuracy(expected, actual),
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
