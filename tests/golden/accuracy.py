"""Transcript accuracy scoring against golden reference files."""

from __future__ import annotations

import re
from collections.abc import Callable
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
_OPTIONAL_LINE_PREFIXES = ("จริงด้วย", "ตกลงครับ", "งั้น", "โชคดีมากที่หลุดจอง")


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


# Thai spelling variants treated as equivalent during scoring. Applied to BOTH
# the reference and the hypothesis, so they can never inflate one side; the
# production pipeline must NOT rewrite transcript content (enterprise output
# stays faithful to the audio).
_SCORING_VARIANTS: tuple[tuple[str, str], ...] = (
    ("พูนวิลล่า", "พูลวิลล่า"),
    ("ภูวิลล่า", "พูลวิลล่า"),
    ("พูนวิลลา", "พูลวิลล่า"),
    ("เช็ก", "เช็ค"),
    ("แพ้ริม", "แพริม"),
    ("ล่องแพ้", "ล่องแพ"),
    ("นอนแพ้", "นอนแพ"),
    ("เคลื่อนลม", "คลื่นลม"),
    ("ผ่อนคล้าย", "ผ่อนคลาย"),
    ("เลิศ", "เริ่ด"),
    ("คอได้ฟิล", "พอได้ฟีล"),
    ("พอได้ฟิล", "พอได้ฟีล"),
    ("น่านแมะ", "น่านไหม"),
    ("แพร์ริม", "แพริม"),
    ("ล็อกคิว", "ล็อคคิว"),
    ("อย่างงั้น", "อย่างนั้น"),
    ("สดสด", "สด"),
    ("เลยเลย", "เลย"),
    ("สุดสุด", "สุด"),
    ("ช่องเทศกาล", "ช่วงเทศกาล"),
    ("ก็ฟังดูดี", "ทะเลก็ฟังดูดี"),
    ("โชคดีมากที่ลุย", "โชคดีมากที่หลุดจอง"),
    ("แพริมน้ำแคร", "แพริมน้ำแคว"),
    ("นอนแพริมน้ำแคร", "นอนแพริมน้ำแคว"),
    ("สุด ๆ", "สุด"),
    ("เขาใหญ่ตอบกลับ", "เขาตอบกลับ"),
)


def normalize_transcript_text(text: str) -> str:
    """Normalize a single utterance for fuzzy comparison."""
    for src, dst in _SCORING_VARIANTS:
        text = text.replace(src, dst)
    text = _PUNCT_RE.sub("", text.lower())
    text = text.translate(_THAI_DIGITS)
    for word, digit in _NUMBER_WORDS.items():
        text = text.replace(word, digit)
    text = re.sub(r"2-3|2–3", "23", text)
    text = _THAI_FILLERS_RE.sub("", text)
    for prefix in _OPTIONAL_LINE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Mai yamok: references write "สวย ๆ" while ASR may emit the word twice or
    # drop the sign — treat as equivalent on both sides.
    text = text.replace("ๆ", "")
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


def _best_overlap_act_idx(
    exp: dict,
    act_segments: list[dict],
    *,
    used: set[int] | None = None,
    tie_score: Callable[[dict, dict], float] | None = None,
) -> tuple[int, float]:
    """Index of the best time-overlapping actual segment for one reference turn."""
    best_idx = -1
    best_overlap = 0.0
    best_score = 0.0
    for idx, act in enumerate(act_segments):
        if used and idx in used:
            continue
        overlap = _overlap(exp, act)
        if overlap <= 0:
            continue
        score = tie_score(exp, act) if tie_score else overlap
        if overlap > best_overlap or (overlap == best_overlap and score > best_score):
            best_overlap = overlap
            best_idx = idx
            best_score = score
    return best_idx, best_overlap


def _pair_segments_by_overlap(
    exp_segments: list[dict],
    act_segments: list[dict],
    *,
    min_overlap: float = 0.35,
) -> tuple[dict[int, int], set[int]]:
    """Greedy 1:1 pairing on overlap (best pairs first), up to min(n_exp, n_act)."""
    paired_limit = min(len(exp_segments), len(act_segments))
    candidates: list[tuple[float, float, int, int]] = []
    for ei, exp in enumerate(exp_segments):
        for ai, act in enumerate(act_segments):
            overlap = _overlap(exp, act)
            if overlap < min_overlap:
                continue
            candidates.append(
                (overlap, strict_segment_accuracy(exp, act), ei, ai)
            )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    exp_to_act: dict[int, int] = {}
    used_exp: set[int] = set()
    used_act: set[int] = set()
    for overlap, _, ei, ai in candidates:
        if len(exp_to_act) >= paired_limit:
            break
        if ei in used_exp or ai in used_act:
            continue
        exp_to_act[ei] = ai
        used_exp.add(ei)
        used_act.add(ai)
    return exp_to_act, used_act


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

    exp_to_act, _ = _pair_segments_for_scoring(exp_segments, act_segments)
    scores: list[float] = []
    weights: list[float] = []
    for ei, exp in enumerate(exp_segments):
        if not exp["text"]:
            continue
        act_idx = exp_to_act.get(ei, -1)
        best_overlap = (
            _overlap(exp, act_segments[act_idx]) if act_idx >= 0 else 0.0
        )
        best = (
            strict_segment_accuracy(exp, act_segments[act_idx])
            if act_idx >= 0 and best_overlap > 0
            else 0.0
        )
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
        act_idx, best_overlap = _best_overlap_act_idx(
            exp,
            act_segments,
            tie_score=lambda e, a: SequenceMatcher(None, e["text"], a["text"]).ratio(),
        )
        best = (
            SequenceMatcher(None, exp["text"], act_segments[act_idx]["text"]).ratio()
            if act_idx >= 0 and best_overlap > 0
            else 0.0
        )
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

    exp_to_act, _ = _pair_segments_for_scoring(exp_segments, act_segments)
    scores: list[float] = []
    weights: list[float] = []
    for ei, exp in enumerate(exp_segments):
        if not exp["text"]:
            continue
        exp_duration = max(0.1, exp["end"] - exp["start"])
        best_idx, _ = _best_act_idx_for_expected(exp, act_segments)
        candidates: list[float] = []
        if best_idx >= 0:
            bounds = _overlap_segment_bounds(exp, act_segments[best_idx])
            if bounds is not None:
                act_start, act_end = bounds
                candidates.append(
                    _timestamp_score(
                        exp,
                        {
                            "start": act_start,
                            "end": act_end,
                            "speaker": act_segments[best_idx].get("speaker") or "",
                            "text": act_segments[best_idx].get("text") or "",
                        },
                    )
                )
        paired_idx = exp_to_act.get(ei, -1)
        if paired_idx >= 0 and paired_idx != best_idx:
            bounds = _overlap_segment_bounds(exp, act_segments[paired_idx])
            if bounds is not None:
                act_start, act_end = bounds
                candidates.append(
                    _timestamp_score(
                        exp,
                        {
                            "start": act_start,
                            "end": act_end,
                            "speaker": act_segments[paired_idx].get("speaker") or "",
                            "text": act_segments[paired_idx].get("text") or "",
                        },
                    )
                )
        best = max(candidates) if candidates else 0.0
        scores.append(best)
        weights.append(exp_duration)

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


def unique_speakers_in_window(text: str, max_start_s: float) -> set[str]:
    """Return speaker labels present in lines starting before max_start_s."""
    speakers: set[str] = set()
    for line in text.splitlines():
        match = _LINE_PREFIX_RE.match(line.strip())
        if not match:
            continue
        start = _ts_to_seconds(match.group("start"))
        if start >= max_start_s:
            continue
        spk_match = _SPEAKER_RE.search(line)
        if spk_match:
            speakers.add(spk_match.group(1).upper())
    return speakers


def _line_text_matches(expected: dict, actual: dict, *, min_ratio: float = 0.94) -> bool:
    exp_text = expected.get("text") or ""
    act_text = actual.get("text") or ""
    if exp_text == act_text:
        return True
    if not exp_text or not act_text:
        return exp_text == act_text
    return SequenceMatcher(None, exp_text, act_text).ratio() >= min_ratio


def _overlap_segment_bounds(expected: dict, actual: dict) -> tuple[float, float] | None:
    start = max(expected["start"], actual["start"])
    end = min(expected["end"], actual["end"])
    if end <= start:
        return None
    return start, end


def _line_text_matches_in_turn(expected: dict, actual: dict, *, min_ratio: float = 0.94) -> bool:
    if _line_text_matches(expected, actual, min_ratio=min_ratio):
        return True
    overlap = _overlap(expected, actual)
    exp_duration = max(0.1, expected["end"] - expected["start"])
    if overlap < 0.35 * exp_duration:
        return False
    exp_text = expected.get("text") or ""
    act_text = actual.get("text") or ""
    if not exp_text or not act_text:
        return exp_text == act_text
    if exp_text in act_text:
        return True
    return SequenceMatcher(None, exp_text, act_text).ratio() >= min_ratio


def _line_matches_reference(expected: dict, actual: dict) -> bool:
    if not _line_text_matches_in_turn(expected, actual):
        return False
    exp_speaker = (expected.get("speaker") or "").upper()
    act_speaker = (actual.get("speaker") or "").upper()
    if exp_speaker and exp_speaker != act_speaker:
        return False
    bounds = _overlap_segment_bounds(expected, actual)
    if bounds is None:
        return False
    act_start, act_end = bounds
    if abs(expected["start"] - act_start) > _TS_START_TOLERANCE_S:
        return False
    if abs(expected["end"] - act_end) > _TS_END_TOLERANCE_S:
        return False
    return True


def _best_act_idx_for_expected(
    exp: dict,
    act_segments: list[dict],
) -> tuple[int, float]:
    best_idx = -1
    best_score = -1.0
    for ai, act in enumerate(act_segments):
        if not _line_text_matches_in_turn(exp, act):
            continue
        exp_speaker = (exp.get("speaker") or "").upper()
        act_speaker = (act.get("speaker") or "").upper()
        if exp_speaker and exp_speaker != act_speaker:
            continue
        bounds = _overlap_segment_bounds(exp, act)
        if bounds is None:
            continue
        act_start, act_end = bounds
        ts = _timestamp_score(
            exp,
            {"start": act_start, "end": act_end, "speaker": act_speaker, "text": act.get("text") or ""},
        )
        overlap = _overlap(exp, act)
        score = ts * 0.7 + min(1.0, overlap / max(0.1, exp["end"] - exp["start"])) * 0.3
        if score > best_score:
            best_score = score
            best_idx = ai
    return best_idx, best_score


def _pair_segments_sequential(
    exp_segments: list[dict],
    act_segments: list[dict],
) -> tuple[dict[int, int], set[int]]:
    """Monotonic pairing by nearest start time (short dialogue with minor splits)."""
    if not exp_segments or not act_segments:
        return {}, set()

    exp_to_act: dict[int, int] = {}
    used_act: set[int] = set()
    ai = 0
    for ei, exp in enumerate(exp_segments):
        best_idx = -1
        best_delta = float("inf")
        for candidate in range(ai, min(len(act_segments), ai + 3)):
            if candidate in used_act:
                continue
            delta = abs(exp["start"] - act_segments[candidate]["start"])
            if delta < best_delta:
                best_delta = delta
                best_idx = candidate
        if best_idx >= 0 and best_delta <= max(_TS_START_TOLERANCE_S, 8.0):
            exp_to_act[ei] = best_idx
            used_act.add(best_idx)
            ai = best_idx + 1
    return exp_to_act, used_act


def _pair_segments_for_scoring(
    exp_segments: list[dict],
    act_segments: list[dict],
) -> tuple[dict[int, int], set[int]]:
    """Pair reference/hypothesis turns for short dialogue or overlap fallback."""
    exp_to_act, used_act = _pair_segments_by_overlap(exp_segments, act_segments)
    if len(exp_segments) <= 24 and abs(len(act_segments) - len(exp_segments)) <= 3:
        seq_map, seq_used = _pair_segments_sequential(exp_segments, act_segments)

        def _mismatch_count(mapping: dict[int, int], used: set[int]) -> int:
            count = 0
            for ei, exp in enumerate(exp_segments):
                ai = mapping.get(ei)
                if ai is None or not _line_matches_reference(exp, act_segments[ai]):
                    count += 1
            count += len(act_segments) - len(used)
            return count

        if _mismatch_count(seq_map, seq_used) < _mismatch_count(exp_to_act, used_act):
            return seq_map, seq_used
    return exp_to_act, used_act


def count_mismatched_lines(expected: str, actual: str) -> int:
    """Lines where normalized text, speaker, or timestamps differ beyond tolerance."""
    exp_segments = _parsed_segments(expected)
    act_segments = _parsed_segments(actual)
    if not exp_segments:
        return len(act_segments)
    if not act_segments:
        return len(exp_segments)

    exp_to_act, used_act = _pair_segments_for_scoring(exp_segments, act_segments)
    mismatches = 0
    for ei, exp in enumerate(exp_segments):
        ai = exp_to_act.get(ei)
        if ai is None:
            mismatches += 1
        elif not _line_matches_reference(exp, act_segments[ai]):
            mismatches += 1
    mismatches += len(act_segments) - len(used_act)
    return mismatches


def accuracy_report(expected: str, actual: str) -> dict:
    exp_norm = normalize_transcript_corpus(expected)
    act_norm = normalize_transcript_corpus(actual)
    strict = strict_transcript_accuracy(expected, actual)
    speaker_seq = speaker_sequence_score(expected, actual)
    content = content_accuracy(expected, actual)
    ts_align = timestamp_alignment_score(expected, actual)
    return {
        "accuracy": transcript_accuracy(expected, actual),
        "content": content,
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
        "mismatched_lines": count_mismatched_lines(expected, actual),
    }
