"""Speaker-diarization evaluation against named-speaker meeting references.

Reference format (e.g. tests/309.txt):

    HH:MM:SS Speaker Name
    utterance text...
    HH:MM:SS Next Speaker
    ...

Turn end times are implied by the next turn's start. Hypotheses are either raw
diarization segments ({start, end, speaker}) or app transcript lines in the
``[HH:MM:SS → HH:MM:SS] [SPEAKER_XX]: text`` format.

Speaker labels are matched to reference names with an optimal (Hungarian)
assignment on overlapped speech time, so scores measure diarization quality
independent of arbitrary SPEAKER_XX numbering.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import re
from pathlib import Path

_NAMED_TURN_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2})\s+(\S.*)$")
_HYP_LINE_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2})\s*→\s*(?P<end>\d{2}:\d{2}:\d{2})\]\s*"
    r"(?:\[(?P<speaker>SPEAKER_\d+)\]:\s*)?(?P<body>.*)$",
    re.IGNORECASE,
)

# Canonical-name folding for reference speaker aliases/typos (309 fixture).
_ALIAS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"เกยรติศักดิ์", "เกียรติศักดิ์"),
)
_ALIAS_CANONICAL: tuple[tuple[str, str], ...] = (
    ("เจ้าหน้าที่ สพก.5 (อนุพงษ์)", "อนุพงษ์"),
    ("เจ้าหน้าที่ ศรชล. (สพก.5)", "อนุพงษ์"),
)
_LAST_TURN_FALLBACK_S = 8.0


def _ts_to_seconds(ts: str) -> float:
    hours, minutes, seconds = (int(part) for part in ts.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def canonical_speaker_name(name: str) -> str:
    name = name.strip()
    for pattern, repl in _ALIAS_PATTERNS:
        name = re.sub(pattern, repl, name)
    for alias, canonical in _ALIAS_CANONICAL:
        if name == alias:
            return canonical
    return name


def parse_named_reference(text: str, total_duration_s: float = 0.0) -> list[dict]:
    """Parse ``HH:MM:SS Speaker`` turn headers and utterance bodies."""
    lines = text.splitlines()
    raw: list[dict] = []
    idx = 0
    while idx < len(lines):
        match = _NAMED_TURN_RE.match(lines[idx].strip())
        if not match:
            idx += 1
            continue
        body_parts: list[str] = []
        idx += 1
        while idx < len(lines) and not _NAMED_TURN_RE.match(lines[idx].strip()):
            part = lines[idx].strip()
            if part:
                body_parts.append(part)
            idx += 1
        raw.append({
            "start": _ts_to_seconds(match.group(1)),
            "speaker": canonical_speaker_name(match.group(2)),
            "text": " ".join(body_parts).strip(),
        })
    turns: list[dict] = []
    for turn_idx, turn in enumerate(raw):
        if turn_idx + 1 < len(raw):
            end = max(turn["start"] + 0.5, raw[turn_idx + 1]["start"])
        elif total_duration_s > turn["start"]:
            end = total_duration_s
        else:
            end = turn["start"] + _LAST_TURN_FALLBACK_S
        turns.append({
            "start": turn["start"],
            "end": end,
            "speaker": turn["speaker"],
            "text": turn.get("text", ""),
        })
    return turns


def parse_hypothesis_transcript(text: str) -> list[dict]:
    """Parse app transcript lines into diarization segments with text bodies."""
    segments: list[dict] = []
    for line in text.splitlines():
        match = _HYP_LINE_RE.match(line.strip())
        if not match or not match.group("speaker"):
            continue
        segments.append({
            "start": _ts_to_seconds(match.group("start")),
            "end": _ts_to_seconds(match.group("end")),
            "speaker": match.group("speaker").upper(),
            "text": (match.group("body") or "").strip(),
        })
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _overlap_matrix(
    ref_turns: list[dict], hyp_segments: list[dict],
) -> tuple[list[str], list[str], list[list[float]]]:
    ref_names = sorted({t["speaker"] for t in ref_turns})
    hyp_names = sorted({s["speaker"] for s in hyp_segments})
    ref_idx = {name: i for i, name in enumerate(ref_names)}
    hyp_idx = {name: i for i, name in enumerate(hyp_names)}
    matrix = [[0.0] * len(hyp_names) for _ in ref_names]
    for ref in ref_turns:
        for hyp in hyp_segments:
            ov = _overlap(ref["start"], ref["end"], hyp["start"], hyp["end"])
            if ov > 0:
                matrix[ref_idx[ref["speaker"]]][hyp_idx[hyp["speaker"]]] += ov
    return ref_names, hyp_names, matrix


def optimal_speaker_mapping(
    ref_turns: list[dict], hyp_segments: list[dict],
) -> dict[str, str]:
    """Best 1:1 hyp-label -> ref-name mapping by overlapped speech time."""
    ref_names, hyp_names, matrix = _overlap_matrix(ref_turns, hyp_segments)
    if not ref_names or not hyp_names:
        return {}
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = -np.array(matrix)
        rows, cols = linear_sum_assignment(cost)
        return {
            hyp_names[c]: ref_names[r]
            for r, c in zip(rows, cols)
            if matrix[r][c] > 0
        }
    except ImportError:
        pairs = [
            (matrix[r][c], r, c)
            for r in range(len(ref_names))
            for c in range(len(hyp_names))
            if matrix[r][c] > 0
        ]
        pairs.sort(reverse=True)
        mapping: dict[str, str] = {}
        used_ref: set[int] = set()
        for _, r, c in pairs:
            if r in used_ref or hyp_names[c] in mapping:
                continue
            mapping[hyp_names[c]] = ref_names[r]
            used_ref.add(r)
        return mapping


def _hyp_speaker_at(
    hyp_segments: list[dict], start: float, end: float,
) -> str | None:
    """Hypothesis speaker with the largest overlap inside [start, end]."""
    totals: dict[str, float] = {}
    for seg in hyp_segments:
        ov = _overlap(start, end, seg["start"], seg["end"])
        if ov > 0:
            totals[seg["speaker"]] = totals.get(seg["speaker"], 0.0) + ov
    if not totals:
        return None
    return max(totals, key=totals.get)


def _boundary_deltas(ref_turns: list[dict], hyp_segments: list[dict]) -> list[float]:
    """Distance from each reference speaker-change to nearest hyp boundary."""
    hyp_bounds = sorted({round(seg["start"], 2) for seg in hyp_segments}
                        | {round(seg["end"], 2) for seg in hyp_segments})
    deltas: list[float] = []
    prev_speaker: str | None = None
    for turn in ref_turns:
        if turn["speaker"] == prev_speaker:
            prev_speaker = turn["speaker"]
            continue
        prev_speaker = turn["speaker"]
        if not hyp_bounds:
            deltas.append(float("inf"))
            continue
        target = turn["start"]
        deltas.append(min(abs(b - target) for b in hyp_bounds))
    return deltas


def parse_reference_turns_with_text(
    text: str, total_duration_s: float = 0.0,
) -> list[dict]:
    """Parse named turns including utterance body text for ASR parity scoring."""
    base_turns = parse_named_reference(text, total_duration_s=total_duration_s)
    if not base_turns:
        return []

    bodies: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _NAMED_TURN_RE.match(stripped):
            bodies.append(" ".join(current).strip())
            current = []
            continue
        if stripped:
            current.append(stripped)
    bodies.append(" ".join(current).strip())
    if len(bodies) == len(base_turns) + 1 and not bodies[0]:
        bodies = bodies[1:]

    turns: list[dict] = []
    for idx, turn in enumerate(base_turns):
        enriched = dict(turn)
        enriched["text"] = bodies[idx].strip() if idx < len(bodies) else ""
        turns.append(enriched)
    return turns


def parse_hypothesis_lines_with_text(text: str) -> list[dict]:
    """Parse app transcript lines into segments with utterance text."""
    return parse_hypothesis_transcript(text)


def _hyp_text_for_interval(
    hyp_segments: list[dict],
    start: float,
    end: float,
) -> str:
    """Collect hypothesis transcript text overlapping a reference interval."""
    parts: list[tuple[float, str]] = []
    for seg in hyp_segments:
        ov_start = max(start, seg["start"])
        ov_end = min(end, seg["end"])
        if ov_end - ov_start < 0.04:
            continue
        text = (seg.get("text") or "").strip()
        if text:
            parts.append((ov_start, text))
    parts.sort(key=lambda pair: pair[0])
    return " ".join(fragment for _, fragment in parts).strip()


def _char_ngram_f1(ref: str, hyp: str, n: int = 2) -> float:
    """Character n-gram F1 for Thai ASR parity (spacing-free, order-robust)."""
    from collections import Counter

    from tests.golden.accuracy import normalize_transcript_text

    ref_norm = normalize_transcript_text(ref)
    hyp_norm = normalize_transcript_text(hyp)
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0
    if len(ref_norm) < n or len(hyp_norm) < n:
        return SequenceMatcher(None, ref_norm, hyp_norm).ratio()
    ref_counts = Counter(ref_norm[i : i + n] for i in range(len(ref_norm) - n + 1))
    hyp_counts = Counter(hyp_norm[i : i + n] for i in range(len(hyp_norm) - n + 1))
    overlap = sum((ref_counts & hyp_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def turn_text_accuracy(
    ref_turns: list[dict],
    hyp_segments: list[dict],
    mapping: dict[str, str] | None = None,
) -> tuple[float, list[dict]]:
    """Score meeting transcript text against named reference turns.

    Uses corpus-level character bigram F1 after normalization (same Thai cleanup
    as sample01). The full hypothesis transcript is compared to the full
    reference corpus; per-turn interval snippets are reported for debugging only.
    """
    del mapping
    ref_parts: list[str] = []
    hyp_corpus_parts: list[str] = []
    per_turn_diffs: list[dict] = []
    for turn in ref_turns:
        ref_text = (turn.get("text") or "").strip()
        if not ref_text:
            continue
        ref_parts.append(ref_text)
        hyp_text = _hyp_text_for_interval(hyp_segments, turn["start"], turn["end"])
        turn_score = _char_ngram_f1(ref_text, hyp_text)
        per_turn_diffs.append({
            "start": turn["start"],
            "speaker": turn["speaker"],
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "ratio": round(turn_score, 4),
        })
    for seg in hyp_segments:
        text = (seg.get("text") or "").strip()
        if text:
            hyp_corpus_parts.append(text)
    if not ref_parts:
        return 1.0, per_turn_diffs
    corpus_score = _char_ngram_f1(
        " ".join(ref_parts),
        " ".join(hyp_corpus_parts),
    )
    return corpus_score, per_turn_diffs


def evaluate_meeting_diarization(
    ref_turns: list[dict],
    hyp_segments: list[dict],
    *,
    ref_text_turns: list[dict] | None = None,
    hyp_text_segments: list[dict] | None = None,
) -> dict:
    """Score hypothesis speakers/timestamps against a named reference."""
    ref_names, hyp_names, matrix = _overlap_matrix(ref_turns, hyp_segments)
    mapping = optimal_speaker_mapping(ref_turns, hyp_segments)

    ref_time: dict[str, float] = {}
    for turn in ref_turns:
        ref_time[turn["speaker"]] = ref_time.get(turn["speaker"], 0.0) + (
            turn["end"] - turn["start"]
        )
    total_ref_time = sum(ref_time.values()) or 1.0

    # Time-weighted attribution: reference time covered by the mapped speaker.
    attributed = 0.0
    covered = 0.0
    for turn in ref_turns:
        for seg in hyp_segments:
            ov = _overlap(turn["start"], turn["end"], seg["start"], seg["end"])
            if ov <= 0:
                continue
            covered += ov
            if mapping.get(seg["speaker"]) == turn["speaker"]:
                attributed += ov

    # Turn-level: majority hypothesis speaker of each reference turn.
    turn_hits = 0
    scored_turns = 0
    per_speaker_turns: dict[str, list[int]] = {}
    for turn in ref_turns:
        scored_turns += 1
        hyp_spk = _hyp_speaker_at(hyp_segments, turn["start"], turn["end"])
        hit = int(hyp_spk is not None and mapping.get(hyp_spk) == turn["speaker"])
        turn_hits += hit
        bucket = per_speaker_turns.setdefault(turn["speaker"], [0, 0])
        bucket[0] += hit
        bucket[1] += 1

    deltas = [d for d in _boundary_deltas(ref_turns, hyp_segments) if d != float("inf")]
    deltas_sorted = sorted(deltas)
    median_delta = deltas_sorted[len(deltas_sorted) // 2] if deltas_sorted else float("inf")

    def _within(tol: float) -> float:
        if not deltas:
            return 0.0
        return sum(1 for d in deltas if d <= tol) / len(deltas)

    per_speaker = {
        name: {
            "ref_time_s": round(ref_time.get(name, 0.0), 1),
            "turns": per_speaker_turns.get(name, [0, 0])[1],
            "turn_acc": round(
                per_speaker_turns.get(name, [0, 1])[0]
                / max(1, per_speaker_turns.get(name, [0, 1])[1]),
                3,
            ),
            "mapped_hyp": next(
                (hyp for hyp, ref in mapping.items() if ref == name), None
            ),
        }
        for name in ref_names
    }

    text_acc = 0.0
    per_turn_diffs: list[dict] = []
    ref_for_text = ref_text_turns if ref_text_turns is not None else ref_turns
    hyp_for_text = hyp_text_segments if hyp_text_segments is not None else hyp_segments
    if any((t.get("text") or "").strip() for t in ref_for_text):
        text_acc, per_turn_diffs = turn_text_accuracy(ref_for_text, hyp_for_text, mapping)

    return {
        "expected_speakers": len(ref_names),
        "detected_speakers": len(hyp_names),
        "speaker_count_match": len(hyp_names) == len(ref_names),
        "mapping": mapping,
        "speaker_time_accuracy": round(attributed / total_ref_time, 4),
        "speaker_time_accuracy_covered": round(attributed / (covered or 1.0), 4),
        "coverage": round(covered / total_ref_time, 4),
        "turn_accuracy": round(turn_hits / max(1, scored_turns), 4),
        "turns_scored": scored_turns,
        "boundary_median_s": round(median_delta, 2),
        "boundary_within_1s": round(_within(1.0), 4),
        "boundary_within_2s": round(_within(2.0), 4),
        "boundary_within_3s": round(_within(3.0), 4),
        "turn_text_accuracy": round(text_acc, 4),
        "per_turn_diffs": per_turn_diffs,
        "per_speaker": per_speaker,
    }


def load_reference_turns(path: Path, total_duration_s: float = 0.0) -> list[dict]:
    return parse_named_reference(
        path.read_text(encoding="utf-8"), total_duration_s=total_duration_s
    )
