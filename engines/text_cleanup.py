"""Transcript text cleanup — Whisper repetition / tail hallucination removal."""

from __future__ import annotations

import re


def _collapse_char_runs(text: str, min_run: int = 5) -> str:
    """Collapse garbled same-character runs (e.g. สสสสสสส → ส)."""
    if min_run < 2:
        return text
    return re.sub(rf"(.)\1{{{min_run - 1},}}", r"\1", text)


def _collapse_spaced_phrase_repeats(text: str) -> str:
    """Collapse 2+ consecutive identical space-separated phrases."""
    for n in range(12, 0, -1):
        inner = r"(?:\S+[ \t]+)" * (n - 1) + r"\S+"
        pattern = rf"({inner})(?:[ \t]+\1){{1,}}"
        text = re.sub(pattern, r"\1", text)
    return text


def _collapse_compact_repeats(text: str) -> str:
    """Collapse consecutive identical substrings (Thai often has no spaces)."""
    for unit_len in range(25, 1, -1):
        min_extra = 1 if unit_len >= 4 else 2
        pattern = rf"(.{{{unit_len}}}?)\1{{{min_extra},}}"
        text = re.sub(pattern, r"\1", text)
    return text


def _collapse_repeated_suffix(text: str, min_repeats: int = 2) -> str:
    """Trim tail where the same phrase repeats many times (Whisper loop)."""
    text = text.strip()
    if len(text) < 6:
        return text
    for unit_len in range(min(30, len(text) // min_repeats), 1, -1):
        unit = text[-unit_len:]
        count = 0
        idx = len(text)
        while idx >= unit_len and text[idx - unit_len:idx] == unit:
            count += 1
            idx -= unit_len
        if count >= min_repeats:
            trimmed = text[: idx + unit_len].strip()
            if trimmed and len(trimmed) < len(text):
                return trimmed
    return text


def clean_transcript_text(text: str) -> str:
    """Remove repetition loops and garbled tails from ASR output."""
    if not text or not text.strip():
        return text
    cleaned = text.strip()
    cleaned = _collapse_char_runs(cleaned)
    cleaned = _collapse_spaced_phrase_repeats(cleaned)
    cleaned = _collapse_compact_repeats(cleaned)
    cleaned = _collapse_repeated_suffix(cleaned)
    cleaned = _collapse_spaced_phrase_repeats(cleaned)
    cleaned = _collapse_compact_repeats(cleaned)
    return cleaned.strip()


def _format_prefixed_line(match: re.Match[str]) -> str:
    prefix = (match.group(1) or "") + (match.group(2) or "")
    body = clean_transcript_text(match.group(3) or "")
    return f"{prefix}{body}" if body else prefix.rstrip()


def _clean_single_transcript_line(line: str, ts_speaker_re: re.Pattern[str]) -> str:
    if not line.strip():
        return line
    match = ts_speaker_re.match(line)
    if match:
        return _format_prefixed_line(match)
    match = re.match(r"^(\[[^\]]+\]\s*)*(\[SPEAKER_\d+\]:\s*)?(.*)$", line)
    if match:
        return _format_prefixed_line(match)
    return clean_transcript_text(line)


def clean_transcript_lines(text: str) -> str:
    """Clean each line of a multi-line transcript (speaker blocks preserved)."""
    if not text:
        return text
    ts_speaker_re = re.compile(
        r"^(\[\d{2}:\d{2}:\d{2} → \d{2}:\d{2}:\d{2}\] )?"
        r"(\[SPEAKER_\d+\]: )?"
        r"(.*)$",
    )
    lines = [_clean_single_transcript_line(line, ts_speaker_re) for line in text.splitlines()]
    return "\n".join(lines)
