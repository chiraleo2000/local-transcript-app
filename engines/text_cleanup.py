"""Transcript text cleanup — Whisper repetition / tail hallucination removal."""

from __future__ import annotations

import os
import re

_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
_LATIN_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")

# Frequent Thai ASR spelling variants (Typhoon/Whisper).
_THAI_ASR_VARIANTS: tuple[tuple[str, str], ...] = (
    ("พูนวิลล่า", "พูลวิลล่า"),
    ("ภูวิลล่า", "พูลวิลล่า"),
    ("พูนวิลลา", "พูลวิลล่า"),
    ("เช็ก", "เช็ค"),
    ("list", "ลิสต์"),
    ("เลยเลย", "เลย"),
    ("แพ้ริม", "แพริม"),
    ("ล่องแพ้เปียก", "ล่องแพเปียก"),
    ("ล่องแพ้", "ล่องแพ"),
    ("นอนแพ้", "นอนแพ"),
    ("คลื่นลม", "คลื่นลม"),
    ("สดสด", "สด ๆ"),
    ("บ้านหลังหนึ่ง", "บ้านหลังนึง"),
    ("2 ห้องนอน", "2 ห้องนอน"),
    ("คอได้ฟิล", "พอได้ฟีล"),
    ("พอได้ฟิล", "พอได้ฟีล"),
)


def fix_common_thai_asr_variants(text: str) -> str:
    """Fix frequent Thai ASR spelling variants."""
    if not text:
        return text
    for src, dst in _THAI_ASR_VARIANTS:
        text = text.replace(src, dst)
    return text


def _collapse_char_runs(text: str, min_run: int = 4) -> str:
    """Collapse garbled same-character runs (e.g. สสสสสสส → ส)."""
    if min_run < 2:
        return text
    return re.sub(rf"(.)\1{{{min_run - 1},}}", r"\1", text)


def _collapse_spaced_phrase_repeats(text: str) -> str:
    """Collapse 2+ consecutive identical space-separated phrases."""
    for n in range(20, 0, -1):
        inner = r"(?:\S+[ \t]+)" * (n - 1) + r"\S+"
        pattern = rf"({inner})(?:[ \t]+\1){{1,}}"
        text = re.sub(pattern, r"\1", text)
    return text


def _collapse_compact_repeats(text: str) -> str:
    """Collapse consecutive identical substrings (Thai often has no spaces)."""
    for unit_len in range(40, 1, -1):
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


def _collapse_thai_token_spacing(text: str) -> str:
    """Join over-segmented Thai ASR tokens while keeping Latin words separate."""
    if not _env_bool("ASR_CLEANUP_THAI_SPACING", True):
        return text
    tokens = text.split()
    if len(tokens) < 4:
        return text
    thai_tokens = [token for token in tokens if _THAI_CHAR_RE.search(token)]
    if len(thai_tokens) < len(tokens) * 0.8:
        return text
    avg_len = sum(len(token) for token in thai_tokens) / len(thai_tokens)
    if avg_len > 3.5:
        return text
    parts: list[str] = []
    buffer = ""
    for token in tokens:
        if _THAI_CHAR_RE.search(token) and not _LATIN_OR_DIGIT_RE.search(token):
            buffer += token
            continue
        if buffer:
            parts.append(buffer)
            buffer = ""
        parts.append(token)
    if buffer:
        parts.append(buffer)
    return " ".join(parts)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clean_transcript_text(text: str) -> str:
    """Remove repetition loops and garbled tails from ASR output."""
    if not text or not text.strip():
        return text
    cleaned = text.strip()
    cleaned = _collapse_char_runs(cleaned)
    cleaned = _collapse_thai_token_spacing(cleaned)
    cleaned = _collapse_spaced_phrase_repeats(cleaned)
    cleaned = _collapse_compact_repeats(cleaned)
    cleaned = _collapse_repeated_suffix(cleaned)
    cleaned = _collapse_spaced_phrase_repeats(cleaned)
    cleaned = _collapse_compact_repeats(cleaned)
    cleaned = fix_common_thai_asr_variants(cleaned)
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
