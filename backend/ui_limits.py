"""Browser UI memory limits — keep large media/transcripts out of the Gradio tab."""

from __future__ import annotations

import os

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".ts"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def ui_preview_max_mb() -> float:
    return float(_env_int("UI_PREVIEW_MAX_MB", 32))


def ui_preview_max_duration_s() -> float:
    return float(_env_int("UI_PREVIEW_MAX_DURATION_S", 300))


def ui_transcript_max_chars() -> int:
    return max(10_000, _env_int("UI_TRANSCRIPT_MAX_CHARS", 100_000))


def ui_transcript_max_lines() -> int:
    return max(1, _env_int("UI_TRANSCRIPT_MAX_LINES", 500))


def media_too_large_for_browser(path: str) -> tuple[bool, str]:
    """Return True when embedding this file in gr.Audio/gr.Video would crash the tab."""
    if not path or not os.path.isfile(path):
        return False, ""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    max_mb = ui_preview_max_mb()
    if size_mb > max_mb:
        return True, (
            f"Browser preview disabled ({size_mb:.0f} MB > {max_mb:.0f} MB). "
            "Large files crash Chrome if loaded into the player."
        )
    from backend.services.media_pipeline import audio_duration_seconds

    duration_s = audio_duration_seconds(path)
    max_dur = ui_preview_max_duration_s()
    if duration_s > max_dur:
        return True, (
            f"Browser preview disabled ({duration_s / 60:.0f} min > {max_dur / 60:.0f} min). "
            "Transcription still runs on the server — use Download .txt when done."
        )
    return False, ""


def format_media_info(path: str | None) -> str:
    if not path:
        return "No file selected."
    name = os.path.basename(path)
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    from backend.services.media_pipeline import audio_duration_seconds

    duration_s = audio_duration_seconds(path)
    too_large, reason = media_too_large_for_browser(path)
    lines = [
        f"**File:** `{name}`",
        f"**Size:** {size_mb:.1f} MB",
        f"**Duration:** {duration_s / 60:.1f} min",
    ]
    if too_large:
        lines.append(f"*{reason}*")
    else:
        lines.append("*Short clip — browser preview enabled below.*")
    return "\n".join(lines)


def display_transcript_text(text: str) -> str:
    """Cap transcript size shown in the browser to avoid tab OOM."""
    if not text:
        return text
    line_limit = ui_transcript_max_lines()
    lines = text.splitlines()
    if len(lines) > line_limit:
        omitted_lines = len(lines) - line_limit
        lines = lines[-line_limit:]
        text = "\n".join(lines)
        text = (
            f"{text}\n\n"
            f"… [Displaying last {line_limit} lines — {omitted_lines:,} earlier lines omitted — "
            "use **Download .txt** for the full transcript]"
        )
    limit = ui_transcript_max_chars()
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        f"{text[:limit]}\n\n"
        f"… [{omitted:,} characters omitted from browser view — use **Download .txt** for the full transcript]"
    )
