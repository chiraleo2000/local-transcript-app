"""Encoding-tolerant .env loading (UTF-8 first, Windows-1252 fallback)."""

from __future__ import annotations

import io
from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_safe(path: str | Path, *, override: bool = False) -> bool:
    """Load a .env file even when it was saved as Windows-1252 / CP1252."""
    env_path = Path(path)
    if not env_path.is_file():
        return False

    raw = env_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
        # Persist UTF-8 so later tools (and Docker) do not hit the same error.
        env_path.write_bytes(text.encode("utf-8"))

    return bool(load_dotenv(stream=io.StringIO(text), override=override))
