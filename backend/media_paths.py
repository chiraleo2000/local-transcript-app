"""Normalize Gradio / API media upload values into filesystem path strings."""

from __future__ import annotations

import os
from typing import Any


def coerce_media_path(item: Any) -> str | None:
    """Extract a filesystem path from Gradio File / list / FileData values."""
    if item is None or item is False:
        return None
    if isinstance(item, (list, tuple)):
        for sub in item:
            path = coerce_media_path(sub)
            if path:
                return path
        return None
    if isinstance(item, dict):
        for key in ("path", "name"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    if isinstance(item, (str, bytes, os.PathLike)):
        text = os.fspath(item)
        return text if text else None
    name = getattr(item, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def normalize_media_paths(media: Any) -> list[str]:
    """Return a flat list of path strings (empty when nothing usable)."""
    if not media:
        return []
    items = list(media) if isinstance(media, (list, tuple)) else [media]
    paths: list[str] = []
    for item in items:
        path = coerce_media_path(item)
        if path:
            paths.append(path)
    return paths


__all__ = ["coerce_media_path", "normalize_media_paths"]
