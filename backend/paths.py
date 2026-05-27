"""Install-root resolution for dev, Docker, and PyInstaller desktop builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Writable install directory (models/, storage/, .env live here).

    PyInstaller onedir: the folder containing LocalTranscriptApp.exe.
    Development: repository root (parent of backend/).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    """Read-only bundled code root (_internal/ when frozen)."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        internal = app_root() / "_internal"
        if internal.is_dir():
            return internal
    return app_root()


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a config path relative to app_root when not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (app_root() / path).resolve()


def ensure_bundle_on_path() -> None:
    """Make bundled packages importable when running from PyInstaller."""
    if not is_frozen():
        return
    root = str(bundle_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    app = str(app_root())
    if app not in sys.path:
        sys.path.insert(0, app)
