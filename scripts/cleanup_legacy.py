#!/usr/bin/env python3
"""Remove legacy release artifacts from older version lines."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LEGACY_DIRS = ("release/v1.2.1", "release/v1.2.2", "release/v1.2.3", "release/v1.2.4")
LEGACY_FILES = (
    "RELEASE_NOTES_v1.2.0.md",
    "RELEASE_NOTES_v1.2.1.md",
    "RELEASE_NOTES_v1.2.2.md",
    "RELEASE_NOTES_v1.2.3.md",
    "RELEASE_NOTES_v1.2.4.md",
    "RUN_INSTRUCTIONS.md",
)


def main() -> int:
    removed = 0
    for rel in LEGACY_DIRS:
        path = REPO_ROOT / rel
        if path.is_dir():
            shutil.rmtree(path)
            print(f"removed dir: {rel}")
            removed += 1
    for rel in LEGACY_FILES:
        path = REPO_ROOT / rel
        if path.is_file():
            path.unlink()
            print(f"removed file: {rel}")
            removed += 1
    for pattern in ("release/v1.2.2/*.exe", "release/v1.2.2/*.zip"):
        for path in REPO_ROOT.glob(pattern):
            path.unlink()
            print(f"removed file: {path.relative_to(REPO_ROOT)}")
            removed += 1
    print(f"cleanup done ({removed} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
