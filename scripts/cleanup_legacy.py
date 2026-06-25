#!/usr/bin/env python3
"""Remove legacy release binaries and duplicate docs (dry-run by default)."""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEGACY_DIRS = ("release/v1.2.1",)
LEGACY_FILES = (
    "RELEASE_NOTES_v1.2.0.md",
    "RELEASE_NOTES_v1.2.1.md",
)
LEGACY_GLOBS = (
    "release/v1.2.2/*.exe",
    "release/v1.2.2/*.zip",
)


def _targets() -> list[Path]:
    out: list[Path] = []
    for rel in LEGACY_DIRS:
        out.append(ROOT / rel)
    for rel in LEGACY_FILES:
        out.append(ROOT / rel)
    for pattern in LEGACY_GLOBS:
        out.extend(Path(p) for p in glob.glob(str(ROOT / pattern)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove legacy release artifacts.")
    parser.add_argument("--apply", action="store_true", help="Delete files (default is dry-run).")
    args = parser.parse_args()

    for target in _targets():
        if not target.exists():
            continue
        if not args.apply:
            print(f"[dry-run] would remove: {target.relative_to(ROOT)}")
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"removed: {target.relative_to(ROOT)}")

    if not args.apply:
        print("Dry run complete. Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
