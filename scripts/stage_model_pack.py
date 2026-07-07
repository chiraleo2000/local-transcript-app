"""Stage a built Model Pack into a release/install directory.

This is packaging glue:
- Slim releases ship without models; user places a Model Pack into ./models.
- Full releases copy ./models from a Model Pack into dist/LocalTranscriptApp/models.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage Model Pack into a target directory.")
    parser.add_argument("--pack", required=True, help="Path to a built models/ directory (Model Pack).")
    parser.add_argument("--target", required=True, help="Target app dir (e.g. dist/LocalTranscriptApp).")
    args = parser.parse_args()

    pack = Path(args.pack).resolve()
    target = Path(args.target).resolve()
    if not pack.is_dir():
        raise SystemExit(f"--pack is not a directory: {pack}")
    if not (pack / "manifest.json").is_file():
        raise SystemExit(f"Model Pack missing manifest.json: {pack}")
    if not target.is_dir():
        raise SystemExit(f"--target is not a directory: {target}")

    dst = target / "models"
    _copytree(pack, dst)
    # Make sure runtime uses offline-only.
    env_path = target / ".env"
    if env_path.is_file():
        txt = env_path.read_text(encoding="utf-8")
        if "HF_HUB_OFFLINE=1" not in txt:
            txt += "\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\nAPP_AUTO_DOWNLOAD_MISSING_MODELS=false\n"
            env_path.write_text(txt, encoding="utf-8")
    print(f"Staged Model Pack into {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

