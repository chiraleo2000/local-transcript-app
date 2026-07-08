#!/usr/bin/env python3
"""Stop all local GPU work and clear VRAM before a validation queue run.

Run this first — never start two GPU Python scripts in parallel on 8 GB VRAM.

Examples:
    python scripts/stop_gpu_work.py
    python scripts/stop_gpu_work.py --docker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _gpu_queue import (  # noqa: E402
    clear_gpu_state,
    kill_gpu_worker_processes,
    stop_docker_transcription,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop GPU workers and clear VRAM")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="also stop docker compose transcription service",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="terminate all Python processes in this repo (aggressive)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    killed = kill_gpu_worker_processes(include_all_python=args.all)
    if killed:
        print(f"Stopped GPU worker PIDs: {killed}", flush=True)
    else:
        print("No other GPU worker processes found.", flush=True)
    if args.docker:
        if stop_docker_transcription():
            print("Docker transcription service stopped.", flush=True)
        else:
            print("Docker transcription service was not running.", flush=True)
    clear_gpu_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
