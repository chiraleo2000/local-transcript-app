#!/usr/bin/env python3
"""Write or merge runtime .env keys (token + resource settings) after install.

Usage:
  python write_runtime_env.py --env-path PREFIX/.env \\
      --hf-token hf_xxx --cpu-threads 0 --force-backend openvino \\
      --min-ram-mb 8192 --min-cpu-threads 4 --min-vram-mb 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_env(text: str) -> list[tuple[str | None, str]]:
    """Return ordered (key|None, raw_line) pairs; key is None for comments/blank."""
    rows: list[tuple[str | None, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rows.append((None, line))
            continue
        key = stripped.split("=", 1)[0].strip()
        rows.append((key, line))
    return rows


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    rows = _parse_env(existing)
    seen: set[str] = set()
    out: list[str] = []
    for key, line in rows:
        if key is not None and key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- Installer / deploy resource settings ---")
        for key in missing:
            out.append(f"{key}={updates[key]}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge HF token + resource settings into .env")
    parser.add_argument("--env-path", required=True, type=Path)
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--cpu-threads", default="")
    parser.add_argument("--force-backend", default="")
    parser.add_argument("--ov-device", default="")
    parser.add_argument("--min-ram-mb", default="")
    parser.add_argument("--min-cpu-threads", default="")
    parser.add_argument("--min-vram-mb", default="")
    parser.add_argument("--ui-max-jobs", default="")
    parser.add_argument("--ui-gradio-concurrency", default="")
    args = parser.parse_args()

    updates: dict[str, str] = {}
    if args.hf_token.strip():
        updates["HF_TOKEN"] = args.hf_token.strip()
    if args.cpu_threads.strip() != "":
        updates["APP_CPU_THREADS"] = args.cpu_threads.strip()
    if args.force_backend.strip():
        updates["APP_FORCE_BACKEND"] = args.force_backend.strip().lower()
    if args.ov_device.strip():
        updates["OV_DEVICE"] = args.ov_device.strip().upper()
    if args.min_ram_mb.strip():
        updates["MIN_SYSTEM_RAM_MB"] = args.min_ram_mb.strip()
    if args.min_cpu_threads.strip():
        updates["MIN_CPU_THREADS"] = args.min_cpu_threads.strip()
    if args.min_vram_mb.strip():
        updates["MIN_NVIDIA_VRAM_MB"] = args.min_vram_mb.strip()
    if args.ui_max_jobs.strip():
        updates["UI_MAX_CONCURRENT_JOBS"] = args.ui_max_jobs.strip()
    if args.ui_gradio_concurrency.strip():
        updates["UI_GRADIO_TRANSCRIBE_CONCURRENCY"] = args.ui_gradio_concurrency.strip()

    # Always ensure documented floors when nothing else set.
    updates.setdefault("MIN_SYSTEM_RAM_MB", "8192")
    updates.setdefault("MIN_CPU_THREADS", "4")
    updates.setdefault("MIN_NVIDIA_VRAM_MB", "8192")
    updates.setdefault("APP_CPU_THREADS", "0")

    upsert_env(args.env_path, updates)
    print(f"Wrote resource settings to {args.env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
