#!/usr/bin/env python3
"""Run enterprise acceptance in Docker (plan v3).

Uses a one-off `docker compose run` job so validation owns the GPU
exclusively (never `docker exec` into the live Gradio service).

Examples:
    python scripts/run_docker_acceptance.py --only sample01 --tag verify
    python scripts/run_docker_acceptance.py --tag final
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO / "docker-compose.gpu.yml"
SERVICE = "transcription"
CONTAINER = "transcription-service"
COMPOSE_PROJECT = "local-transcript-app"

CUDA_RESTART_EXIT_CODE = 42

_ERROR_MARKERS = (
    "CUDA out of memory",
    "cuda out of memory",
    "ERROR:",
    "CUDA error:",
    "cudaErrorUnknown",
    "INTERNAL ASSERT FAILED",
)

_REQUIRED_ENV = {
    "ASR_CUDA_MEMORY_FRACTION": "0.92",
    "ASR_CUDA_BATCH_SIZE": "1",
    "ASR_NUM_BEAMS": "4",
    "UI_MAX_CONCURRENT_JOBS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "APP_AUTO_DOWNLOAD_MISSING_MODELS": "false",
    "APP_REQUIRE_DIARIZATION_MODELS": "true",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Docker-only enterprise acceptance")
    parser.add_argument("--tag", default="docker", help="output filename prefix")
    parser.add_argument(
        "--only",
        choices=("sample01", "meeting309", "all"),
        default="all",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failing fixture",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="skip docker compose build (image already up-to-date)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="forward --fast to in-container validation",
    )
    parser.add_argument(
        "--leave-stopped",
        action="store_true",
        help="do not restart the Gradio service after validation",
    )
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    check: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    kwargs: dict = {
        "cwd": REPO,
        "check": check,
    }
    if capture:
        kwargs["text"] = True
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, **kwargs)


def _run_streaming(cmd: list[str]) -> tuple[int, str]:
    """Run a command and stream merged stdout/stderr line-by-line."""
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    code = proc.wait()
    return code, "".join(chunks)


def _stop_host_gpu_workers() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from _gpu_queue import kill_gpu_worker_processes

    killed = kill_gpu_worker_processes()
    if killed:
        print(f"Stopped host GPU worker PIDs: {killed}", flush=True)


def _cleanup_stale_run_containers() -> list[str]:
    """Remove orphaned `docker compose run` containers that still hold the GPU."""
    proc = subprocess.run(
        [
            "docker", "ps", "-aq",
            "--filter", f"name={COMPOSE_PROJECT}-transcription-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    ids = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    removed: list[str] = []
    for cid in ids:
        stop = subprocess.run(
            ["docker", "rm", "-f", cid],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if stop.returncode == 0:
            removed.append(cid[:12])
    if removed:
        print(
            f"Removed stale compose-run container(s): {', '.join(removed)}",
            flush=True,
        )
        time.sleep(5)
    return removed


def _build_image(*, skip_build: bool) -> None:
    if skip_build:
        return
    env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    build = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "build"],
        cwd=REPO,
        env=env,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError("docker compose build failed")


def _stop_service() -> None:
    _run(["docker", "compose", "-f", str(COMPOSE_FILE), "stop", SERVICE], capture=False)


def _start_service() -> None:
    _run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", SERVICE], capture=False)


def _verify_image_env() -> list[str]:
    keys = list(_REQUIRED_ENV.keys())
    py = (
        "import os; keys=" + repr(keys) + "; "
        "print('\\n'.join(f'{k}='+repr(os.getenv(k,'')) for k in keys))"
    )
    proc = _run(
        [
            "docker", "compose", "-f", str(COMPOSE_FILE),
            "run", "-T", "--rm", "--no-deps", SERVICE,
            "python3", "-c", py,
        ],
    )
    actual_map: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        actual_map[key] = value.strip().strip("'\"")
    mismatches: list[str] = []
    for key, expected in _REQUIRED_ENV.items():
        actual = actual_map.get(key, "")
        if actual != expected:
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    return mismatches


def _run_validation(only: str, tag: str, *, fast: bool) -> tuple[int, str]:
    cmd = [
        "docker", "compose", "-f", str(COMPOSE_FILE),
        "run", "-T", "--rm", "--no-deps",
        SERVICE,
        "python3", "-u", "scripts/run_enterprise_validation.py",
        "--tag", tag,
        "--only", only,
    ]
    if fast:
        cmd.append("--fast")
    print(
        f"Starting compose run for {only} (logs stream below; meeting309 ~60–70 min) …",
        flush=True,
    )
    return _run_streaming(cmd)


def _output_has_errors(combined: str, tag: str, fixture: str) -> list[str]:
    hits = [line for line in combined.splitlines() if any(m in line for m in _ERROR_MARKERS)]
    path = REPO / "tests" / "output" / f"{tag}_{fixture}_actual.txt"
    if path.is_file():
        body = path.read_text(encoding="utf-8", errors="replace").lstrip()
        if body.startswith("ERROR:"):
            hits.append(f"transcript ERROR: prefix in {path.name}")
    return hits


def _run_fixture_acceptance(
    fixture: str,
    tag: str,
    *,
    fast: bool,
) -> int:
    """Run one fixture and return a non-zero exit code on failure."""
    print(f"\n=== DOCKER ACCEPTANCE: {fixture} ===", flush=True)
    code, combined = _run_validation(fixture, tag, fast=fast)
    if code == CUDA_RESTART_EXIT_CODE:
        print(
            f"CUDA auto-restart (exit {CUDA_RESTART_EXIT_CODE}); retrying {fixture} once …",
            flush=True,
        )
        _cleanup_stale_run_containers()
        _clear_stale_gpu_lock()
        time.sleep(15)
        code, combined = _run_validation(fixture, tag, fast=fast)
    hits = _output_has_errors(combined, tag, fixture)
    if hits:
        print("Errors detected:", file=sys.stderr)
        for line in hits[:20]:
            print(f"  {line}", file=sys.stderr)
    exit_code = 1 if hits else 0
    if code != 0:
        exit_code = code
    summary = REPO / "tests" / "output" / f"{tag}_enterprise_summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if not data.get("pass"):
            exit_code = 1
    return exit_code


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _verify_host_model_cache() -> bool:
    """Verify ./models/ on the host before stopping the Gradio service."""
    proc = subprocess.run(
        [sys.executable, "scripts/ensure_model_cache.py", "--strict-diarization"],
        cwd=REPO,
        check=False,
    )
    return proc.returncode == 0


def _clear_stale_gpu_lock() -> None:
    """Remove a leftover gpu.lock from a crashed compose-run container."""
    lock = REPO / ".cache" / "local-transcript-app" / "gpu.lock"
    if not lock.is_file():
        return
    try:
        lock.unlink()
        print(f"Removed stale GPU lock: {lock}", flush=True)
    except OSError as exc:
        print(f"Warning: could not remove GPU lock {lock} ({exc})", flush=True)


def _prepare_output_files(tag: str, fixtures: list[str]) -> None:
    """Remove stale acceptance outputs that block non-root Docker writes."""
    out = REPO / "tests" / "output"
    out.mkdir(parents=True, exist_ok=True)
    names = [f"{tag}_enterprise_summary.json"]
    names.extend(f"{tag}_{fx}_actual.txt" for fx in fixtures)
    for name in names:
        path = out / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def _prepare_acceptance_run(args: argparse.Namespace) -> list[str] | None:
    """Build image and verify env. Returns env mismatches, or None on success."""
    _configure_stdout()
    if not _verify_host_model_cache():
        print(
            "Local model cache incomplete. Run scripts/bootstrap_models.py once, "
            "then retry Docker acceptance.",
            file=sys.stderr,
        )
        return ["model_cache_incomplete"]
    _prepare_output_files(args.tag, _fixture_order(args.only))
    _stop_host_gpu_workers()
    _cleanup_stale_run_containers()
    _clear_stale_gpu_lock()
    _build_image(skip_build=args.skip_build)
    _stop_service()
    time.sleep(15)
    print("Verifying Docker image env …", flush=True)
    return _verify_image_env()


def _fixture_order(only: str) -> list[str]:
    if only == "all":
        return ["sample01", "meeting309"]
    return [only]


def main() -> int:
    args = _parse_args()
    env_bad = _prepare_acceptance_run(args)
    if env_bad:
        print("Docker image env mismatch:", file=sys.stderr)
        for line in env_bad:
            print(f"  {line}", file=sys.stderr)
        if not args.leave_stopped:
            _start_service()
        return 2

    exit_code = 0
    for fixture in _fixture_order(args.only):
        fixture_code = _run_fixture_acceptance(fixture, args.tag, fast=args.fast)
        if fixture_code != 0:
            exit_code = fixture_code
        if exit_code != 0 and args.fail_fast:
            break

    if not args.leave_stopped:
        print("Restarting Gradio service …", flush=True)
        _start_service()
        print("UI: http://localhost:7988", flush=True)

    print(f"\nDOCKER ACCEPTANCE exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
