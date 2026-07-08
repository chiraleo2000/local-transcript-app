"""Shared GPU queue helpers — one 8 GB VRAM owner at a time (host Python or Docker)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.gpu.yml"
CONTAINER_NAME = "transcription-service"

_GPU_SCRIPT_MARKERS = (
    "run_enterprise_validation",
    "sweep_309_vbx",
    "diarize_count",
    "smoke_transcribe",
    "run_309_diar",
    "probe_309_centroids",
    "score_fixture",
    "check_sm_pipeline_speakers",
    "check_sm_speakers",
)

_T = TypeVar("_T")


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def docker_transcription_running() -> bool:
    probe = subprocess.run(
        [
            "docker", "ps",
            "--filter", f"name={CONTAINER_NAME}",
            "--filter", "status=running",
            "--format", "{{.Names}}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return CONTAINER_NAME in (probe.stdout or "")


def stop_docker_transcription() -> bool:
    """Stop transcription-service if running. Returns True when it was stopped."""
    if not docker_transcription_running():
        return False
    print(f"Stopping {CONTAINER_NAME} for exclusive host GPU access …", flush=True)
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "stop", "transcription"],
        cwd=REPO_ROOT,
        check=False,
    )
    return True


def restart_docker_transcription() -> None:
    print(f"Restarting {CONTAINER_NAME} …", flush=True)
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "transcription"],
        cwd=REPO_ROOT,
        check=False,
    )


def _restart_docker_enabled() -> bool:
    return os.getenv("GPU_QUEUE_RESTART_DOCKER", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _running_inside_docker() -> bool:
    return Path("/.dockerenv").exists()


@contextmanager
def exclusive_host_gpu_queue(*, stop_docker: bool = True):
    """Acquire cross-process GPU lock; pause Docker GPU service while held."""
    ensure_repo_on_path()
    from backend import vram_state

    if _running_inside_docker():
        stop_docker = False
    paused = False
    if stop_docker:
        paused = stop_docker_transcription()
    vram_state.acquire_gpu_job_lock()
    try:
        yield
    finally:
        vram_state.release_gpu_job_lock()
        if paused and _restart_docker_enabled():
            restart_docker_transcription()


def run_locked(main_fn: Callable[[], _T], *, stop_docker: bool = True) -> _T:
    with exclusive_host_gpu_queue(stop_docker=stop_docker):
        return main_fn()


def _is_gpu_worker_process(
    proc,
    *,
    my_pid: int,
    repo_lower: str,
    include_all_python: bool,
) -> bool:
    pid = int(proc.info["pid"] or 0)
    if pid == my_pid or pid <= 0:
        return False
    name = (proc.info.get("name") or "").lower()
    if "python" not in name:
        return False
    if include_all_python:
        return True
    cmdline = proc.info.get("cmdline") or []
    cmd = " ".join(str(part) for part in cmdline).lower()
    cwd = str(proc.info.get("cwd") or "").lower()
    if repo_lower not in cwd and repo_lower not in cmd:
        return False
    return any(marker in cmd for marker in _GPU_SCRIPT_MARKERS)


def kill_gpu_worker_processes(*, include_all_python: bool = False) -> list[int]:
    """Terminate other Python processes running GPU scripts in this repo."""
    import psutil

    ensure_repo_on_path()
    my_pid = os.getpid()
    repo_lower = str(REPO_ROOT).lower()
    killed: list[int] = []

    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            if not _is_gpu_worker_process(
                proc,
                my_pid=my_pid,
                repo_lower=repo_lower,
                include_all_python=include_all_python,
            ):
                continue
            pid = int(proc.info["pid"] or 0)
            proc.terminate()
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed:
        alive: list[psutil.Process] = []
        for pid in killed:
            try:
                alive.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        if alive:
            psutil.wait_procs(alive, timeout=3)
    return killed


def clear_gpu_state() -> None:
    """Release lock, unload models, and empty CUDA cache."""
    ensure_repo_on_path()
    from backend import vram_state
    from backend.model_registry import unload_all_models

    vram_state.release_gpu_job_lock()
    try:
        from engines.diarization import unload_model

        unload_model()
    except (ImportError, RuntimeError, AttributeError):
        pass
    unload_all_models()
    vram_state.teardown(aggressive=True)
    vram_state.recover_cuda()
    snap = vram_state.snapshot()
    print(
        f"VRAM cleared: free={snap.get('free_mb', 0)}MB "
        f"alloc={snap.get('allocated_mb', 0)}MB",
        flush=True,
    )
