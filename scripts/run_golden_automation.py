#!/usr/bin/env python3
"""Deploy Docker, then run golden-file automation for test-sample01."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.golden.bootstrap import bootstrap_golden_runtime

bootstrap_golden_runtime()

from tests.golden.config import GOLDEN_ACCURACY_ENV, apply_golden_env
from tests.golden.debug_log import debug_log

COMPOSE_FILE = REPO_ROOT / "docker-compose.gpu.yml"
SERVICE_NAME = "transcription"
CONTAINER_NAME = "transcription-service"


def _run_pytest() -> int:
    apply_golden_env()
    os.environ["RUN_GPU_INTEGRATION"] = "1"

    try:
        import pytest  # noqa: F401
    except ImportError:
        return _run_direct()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_golden_sample01.py",
        "-v",
        "--tb=short",
    ]
    print("Golden automation config:")
    for key, value in sorted(GOLDEN_ACCURACY_ENV.items()):
        print(f"  {key}={value}")
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def _run_direct() -> int:
    from tests.golden.config import CONFIG_PROFILES, apply_golden_env
    from tests.golden.runner import run_golden_fixture

    os.environ["RUN_GPU_INTEGRATION"] = "1"
    threshold = float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.95"))

    print("Golden automation profiles:", len(CONFIG_PROFILES))
    best_score = 0.0
    best_profile = 0

    for idx, profile_extra in enumerate(CONFIG_PROFILES):
        apply_golden_env(profile_extra)
        print(f"\n--- Profile {idx + 1}/{len(CONFIG_PROFILES)} ---")
        for key in sorted({**GOLDEN_ACCURACY_ENV, **profile_extra}):
            print(f"  {key}={os.getenv(key)}")

        try:
            outcome = run_golden_fixture(
                threshold=threshold,
                run_id=f"automation-p{idx + 1}",
                profile_extra=profile_extra,
            )
        except Exception as exc:
            debug_log(
                hypothesis_id="H2",
                location="scripts/run_golden_automation.py:error",
                message="golden run failed",
                data={"profile": idx + 1, "error": str(exc)[:500]},
                run_id=f"automation-p{idx + 1}",
            )
            print(f"\nProfile {idx + 1} FAILED: {exc}")
            continue

        score = outcome["report"]["accuracy"]
        best_score = max(best_score, score)
        if score >= best_score:
            best_profile = idx + 1
        elapsed = outcome.get("elapsed_s", 0)
        gpu = outcome.get("gpu_status", {})
        print(f"\nAccuracy: {score:.1%} (threshold {threshold:.0%})")
        print(
            f"Elapsed: {elapsed:.1f}s | GPU diar={gpu.get('diarization_device')} "
            f"asr={gpu.get('asr_device')}"
        )
        print(f"Report: {outcome['report']}")
        print(f"Output: {outcome['output_path']}")
        if outcome["passed"]:
            print(f"PASSED on profile {idx + 1}")
            return 0
        print(f"Profile {idx + 1} below threshold; trying next config...")

    print(f"\nFAILED: best accuracy {best_score:.1%} (profile {best_profile})")
    return 1


def _docker_service_running() -> bool:
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


def _wait_for_healthy(timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = subprocess.run(
            [
                "docker", "inspect",
                "--format", "{{.State.Health.Status}}",
                CONTAINER_NAME,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        status = (probe.stdout or "").strip()
        if status == "healthy":
            return True
        if status in {"", "none"} and _docker_service_running():
            return True
        time.sleep(5)
    return False


def _sync_into_container() -> None:
    """Copy golden harness + latest pipeline code into deployed container."""
    pairs = [
        ("tests", "/app/tests"),
        ("scripts/run_golden_automation.py", "/app/scripts/run_golden_automation.py"),
        ("backend", "/app/backend"),
        ("engines", "/app/engines"),
    ]
    for src, dst in pairs:
        subprocess.run(
            ["docker", "cp", str(REPO_ROOT / src), f"{CONTAINER_NAME}:{dst}"],
            cwd=REPO_ROOT,
            check=True,
        )


def _deploy_docker() -> None:
    print("Building and deploying Docker GPU stack...")
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("Waiting for transcription-service to become healthy...")
    if not _wait_for_healthy():
        raise RuntimeError(f"{CONTAINER_NAME} did not become healthy in time")
    print(f"{CONTAINER_NAME} is ready.")


def _run_exclusive_gpu_container() -> int:
    """Stop live service and run golden test with exclusive GPU access."""
    repo = str(REPO_ROOT)
    paused = False
    if _docker_service_running():
        print("Stopping transcription-service for exclusive GPU golden run...")
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "stop", "transcription"],
            cwd=REPO_ROOT,
            check=True,
        )
        paused = True

    cmd = [
        "docker", "compose", "-f", str(COMPOSE_FILE),
        "run", "--rm",
        "-e", "RUN_GPU_INTEGRATION=1",
        "-e", "GOLDEN_FIXTURE=sample01",
        "-e", "GOLDEN_REQUIRE_GPU=1",
        "-v", f"{repo}:/app",
        SERVICE_NAME,
        "python3", "scripts/run_golden_automation.py", "--in-container",
    ]
    print("Exclusive GPU golden run:")
    print(" ", " ".join(cmd))
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
    finally:
        if paused:
            print("Restarting transcription-service...")
            subprocess.run(
                ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "transcription"],
                cwd=REPO_ROOT,
                check=False,
            )


def _run_in_deployed_container() -> int:
    if not _docker_service_running():
        _deploy_docker()
    else:
        print(f"{CONTAINER_NAME} already running; rebuilding image...")
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
            cwd=REPO_ROOT,
            check=True,
        )
        if not _wait_for_healthy():
            raise RuntimeError(f"{CONTAINER_NAME} did not become healthy after rebuild")

    return _run_exclusive_gpu_container()


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden transcript automation")
    parser.add_argument("--in-container", action="store_true")
    parser.add_argument("--docker", action="store_true", help="run via ephemeral compose run")
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="build/start GPU stack, sync code, exec golden test in live container",
    )
    args = parser.parse_args()

    if args.in_container:
        return _run_direct()
    if args.deploy or os.getenv("GOLDEN_DEPLOY", "").strip().lower() in {
        "1", "true", "yes",
    }:
        return _run_in_deployed_container()
    if args.docker or os.getenv("GOLDEN_USE_DOCKER", "").strip().lower() in {
        "1", "true", "yes",
    }:
        return _run_in_deployed_container()
    return _run_direct()


if __name__ == "__main__":
    raise SystemExit(main())
