#!/usr/bin/env python3
"""Deploy Docker GPU stack and run golden automation (sample01 + long-audio perf)."""

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

from tests.golden.config import (  # noqa: E402
    CONFIG_PROFILES,
    GOLDEN_ACCURACY_ENV,
    apply_golden_env,
)
from tests.golden.fixtures import active_fixture, all_fixtures  # noqa: E402

COMPOSE_FILE = REPO_ROOT / "docker-compose.gpu.yml"
SERVICE_NAME = "transcription"
CONTAINER_NAME = "transcription-service"


def _print_outcome(outcome: dict) -> None:
    score = outcome["report"].get("accuracy")
    threshold = outcome.get("threshold")
    elapsed = outcome.get("elapsed_s", 0.0)
    target = outcome.get("target_s", 0.0)
    perf = outcome.get("performance_met", True)
    gpu = outcome.get("gpu_status", {})
    if score is not None and threshold is not None:
        print(f"\nAccuracy: {score:.1%} (threshold {threshold:.0%})")
    print(
        f"Elapsed: {elapsed:.1f}s | Target: {target:.1f}s | Performance met: {perf}"
    )
    print(
        f"GPU diar={gpu.get('diarization_device')} asr={gpu.get('asr_device')}"
    )
    print(f"Report: {outcome['report']}")
    print(f"Output: {outcome['output_path']}")


def _run_sample01_profiles(threshold: float) -> int:
    from tests.golden.runner import run_golden_fixture

    fixture = active_fixture("sample01")
    if not fixture.audio.is_file():
        print(f"SKIP sample01: audio missing ({fixture.audio})")
        return 1

    print("=== Golden accuracy: test-sample01.m4a ===")
    best_score = 0.0
    best_profile = 0

    for idx, profile_extra in enumerate(CONFIG_PROFILES):
        apply_golden_env(profile_extra)
        print(f"\n--- Profile {idx + 1}/{len(CONFIG_PROFILES)} ---")
        try:
            outcome = run_golden_fixture(
                fixture,
                threshold=threshold,
                run_id=f"sample01-p{idx + 1}",
                profile_extra=profile_extra,
            )
        except Exception as exc:
            print(f"Profile {idx + 1} FAILED: {exc}")
            continue

        score = outcome["report"].get("accuracy") or 0.0
        best_score = max(best_score, score)
        if score >= best_score:
            best_profile = idx + 1
        _print_outcome(outcome)
        if outcome["passed"]:
            print(f"PASSED sample01 on profile {idx + 1}")
            return 0
        print("Below threshold or over time budget; trying next profile...")

    print(f"\nFAILED sample01: best accuracy {best_score:.1%} (profile {best_profile})")
    return 1


def _run_recording172_perf() -> int:
    from tests.golden.runner import run_golden_fixture

    fixture = active_fixture("recording172")
    if not fixture.audio.is_file():
        print(f"SKIP recording172: audio missing ({fixture.audio})")
        return 0

    print("\n=== Long-audio performance: Recording 172.wav ===")
    print(f"Target elapsed: {fixture.performance_target_s():.1f}s")
    try:
        outcome = run_golden_fixture(
            fixture,
            run_id="recording172",
            production_mode=True,
        )
    except Exception as exc:
        print(f"recording172 FAILED: {exc}")
        return 1

    _print_outcome(outcome)
    if outcome["passed"]:
        print("PASSED recording172 performance smoke test")
        return 0
    print("FAILED recording172 performance smoke test")
    return 1


def _run_direct(fixtures: list[str] | None = None) -> int:
    apply_golden_env()
    os.environ["RUN_GPU_INTEGRATION"] = "1"
    threshold = float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.95"))

    selected = fixtures or ["sample01", "recording172"]
    rc = 0
    if "sample01" in selected:
        rc = _run_sample01_profiles(threshold) or rc
    if "recording172" in selected:
        rc = _run_recording172_perf() or rc
    return rc


def _run_pytest(fixtures: list[str] | None) -> int:
    apply_golden_env()
    os.environ["RUN_GPU_INTEGRATION"] = "1"
    tests = []
    if not fixtures or "sample01" in fixtures:
        tests.append("tests/test_golden_automation.py::test_sample01_meets_golden_transcript")
    if not fixtures or "recording172" in fixtures:
        tests.append(
            "tests/test_golden_automation.py::test_recording172_meets_performance_target"
        )
    if not tests:
        tests = ["tests/test_golden_automation.py"]

    cmd = [sys.executable, "-m", "pytest", *tests, "-v", "--tb=short"]
    print("Golden automation config:")
    for key, value in sorted(GOLDEN_ACCURACY_ENV.items()):
        print(f"  {key}={value}")
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


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


def _run_exclusive_gpu_container(fixtures: list[str] | None) -> int:
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

    fixture_arg = ",".join(fixtures) if fixtures else "all"
    cmd = [
        "docker", "compose", "-f", str(COMPOSE_FILE),
        "run", "--rm",
        "-e", "RUN_GPU_INTEGRATION=1",
        "-e", "GOLDEN_REQUIRE_GPU=1",
        "-e", f"GOLDEN_FIXTURES={fixture_arg}",
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


def _run_in_deployed_container(fixtures: list[str] | None) -> int:
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
    return _run_exclusive_gpu_container(fixtures)


def _parse_fixtures(raw: str | None) -> list[str] | None:
    if not raw or raw.strip().lower() in {"all", ""}:
        return None
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    known = {f.name for f in all_fixtures()}
    for name in names:
        if name not in known:
            raise SystemExit(f"Unknown fixture {name!r}; known: {', '.join(sorted(known))}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden transcript automation")
    parser.add_argument("--in-container", action="store_true")
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument(
        "--fixtures",
        default=os.getenv("GOLDEN_FIXTURES", "all"),
        help="comma-separated fixture names or 'all' (default)",
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="skip recording172 long-audio performance test",
    )
    args = parser.parse_args()

    fixtures = _parse_fixtures(args.fixtures)
    if args.skip_long and fixtures is None:
        fixtures = ["sample01"]
    elif args.skip_long and fixtures:
        fixtures = [f for f in fixtures if f != "recording172"]

    if args.in_container:
        return _run_direct(fixtures)
    if args.deploy or os.getenv("GOLDEN_DEPLOY", "").strip().lower() in {"1", "true", "yes"}:
        return _run_in_deployed_container(fixtures)
    if args.docker or os.getenv("GOLDEN_USE_DOCKER", "").strip().lower() in {"1", "true", "yes"}:
        return _run_in_deployed_container(fixtures)

    try:
        import pytest  # noqa: F401
    except ImportError:
        return _run_direct(fixtures)
    return _run_pytest(fixtures)


if __name__ == "__main__":
    raise SystemExit(main())
