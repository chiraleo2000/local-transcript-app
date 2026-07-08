#!/usr/bin/env python3
"""Run pytest coverage then SonarQube scanner (SonarCloud or self-hosted).

Requires:
  SONAR_HOST_URL  e.g. https://sonarcloud.io
  SONAR_TOKEN     analysis token

Optional:
  SONAR_ORGANIZATION  required for SonarCloud

Example:
    python scripts/run_sonar_scan.py
    python scripts/run_sonar_scan.py --skip-scan   # coverage only (CI without token)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonarQube scan with pytest coverage")
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="run coverage only; do not invoke sonar-scanner",
    )
    parser.add_argument(
        "--pytest-args",
        default="tests/test_asr_performance.py tests/test_asr_quality.py "
        "tests/test_meeting_eval.py tests/test_golden_automation.py "
        "tests/test_whisper_decode.py tests/test_cuda_recovery.py -q",
        help="pytest arguments (default: fast unit suite)",
    )
    return parser.parse_args()


def _run_coverage(pytest_args: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_args.split(),
        "--cov=backend",
        "--cov=engines",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing:skip-covered",
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, check=False)
    return proc.returncode


def _run_sonar_scanner() -> int:
    host = os.getenv("SONAR_HOST_URL", "").strip()
    token = os.getenv("SONAR_TOKEN", "").strip()
    if not host or not token:
        print(
            "SONAR_HOST_URL and SONAR_TOKEN must be set to run sonar-scanner.",
            file=sys.stderr,
        )
        return 2

    scanner = shutil.which("sonar-scanner")
    if not scanner:
        print(
            "sonar-scanner not found on PATH. Install from "
            "https://docs.sonarqube.org/latest/analyzing-source-code/scanners/sonarscanner/",
            file=sys.stderr,
        )
        return 2

    env = {**os.environ, "SONAR_HOST_URL": host, "SONAR_TOKEN": token}
    org = os.getenv("SONAR_ORGANIZATION", "").strip()
    cmd = [scanner, f"-Dsonar.host.url={host}"]
    if org:
        cmd.append(f"-Dsonar.organization={org}")

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env, check=False)
    return proc.returncode


def main() -> int:
    args = _parse_args()
    cov_code = _run_coverage(args.pytest_args)
    if cov_code != 0:
        print(f"pytest/coverage failed (exit {cov_code})", file=sys.stderr)
        return cov_code

    if args.skip_scan:
        print("Coverage OK; --skip-scan set, skipping sonar-scanner.", flush=True)
        return 0

    return _run_sonar_scanner()


if __name__ == "__main__":
    raise SystemExit(main())
