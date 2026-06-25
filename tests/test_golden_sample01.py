"""Golden-file integration test: test-sample01.m4a vs expected transcript."""

from __future__ import annotations

import os

import pytest

from tests.golden.fixtures import active_fixture
from tests.golden.runner import run_golden_fixture

ACCURACY_THRESHOLD = float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.95"))


def _gpu_integration_enabled() -> bool:
    return os.getenv("RUN_GPU_INTEGRATION", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


@pytest.fixture(scope="module")
def golden_fixture():
    fixture = active_fixture("sample01")
    if not fixture.audio.is_file():
        pytest.skip(f"golden audio missing: {fixture.audio}")
    if not fixture.expected.is_file():
        pytest.skip(f"golden transcript missing: {fixture.expected}")
    return fixture


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_sample01_meets_golden_transcript(golden_fixture):
    if not _gpu_integration_enabled():
        pytest.skip("set RUN_GPU_INTEGRATION=1 to run golden GPU integration test")

    outcome = run_golden_fixture(golden_fixture, threshold=ACCURACY_THRESHOLD)
    assert outcome["passed"], (
        f"transcript accuracy {outcome['report']['accuracy']:.1%} < "
        f"{ACCURACY_THRESHOLD:.0%}\n"
        f"report={outcome['report']}\n"
        f"actual saved to {outcome['output_path']}"
    )
