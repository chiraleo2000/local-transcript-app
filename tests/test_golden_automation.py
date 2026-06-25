"""Golden automation: sample01 accuracy + long-audio performance fixtures."""

from __future__ import annotations

import os

import pytest

from tests.golden.config import CONFIG_PROFILES, apply_golden_env
from tests.golden.fixtures import active_fixture
from tests.golden.runner import run_golden_fixture

ACCURACY_THRESHOLD = float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.95"))


def _gpu_integration_enabled() -> bool:
    return os.getenv("RUN_GPU_INTEGRATION", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _run_perf(fixture_name: str) -> None:
    fixture = active_fixture(fixture_name)
    if not fixture.audio.is_file():
        pytest.skip(f"audio missing: {fixture.audio}")
    if not _gpu_integration_enabled():
        pytest.skip("set RUN_GPU_INTEGRATION=1")
    outcome = run_golden_fixture(
        fixture,
        run_id=f"pytest-{fixture_name}",
        production_mode=True,
    )
    assert outcome["passed"], (
        f"{fixture_name} performance failed\n"
        f"elapsed={outcome['elapsed_s']:.1f}s target={outcome['target_s']:.1f}s\n"
        f"performance_met={outcome['performance_met']}\n"
        f"output={outcome['output_path']}"
    )


@pytest.fixture(scope="module")
def sample01_fixture():
    fixture = active_fixture("sample01")
    if not fixture.audio.is_file():
        pytest.skip(f"golden audio missing: {fixture.audio}")
    if fixture.expected is None or not fixture.expected.is_file():
        pytest.skip(f"golden transcript missing: {fixture.expected}")
    return fixture


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_sample01_meets_golden_transcript(sample01_fixture):
    if not _gpu_integration_enabled():
        pytest.skip("set RUN_GPU_INTEGRATION=1 to run golden GPU integration test")

    last_outcome = None
    for idx, profile_extra in enumerate(CONFIG_PROFILES):
        apply_golden_env(profile_extra)
        outcome = run_golden_fixture(
            sample01_fixture,
            threshold=ACCURACY_THRESHOLD,
            run_id=f"pytest-sample01-p{idx + 1}",
            profile_extra=profile_extra,
        )
        last_outcome = outcome
        if outcome["passed"]:
            return

    assert last_outcome is not None
    assert last_outcome["passed"], (
        f"transcript accuracy {last_outcome['report'].get('accuracy', 0):.1%} < "
        f"{ACCURACY_THRESHOLD:.0%} or exceeded time budget\n"
        f"report={last_outcome['report']}\n"
        f"elapsed={last_outcome['elapsed_s']:.1f}s target={last_outcome['target_s']:.1f}s\n"
        f"actual saved to {last_outcome['output_path']}"
    )


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_recording172_meets_performance_target():
    _run_perf("recording172")


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_recording19_meets_performance_target():
    _run_perf("recording19")


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_sample47_meets_performance_target():
    _run_perf("sample47")
