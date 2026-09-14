"""Golden automation for the two audio files in this checkout.

- tests/test-sample01.m4a against tests/test-sample01.txt
- tests/309.m4a against tests/309.txt
"""

from __future__ import annotations

import os

import pytest

from backend.enterprise_config import (
    ENTERPRISE_FIXTURE_OVERRIDES,
    ENTERPRISE_LONG_AUDIO_ENV,
)
from tests.golden.config import CONFIG_PROFILES, apply_golden_env
from tests.golden.fixtures import active_fixture
from tests.golden.runner import run_golden_fixture

ACCURACY_THRESHOLD = float(os.getenv("GOLDEN_ACCURACY_THRESHOLD", "0.90"))


def _gpu_integration_enabled() -> bool:
    return os.getenv("RUN_GPU_INTEGRATION", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _require_gpu() -> None:
    if not _gpu_integration_enabled():
        pytest.fail("RUN_GPU_INTEGRATION is required; GPU tests must not be skipped")


def _require_media(fixture) -> None:
    if not fixture.audio.is_file():
        pytest.fail(f"golden audio missing: {fixture.audio}")
    if fixture.expected is None or not fixture.expected.is_file():
        pytest.fail(f"golden transcript missing: {fixture.expected}")


def _fixture_profile(name: str) -> dict[str, str]:
    extra = dict(ENTERPRISE_FIXTURE_OVERRIDES.get(name, {}))
    if name == "meeting309":
        extra = {**ENTERPRISE_LONG_AUDIO_ENV, **extra}
    return extra


@pytest.fixture(scope="module")
def sample01_fixture():
    fixture = active_fixture("sample01")
    _require_media(fixture)
    return fixture


@pytest.fixture(scope="module")
def meeting309_fixture():
    fixture = active_fixture("meeting309")
    _require_media(fixture)
    return fixture


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_sample01_meets_golden_transcript(sample01_fixture):
    _require_gpu()

    last_outcome = None
    for idx, profile_extra in enumerate(CONFIG_PROFILES):
        extra = {**_fixture_profile("sample01"), **profile_extra}
        apply_golden_env(extra)
        outcome = run_golden_fixture(
            sample01_fixture,
            threshold=ACCURACY_THRESHOLD,
            run_id=f"pytest-sample01-p{idx + 1}",
            profile_extra=extra,
        )
        last_outcome = outcome
        if outcome["passed"]:
            return

    assert last_outcome is not None
    report = last_outcome["report"]
    assert last_outcome["passed"], (
        f"sample01 failed golden gate\n"
        f"content={report.get('content_accuracy', 0):.1%} "
        f"speaker={report.get('speaker_sequence', 0):.1%} "
        f"timestamp={report.get('timestamp_accuracy', 0):.1%} "
        f"strict={report.get('strict_accuracy', 0):.1%} "
        f"mismatched={report.get('mismatched_lines')}\n"
        f"elapsed={last_outcome['elapsed_s']:.1f}s "
        f"target={last_outcome['target_s']:.1f}s\n"
        f"actual saved to {last_outcome['output_path']}"
    )


@pytest.mark.golden
@pytest.mark.gpu
@pytest.mark.slow
def test_meeting309_meets_golden_transcript(meeting309_fixture):
    _require_gpu()
    extra = _fixture_profile("meeting309")
    apply_golden_env(extra)
    outcome = run_golden_fixture(
        meeting309_fixture,
        run_id="pytest-meeting309",
        profile_extra=extra,
    )
    report = outcome["report"]
    assert outcome["passed"], (
        f"meeting309 failed golden gate\n"
        f"speakers={report.get('detected_speakers')}/{report.get('expected_speakers')} "
        f"time_acc={report.get('speaker_time_accuracy')} "
        f"turn_acc={report.get('turn_accuracy')} "
        f"text_acc={report.get('turn_text_accuracy')}\n"
        f"checks={report.get('meeting_checks')}\n"
        f"elapsed={outcome['elapsed_s']:.1f}s target={outcome['target_s']:.1f}s\n"
        f"actual saved to {outcome['output_path']}"
    )
