"""Golden fixture definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.asr_performance import performance_target_seconds

REPO_ROOT = Path(__file__).resolve().parents[2]

_SM_KMUTT = (
    REPO_ROOT / "tests" / "SM-พี่วีระ x รองยุ่น ศรชล x ดร เบิร์ด kmutt.m4a"
)


@dataclass(frozen=True)
class GoldenFixture:
    name: str
    audio: Path
    expected: Path | None
    output: Path
    max_speakers: int = 0
    expected_speakers: int = 0
    min_speakers_first_minute: int = 0
    accuracy_threshold: float | None = 0.95
    check_performance: bool = True
    production_mode: bool = False
    # Named-speaker meeting reference (HH:MM:SS + Thai name headers) scored
    # with tests/golden/meeting_eval.py instead of the [SPEAKER_XX] scorer.
    named_reference: bool = False

    def requires_accuracy(self) -> bool:
        return self.accuracy_threshold is not None and self.expected is not None

    def performance_target_s(self) -> float:
        if not self.check_performance:
            return 0.0
        return performance_target_seconds(_probe_audio_duration_s(self.audio))

    def audio_duration_s(self) -> float:
        return _probe_audio_duration_s(self.audio)


def _probe_audio_duration_s(path: Path) -> float:
    try:
        import subprocess

        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return float((proc.stdout or "").strip())
    except (OSError, ValueError):
        return 0.0


FIXTURES: dict[str, GoldenFixture] = {
    "sample01": GoldenFixture(
        name="sample01",
        audio=REPO_ROOT / "tests" / "test-sample01.m4a",
        expected=REPO_ROOT / "tests" / "test-sample01.txt",
        output=REPO_ROOT / "tests" / "output" / "test-sample01_actual.txt",
        max_speakers=4,
        expected_speakers=4,
        accuracy_threshold=0.95,
        check_performance=True,
        production_mode=False,
    ),
    "recording172": GoldenFixture(
        name="recording172",
        audio=REPO_ROOT / "tests" / "Recording 172.wav",
        expected=None,
        output=REPO_ROOT / "tests" / "output" / "Recording_172_actual.txt",
        max_speakers=2,
        expected_speakers=2,
        accuracy_threshold=None,
        check_performance=True,
        production_mode=True,
    ),
    "recording19": GoldenFixture(
        name="recording19",
        audio=REPO_ROOT / "tests" / "Recording 19.wav",
        expected=None,
        output=REPO_ROOT / "tests" / "output" / "Recording_19_actual.txt",
        max_speakers=2,
        expected_speakers=2,
        accuracy_threshold=None,
        check_performance=True,
        production_mode=True,
    ),
    "recording6250": GoldenFixture(
        name="recording6250",
        audio=REPO_ROOT / "tests" / "Recording 6250.wav",
        expected=REPO_ROOT / "tests" / "Recording_6250_transcript.txt",
        output=REPO_ROOT / "tests" / "output" / "Recording_6250_actual.txt",
        max_speakers=2,
        expected_speakers=2,
        accuracy_threshold=0.95,
        check_performance=True,
        production_mode=True,
    ),
    "sm_kmutt": GoldenFixture(
        name="sm_kmutt",
        audio=_SM_KMUTT,
        expected=None,
        output=REPO_ROOT / "tests" / "output" / "SM_kmutt_actual.txt",
        max_speakers=4,
        expected_speakers=4,
        min_speakers_first_minute=2,
        accuracy_threshold=None,
        check_performance=True,
        production_mode=True,
    ),
    "recording47": GoldenFixture(
        name="recording47",
        audio=REPO_ROOT / "tests" / "Recording 47.wav",
        expected=None,
        output=REPO_ROOT / "tests" / "output" / "Recording_47_actual.txt",
        max_speakers=5,
        expected_speakers=5,
        accuracy_threshold=None,
        check_performance=True,
        production_mode=True,
    ),
    # 89.7-min 11-speaker Thai meeting (ศรชล x มจธ). Named-speaker reference;
    # scored on speaker count / attribution / boundary alignment.
    "meeting309": GoldenFixture(
        name="meeting309",
        audio=REPO_ROOT / "tests" / "309.m4a",
        expected=REPO_ROOT / "tests" / "309.txt",
        output=REPO_ROOT / "tests" / "output" / "309_actual.txt",
        max_speakers=11,
        expected_speakers=11,
        accuracy_threshold=None,
        check_performance=True,
        production_mode=True,
        named_reference=True,
    ),
}

ACCURACY_FIXTURES = ("sample01",)
LONG_PERF_FIXTURES = (
    "recording172",
    "recording19",
    "sm_kmutt",
    "recording47",
    "meeting309",
)


def active_fixture(name: str | None = None) -> GoldenFixture:
    key = (name or "sample01").strip().lower()
    if key not in FIXTURES:
        known = ", ".join(sorted(FIXTURES))
        raise KeyError(f"unknown golden fixture {key!r}; known: {known}")
    return FIXTURES[key]


def all_fixtures() -> list[GoldenFixture]:
    return [FIXTURES[name] for name in (*ACCURACY_FIXTURES, *LONG_PERF_FIXTURES)]


def long_perf_fixtures() -> list[GoldenFixture]:
    return [FIXTURES[name] for name in LONG_PERF_FIXTURES]
