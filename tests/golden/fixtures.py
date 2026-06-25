"""Golden fixture definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GoldenFixture:
    name: str
    audio: Path
    expected: Path
    output: Path
    max_speakers: int


FIXTURES: dict[str, GoldenFixture] = {
    "sample01": GoldenFixture(
        name="sample01",
        audio=REPO_ROOT / "tests" / "test-sample01.m4a",
        expected=REPO_ROOT / "tests" / "test-sample01.txt",
        output=REPO_ROOT / "tests" / "output" / "test-sample01_actual.txt",
        max_speakers=4,
    ),
}


def active_fixture(name: str | None = None) -> GoldenFixture:
    key = (name or "sample01").strip().lower()
    if key not in FIXTURES:
        known = ", ".join(sorted(FIXTURES))
        raise KeyError(f"unknown golden fixture {key!r}; known: {known}")
    return FIXTURES[key]
