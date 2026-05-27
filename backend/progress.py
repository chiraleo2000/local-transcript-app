"""Thread-safe transcription job progress for UI polling and /job/progress."""

from __future__ import annotations

import threading
import time
from typing import Any


class JobProgress:
    """Tracks phase, percent, elapsed time, and ETA for the active transcription job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.active = False
            self.job_id = ""
            self.phase = "idle"
            self.message = "Idle."
            self.percent = 0.0
            self.elapsed_s = 0.0
            self.remaining_s: float | None = None
            self.audio_duration_s = 0.0
            self._started_at: float | None = None
            self._asr_base = 45.0
            self._asr_span = 50.0

    def start(self, job_id: str = "") -> None:
        with self._lock:
            self.active = True
            self.job_id = job_id
            self.phase = "starting"
            self.message = "Starting transcription\u2026"
            self.percent = 1.0
            self._started_at = time.perf_counter()
            self.elapsed_s = 0.0
            self.remaining_s = None
            self.audio_duration_s = 0.0

    def set_job_id(self, job_id: str) -> None:
        with self._lock:
            self.job_id = job_id

    def set_phase(self, phase: str, message: str, percent: float) -> None:
        with self._lock:
            self.phase = phase
            self.message = message
            self.percent = min(99.0, max(self.percent, float(percent)))
            self._tick_elapsed()

    def set_audio_duration(self, duration_s: float) -> None:
        with self._lock:
            self.audio_duration_s = max(0.0, float(duration_s))

    def set_asr_window(self, current: int, total: int, message: str | None = None) -> None:
        with self._lock:
            frac = current / max(total, 1)
            self.phase = "asr"
            self.percent = self._asr_base + frac * self._asr_span
            self.message = message or f"Transcribing window {current}/{total}\u2026"
            self._tick_elapsed()

    def finish(self, message: str = "Complete.") -> None:
        with self._lock:
            self.phase = "done"
            self.message = message
            self.percent = 100.0
            self.active = False
            self.remaining_s = 0.0
            self._tick_elapsed()

    def fail(self, message: str) -> None:
        with self._lock:
            self.phase = "error"
            self.message = message
            self.active = False
            self._tick_elapsed()

    def _tick_elapsed(self) -> None:
        if self._started_at is not None:
            self.elapsed_s = time.perf_counter() - self._started_at
        if 0.0 < self.percent < 100.0:
            self.remaining_s = max(
                0.0,
                self.elapsed_s * (100.0 - self.percent) / self.percent,
            )
        elif self.percent >= 100.0:
            self.remaining_s = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._tick_elapsed()
            return {
                "active": self.active,
                "job_id": self.job_id,
                "phase": self.phase,
                "message": self.message,
                "percent": round(self.percent, 1),
                "elapsed_s": round(self.elapsed_s, 1),
                "remaining_s": (
                    None if self.remaining_s is None else round(self.remaining_s, 1)
                ),
                "audio_duration_s": round(self.audio_duration_s, 2),
            }


_PROGRESS = JobProgress()


def get_job_progress() -> JobProgress:
    return _PROGRESS
