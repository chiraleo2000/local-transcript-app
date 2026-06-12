"""ASR inference semaphore serializes CUDA calls across job threads."""

from __future__ import annotations

import threading
import time

from backend.services.asr_local import asr_inference_slot, max_concurrent_asr_inference


def test_max_concurrent_asr_inference_default():
    assert max_concurrent_asr_inference() >= 1


def test_asr_inference_slot_serializes(monkeypatch):
    monkeypatch.setenv("ASR_MAX_CONCURRENT_INFERENCE", "1")
    order: list[str] = []

    def worker(name: str, hold_s: float) -> None:
        with asr_inference_slot():
            order.append(f"{name}_start")
            time.sleep(hold_s)
            order.append(f"{name}_end")

    t1 = threading.Thread(target=worker, args=("a", 0.08))
    t2 = threading.Thread(target=worker, args=("b", 0))
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert order.index("a_end") < order.index("b_start")
