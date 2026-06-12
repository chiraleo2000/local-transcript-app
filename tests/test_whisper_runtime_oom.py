"""Tests for CUDA OOM retry chunk halving in whisper_runtime."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from engines.whisper_runtime import (
    _retry_pipe_on_oom,
    is_cuda_oom,
    is_cuda_recoverable,
    is_cuda_unknown_error,
    run_pipe_with_oom_retry,
)


class _CudaOomError(Exception):
    pass


def test_is_cuda_oom_matches_message():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))


def test_is_cuda_unknown_error_matches_accelerator_message():
    exc = RuntimeError(
        "CUDA error: unknown error\n"
        "Search for `cudaErrorUnknown' in https://docs.nvidia.com/cuda/cuda-runtime-api/"
    )
    assert is_cuda_unknown_error(exc)
    assert is_cuda_recoverable(exc)
    assert not is_cuda_oom(exc)


def test_run_pipe_with_oom_retry_recovers_cuda_unknown():
    calls: list[tuple[int, int | None]] = []

    def run_pipe(_pipe, _audio, _lang, _ts, batch, chunk=None):
        calls.append((batch, chunk))
        if len(calls) == 1:
            raise RuntimeError("CUDA error: unknown error")
        return {"text": "ok", "chunks": []}

    runtime = MagicMock()
    runtime.engine_name = "Pathumma"
    runtime.retry_chunk_length_s.return_value = 10
    runtime.clear_cuda_cache = MagicMock()
    runtime.reload_pipeline = MagicMock(return_value="reloaded-pipe")

    result, pipe = run_pipe_with_oom_retry(
        run_pipe,
        pipe="original-pipe",
        audio_input={"raw": []},
        language="thai",
        timestamp_mode=True,
        batch_size=4,
        runtime=runtime,
        audio_duration_s=120.0,
        chunk_length_s=30,
    )

    assert result == {"text": "ok", "chunks": []}
    assert pipe == "reloaded-pipe"
    assert calls == [(4, 30), (1, 10)]
    runtime.reload_pipeline.assert_called_once()


def test_retry_pipe_on_oom_halves_chunk_before_floor():
    calls: list[int | None] = []

    def run_pipe(_pipe, _audio, _lang, _ts, batch, chunk=None):
        calls.append(chunk)
        raise _CudaOomError("CUDA out of memory")

    runtime = MagicMock()
    runtime.engine_name = "TestEngine"
    runtime.retry_chunk_length_s.return_value = 10
    runtime.clear_cuda_cache = MagicMock()

    with pytest.raises(_CudaOomError):
        _retry_pipe_on_oom(
            run_pipe,
            pipe=object(),
            audio_input={"raw": []},
            language="thai",
            timestamp_mode=True,
            batch_size=1,
            runtime=runtime,
            chunk_s=32,
            retry_chunk_s=10,
            window_index=3,
        )

    assert calls[0] == 16
    assert calls[-1] == 10
