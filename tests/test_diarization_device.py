"""Tests for pyannote pipeline device tracking (parameters() is hyperparams, not tensors)."""

from __future__ import annotations

import engines.diarization as diarization


class _FakePipeline:
    """Minimal stand-in: pyannote Pipeline.parameters() returns a dict, not nn.Parameter."""

    def __init__(self):
        self.moves: list[str] = []

    def parameters(self, instantiated=True):  # noqa: ARG002
        return {"segmentation": {"threshold": 0.42}}

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_move_pipeline_tracks_device_without_iterating_parameters(monkeypatch):
    monkeypatch.setattr(diarization, "_tracked_device", "cpu")
    pipe = _FakePipeline()

    diarization._move_pipeline_to_device(pipe, "cuda", "test")

    assert pipe.moves == ["cuda"]
    assert diarization._tracked_device == "cuda"


def test_move_pipeline_skips_redundant_to(monkeypatch):
    monkeypatch.setattr(diarization, "_tracked_device", "cpu")
    pipe = _FakePipeline()

    diarization._move_pipeline_to_device(pipe, "cpu", "test")

    assert pipe.moves == []


def test_release_after_job_keeps_cuda_when_co_resident(monkeypatch):
    pipe = _FakePipeline()
    monkeypatch.setattr(diarization, "_pipeline_cache", [pipe])
    monkeypatch.setattr(diarization, "_tracked_device", "cuda")
    monkeypatch.setenv("DIARIZATION_PRELOAD_MODE", "eager")
    monkeypatch.setenv("DIARIZATION_PRELOAD_DEVICE", "cuda")
    monkeypatch.setattr(
        "backend.services.asr_local.requires_sequential_gpu_models",
        lambda: False,
    )

    diarization.release_after_job()

    assert diarization._pipeline_cache == [pipe]
    assert pipe.moves == []
    assert diarization._tracked_device == "cuda"


def test_release_after_job_moves_cuda_to_preload_when_keep_preloaded(monkeypatch):
    pipe = _FakePipeline()
    monkeypatch.setattr(diarization, "_pipeline_cache", [pipe])
    monkeypatch.setattr(diarization, "_tracked_device", "cuda")
    monkeypatch.setenv("DIARIZATION_KEEP_PRELOADED", "true")
    monkeypatch.setenv("DIARIZATION_PRELOAD_DEVICE", "cpu")
    monkeypatch.setattr(diarization, "_device_for_preload", lambda _torch: "cpu")

    diarization.release_after_job()

    assert pipe.moves == ["cpu"]
    assert diarization._tracked_device == "cpu"


def test_select_diarization_device_cpu_on_8gb_class(monkeypatch):
    class _Cuda:
        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def device(name):
            return name

    monkeypatch.setenv("DIARIZATION_DEVICE", "auto")
    monkeypatch.setenv("DIARIZATION_GPU_CO_RESIDENT", "true")
    monkeypatch.setattr(diarization, "_cuda_vram_mb", lambda _t: 8192)
    monkeypatch.setattr(diarization, "_cuda_free_mb", lambda _t: 500)
    monkeypatch.setattr(
        "backend.services.asr_local.model_is_loaded",
        lambda _e: True,
    )

    device = diarization._select_diarization_device(_Cuda)
    assert str(device) == "cpu"


def test_select_diarization_device_cuda_when_asr_staged_off_gpu(monkeypatch):
    class _Cuda:
        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def device(name):
            return name

    monkeypatch.setenv("DIARIZATION_DEVICE", "auto")
    monkeypatch.setenv("DIARIZATION_GPU_CO_RESIDENT", "true")
    monkeypatch.setenv("DIARIZATION_CUDA_MIN_FREE_MB", "1536")
    monkeypatch.setattr(diarization, "_cuda_vram_mb", lambda _t: 8192)
    monkeypatch.setattr(diarization, "_cuda_free_mb", lambda _t: 5200)
    monkeypatch.setattr(
        "backend.services.asr_local.model_is_loaded",
        lambda _e: False,
    )

    device = diarization._select_diarization_device(_Cuda)
    assert str(device) == "cuda"


def test_select_diarization_device_cuda_when_co_resident_vram_ok(monkeypatch):
    class _Cuda:
        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def device(name):
            return name

    monkeypatch.setenv("DIARIZATION_DEVICE", "auto")
    monkeypatch.setenv("DIARIZATION_GPU_CO_RESIDENT", "true")
    monkeypatch.setenv("DIARIZATION_CUDA_MIN_FREE_MB", "3072")
    monkeypatch.setattr(diarization, "_cuda_vram_mb", lambda _t: 8192)
    monkeypatch.setattr(diarization, "_cuda_free_mb", lambda _t: 3500)
    monkeypatch.setattr(
        "backend.services.asr_local.model_is_loaded",
        lambda _e: True,
    )

    device = diarization._select_diarization_device(_Cuda)
    assert str(device) == "cuda"


def test_move_pipeline_falls_back_to_cpu_on_cuda_error(monkeypatch):
    class _FailCudaPipe(_FakePipeline):
        def to(self, device):
            if str(device) == "cuda":
                raise RuntimeError("CUDA error: unknown error")
            return super().to(device)

    monkeypatch.setattr(diarization, "_tracked_device", "cpu")
    monkeypatch.setattr(diarization, "_recover_cuda_after_failure", lambda _t: None)
    pipe = _FailCudaPipe()

    diarization._move_pipeline_to_device(pipe, "cuda", "test")

    assert pipe.moves == ["cpu"]
    assert diarization._tracked_device == "cpu"


def test_release_after_job_moves_to_cpu_when_preload_cpu(monkeypatch):
    pipe = _FakePipeline()
    monkeypatch.setattr(diarization, "_pipeline_cache", [pipe])
    monkeypatch.setattr(diarization, "_tracked_device", "cuda")
    monkeypatch.setenv("DIARIZATION_PRELOAD_MODE", "eager")
    monkeypatch.setenv("DIARIZATION_PRELOAD_DEVICE", "cpu")
    monkeypatch.setattr(diarization, "_device_for_preload", lambda _torch: "cpu")

    diarization.release_after_job()

    assert diarization._pipeline_cache == [pipe]
    assert pipe.moves == ["cpu"]
    assert diarization._tracked_device == "cpu"
