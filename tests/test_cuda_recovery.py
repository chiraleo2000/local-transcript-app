"""Unit tests for CUDA context self-heal / restart logic (no GPU required)."""

from __future__ import annotations

import importlib

import pytest

vram_state = importlib.import_module("backend.vram_state")


class TestCudaAutoRestartEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_enabled_values(self, monkeypatch, value):
        monkeypatch.setenv("CUDA_AUTO_RESTART", value)
        assert vram_state.cuda_auto_restart_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
    def test_disabled_values(self, monkeypatch, value):
        monkeypatch.setenv("CUDA_AUTO_RESTART", value)
        assert vram_state.cuda_auto_restart_enabled() is False

    def test_unset_is_disabled(self, monkeypatch):
        monkeypatch.delenv("CUDA_AUTO_RESTART", raising=False)
        assert vram_state.cuda_auto_restart_enabled() is False


class TestRequestCudaRestart:
    def test_returns_false_when_disabled(self, monkeypatch):
        monkeypatch.setenv("CUDA_AUTO_RESTART", "0")
        # Must not exit the process when auto-restart is off.
        assert vram_state.request_cuda_restart("test") is False

    def test_exits_process_when_enabled(self, monkeypatch):
        monkeypatch.setenv("CUDA_AUTO_RESTART", "1")
        calls: list[int] = []

        def fake_exit(code: int):
            calls.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(vram_state.os, "_exit", fake_exit)
        with pytest.raises(SystemExit):
            vram_state.request_cuda_restart("dead context")
        assert calls == [vram_state.CUDA_RESTART_EXIT_CODE]


def _force_cuda_available(monkeypatch) -> None:
    """Inject a fake torch whose CUDA is 'available' so the guard runs its body."""
    import sys
    import types

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


class TestEnsureCudaHealthyOrRestart:
    def test_healthy_context_returns_true(self, monkeypatch):
        _force_cuda_available(monkeypatch)
        monkeypatch.setattr(vram_state, "cuda_device_healthy", lambda: True)
        assert vram_state.ensure_cuda_healthy_or_restart("asr") is True

    def test_recovers_on_second_probe(self, monkeypatch):
        _force_cuda_available(monkeypatch)
        probes = iter([False, True])
        monkeypatch.setattr(vram_state, "cuda_device_healthy", lambda: next(probes))
        recovered: list[bool] = []
        monkeypatch.setattr(vram_state, "recover_cuda", lambda: recovered.append(True))
        assert vram_state.ensure_cuda_healthy_or_restart("asr") is True
        assert recovered == [True]

    def test_dead_context_restarts_when_enabled(self, monkeypatch):
        _force_cuda_available(monkeypatch)
        monkeypatch.setenv("CUDA_AUTO_RESTART", "1")
        monkeypatch.setattr(vram_state, "cuda_device_healthy", lambda: False)
        monkeypatch.setattr(vram_state, "recover_cuda", lambda: None)

        exits: list[int] = []

        def fake_exit(code: int):
            exits.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(vram_state.os, "_exit", fake_exit)
        with pytest.raises(SystemExit):
            vram_state.ensure_cuda_healthy_or_restart("asr")
        assert exits == [vram_state.CUDA_RESTART_EXIT_CODE]

    def test_dead_context_returns_false_when_disabled(self, monkeypatch):
        _force_cuda_available(monkeypatch)
        monkeypatch.setenv("CUDA_AUTO_RESTART", "0")
        monkeypatch.setattr(vram_state, "cuda_device_healthy", lambda: False)
        monkeypatch.setattr(vram_state, "recover_cuda", lambda: None)
        assert vram_state.ensure_cuda_healthy_or_restart("asr") is False
