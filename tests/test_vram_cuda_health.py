"""CUDA health probe used to avoid corrupt-context ASR reloads."""

from unittest.mock import MagicMock, patch

import pytest


def test_cuda_device_healthy_success():
    from backend import vram_state

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    probe = MagicMock()
    torch.zeros.return_value = probe

    with patch.dict("sys.modules", {"torch": torch}):
        assert vram_state.cuda_device_healthy() is True
    torch.zeros.assert_called_once_with(1, device="cuda")
    torch.cuda.synchronize.assert_called_once()


def test_cuda_device_healthy_failure_recovers():
    from backend import vram_state

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.zeros.side_effect = RuntimeError("cudaErrorUnknown")

    with patch.object(vram_state, "recover_cuda") as recover:
        with patch.dict("sys.modules", {"torch": torch}):
            assert vram_state.cuda_device_healthy() is False
    recover.assert_called_once()


def test_recover_cuda_no_torch():
    from backend import vram_state

    with patch.dict("sys.modules", {"torch": None}):
        vram_state.recover_cuda()  # should not raise
