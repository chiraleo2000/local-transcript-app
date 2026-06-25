"""Pytest hooks for golden automation."""

from __future__ import annotations

import pytest

from tests.golden.bootstrap import bootstrap_golden_runtime
from tests.golden.config import apply_golden_env

bootstrap_golden_runtime()


@pytest.fixture(scope="session", autouse=True)
def _golden_env_for_integration(request):
    """Apply accuracy env when running golden integration tests."""
    if request.node.get_closest_marker("golden"):
        apply_golden_env()
