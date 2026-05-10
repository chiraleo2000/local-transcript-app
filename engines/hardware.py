"""Compatibility wrapper for the backend hardware policy."""

from backend.services.hardware_policy import detect_hardware, hardware_summary


__all__ = ["detect_hardware", "hardware_summary"]
