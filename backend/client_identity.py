"""Resolve client identity for multi-user workstation history and logging."""

from __future__ import annotations

from typing import Any


def client_ip_from_request(request: Any) -> str:
    """Best-effort client IP from Gradio/Starlette request (honors X-Forwarded-For)."""
    if request is None:
        return ""
    headers = getattr(request, "headers", None)
    if headers is not None:
        forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if forwarded:
            return str(forwarded).split(",")[0].strip()
        real_ip = headers.get("x-real-ip") or headers.get("X-Real-IP")
        if real_ip:
            return str(real_ip).strip()
    client = getattr(request, "client", None)
    if client is not None:
        host = getattr(client, "host", None)
        if host:
            return str(host).strip()
    return ""
