"""Optional local-only transcript correction."""

from __future__ import annotations

import json
import os
import time
import urllib.request


DEFAULT_THAI_SMALL_MODEL = "typhoon2-8b-instruct-q4"


def _prompt(text: str) -> str:
    return (
        "Correct this Thai/English transcript while preserving every timestamp "
        "and speaker label exactly. Improve punctuation, spacing, and obvious ASR "
        "errors only. Return only the corrected transcript.\n\n"
        f"{text}"
    )


def _post_json(endpoint: str, payload: dict, timeout: int = 600) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _correct_with_llamacpp(text: str, model_name: str) -> str:
    endpoint = os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
    max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful Thai transcript correction assistant.",
            },
            {"role": "user", "content": _prompt(text)},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = _post_json(endpoint, payload)
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()
    return ""


def _correct_with_ollama(text: str, model_name: str) -> str:
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
    payload = {"model": model_name, "prompt": _prompt(text), "stream": False}
    data = _post_json(endpoint, payload)
    return (data.get("response") or "").strip()


def correct_with_local_llm(text: str, model: str | None = None) -> tuple[str, float, str]:
    """Correct transcript text through a local llama.cpp or Ollama server."""
    started = time.perf_counter()
    provider = os.getenv("LOCAL_LLM_PROVIDER", "llamacpp").strip().lower()
    model_name = model or os.getenv("LOCAL_LLM_MODEL", DEFAULT_THAI_SMALL_MODEL)
    try:
        if provider in {"ollama", "ollama-compatible"}:
            corrected = _correct_with_ollama(text, model_name)
        else:
            corrected = _correct_with_llamacpp(text, model_name)
        if corrected:
            return (
                corrected,
                time.perf_counter() - started,
                f"Corrected locally with {provider}/{model_name}.",
            )
        return (
            text,
            time.perf_counter() - started,
            "Local LLM returned no text; kept original transcript.",
        )
    except (OSError, json.JSONDecodeError) as exc:
        return text, time.perf_counter() - started, f"Local correction skipped: {exc}"
