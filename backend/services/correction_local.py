"""Optional local-only transcript correction."""

from __future__ import annotations

import json
import os
import time
import urllib.request


def correct_with_local_llm(text: str, model: str | None = None) -> tuple[str, float, str]:
    """Correct transcript text through a local Ollama server when available."""
    started = time.perf_counter()
    model_name = model or os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b")
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")
    prompt = (
        "Correct this transcript while preserving timestamps and speaker labels. "
        "Return only the corrected transcript.\n\n"
        f"{text}"
    )
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        corrected = (data.get("response") or "").strip()
        if corrected:
            return corrected, time.perf_counter() - started, f"Corrected locally with {model_name}."
        return text, time.perf_counter() - started, "Local LLM returned no text; kept original transcript."
    except (OSError, json.JSONDecodeError) as exc:
        return text, time.perf_counter() - started, f"Local correction skipped: {exc}"
