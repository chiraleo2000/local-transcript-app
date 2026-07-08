#!/bin/bash
set -eu

cd /app

# One-off jobs (acceptance, env checks) bypass Gradio startup.
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

echo "[entrypoint] Verifying local model cache (offline)..."
python3 scripts/ensure_model_cache.py

echo "[entrypoint] Starting Local Transcript App..."
exec python3 app.py