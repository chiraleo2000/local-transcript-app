#!/bin/bash
set -eu

cd /app

# One-off jobs (acceptance, env checks) bypass Gradio startup.
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  echo "[entrypoint] HF token present — downloading any missing models..."
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 scripts/bootstrap_missing_models.py
fi

echo "[entrypoint] Verifying local model cache (offline)..."
python3 scripts/ensure_model_cache.py

echo "[entrypoint] Starting Local Transcript App..."
exec python3 app.py