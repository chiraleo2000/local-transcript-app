#!/bin/bash
set -eu

cd /app

echo "[entrypoint] Verifying local model cache (offline)..."
python3 scripts/ensure_model_cache.py

echo "[entrypoint] Starting Local Transcript App..."
exec python3 app.py
