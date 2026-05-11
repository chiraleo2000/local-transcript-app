#!/bin/bash
set -e

echo "============================================================"
echo " Transcription Service - Run (Linux / Mac)"
echo " Usage:  ./run.sh          <-- run app directly (default)"
echo "         ./run.sh gui      <-- native desktop window (pywebview)"
echo "         ./run.sh docker   <-- run via Docker on port 7896"
echo "============================================================"
echo

# Resolve script directory so it works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
#  GUI / DESKTOP WINDOW MODE
# ============================================================
if [ "${1:-}" = "gui" ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment not found. Run ./setup.sh first."
        exit 1
    fi
    echo "Starting Local Transcript App in native desktop window..."
    echo
    export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    source venv/bin/activate
    python launcher.py
    exit 0
fi

# ============================================================
#  DOCKER MODE
# ============================================================
if [ "${1:-}" = "docker" ]; then
    echo "[1/3] Checking Docker..."
    command -v docker &>/dev/null || { echo "[ERROR] Docker not installed."; exit 1; }
    docker info &>/dev/null     || { echo "[ERROR] Docker daemon not running."; exit 1; }

    echo "[2/3] Detecting GPU support in Docker..."
    COMPOSE_FILES="-f docker-compose.yml"
    if docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi &>/dev/null; then
        echo "      GPU available in Docker — using CUDA mode."
        COMPOSE_FILES="-f docker-compose.gpu.yml"
    else
        echo "      GPU not available in Docker — using CPU/OpenVINO mode."
        echo "      To enable GPU: install nvidia-container-toolkit and restart Docker."
    fi

    echo "[3/3] Starting Docker container on http://localhost:7896 ..."
    echo
    docker compose $COMPOSE_FILES up --build -d
    docker logs -f transcription-service
    exit 0
fi

# ============================================================
#  DIRECT RUN MODE
# ============================================================
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo "[1/4] Checking Python..."
venv/bin/python --version

echo "[2/4] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "      nvidia-smi not found — will use OpenVINO / CPU fallback."
fi

echo "[3/4] Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "      FFmpeg OK"
else
    echo "[WARNING] ffmpeg not found. Install: sudo apt install ffmpeg  (Mac: brew install ffmpeg)"
fi

echo "[4/4] Starting local transcript app on http://localhost:7896 ..."
echo
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
source venv/bin/activate
python app.py
