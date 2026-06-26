#!/bin/bash
set -e

echo "============================================================"
echo " Transcription Service - Run (Linux / Mac)"
echo " Usage:  ./run.sh          <-- run app directly (default)"
echo "         ./run.sh gui      <-- native desktop window (pywebview)"
echo "         ./run.sh docker   <-- Docker (GPU :7988 or OpenVINO :7987)"
echo "============================================================"
echo

# Resolve script directory so it works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set_model_env() {
    export APP_MODEL_ROOT="$SCRIPT_DIR/models"
    export HF_HOME="$SCRIPT_DIR/models/hf_cache"
    export HF_HUB_CACHE="$SCRIPT_DIR/models/hf_cache/hub"
    export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/models/hf_cache/hub"
    export TORCH_HOME="$SCRIPT_DIR/models/torch"
    export OV_CACHE_DIR="$SCRIPT_DIR/models/ov_cache"
    export HF_HUB_DISABLE_SYMLINKS_WARNING=1
    export TRANSFORMERS_OFFLINE=0
    export HF_HUB_OFFLINE=0
    export APP_AUTO_DOWNLOAD_MISSING_MODELS=1
    export ASR_ADAPTIVE_PERFORMANCE=1
    export ASR_TARGET_SHORT_MAX_S=600
    export ASR_TARGET_LONG_MAX_S=1800
    export ASR_TARGET_LONG_AUDIO_S=3600
    export ASR_TARGET_RT_RATIO_LONG=4
    export ASR_NUM_BEAMS_MAX=8
    export ASR_NUM_BEAMS_MIN=4
    export ASR_NUM_BEAMS=6
    export ASR_QUALITY_PROFILE=high
    export ASR_KEEP_PRELOADED=1
    export ASR_TURN_GUIDED_MERGE_GAP_S=0.55
    export DIARIZATION_PRELOAD_MODE=eager
    export DIARIZATION_KEEP_PRELOADED=1
    export DIARIZATION_PREPROCESS_SR=16000
    export DIARIZATION_GPU_CO_RESIDENT=0
    export DIARIZATION_DEVICE=cuda
    export DIARIZATION_PRELOAD_DEVICE=cpu
    export DIARIZATION_ALLOW_8GB_CUDA=1
    export DIARIZATION_CUDA_MIN_FREE_MB=768
    export DIARIZATION_CUDA_RUN_MIN_FREE_MB=512
    export ASR_UNLOAD_FOR_DIARIZATION=1
    export ASR_DEFAULT_ENGINES=Auto
    export ASR_AUTO_POLICY=quality
}

ensure_model_cache() {
    echo "[cache] Verifying local model cache under ./models/hf_cache/hub ..."
    python scripts/ensure_model_cache.py || {
        echo "[ERROR] Local model cache is incomplete. Check HF_TOKEN in .env and retry."
        exit 1
    }
}

# ============================================================
#  GUI / DESKTOP WINDOW MODE
# ============================================================
if [ "${1:-}" = "gui" ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment not found. Run ./setup.sh first."
        exit 1
    fi
    set_model_env
    export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    source venv/bin/activate
    ensure_model_cache
    echo "Starting Local Transcript App in native desktop window..."
    echo
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

    echo "[2/3] Detecting accelerator support in Docker..."
    if docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi &>/dev/null; then
        echo "      NVIDIA GPU available — using CUDA compose."
        COMPOSE_FILES="-f docker-compose.gpu.yml"
    elif [[ -f docker-compose.openvino.yml ]]; then
        echo "      No NVIDIA GPU in Docker — using OpenVINO/CPU AI compose."
        COMPOSE_FILES="-f docker-compose.openvino.yml"
    else
        echo "      GPU not available — using generic CPU compose."
        COMPOSE_FILES="-f docker-compose.yml"
        echo "      To enable NVIDIA GPU: install nvidia-container-toolkit and restart Docker."
    fi

    echo "[3/3] Docker deployment (BuildKit cached build)..."
    echo
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    docker compose $COMPOSE_FILES build
    docker compose $COMPOSE_FILES up -d
    if [[ "$COMPOSE_FILES" == *"gpu.yml"* ]]; then
        echo "Waiting for transcription-service health..."
        for _ in $(seq 1 24); do
            status=$(docker inspect --format '{{.State.Health.Status}}' transcription-service 2>/dev/null || true)
            if [ "$status" = "healthy" ]; then break; fi
            sleep 5
        done
        echo "GPU stack: http://localhost:7988"
        docker logs -f transcription-service
    else
        echo "OpenVINO/CPU stack: http://localhost:7987"
        docker logs -f transcription-service-openvino
    fi
    exit 0
fi

# ============================================================
#  DIRECT RUN MODE
# ============================================================
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

set_model_env

echo "[1/5] Checking Python..."
venv/bin/python --version

echo "[2/5] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "      nvidia-smi not found — will use OpenVINO / CPU fallback."
fi

echo "[3/5] Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    echo "      FFmpeg OK"
else
    echo "[WARNING] ffmpeg not found. Install: sudo apt install ffmpeg  (Mac: brew install ffmpeg)"
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
source venv/bin/activate
ensure_model_cache

echo "[5/5] Starting local transcript app on http://localhost:7896 ..."
echo
python app.py
