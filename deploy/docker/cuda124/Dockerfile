# syntax=docker/dockerfile:1
# Build context MUST be the repository root.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    NVIDIA_REQUIRE_CUDA="cuda>=12.4" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libsndfile1 git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p models/hf_cache models/ov_cache storage/input storage/audio \
             storage/transcripts storage/jobs storage/logs config

COPY deploy/docker/cuda124/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && (python3 -m pip uninstall -y torchcodec || true) \
    && python3 -c "import site,os; \
stub=[os.path.join(p,'torchcodec-0.0.1.dist-info') for p in site.getsitepackages() if os.path.isdir(p)]; \
d=stub[0] if stub else '/usr/local/lib/python3.10/dist-packages/torchcodec-0.0.1.dist-info'; \
os.makedirs(d,exist_ok=True); \
open(os.path.join(d,'METADATA'),'w').write('Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n'); \
open(os.path.join(d,'RECORD'),'w').write(''); \
open(os.path.join(d,'INSTALLER'),'w').write('pip'); \
print('torchcodec stub created at',d)"

COPY engines/ engines/
COPY backend/ backend/
COPY torchcodec/ torchcodec/
COPY scripts/ scripts/
COPY tests/ tests/
COPY app.py .
COPY sitecustomize.py .

ENV PYTHONPATH=/app \
    APP_MODEL_ROOT=/app/models \
    HF_HOME=/app/models/hf_cache \
    HF_HUB_CACHE=/app/models/hf_cache/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    APP_AUTO_DOWNLOAD_MISSING_MODELS=false \
    HUGGINGFACE_HUB_CACHE=/app/models/hf_cache/hub \
    TORCH_HOME=/app/models/torch \
    OV_CACHE_DIR=/app/models/ov_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY

RUN sed -i 's/\r$//' scripts/docker_entrypoint.sh && chmod +x scripts/docker_entrypoint.sh \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7896

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7896/startup-events')" || exit 1

ENTRYPOINT ["/bin/bash", "scripts/docker_entrypoint.sh"]
