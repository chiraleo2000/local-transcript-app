# syntax=docker/dockerfile:1
# ---------- Stage 1: Build dependencies ----------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# System dependencies (ffmpeg needed for audio preprocessing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libsndfile1 git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-create persistent-mount directories so volumes work without root
RUN mkdir -p models/hf_cache models/ov_cache storage/input storage/audio \
             storage/transcripts storage/jobs storage/logs config

# Install Python deps first (layer cache; pip cache mount speeds rebuilds)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages -r requirements.txt \
    && (python3 -m pip uninstall --break-system-packages -y torchcodec || true) \
    && python3 -c "import site,os; \
stub=[os.path.join(p,'torchcodec-0.0.1.dist-info') for p in site.getsitepackages() if os.path.isdir(p)]; \
d=stub[0] if stub else '/usr/local/lib/python3.12/dist-packages/torchcodec-0.0.1.dist-info'; \
os.makedirs(d,exist_ok=True); \
open(os.path.join(d,'METADATA'),'w').write('Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n'); \
open(os.path.join(d,'RECORD'),'w').write(''); \
open(os.path.join(d,'INSTALLER'),'w').write('pip'); \
print('torchcodec stub created at',d)"

# Copy application code
COPY engines/ engines/
COPY backend/ backend/
COPY torchcodec/ torchcodec/
COPY scripts/ scripts/
COPY tests/ tests/
COPY app.py .
COPY sitecustomize.py .
# Runtime env comes from docker-compose env_file / environment (not baked into image).

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
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run as non-root (Sonar docker:S6471); bind-mounted volumes remain writable on Windows hosts.
RUN sed -i 's/\r$//' scripts/docker_entrypoint.sh && chmod +x scripts/docker_entrypoint.sh \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Gradio listens on 7896 inside the container (host maps 7987:7896 in compose)
EXPOSE 7896

# Health check — Gradio exposes startup-events at /startup-events (not /gradio_api/...)
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7896/startup-events')" || exit 1

ENTRYPOINT ["/bin/bash", "scripts/docker_entrypoint.sh"]
