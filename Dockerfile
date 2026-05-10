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

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && python3 -m pip uninstall -y torchcodec || true \
    && python3 -c "import site,os; \
stub=[os.path.join(p,'torchcodec-0.0.1.dist-info') for p in site.getsitepackages() if os.path.isdir(p)]; \
d=stub[0] if stub else '/usr/lib/python3/dist-packages/torchcodec-0.0.1.dist-info'; \
os.makedirs(d,exist_ok=True); \
open(os.path.join(d,'METADATA'),'w').write('Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n'); \
open(os.path.join(d,'RECORD'),'w').write(''); \
open(os.path.join(d,'INSTALLER'),'w').write('pip'); \
print('torchcodec stub created at',d)"

# Copy application code
COPY engines/ engines/
COPY backend/ backend/
COPY scripts/ scripts/
COPY app.py .
COPY .env* ./

# Gradio listens on 7896
EXPOSE 7896

# Health check — Gradio exposes a REST API at /gradio_api/
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7896/gradio_api/startup-events')" || exit 1

CMD ["python3", "app.py"]
