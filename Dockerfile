# ---------- Stage 1: Build dependencies ----------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (ffmpeg needed for audio preprocessing)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg libsndfile1 git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && python3 -m pip uninstall -y torchcodec || true

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
