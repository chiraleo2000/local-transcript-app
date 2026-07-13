# Local Transcript App

GPU-accelerated local audio/video transcription with speaker diarization.  
No cloud APIs. No telemetry. All processing stays on your machine.

**Version 1.2.6**

---

## Features

- **Typhoon Whisper Large v3** — default ASR engine (Thai-optimised, high accuracy)
- **Pathumma Whisper Thai Large v3** — fast alternative ASR engine
- **Speaker diarization** — `pyannote/speaker-diarization-community-1` (Sep 2025), fully local
- **Advanced diarization tuning** — segmentation threshold, clustering threshold, min cluster size, silence gap — adjustable live in the UI
- **Audio enhancement** — bandpass filter → spectral noise reduction → gate / compress / limiter chain (louder speech, minimal background)
- **Native desktop window** — pywebview wraps the Gradio UI; no browser required when using `launcher.py` or `LocalTranscriptApp.exe`
- **Docker GPU mode** — NVIDIA CUDA 13 + PyTorch; models cached in `./models/`
- **Strict 8 GB VRAM policy** — safe on RTX 4060 Laptop (8 GB); one model at a time, sequential engines, capped chunk size
- **OOM-safe long jobs** — disk-window ASR streaming (one slice in RAM), iterative CUDA chunk halving, UI transcript line/char caps, co-resident GPU preload with phase teardown
- **Public/LAN reverse proxy** — nginx or Windows IIS in front of Docker `:7988` (see [`deploy/README.md`](deploy/README.md))

---

## Networking

| Mode | URL | Notes |
| --- | --- | --- |
| Direct Python / GUI (`run.bat`, `launcher.py`) | `http://localhost:7896` | `GRADIO_SERVER_PORT=7896` |
| Docker NVIDIA GPU | `http://localhost:7988` | `docker-compose.gpu.yml` |
| Docker OpenVINO / CPU | `http://localhost:7987` | `docker-compose.openvino.yml` |
| Public / LAN (nginx or IIS) | your hostname | Reverse-proxy to `127.0.0.1:7988` — [`deploy/README.md`](deploy/README.md) |

`models/ov_cache` is **OpenVINO-only**. GPU Docker does not need it.

---

## Quick Start

### Option A — Docker (recommended)

> **NVIDIA GPU:** `docker compose -f docker-compose.gpu.yml up -d --build` → http://localhost:7988  
> **OpenVINO / CPU (no NVIDIA):** `docker compose -f docker-compose.openvino.yml up -d --build` → http://localhost:7987

```bat
REM Windows — auto-detect GPU vs OpenVINO
run.bat docker

# Linux / macOS
./run.sh docker
```

Open the native desktop window automatically by running the launcher instead:

```bat
REM Windows — native window (no browser needed)
run.bat gui

# Linux / macOS
./run.sh gui
```

Or open manually: **Docker GPU** → <http://localhost:7988> · **Docker OpenVINO** → <http://localhost:7987> · **Direct/GUI** → <http://localhost:7896>

---

### Option B — Direct Python (venv)

**First-time setup:**

```bat
REM Windows
setup.bat
copy .env.example .env
REM  Edit .env — set HF_TOKEN=hf_your_token_here
venv\Scripts\activate
python scripts\bootstrap_models.py
```

```bash
# Linux / macOS
./setup.sh
cp .env.example .env
# Edit .env — set HF_TOKEN=hf_your_token_here
source venv/bin/activate
python scripts/bootstrap_models.py
```

**Run the app:**

```bat
REM Windows — browser tab
run.bat

REM Windows — native desktop window
run.bat gui

REM Windows — force native OpenVINO (Intel Arc GPU / Intel NPU / CPU)
run.bat ov
REM Force GPU or NPU explicitly (if available)
run.bat ov-gpu
run.bat ov-npu
```

```bash
# Linux / macOS — browser tab
./run.sh

# Linux / macOS — native desktop window
./run.sh gui
```

---

## Desktop Shortcut / Icon

### Windows

```powershell
# Creates a shortcut on your Desktop
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

# Also add to Start Menu
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1 -StartMenu
```

Double-click **"Local Transcript App"** on your Desktop — it starts Docker (or the venv) automatically and opens the app in a native window.

### Linux

```bash
chmod +x scripts/create_shortcut.sh
./scripts/create_shortcut.sh               # Desktop shortcut only
./scripts/create_shortcut.sh --apps        # Also install to application menu
```

---

## How to Use the App

### 1. Upload a file

Drag-and-drop or click **"Audio or Video File"** to upload.  
Supported: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, and more.

### 2. Configure options

| Setting | Default | Notes |
| --- | --- | --- |
| **Language** | Thai | Dropdown — choose the spoken language |
| **Local ASR Engines** | Typhoon Whisper | Tick one or both; dual-engine runs sequentially on 8 GB GPUs |
| **Audio Enhancement** | Off | Bandpass + loudnorm → spectral NR → gate/compress/limit (stronger defaults in Docker GPU profile) |
| **Speaker Diarization** | Off | Identify and label individual speakers |
| **Max Speakers** | 3 | Appears when Diarization is enabled; pyannote auto-detects from 1 up to this limit |
| **Advanced Diarization Settings** | (accordion) | Short-clip adaptive tuning (automatic) or manual overrides — see below |

### 3. Advanced Diarization Settings (accordion)

Visible only when **Speaker Diarization** is checked.

**Automatic (recommended):** Clips under 90 seconds use adaptive pyannote params (lower `min_cluster_size`, looser clustering) and a single retry if only one speaker is detected. No manual setup required.

| Slider | Default (manual override) | Tune when… |
| --- | --- | --- |
| **Short clip / multi-speaker preset** | Off | One-click aggressive settings for &lt; 90 s clips with 2–3 speakers |
| **Segmentation Threshold** (0.42) | Activity threshold; lower = catches quieter / shorter turns | Speakers are missed or turns are cut short |
| **Min Silence Gap** (0.10 s) | Minimum gap before splitting a turn | Too many spurious splits — raise it |
| **Clustering Threshold** (0.60) | Speaker embedding distance; lower = more speakers separated | Different speakers are merged into one — lower it |
| **Min Cluster Size** (6) | Minimum segments to form a speaker cluster | Rare/short speakers are dropped — lower it |

### 4. Transcribe

Click **Transcribe**. A progress indicator appears while models run.  
Results appear in the **Typhoon Whisper** and **Pathumma Whisper** tabs.

### 5. Results

Each engine tab shows:

- **Transcript** — full text with speaker labels and timestamps when diarization is on
- **Elapsed Time** — wall time for that engine
- **Output name** — custom download filename (defaults to uploaded file stem)
- **Download .txt** — save the transcript

The **Previous transcripts** panel lists past jobs from `storage/jobs/` — load into the editor or download without re-running ASR. On workstation deployments with `UI_HISTORY_PER_CLIENT_IP=true`, the list is filtered by client IP (`X-Forwarded-For` behind nginx/IIS). Refresh the page during a long job to recover live progress (same browser tab).

Job metadata is saved to `storage/jobs/` as JSON (status: running → completed/cancelled/failed).

---

## Project Structure

```text
local-transcript-app/
├── app.py                        # Gradio UI
├── launcher.py                   # Desktop launcher (pywebview native window)
├── run.bat                       # Windows: run.bat | run.bat gui | run.bat docker
├── run.sh                        # Linux/Mac: ./run.sh | ./run.sh gui | ./run.sh docker
├── setup.bat / setup.sh          # First-time dependency install
├── docker-compose.gpu.yml        # Docker + NVIDIA GPU (CUDA)
├── docker-compose.openvino.yml   # Docker + OpenVINO / CPU
├── Dockerfile                    # CUDA 13 image (GPU compose)
├── Dockerfile.openvino           # OpenVINO / CPU AI image
├── .env                          # Runtime configuration (copy from .env.example)
├── .env.example                  # Documented configuration template
├── backend/
│   ├── pipeline.py               # Main transcription workflow
│   ├── storage.py                # Folder structure, transcript/job writes
│   └── services/
│       ├── asr_local.py          # ASR engine facade
│       ├── media_pipeline.py     # Normalise, enhance, diarize
│       └── hardware_policy.py    # GPU/CPU backend detection
├── engines/
│   ├── typhoon_asr.py            # Typhoon Whisper Large v3
│   ├── pathumma_asr.py           # Pathumma Whisper Thai Large v3
│   ├── diarization.py            # pyannote speaker diarization
│   ├── preprocess.py             # 3-stage audio enhancement
│   └── hardware.py               # Device helpers
├── scripts/
│   ├── bootstrap_models.py       # Download / cache all models
│   ├── create_shortcut.ps1       # Windows desktop shortcut generator
│   └── create_shortcut.sh        # Linux .desktop file generator
├── models/                       # Created at runtime — model cache (not in git)
└── storage/                      # Created at runtime — transcripts/jobs/logs
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and edit before first run.

### Required

```dotenv
HF_TOKEN=hf_your_token_here     # Required for gated models (Typhoon Whisper, pyannote)
```

### ASR

**Default engine:** Pathumma Whisper (Thai Large v3). Typhoon Whisper remains available in the UI.

```dotenv
ASR_DEFAULT_ENGINES=Pathumma Whisper
ASR_QUALITY_PROFILE=high               # high (default) | balanced — emergency low-VRAM only
ASR_PRELOAD_MODE=eager
MIN_NVIDIA_VRAM_MB=8192
ASR_HARD_MEMORY_SAFE=true
ASR_8GB_CLASS_MAX_MB=9000
PATHUMMA_MODEL_ID=nectec/Pathumma-whisper-th-large-v3
TYPHOON_MODEL_ID=typhoon-ai/typhoon-whisper-large-v3
```

### Quality profiles (accuracy-first default)

Profiles apply defaults **only when a variable is not already set** in `.env` or Docker `environment:`.

| Profile | Use case | ASR window | Diarization | VRAM / speed |
|---------|----------|------------|-------------|--------------|
| `high` (default) | Best speaker separation & continuity | 300s | Full multi-sample grid, 44.1 kHz preprocess, enhance always-on | Slow diarization by design |
| `balanced` | Emergency low-VRAM / OOM recovery only | 60s | Single-pass, faster | Lower accuracy |

Diarization with `DIARIZATION_ACCURACY_MODE=true` may take 2–5× longer than legacy fast paths — that is intentional.

```dotenv
ASR_QUALITY_PROFILE=high
ASR_LONG_FORM_WINDOW_S=300
ASR_LONG_FORM_OVERLAP_S=45
DIARIZATION_ACCURACY_MODE=true
DIARIZATION_MULTI_SAMPLE=true
DIARIZATION_MULTI_SAMPLE_PASSES=0        # 0 = full parameter grid
DIARIZATION_PREPROCESS_SR=44100
AUDIO_ENHANCE_WHEN_DIARIZATION=true
DIARIZATION_TRANSCRIPT_MERGE_GAP_S=1.0
PATHUMMA_WORD_TIMESTAMPS_ON_8GB=true
```

For OOM on 8 GB only: switch to `ASR_QUALITY_PROFILE=balanced` temporarily.

```dotenv
ASR_CHUNK_LENGTH_S=300
ASR_8GB_MAX_CHUNK_LENGTH_S=120
ASR_LONG_FORM_WINDOW_S=300
ASR_LONG_FORM_OVERLAP_S=45
ASR_CUDA_MEMORY_FRACTION=0.90
TYPHOON_WORD_TIMESTAMPS_ON_8GB=false
```

### Job history & refresh recovery

Completed and in-progress jobs write manifests to `storage/jobs/` (including `client_ip` when available). The **Previous transcripts** panel lists past jobs for your IP when `UI_HISTORY_PER_CLIENT_IP=true`; refresh the browser during a long run to reattach progress (same tab id in sessionStorage). **Cancel** stops that tab’s job (including while queued) and clears CUDA cache for the next queued user (`UI_CANCEL_FREES_GPU_FOR_QUEUE`).

On **8 GB** keep `UI_MAX_CONCURRENT_JOBS=1` and set `UI_GRADIO_TRANSCRIBE_CONCURRENCY=2`–`4` so multiple users queue instead of OOM. True 2–4 parallel GPU ASR+diar jobs need **16 GB+**.

### Audio Enhancement

```dotenv
AUDIO_ENHANCE_DEFAULT=true
AUDIO_ENHANCE_WHEN_DIARIZATION=true
AUDIO_ENHANCE_TARGET_PEAK_DB=-1.5
AUDIO_ENHANCE_MAX_GAIN_DB=18
AUDIO_ENHANCE_NOISE_REDUCTION=0.92
AUDIO_ENHANCE_ATEMPO=0.92
```

### Speaker Diarization

```dotenv
DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1
DIARIZATION_DEVICE=cuda
DIARIZATION_PREPROCESS_SR=44100
DIARIZATION_NOISE_REDUCTION=0.0
DIARIZATION_SEGMENT_S=300
DIARIZATION_SEGMENT_OVERLAP_S=60
DIARIZATION_MAX_ASR_WINDOW_S=300

# Pyannote accuracy tuning (also editable live in the UI)
DIARIZATION_SEGMENTATION_THRESHOLD=0.42
DIARIZATION_MIN_DURATION_OFF=0.1
DIARIZATION_CLUSTERING_THRESHOLD=0.60
DIARIZATION_MIN_CLUSTER_SIZE=6
```

---

## Hardware Support

**Minimum host:** 4 CPU threads, **8 GB RAM**. NVIDIA CUDA path also needs **≥ 8 GB VRAM**.

| Hardware | Mode |
| --- | --- |
| NVIDIA GPU ≥ 8 GB VRAM | CUDA (`docker-compose.gpu.yml`) |
| NVIDIA GPU < 8 GB VRAM | OpenVINO / CPU fallback |
| Intel Core Ultra / Arc / iGPU | OpenVINO GPU / NPU (`docker-compose.openvino.yml` or `run.bat ov-gpu`) |
| Intel NPU | OpenVINO NPU (`OV_DEVICE=NPU`) |
| AMD AI CPU / x86 CPU | OpenVINO CPU |
| AMD GPU (Windows) | DirectML when `torch-directml` is installed; else OpenVINO CPU |
| AMD GPU (Linux) | ROCm PyTorch when available; else OpenVINO CPU |
| ARM64 (Apple Silicon / Ampere) | OpenVINO (CPU; GPU when exposed) |

**Docker note (Intel GPU/NPU):** to use Intel **GPU/NPU inside Docker**, the container needs host device nodes (Linux `/dev/dri`). On Windows Docker Desktop, OpenVINO is usually **CPU-only** in the Linux container — use native `run.bat ov-gpu` for Arc/NPU.

---

## Model Bootstrap

Run once after setup to download and cache all models locally:

```bash
python scripts/bootstrap_models.py
```

`pyannote/speaker-diarization-community-1` is a gated model — accept its terms on Hugging Face, then set `HF_TOKEN` in `.env`.

**Maintainer setup only:** keep `HF_HUB_OFFLINE=0` while running `bootstrap_models.py` so missing weights can be downloaded once. **Production and Docker defaults are offline** (`HF_HUB_OFFLINE=1`, `APP_AUTO_DOWNLOAD_MISSING_MODELS=false` in `.env.production` and `docker-compose.gpu.yml`). After bootstrap, the app reads only from `./models/hf_cache/` — no repeated hub downloads at runtime.

---

## Runtime Storage

```text
config/app_config.json        — hardware policy written at first run
models/hf_cache/              — Hugging Face model weights
models/ov_cache/              — OpenVINO exported models
storage/input/                — uploaded media (temporary)
storage/audio/                — processed audio
storage/transcripts/          — .txt output files
storage/jobs/                 — JSON job manifests
storage/logs/                 — optional log files
```

---

## Stopping the App

- **Desktop window**: close the window — the backend shuts down automatically
- **Terminal**: `Ctrl+C`
- **Docker**: `docker compose -f docker-compose.gpu.yml down`

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| GPU not used | Requires NVIDIA GPU ≥ 8 GB VRAM and CUDA drivers |
| Missing model fails in offline mode | Set `HF_HUB_OFFLINE=0`, run bootstrap or start the app once, then switch back to offline only after the model is cached |
| Video upload fails | Install FFmpeg and add it to PATH |
| Speaker diarization misses a speaker | Raise **Max Speakers**, lower **Clustering Threshold**, and lower **Min Cluster Size** in the UI |
| Speakers merged into one | Raise **Max Speakers** and lower **Clustering Threshold** (try 0.45–0.55) |
| Desktop window shows blank / loading forever | Check `docker ps` or `venv/Scripts/python app.py` in a terminal |
| CUDA out of memory | Strict 8 GB mode halves batch then chunk length (min 8s). Reduce `ASR_8GB_CHUNK_LENGTH_S` / `ASR_8GB_MAX_CHUNK_LENGTH_S`. Set `ASR_CLEAR_VRAM_ON_MEDIA_CHANGE=true` to free GPU when switching files. |
| Gradio tab freezes on long transcript | UI shows last `UI_TRANSCRIPT_MAX_LINES` (default 500) — use **Download .txt** for the full file |
| Back-to-back 3h jobs OOM | Check logs for `VRAM [phase]` lines; cancel unloads all models; co-resident mode keeps weights but runs phase teardown between stages |
| Whisper repeats words at end of segments | Enabled by default: `ASR_SUPPRESS_HALLUCINATIONS=true` plus post-processing in `engines/text_cleanup.py` |
| Want better diarization than community-1 | See **Speaker diarization models** below — fully open-source options are limited; community-1 is the best local HF pipeline today |

### Speaker diarization models

The app uses **`pyannote/speaker-diarization-community-1`** (Sep 2025), which is the strongest **fully open, local** speaker-diarization pipeline on Hugging Face today. It beats the older `speaker-diarization-3.1` pipeline on most benchmarks and runs entirely on your machine with `HF_TOKEN`.

**What is *not* included (and why):**

| Model | Status |
| --- | --- |
| `pyannote/speaker-diarization-precision-2` | Commercial / pyannoteAI cloud API — not a drop-in local OSS replacement |
| NVIDIA NeMo Sortformer | Heavy separate stack; not integrated in this app |
| Legacy `speaker-diarization-3.1` | Still works via `DIARIZATION_MODEL_ID` but is generally weaker than community-1 |

For short clips with several speakers, enable **Short clip / multi-speaker preset** in the UI or tune segmentation/clustering sliders (defaults: 0.42 / 0.10 / 0.60 / 6). Adaptive logic also retries with lower clustering when too few speakers are detected.

---

## Privacy

All processing is local. No audio, video, or transcript data is sent to any cloud service. Internet access is used only during model download (bootstrap).

## Full Feature List

- Audio file transcription.
- Video file transcription by extracting audio locally with FFmpeg.
- Local ASR engines:
  - Typhoon Whisper Large v3.
  - Pathumma Whisper Thai Large v3.
- Speaker diarization with local `pyannote/speaker-diarization-community-1` pipeline.
- Folder-based storage inside the app path.
- Hardware-aware backend selection for NVIDIA GPU, Intel OpenVINO GPU/NPU, AMD GPU fallback, and CPU-only systems.

## Detailed Project Structure

```text
local-transcript-app/
  app.py                         # Gradio UI only
  backend/
    pipeline.py                  # Main backend workflow
    storage.py                   # App-local folders, config, transcript/job writes
    services/
      asr_local.py               # Local ASR service facade
      media_pipeline.py          # Video/audio normalize, enhancement, diarization
      hardware_policy.py         # Hardware checks and backend selection
  engines/                       # Model implementation compatibility layer
  scripts/
    bootstrap_models.py          # Model download/export bootstrap
  installer/                     # Windows installer specs/scripts
  storage/                       # Created at runtime; user transcripts/jobs/logs
  models/                        # Created at runtime; model/cache folders
  RUN_INSTRUCTIONS.md            # Practical run guide
```

## Hardware Policy

The app detects available resources and chooses the best local backend.

| Hardware | Policy |
| --- | --- |
| NVIDIA GPU with CUDA and at least 8 GB VRAM | Use CUDA |
| NVIDIA GPU with less than 8 GB VRAM | Use OpenVINO/CPU fallback |
| Intel GPU / Core Ultra | Prefer OpenVINO GPU (then NPU) |
| Intel NPU | Prefer OpenVINO NPU when GPU absent |
| AMD GPU (Windows) | Prefer DirectML when installed |
| AMD GPU (Linux ROCm) | Prefer PyTorch HIP |
| ARM64 / AMD AI / CPU only | OpenVINO CPU (or GPU when exposed) |

**Minimum:** `MIN_CPU_THREADS=4`, `MIN_SYSTEM_RAM_MB=8192`. Hosts below that still run with a warning.

The CUDA runtime uses a strict 8 GB-class policy. On GPUs up to `ASR_8GB_CLASS_MAX_MB` (default 9000 MB, covering RTX 4060 Laptop cards that report about 8187 MB), the app keeps configured ASR models resident for reuse, uses sequential ASR unless `ASR_ALLOW_8GB_PARALLEL=true`, and stages diarization vs ASR on 8 GB cards.

## Windows Quick Start For Normal Users

The target release is a single Windows installer `.exe`. The installer flow is planned to:

1. Install the app files.
2. Check hardware and choose CUDA/OpenVINO/CPU backend.
3. Create local folders under the app path.
4. Download/export required models into `models/`.
5. Write `config/app_config.json`.
6. Create Start Menu/Desktop shortcuts.
7. Launch the local Gradio app.

Until the installer build is finalized, use the developer setup flow below.

## Developer Setup On Windows

```bat
cd C:\path\to\local-transcript-app
setup.bat
```

## Starting The App

### Windows (Starting The App)

**First time only** — install dependencies and download models:

```bat
setup.bat
copy .env.example .env
REM Edit .env and set HF_TOKEN=hf_your_token_here
venv\Scripts\activate
python scripts\bootstrap_models.py
```

**Every subsequent run:**

```bat
venv\Scripts\activate
python app.py
```

Or use the one-shot launcher:

```bat
run.bat
```

Then open your browser at:

```text
http://localhost:7896
```

---

### Linux / macOS

**First time only:**

```bash
./setup.sh
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
source venv/bin/activate
python scripts/bootstrap_models.py
```

**Every subsequent run:**

```bash
source venv/bin/activate
python app.py
# or: ./run.sh
```

Then open:

```text
http://localhost:7896
```

---

### What to expect on first start

1. Hardware is detected — NVIDIA CUDA (>=8 GB VRAM) is preferred; falls back to OpenVINO CPU or CPU.
2. The configured default ASR engine preloads from `models/hf_cache/hub` at startup before the UI is served; other engines load on demand.
3. Gradio UI starts after preload; the terminal shows `Running on local URL: http://0.0.0.0:7896`.
4. Upload an audio or video file, choose engine and language, click **Transcribe**.

Keep Hugging Face model snapshots under the canonical hub cache: `models/hf_cache/hub/models--...`. Legacy duplicate folders directly under `models/hf_cache/models--...` are not used by this app and should be moved out of the active cache root.

Optional runtime knobs in `.env`:

```text
APP_SUPPRESS_WARNING_LOGS=true             # keep known dependency warning logs quiet
APP_MODEL_ROOT=./models                   # Docker uses /app/models via bind mount
HF_HOME=./models/hf_cache                 # reuse downloaded Hugging Face models
HF_HUB_CACHE=./models/hf_cache/hub
HF_HUB_OFFLINE=0                          # allow first-run downloads after model changes
HUGGINGFACE_HUB_CACHE=./models/hf_cache/hub
TORCH_HOME=./models/torch
OV_CACHE_DIR=./models/ov_cache            # reuse OpenVINO exports
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # reduce CUDA allocator fragmentation
ASR_HARD_MEMORY_SAFE=true                 # enforce strict 8 GB batch/worker policy
ASR_8GB_CLASS_MAX_MB=9000                 # treat reported 8 GB laptop GPUs as low-VRAM
ASR_PARALLEL_MODE=auto                    # forced parallel is ignored in strict 8 GB mode
ASR_PARALLEL_MIN_VRAM_MB=12288            # auto-parallel threshold for larger GPUs
ASR_CLEAR_VRAM_BETWEEN_ENGINES=false      # keep preloaded ASR models resident between engines
ASR_CLEAR_VRAM_AFTER_JOB=false            # keep ASR models in VRAM for the next transcript round
ASR_DEFAULT_ENGINES=Pathumma Whisper
ASR_QUALITY_PROFILE=high                  # default; use balanced only for OOM emergencies
ASR_8GB_ALLOW_LARGE_CHUNKS=false          # high profile: set true for full 180s on 8 GB
ASR_ALLOW_8GB_PARALLEL=false             # run selected ASR engines sequentially on 8 GB GPUs
ASR_CUDA_BATCH_SIZE=4                     # GPU compose default; capped by ASR_8GB_MAX_BATCH_SIZE
ASR_8GB_MAX_BATCH_SIZE=4
ASR_BATCH_DURATION_CAP=true               # allow batch>1 for 60–120s audio when max batch>1
ASR_MAX_CONCURRENT_INFERENCE=1            # one shared Whisper pipeline on 8 GB
ASR_UNLOAD_ON_CANCEL=true                 # Cancel & Reset frees VRAM before next job
UI_MAX_CONCURRENT_JOBS=1                  # GPU slots; keep 1 on 8 GB (restart after change)
UI_GRADIO_TRANSCRIBE_CONCURRENCY=4        # queued users / tabs
UI_CANCEL_JOIN_TIMEOUT_S=120              # max wait for worker after cancel
UI_HISTORY_PER_CLIENT_IP=true
UI_CANCEL_FREES_GPU_FOR_QUEUE=true
UI_GRADIO_TRANSCRIBE_CONCURRENCY=8        # Gradio streams (tabs); does not add GPU load
ASR_CUDA_MEMORY_FRACTION=0.90             # leave headroom for CUDA/runtime allocations
ASR_CHUNK_LENGTH_S=30                     # balanced profile default
ASR_8GB_CHUNK_LENGTH_S=90
ASR_8GB_MAX_CHUNK_LENGTH_S=90             # cap on 8 GB (raise with ASR_8GB_ALLOW_LARGE_CHUNKS)
ASR_8GB_RETRY_CHUNK_LENGTH_S=10           # final retry after CUDA OOM
ASR_8GB_ALLOW_LARGE_CHUNKS=true           # high profile: larger chunks on 8 GB
ASR_MIN_CHUNKED_DURATION_S=120
DIARIZATION_TRANSCRIPT_MERGE_GAP_S=1.25
PATHUMMA_MODEL_ID=nectec/Pathumma-whisper-th-large-v3
TYPHOON_MODEL_ID=typhoon-ai/typhoon-whisper-large-v3
ASR_WORD_TIMESTAMPS_WITH_DIARIZATION=true # better ASR-to-speaker alignment
TYPHOON_WORD_TIMESTAMPS_ON_8GB=false      # Typhoon uses chunk timestamps on 8 GB to avoid OOM
ASR_ATTENTION_IMPLEMENTATION=sdpa         # prefer memory-efficient torch attention
ASR_PRELOAD_MODE=eager                    # preload the configured ASR engines from models/ at startup

AUDIO_ENHANCE_DEFAULT=false               # keep enhancement unchecked on startup (Docker GPU: true)
AUDIO_ENHANCE_TARGET_PEAK_DB=-3.0
AUDIO_ENHANCE_MAX_GAIN_DB=15.0           # max lift for quiet speech
AUDIO_ENHANCE_NOISE_REDUCTION=0.80      # spectral NR strength (0–1)
AUDIO_ENHANCE_LOUDNORM_I=-16.0            # FFmpeg loudnorm integrated loudness (LUFS)
AUDIO_ENHANCE_NOISE_PROFILE=leading       # leading | adaptive — noise sample from first 0.75 s
AUDIO_ENHANCE_TARGET_PEAK_DB=-3.0         # make quiet speech louder without clipping
AUDIO_ENHANCE_MAX_GAIN_DB=10.0            # cap boost so noise does not explode
AUDIO_ENHANCE_NOISE_REDUCTION=0.65        # less destructive than heavy gating

DIARIZATION_MODEL_ID=pyannote/speaker-diarization-community-1
DIARIZATION_DEVICE=cpu                   # keep pyannote embeddings off 8 GB GPU VRAM
DIARIZATION_CUDA_MIN_FREE_MB=1024        # move diarization to CPU if CUDA memory is crowded
DIARIZATION_CUDA_MIN_VRAM_MB=12288
```

Each job manifest records `audio_duration_s`, `total_elapsed_s`, `target_elapsed_s`, and `target_met`. Targets are tiered: **≤10 min** wall time for audio under 20 minutes, **half realtime** (audio duration ÷ 2) for longer files.

### Enterprise Docker acceptance (production sign-off)

Production GPU sign-off uses **two fixtures only** — never run host GPU validation while Docker holds the GPU:

```powershell
python scripts/stop_gpu_work.py --docker
docker compose -f docker-compose.gpu.yml up -d --build --force-recreate
pytest tests/test_asr_performance.py tests/test_asr_quality.py tests/test_meeting_eval.py tests/test_golden_automation.py -q
python scripts/run_sonar_scan.py --skip-scan
python scripts/run_docker_acceptance.py --tag final
```

| Fixture | Audio | Accuracy gates | Wall time |
|---------|-------|----------------|-----------|
| `sample01` | `tests/test-sample01.m4a` (~3.7 min) | 99/98/98/95% content/speaker/timestamp/strict, 4/4 speakers, 0 mismatched lines | ≤10 min |
| `meeting309` | `tests/309.m4a` (~90 min) | 11/11 speakers + meeting gates | ≤ half audio (~45 min) |

VRAM policy: `ASR_CUDA_MEMORY_FRACTION=0.92` with batch=1, beams=5, single concurrent job. See `backend/enterprise_config.py` and `docker-compose.gpu.yml`.

**Offline models:** Docker and validation never download from Hugging Face Hub. Populate `./models/` once via `scripts/bootstrap_models.py` on a maintainer machine; `ensure_model_cache.py --strict-diarization` verifies ASR + pyannote caches before acceptance runs.

**sample01 baseline:** `ENTERPRISE_FIXTURE_OVERRIDES["sample01"]` is locked to the cal8 Docker profile (diar on raw audio, ASR-only enhance, seg 0.31 / cluster 0.38). Do not change without a fresh Docker re-score.

### Golden automation (GPU regression)

```powershell
python scripts/run_golden_automation.py --deploy
```

Fixtures: `test-sample01.m4a` (accuracy vs `tests/test-sample01.txt`) and optional long-audio perf smoke tests. Enterprise acceptance uses `run_docker_acceptance.py` (see above).

### Stopping the app

Press `Ctrl+C` in the terminal window where `app.py` is running.

## Runtime Storage Layout

```text
config/app_config.json
models/hf_cache/
models/ov_cache/
storage/input/
storage/audio/
storage/transcripts/
storage/jobs/
storage/logs/
```

Transcripts are written to `storage/transcripts/`. Job metadata is written to `storage/jobs/` as JSON.

## Model Bootstrap Script

```bash
python scripts/bootstrap_models.py
```

The bootstrap script:

- Detects hardware and writes backend policy to config.
- Downloads or exports local ASR models.
- Stores model artifacts under app-local `models/` paths.
- Reuses existing model cache on repeated runs.

`pyannote/speaker-diarization-community-1` is a gated Hugging Face model. Accept the model terms on Hugging Face, then set your token in `.env`:

```text
HF_TOKEN=your-token-here
```

## Pyannote Version

The project pins the Python runtime package to `pyannote.audio==4.0.4`. This is different from the diarization model ID, which defaults to `pyannote/speaker-diarization-community-1` (override via `DIARIZATION_MODEL_ID` if you need the legacy `pyannote/speaker-diarization-3.1`).

For normal online installation, do not download GitHub release assets manually. `pip install -r requirements.txt` downloads and installs the correct wheel automatically.

If you need an offline/manual install, use only:

```text
pyannote_audio-4.0.4-py3-none-any.whl
```

The `.sigstore.json` files are optional provenance/signature metadata for verification workflows. The `.tar.gz` and source-code archives are source distributions and are not needed for this app unless you specifically want to build pyannote.audio from source.

Manual offline install example:

```powershell
venv\Scripts\python.exe -m pip install .\vendor\pyannote_audio-4.0.4-py3-none-any.whl
```

## Privacy Statement

Audio/video processing, transcription, diarization, and transcript storage are local. The app does not send files to Azure, OpenAI, or any cloud transcription service. Internet access is used only when the user installs/downloads models.

## Troubleshooting Reference

### NVIDIA GPU Found But Not Used

The app requires at least 8 GB VRAM for the CUDA path. If VRAM is lower, it falls back to OpenVINO/CPU to avoid crashes.

### AMD GPU Found

AMD acceleration is not enabled in v1. The app uses OpenVINO CPU/CPU fallback.

### Video Upload Fails

Install FFmpeg and make sure it is available on PATH.

### Models Fail To Download

Check internet access, disk space, and `HF_TOKEN` if the model is gated.

### First Run Is Slow

The first run may download and export large models. Later runs reuse files in `models/`. ASR models load lazily by default, so the UI opens quickly and VRAM is only used during actual transcription.
