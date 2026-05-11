# Local Transcript App

GPU-accelerated local audio/video transcription with speaker diarization.  
No cloud APIs. No telemetry. All processing stays on your machine.

---

## Features

- **Typhoon Whisper Large v3** — default ASR engine (Thai-optimised, high accuracy)
- **Thonburian / Distill Whisper Thai Large v3** — fast alternative ASR engine
- **Speaker diarization** — `pyannote/speaker-diarization-3.1`, fully local
- **Advanced diarization tuning** — segmentation threshold, clustering threshold, min cluster size, silence gap — adjustable live in the UI
- **Audio enhancement** — bandpass filter → spectral noise reduction → compressor / limiter chain
- **Local LLM correction** — optional post-transcription text cleanup via llama.cpp or Ollama
- **Native desktop window** — pywebview wraps the Gradio UI; no browser required when using `launcher.py` or `LocalTranscriptApp.exe`
- **Docker GPU mode** — NVIDIA CUDA 13 + PyTorch; models cached in `./models/`
- **Strict 8 GB VRAM policy** — safe on RTX 4060 Laptop (8 GB); one model at a time, sequential engines, capped chunk size

---

## Quick Start

### Option A — Docker (recommended, GPU)

> Requires Docker Desktop with GPU enabled, or Linux with `nvidia-container-toolkit`.

```bat
REM Windows
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

Or open manually: **http://localhost:7896**

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
|---|---|---|
| **Language** | Thai | Dropdown — choose the spoken language |
| **Local ASR Engines** | Typhoon Whisper | Tick one or both; dual-engine runs sequentially on 8 GB GPUs |
| **Audio Enhancement** | Off | Noise gate → spectral NR → compressor → limiter chain |
| **Speaker Diarization** | Off | Identify and label individual speakers |
| **Max Speakers** | 3 | Appears when Diarization is enabled |
| **Advanced Diarization Settings** | (accordion) | Fine-tune pyannote accuracy per run — see below |

### 3. Advanced Diarization Settings (accordion)

Visible only when **Speaker Diarization** is checked.

| Slider | What it does | Tune when… |
|---|---|---|
| **Segmentation Threshold** (0.42) | Activity threshold; lower = catches quieter / shorter turns | Speakers are missed or turns are cut short |
| **Min Silence Gap** (0.10 s) | Minimum gap before splitting a turn | Too many spurious splits — raise it |
| **Clustering Threshold** (0.60) | Speaker embedding distance; lower = more speakers separated | Different speakers are merged into one — lower it |
| **Min Cluster Size** (6) | Minimum segments to form a speaker cluster | Rare/short speakers are dropped — lower it |

### 4. Transcribe

Click **Transcribe**. A progress indicator appears while models run.  
Results appear in the **Typhoon Whisper** and **Thonburian Whisper** tabs.

### 5. Results

Each engine tab shows:
- **Transcript** — full text with speaker labels and timestamps when diarization is on
- **Elapsed Time** — wall time for that engine
- **Download .txt** — save the transcript
- **Run Local LLM Correction** — optional post-processing (requires a running llama.cpp / Ollama server)

Job metadata is saved to `storage/jobs/` as JSON.

---

## Project Structure

```text
local-transcript-app/
├── app.py                        # Gradio UI
├── launcher.py                   # Desktop launcher (pywebview native window)
├── run.bat                       # Windows: run.bat | run.bat gui | run.bat docker
├── run.sh                        # Linux/Mac: ./run.sh | ./run.sh gui | ./run.sh docker
├── setup.bat / setup.sh          # First-time dependency install
├── docker-compose.gpu.yml        # Docker + NVIDIA GPU (recommended)
├── docker-compose.yml            # Docker CPU/OpenVINO fallback
├── Dockerfile                    # CUDA 13 image, PyTorch, all dependencies
├── .env                          # Runtime configuration (copy from .env.example)
├── .env.example                  # Documented configuration template
├── backend/
│   ├── pipeline.py               # Main transcription workflow
│   ├── storage.py                # Folder structure, transcript/job writes
│   └── services/
│       ├── asr_local.py          # ASR engine facade
│       ├── media_pipeline.py     # Normalise, enhance, diarize
│       ├── correction_local.py   # LLM correction
│       └── hardware_policy.py    # GPU/CPU backend detection
├── engines/
│   ├── typhoon_asr.py            # Typhoon Whisper Large v3
│   ├── thonburian_asr.py         # Distill Whisper Thai Large v3
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

```dotenv
ASR_DEFAULT_ENGINES=Typhoon Whisper   # Default engine shown checked in UI
ASR_PRELOAD_MODE=eager                # eager = load model at startup; lazy = on first use
ASR_HARD_MEMORY_SAFE=true             # One model at a time on 8 GB GPUs
ASR_8GB_CLASS_MAX_MB=9000             # GPUs up to this VRAM use strict 8 GB policy
ASR_CHUNK_LENGTH_S=30                 # Audio chunk size (larger GPUs)
ASR_8GB_CHUNK_LENGTH_S=20             # Chunk size for 8 GB GPUs
ASR_CUDA_MEMORY_FRACTION=0.90         # Leave headroom for CUDA allocator
TYPHOON_WORD_TIMESTAMPS_ON_8GB=false  # Disable word timestamps on 8 GB (saves VRAM)
```

### Audio Enhancement

```dotenv
AUDIO_ENHANCE_DEFAULT=false           # Enhancement unchecked on startup
AUDIO_ENHANCE_TARGET_PEAK_DB=-3.0     # Target loudness after gain
AUDIO_ENHANCE_MAX_GAIN_DB=10.0        # Cap gain so noise doesn't explode
AUDIO_ENHANCE_NOISE_REDUCTION=0.65    # Spectral gating strength (0–1)
```

### Speaker Diarization

```dotenv
DIARIZATION_MODEL_ID=pyannote/speaker-diarization-3.1
DIARIZATION_DEVICE=cpu                # cpu = reserve GPU for ASR
DIARIZATION_PREPROCESS_SR=44100       # Intermediate rate before 16 kHz downsample
DIARIZATION_NOISE_REDUCTION=0.60      # NR strength for diarization audio

# Pyannote accuracy tuning (also editable live in the UI)
DIARIZATION_SEGMENTATION_THRESHOLD=0.42
DIARIZATION_MIN_DURATION_OFF=0.1
DIARIZATION_CLUSTERING_THRESHOLD=0.60
DIARIZATION_MIN_CLUSTER_SIZE=6
```

### Local LLM Correction (optional)

```dotenv
LOCAL_LLM_PROVIDER=llamacpp
LLAMACPP_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions
LOCAL_LLM_MODEL=typhoon2-8b-instruct-q4
LOCAL_LLM_MAX_TOKENS=4096
```

---

## Hardware Support

| Hardware | Mode |
|---|---|
| NVIDIA GPU ≥ 8 GB VRAM | CUDA (PyTorch) |
| NVIDIA GPU < 8 GB VRAM | OpenVINO / CPU fallback |
| Intel GPU / NPU | OpenVINO GPU or NPU |
| AMD GPU | OpenVINO CPU fallback (v1) |
| CPU only | OpenVINO CPU |

The RTX 4060 Laptop (8 GB) is the reference hardware. Strict 8 GB mode keeps only one Whisper model in VRAM at a time, runs diarization on CPU, and uses sequential multi-engine mode.

---

## Model Bootstrap

Run once after setup to download and cache all models locally:

```bash
python scripts/bootstrap_models.py
```

`pyannote/speaker-diarization-3.1` is a gated model — accept its terms on Hugging Face, then set `HF_TOKEN` in `.env`.

After bootstrap, set `HF_HUB_OFFLINE=1` in `.env` to prevent any runtime downloads.

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
|---|---|
| GPU not used | Requires NVIDIA GPU ≥ 8 GB VRAM and CUDA drivers |
| Models download on every start | Set `HF_HUB_OFFLINE=1` in `.env` after bootstrap |
| Video upload fails | Install FFmpeg and add it to PATH |
| Speaker diarization misses a speaker | Lower **Clustering Threshold** and **Min Cluster Size** in the UI |
| Speakers merged into one | Lower **Clustering Threshold** (try 0.45–0.55) |
| Desktop window shows blank / loading forever | Check `docker ps` or `venv/Scripts/python app.py` in a terminal |
| CUDA out of memory | Already handled by strict 8 GB mode; if it still occurs, reduce `ASR_8GB_CHUNK_LENGTH_S` |

---

## Privacy

All processing is local. No audio, video, or transcript data is sent to any cloud service. Internet access is used only during model download (bootstrap).


## Features

- Audio file transcription.
- Video file transcription by extracting audio locally with FFmpeg.
- Local ASR engines:
  - Typhoon Whisper Large v3.
  - Thonburian / Distill Whisper Thai Large v3.
- Speaker diarization with local `pyannote/speaker-diarization-3.1` pipeline.
- Optional local LLM correction through a local Ollama-compatible endpoint.
- Folder-based storage inside the app path.
- Hardware-aware backend selection for NVIDIA GPU, Intel OpenVINO GPU/NPU, AMD GPU fallback, and CPU-only systems.

## Project Structure

```text
local-transcript-app/
  app.py                         # Gradio UI only
  backend/
    pipeline.py                  # Main backend workflow
    storage.py                   # App-local folders, config, transcript/job writes
    services/
      asr_local.py               # Local ASR service facade
      media_pipeline.py          # Video/audio normalize, enhancement, diarization
      correction_local.py        # Optional local-only LLM correction
      hardware_policy.py         # Hardware checks and backend selection
  engines/                       # Model implementation compatibility layer
  scripts/
    bootstrap_models.py          # Model download/export bootstrap
  installer/                     # Windows installer specs/scripts
  storage/                       # Created at runtime; user transcripts/jobs/logs
  models/                        # Created at runtime; model/cache folders
  RUN_INSTRUCTIONS.md            # Practical run guide
```

The runnable app is now in the main `local-transcript-app` path. The older `speaker-aware-ai/` and `test-transcript-service/` folders are legacy/reference folders and are not needed to run the root app.

## Hardware Policy

The app detects available resources and chooses the best local backend.

| Hardware | Policy |
| --- | --- |
| NVIDIA GPU with CUDA and at least 8 GB VRAM | Use CUDA |
| NVIDIA GPU with less than 8 GB VRAM | Use OpenVINO/CPU fallback |
| Intel GPU | Prefer OpenVINO GPU |
| Intel NPU | Prefer OpenVINO NPU when available |
| AMD GPU | Use CPU/OpenVINO fallback in v1 |
| CPU only | Use OpenVINO CPU or CPU fallback |

The CUDA runtime now uses a strict 8 GB-class policy. On GPUs up to `ASR_8GB_CLASS_MAX_MB` (default 9000 MB, covering RTX 4060 Laptop cards that report about 8187 MB), the app keeps only one Whisper model resident on the GPU, unloads it before the next engine starts, ignores forced parallel ASR, caps CUDA batch size, and clears VRAM between engines/jobs. Speaker diarization and preprocessing run on CPU by default so ASR owns the GPU budget.

For fastest 8 GB operation, the default UI selection is the distilled Thonburian engine. You can still select both engines for comparison, but they run sequentially to stay inside the VRAM budget, so total wall time will be longer.

The selected backend and reason are saved to `config/app_config.json`.

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

### Windows

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

1. Hardware is detected — NVIDIA CUDA (≥8 GB VRAM) is preferred; falls back to OpenVINO CPU or CPU.
2. Models load lazily from `models/hf_cache/` on first use to keep VRAM free.
3. Gradio UI starts; the terminal shows `Running on local URL: http://0.0.0.0:7896`.
4. Upload an audio or video file, choose engine and language, click **Transcribe**.

Optional runtime knobs in `.env`:

```text
APP_SUPPRESS_WARNING_LOGS=true             # keep known dependency warning logs quiet
APP_MODEL_ROOT=./models                   # Docker uses /app/models via bind mount
HF_HOME=./models/hf_cache                 # reuse downloaded Hugging Face models
HF_HUB_CACHE=./models/hf_cache/hub
HF_HUB_OFFLINE=1                          # do not download models at runtime
HUGGINGFACE_HUB_CACHE=./models/hf_cache/hub
TORCH_HOME=./models/torch
OV_CACHE_DIR=./models/ov_cache            # reuse OpenVINO exports
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # reduce CUDA allocator fragmentation
ASR_HARD_MEMORY_SAFE=true                 # enforce one GPU ASR model on 8 GB-class cards
ASR_8GB_CLASS_MAX_MB=9000                 # treat reported 8 GB laptop GPUs as low-VRAM
ASR_PARALLEL_MODE=auto                    # forced parallel is ignored in strict 8 GB mode
ASR_PARALLEL_MIN_VRAM_MB=12288            # auto-parallel threshold for larger GPUs
ASR_CLEAR_VRAM_BETWEEN_ENGINES=true       # unload before loading another engine
ASR_CLEAR_VRAM_AFTER_JOB=true             # final CUDA cache cleanup after each job
ASR_DEFAULT_ENGINES=Thonburian Whisper    # fastest default for 8 GB
ASR_CUDA_BATCH_SIZE=1                     # strict 8 GB safe default
ASR_8GB_MAX_BATCH_SIZE=1
ASR_CUDA_MEMORY_FRACTION=0.90             # leave headroom for CUDA/runtime allocations
ASR_CHUNK_LENGTH_S=30                     # larger GPUs can use this value
ASR_8GB_CHUNK_LENGTH_S=20                 # strict 8 GB load default
ASR_8GB_MAX_CHUNK_LENGTH_S=20             # cap long chunks on 8 GB GPUs
ASR_8GB_RETRY_CHUNK_LENGTH_S=10           # final retry after CUDA OOM
ASR_WORD_TIMESTAMPS_WITH_DIARIZATION=true # better ASR-to-speaker alignment
TYPHOON_WORD_TIMESTAMPS_ON_8GB=false      # Typhoon uses chunk timestamps on 8 GB to avoid OOM
ASR_ATTENTION_IMPLEMENTATION=sdpa         # prefer memory-efficient torch attention
ASR_PRELOAD_MODE=eager                    # preload the strict 8 GB default ASR engine at startup

AUDIO_ENHANCE_DEFAULT=false               # keep enhancement unchecked on startup
AUDIO_ENHANCE_TARGET_PEAK_DB=-3.0         # make quiet speech louder without clipping
AUDIO_ENHANCE_MAX_GAIN_DB=10.0            # cap boost so noise does not explode
AUDIO_ENHANCE_NOISE_REDUCTION=0.65        # less destructive than heavy gating

DIARIZATION_MODEL_ID=pyannote/speaker-diarization-3.1
DIARIZATION_DEVICE=cpu                    # reserve GPU VRAM for ASR
DIARIZATION_CUDA_MIN_VRAM_MB=12288
```

Each job manifest records `audio_duration_s`, `total_elapsed_s`, `target_elapsed_s`, and `target_met`. The target is 180 seconds for audio under 9 minutes and one-third of the audio duration for longer files. Hardware, audio quality, diarization, and selecting both engines can still affect the result, but the defaults are tuned for the fastest single-engine path within the 8 GB VRAM budget.

### Stopping the app

Press `Ctrl+C` in the terminal window where `app.py` is running.

## Runtime Storage

The app creates these folders automatically:

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

## Model Bootstrap

Run this after dependency install:

```bash
python scripts/bootstrap_models.py
```

The bootstrap script:

- Detects hardware and writes backend policy to config.
- Downloads or exports local ASR models.
- Stores model artifacts under app-local `models/` paths.
- Reuses existing model cache on repeated runs.

`pyannote/speaker-diarization-3.1` is a gated Hugging Face model. Accept the model terms on Hugging Face, then set your token in `.env`:

```text
HF_TOKEN=your-token-here
```

## Pyannote Version

The project pins the Python runtime package to `pyannote.audio==4.0.4`. This is different from the diarization model ID, which remains `pyannote/speaker-diarization-3.1`.

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

## Optional Local LLM Correction

Correction is local-only and now runs only after a transcript is generated and the user clicks **Run Local LLM Correction** in an engine tab. This keeps transcription fast and avoids loading an LLM during ASR.

The default adapter targets a local `llama.cpp` OpenAI-compatible server. In strict 8 GB transcription mode, keep the LLM unloaded during ASR or run it on CPU; then click correction after the transcript is generated. If you use GPU LLM correction on the same 8 GB card, start it after ASR finishes so it does not compete with Whisper for VRAM.

Optional `.env` values:

```text
LOCAL_LLM_PROVIDER=llamacpp
LLAMACPP_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions
LOCAL_LLM_MODEL=typhoon2-8b-instruct-q4
LOCAL_LLM_MAX_TOKENS=4096
```

Ollama is still supported:

```text
LOCAL_LLM_PROVIDER=ollama
OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate
LOCAL_LLM_MODEL=llama3.1:8b
```

If the local LLM server or model is unavailable, the app keeps the original transcript and reports that correction was skipped.

## Privacy

Audio/video processing, transcription, diarization, transcript storage, and optional correction are local. The app does not send files to Azure, OpenAI, or any cloud transcription service. Internet access is used only when the user installs/downloads models.

## Troubleshooting

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
