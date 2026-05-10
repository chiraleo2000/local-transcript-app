# Local Transcript App

Local Transcript App is a local-first audio/video transcription tool with speaker diarization. It uses open-source models for transcription and pyannote for local speaker diarization. No Azure Speech, Azure OpenAI, cloud transcript service, or database is used at runtime.

Internet access is only needed when downloading models during setup or bootstrap.

## Features

- Audio file transcription.
- Video file transcription by extracting audio locally with FFmpeg.
- Local ASR engines:
  - Typhoon Whisper Large v3.
  - Thonburian / Distill Whisper Thai Large v3.
- Speaker diarization with local pyannote pipeline.
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
| NVIDIA GPU with CUDA and at least 6 GB VRAM | Use CUDA |
| NVIDIA GPU with less than 6 GB VRAM | Use OpenVINO/CPU fallback |
| Intel GPU | Prefer OpenVINO GPU |
| Intel NPU | Prefer OpenVINO NPU when available |
| AMD GPU | Use CPU/OpenVINO fallback in v1 |
| CPU only | Use OpenVINO CPU or CPU fallback |

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
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
setup.bat
venv\Scripts\activate
python scripts\bootstrap_models.py
run.bat
```

Then open:

```text
http://localhost:7896
```

## Linux/macOS Setup

```bash
cd /path/to/local-transcript-app
./setup.sh
source venv/bin/activate
python scripts/bootstrap_models.py
./run.sh
```

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

Some pyannote or Hugging Face models may require a Hugging Face token depending on model access rules. Set it in `.env` when needed:

```text
HF_TOKEN=your-token-here
```

## Optional Local LLM Correction

Correction is local-only. The first adapter uses an Ollama-compatible local endpoint.

Optional `.env` values:

```text
OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate
LOCAL_LLM_MODEL=llama3.1:8b
```

If Ollama or the model is unavailable, the app keeps the original transcript and reports that correction was skipped.

## Privacy

Audio/video processing, transcription, diarization, transcript storage, and optional correction are local. The app does not send files to Azure, OpenAI, or any cloud transcription service. Internet access is used only when the user installs/downloads models.

## Troubleshooting

### NVIDIA GPU Found But Not Used

The app requires at least 6 GB VRAM for the CUDA path. If VRAM is lower, it falls back to OpenVINO/CPU to avoid crashes.

### AMD GPU Found

AMD acceleration is not enabled in v1. The app uses OpenVINO CPU/CPU fallback.

### Video Upload Fails

Install FFmpeg and make sure it is available on PATH.

### Models Fail To Download

Check internet access, disk space, and `HF_TOKEN` if the model is gated.

### First Run Is Slow

The first run may download and export large models. Later runs reuse files in `models/`.
