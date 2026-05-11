# How To Run Local Transcript App

This file is the practical run guide for the root app in `local-transcript-app`.

## 1. Open The App Folder

On Windows PowerShell:

```powershell
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
```

The runnable app files are in this main folder:

```text
app.py
backend/
engines/
scripts/bootstrap_models.py
requirements.txt
setup.bat
run.bat
```

The old `speaker-aware-ai/` and `test-transcript-service/` folders are reference/legacy folders. You do not need to enter them to run the app.

## 2. Install Dependencies

Windows:

```bat
setup.bat
```

Linux/macOS:

```bash
./setup.sh
```

## 3. Download And Prepare Models

Windows:

```bat
venv\Scripts\activate
python scripts\bootstrap_models.py
```

Linux/macOS:

```bash
source venv/bin/activate
python scripts/bootstrap_models.py
```

This step checks hardware, selects the backend, downloads/exports models, and writes config under:

```text
config/app_config.json
models/
```

## 4. Run The App

Windows:

```bat
run.bat
```

Linux/macOS:

```bash
./run.sh
```

Open this URL in your browser:

```text
http://localhost:7896
```

## 5. Hardware Rules

- NVIDIA GPU with CUDA and at least 6 GB VRAM: CUDA path.
- NVIDIA GPU below 6 GB VRAM: fallback to OpenVINO/CPU.
- Intel GPU/NPU: OpenVINO path when available.
- AMD GPU: CPU/OpenVINO fallback in v1.
- CPU only: CPU/OpenVINO fallback.

On 6-8 GB-class NVIDIA GPUs the app uses strict memory-safe mode:

- Default engine: `Thonburian Whisper` for the fastest 8 GB path.
- One GPU ASR model at a time; both engines can be selected, but they run sequentially.
- Forced parallel ASR is ignored on 8 GB-class GPUs when `ASR_HARD_MEMORY_SAFE=true`.
- Pyannote diarization and preprocessing run on CPU by default.
- CUDA batch size is capped at 1, long-form ASR uses 360-second windows with 30-second overlap, and CUDA cache is cleared between long-form windows.
- Diarization uses `pyannote/speaker-diarization-3.1`; accept the model terms on Hugging Face and set `HF_TOKEN`.
- The pyannote runtime package is pinned to `pyannote.audio==4.0.4`. For offline setup, the only release asset you need is `pyannote_audio-4.0.4-py3-none-any.whl`; source archives are not required.
- Audio Enhancement is enabled by default and applies capped gain toward `-3 dBFS` so quiet speech is louder without clipping. Uploading a file clears the enhanced preview instead of generating it immediately, so transcription does not spend CPU time enhancing the same file twice.

The app records a speed target in each job manifest: 3 minutes for audio under 9 minutes, otherwise one-third of the audio duration. Selecting both engines or enabling diarization can increase total time.

## 6. Output Files

The app stores outputs in the main app folder:

```text
storage/transcripts/
storage/jobs/
storage/audio/
storage/logs/
```

No database is required.

## 7. Optional Local LLM Correction

Correction runs after transcription only when you click **Run Local LLM Correction** in a transcript tab.

Recommended 8 GB VRAM setup is a local llama.cpp CUDA server with a Thai-compatible 7B/8B GGUF model quantized to 4-bit. Set these optional values in `.env`:

```text
LOCAL_LLM_PROVIDER=llamacpp
LLAMACPP_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions
LOCAL_LLM_MODEL=typhoon2-8b-instruct-q4
```

Ollama is still supported with `LOCAL_LLM_PROVIDER=ollama` and `OLLAMA_ENDPOINT=http://127.0.0.1:11434/api/generate`.

If the local LLM is not available, transcription still works and correction is skipped.
