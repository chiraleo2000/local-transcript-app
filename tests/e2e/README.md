# Playwright E2E — Local Transcript App

Automates the Gradio UI against a running instance (Docker or local `python app.py`).

## Prerequisites

1. App running at `http://localhost:7987` for Docker (maps to container port 7896) or `http://localhost:7896` for direct Python
2. Models preloaded (first start may take several minutes)
3. Node.js 18+

## Setup

```bash
cd tests/e2e
npm install
npx playwright install chromium
python generate_fixtures.py
```

Fixtures:

| File | Default | Env override |
|------|---------|----------------|
| `fixtures/small.wav` | 5 s | `E2E_SMALL_SECONDS` |
| `fixtures/large.wav` | 120 s | `E2E_LARGE_SECONDS` |

Use your own files:

```bash
set E2E_LARGE_AUDIO=C:\path\to\long.mp3
npm test
```

## Run

```bash
npm test
```

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `E2E_BASE_URL` | `http://localhost:7987` | Gradio URL |
| `E2E_MODEL_READY_MS` | `1200000` | Wait for Transcribe button |
| `E2E_SMALL_TRANSCRIBE_MS` | `900000` | Small job timeout |
| `E2E_LARGE_TRANSCRIBE_MS` | `2100000` | Large job timeout |

Real-audio spec (`real_audio.spec.ts`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `E2E_SHORT_AUDIO` | `../../tests/Recording 1274.wav` | Fast smoke |
| `E2E_LONG_AUDIO` | `../../tests/sudar-0001.m4a` | Long OOM stress (optional) |
| `E2E_LONG_TRANSCRIBE_MS` | `14400000` | Long job timeout (4 h) |

```bash
npm test -- real_audio.spec.ts
```

## Assertions

- HTTP 200 on `/`
- **Transcribe** enabled after model preload
- Upload shows filename
- `#live-status` reaches `running` then `done` (not `error`)
- Transcript textarea non-empty, not `(failed)` / `ERROR`
- `#elapsed-timer` visible and increments during job (stopwatch UI)
