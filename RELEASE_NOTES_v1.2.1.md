# Release notes — v1.2.1

## Summary

Patch release fixing a GPU out-of-memory crash on the second transcription after cancel, and improving speaker diarization accuracy.

---

## Bug fixes

### Cancel no longer causes OOM on the next run (#critical)

**Root cause:** When a job was cancelled mid-inference the cancel `RuntimeError` raised inside the ASR window loop was silently caught by a broad `except Exception` in `_run_one_asr_engine` and converted into an error dict. The cleanup handler in `run_transcription_job` never fired, so ASR model weights remained pinned in GPU VRAM. Starting a second job then loaded a fresh audio tensor on top of the unreleased VRAM and hit OOM.

**Fix:**
- `_run_one_asr_engine` now has a dedicated `except RuntimeError` branch that re-raises any exception whose message contains "cancelled", letting it propagate to the outer cleanup handler.
- `_cleanup_cancelled_job` now explicitly calls `unload_model()` for all ASR engines and `clear_diarization_model()` before flushing the CUDA cache, guaranteeing a clean VRAM state for the next job.
- `_reset_ui_outputs` (Gradio cancel button handler) still calls `clear_accelerator_cache()` immediately for fast visual feedback, which is complementary to the model-unload on the background thread.

### Cancel now propagates through the full ASR window loop

`cancel_event` is checked at the top of every long-form ASR window in both `typhoon_asr.py` and `pathumma_asr.py`, so pressing Cancel during a 360-second windowed inference stops at the next window boundary (≤ 30 s) rather than waiting for the full file.

---

## Improvements

### Diarization accuracy — audio preprocessing overhaul

The FFmpeg filter chain used to prepare audio for pyannote's WeSpeaker speaker embeddings has been updated from the previous settings to a chain optimised for community-1's ResNet-34 backbone:

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| High-pass | 50 Hz | **80 Hz** | Sub-80 Hz room rumble carries zero speaker-ID information |
| Low-pass | 9000 Hz | **8000 Hz** | 8 kHz is the Nyquist limit of 16 kHz audio; frequencies above add only noise |
| Dynamic compression | none | **acompressor 3:1 at −20 dBFS, attack 5 ms, release 50 ms** | Equalises loud and quiet speakers so their embeddings land in tighter clusters |
| Loudness target | −16 LUFS / LRA 11 | **−23 LUFS / LRA 7** | EBU R128 broadcast standard; tighter LRA gives more consistent inter-speaker levels |

### Diarization accuracy — short-segment post-processing

A new `_merge_short_segments()` pass runs after pyannote returns, before speaker remapping. Sub-0.4 s segments that are flanked on both sides by a different speaker (and have ≤ 0.1 s gaps to each neighbour) are absorbed into the longer adjacent turn. This removes boundary fragments that pyannote produces at speaker transitions and that would otherwise appear as spurious single-chunk speaker changes in the transcript output.

---

## Files changed

- `backend/pipeline.py` — OOM fix: cancel re-raise in `_run_one_asr_engine`; model unload in `_cleanup_cancelled_job`
- `backend/services/asr_local.py` — `cancel_event` parameter threading
- `engines/typhoon_asr.py` — cancel check in long-form window loop
- `engines/pathumma_asr.py` — cancel check in long-form window loop
- `engines/diarization.py` — improved FFmpeg filter chain; `_merge_short_segments` post-processing
- `app.py` — `_reset_ui_outputs` calls `clear_accelerator_cache()` on cancel
