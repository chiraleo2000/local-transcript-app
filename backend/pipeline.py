"""Core backend transcription pipeline."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.asr_local import ALL_ENGINES, transcribe_engine
from backend.services.correction_local import correct_with_local_llm
from backend.services.media_pipeline import diarize_audio, enhance_audio, normalize_media
from backend.storage import new_job_id, now_iso, save_job_manifest, save_transcript


logger = logging.getLogger(__name__)


def run_transcription_job(
    media_path: str,
    selected_engines: list[str],
    language: str,
    diarization: bool,
    min_speakers: int,
    max_speakers: int,
    enhance: bool,
    local_correction: bool = False,
) -> dict:
    """Run the full local transcript pipeline and persist outputs."""
    job_id = new_job_id()
    process_path = normalize_media(media_path, job_id)
    if enhance:
        process_path = enhance_audio(process_path)

    selected = [engine for engine in selected_engines if engine in ALL_ENGINES]
    if not selected:
        selected = list(ALL_ENGINES)

    n_min = max(1, int(min_speakers)) if diarization else 0
    n_max = int(max_speakers) if diarization else 0
    if diarization and n_max < n_min:
        n_max = n_min
    diar_segments = None
    if diarization:
        try:
            diar_segments = diarize_audio(process_path, n_min, n_max)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Diarization failed: %s", exc, exc_info=True)
            diar_segments = None

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        futures = {
            pool.submit(transcribe_engine, engine, process_path, language, diar_segments): engine
            for engine in selected
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                text, elapsed = future.result()
                note = ""
                if local_correction and text and not text.startswith(("(", "ERROR")):
                    text, correction_elapsed, note = correct_with_local_llm(text)
                    elapsed += correction_elapsed
                transcript_path = save_transcript(job_id, engine, text)
                results[engine] = {
                    "text": text,
                    "elapsed": elapsed,
                    "download_path": transcript_path,
                    "note": note,
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("%s failed: %s", engine, exc, exc_info=True)
                results[engine] = {
                    "text": f"ERROR: {exc}",
                    "elapsed": 0.0,
                    "download_path": None,
                    "note": "",
                }

    manifest = {
        "job_id": job_id,
        "created_at": now_iso(),
        "source_path": media_path,
        "processed_path": process_path,
        "selected_engines": selected,
        "language": language,
        "diarization": diarization,
        "min_speakers": n_min,
        "max_speakers": n_max,
        "enhance": enhance,
        "local_correction": local_correction,
        "results": results,
    }
    manifest_path = save_job_manifest(job_id, manifest)
    return {"job_id": job_id, "manifest_path": manifest_path, "results": results}
