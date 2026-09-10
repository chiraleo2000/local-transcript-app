"""Shared fire-and-forget job enqueue for UI and REST API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.job_queue import (
    release_queue_slot,
    snapshot_queue,
    submit_background_job,
    try_reserve_queue_slot,
)
from backend.pipeline import JobMeta, run_transcription_job
from backend.queue_policy import max_batch_files
from backend.storage import (
    copy_input_file,
    new_job_id,
    write_job_record,
)

logger = logging.getLogger(__name__)


@dataclass
class EnqueueOptions:
    language: str = "Thai"
    diarization: bool = True
    enhance: bool = True
    max_speakers: int = 0
    selected_engines: list[str] = field(default_factory=lambda: ["Auto"])
    output_name: str | None = None
    diarize_kwargs: dict[str, Any] | None = None
    tab_id: str = ""
    client_ip: str = ""
    user_id: int = 0
    username: str = ""


@dataclass
class EnqueuedJob:
    job_id: str
    source_filename: str
    display_name: str
    status: str
    error: str = ""


@dataclass
class EnqueueBatchResult:
    accepted: list[EnqueuedJob]
    rejected: list[EnqueuedJob]
    queue: dict[str, Any]


def _display_name(path: str, output_name: str | None) -> tuple[str, str]:
    source_filename = Path(path).name
    stem = Path(source_filename).stem
    display = (output_name or "").strip() or stem
    return source_filename, display


def _run_enqueued_job(
    handle,
    *,
    job_id: str,
    media_path: str,
    source_filename: str,
    display_name: str,
    opts: EnqueueOptions,
) -> None:
    try:
        run_transcription_job(
            media_path=media_path,
            selected_engines=list(opts.selected_engines),
            language=opts.language,
            diarization=opts.diarization,
            max_speakers=int(opts.max_speakers or 0),
            enhance=opts.enhance,
            diarize_kwargs=opts.diarize_kwargs,
            cancel_event=handle.cancel_event,
            progress=handle.progress,
            meta=JobMeta(
                tab_id=opts.tab_id or f"queue:{job_id}",
                display_name=display_name,
                source_filename=source_filename,
                output_name=opts.output_name or display_name,
                client_ip=opts.client_ip,
                user_id=int(opts.user_id or 0),
                username=opts.username or "",
            ),
            job_id=job_id,
        )
    except RuntimeError as exc:
        status = "cancelled" if "cancel" in str(exc).lower() else "failed"
        write_job_record(
            job_id,
            {
                "status": status,
                "error": str(exc),
                "user_id": int(opts.user_id or 0),
                "username": opts.username or "",
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Queued job %s failed", job_id)
        write_job_record(
            job_id,
            {
                "status": "failed",
                "error": str(exc),
                "user_id": int(opts.user_id or 0),
                "username": opts.username or "",
            },
        )


def enqueue_media_job(
    media_path: str,
    opts: EnqueueOptions,
    *,
    job_id: str | None = None,
    already_archived: bool = False,
) -> EnqueuedJob:
    """Archive (unless already), reserve queue slot, persist queued, start worker."""
    jid = job_id or new_job_id()
    source_filename, display_name = _display_name(media_path, opts.output_name)

    if not try_reserve_queue_slot():
        return EnqueuedJob(
            job_id=jid,
            source_filename=source_filename,
            display_name=display_name,
            status="rejected",
            error="Job queue full.",
        )

    if already_archived:
        durable = media_path
    else:
        durable = copy_input_file(media_path, jid, source_filename) or media_path

    write_job_record(
        jid,
        {
            "job_id": jid,
            "status": "queued",
            "display_name": display_name,
            "source_filename": source_filename,
            "source_path": durable,
            "input_path": durable,
            "user_id": int(opts.user_id or 0),
            "username": opts.username or "",
            "language": opts.language,
            "diarization": opts.diarization,
            "enhance": opts.enhance,
            "max_speakers": int(opts.max_speakers or 0),
            "selected_engines": list(opts.selected_engines),
            "client_ip": opts.client_ip or "",
            "tab_id": opts.tab_id or f"queue:{jid}",
            "output_name": opts.output_name or display_name,
        },
    )

    job_opts = EnqueueOptions(
        language=opts.language,
        diarization=opts.diarization,
        enhance=opts.enhance,
        max_speakers=opts.max_speakers,
        selected_engines=list(opts.selected_engines),
        output_name=opts.output_name or display_name,
        diarize_kwargs=opts.diarize_kwargs,
        tab_id=opts.tab_id or f"queue:{jid}",
        client_ip=opts.client_ip,
        user_id=opts.user_id,
        username=opts.username,
    )

    def _worker(handle) -> None:
        _run_enqueued_job(
            handle,
            job_id=jid,
            media_path=durable,
            source_filename=source_filename,
            display_name=display_name,
            opts=job_opts,
        )

    try:
        submit_background_job(jid, _worker)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        release_queue_slot(started=False)
        write_job_record(jid, {"status": "failed", "error": str(exc)})
        return EnqueuedJob(
            job_id=jid,
            source_filename=source_filename,
            display_name=display_name,
            status="failed",
            error=str(exc),
        )

    return EnqueuedJob(
        job_id=jid,
        source_filename=source_filename,
        display_name=display_name,
        status="queued",
    )


def enqueue_media_batch(
    media_paths: list[str],
    opts: EnqueueOptions,
    *,
    max_files: int | None = None,
) -> EnqueueBatchResult:
    """Enqueue up to *max_files* paths; reject the rest with clear errors."""
    cap = max_files if max_files is not None else max_batch_files()
    accepted: list[EnqueuedJob] = []
    rejected: list[EnqueuedJob] = []

    paths = [p for p in media_paths if p]
    if len(paths) > cap:
        for extra in paths[cap:]:
            name = Path(extra).name
            rejected.append(
                EnqueuedJob(
                    job_id="",
                    source_filename=name,
                    display_name=Path(name).stem,
                    status="rejected",
                    error=f"Batch limit is {cap} files.",
                )
            )
        paths = paths[:cap]

    for path in paths:
        # Per-file output name only when single file and opts.output_name set
        file_opts = opts
        if len(paths) > 1:
            file_opts = EnqueueOptions(
                language=opts.language,
                diarization=opts.diarization,
                enhance=opts.enhance,
                max_speakers=opts.max_speakers,
                selected_engines=list(opts.selected_engines),
                output_name=None,
                diarize_kwargs=opts.diarize_kwargs,
                tab_id=opts.tab_id,
                client_ip=opts.client_ip,
                user_id=opts.user_id,
                username=opts.username,
            )
        result = enqueue_media_job(path, file_opts)
        if result.status == "queued":
            accepted.append(result)
        else:
            rejected.append(result)

    return EnqueueBatchResult(
        accepted=accepted,
        rejected=rejected,
        queue=snapshot_queue(),
    )


_OPTION_KEYS = (
    "language",
    "diarization",
    "enhance",
    "max_speakers",
    "selected_engines",
    "output_name",
)


def _input_path_for_resume(manifest: dict, row: dict) -> str:
    return str(
        manifest.get("input_path")
        or manifest.get("source_path")
        or row.get("input_path")
        or ""
    )


def _options_for_resume(manifest: dict, row: dict) -> dict[str, Any]:
    options = row.get("options") if isinstance(row.get("options"), dict) else {}
    if options:
        return options
    return {k: manifest[k] for k in _OPTION_KEYS if k in manifest}


def _engines_for_resume(options: dict, manifest: dict) -> list[str]:
    engines = options.get("selected_engines") or manifest.get("selected_engines") or ["Auto"]
    if isinstance(engines, str):
        return [engines]
    return list(engines)


def _mark_running_as_queued(job_id: str, manifest: dict, row: dict) -> None:
    from backend.jobs_db import upsert_job

    if (manifest.get("status") or row.get("status")) != "running":
        return
    patch = {"status": "queued", "error": "Re-queued after restart."}
    upsert_job(job_id, patch)
    write_job_record(job_id, patch)


def _enqueue_options_for_resume(
    options: dict,
    engines: list[str],
    manifest: dict,
    row: dict,
) -> EnqueueOptions:
    return EnqueueOptions(
        language=str(options.get("language") or "Thai"),
        diarization=bool(options.get("diarization", True)),
        enhance=bool(options.get("enhance", True)),
        max_speakers=int(options.get("max_speakers") or 0),
        selected_engines=engines,
        output_name=options.get("output_name") or row.get("display_name") or None,
        tab_id=str(manifest.get("tab_id") or row.get("tab_id") or ""),
        client_ip=str(manifest.get("client_ip") or row.get("client_ip") or ""),
        user_id=int(manifest.get("user_id") or row.get("user_id") or 0),
        username=str(manifest.get("username") or row.get("username") or ""),
    )


def _tally_resume_result(result: EnqueuedJob, counters: dict[str, int]) -> None:
    if result.status == "queued":
        counters["resumed"] += 1
    elif result.status == "rejected":
        counters["skipped"] += 1
        logger.warning("Could not resume job %s: %s", result.job_id, result.error)
    else:
        counters["failed"] += 1


def _resume_one_job(row: dict[str, Any], counters: dict[str, int]) -> None:
    from backend.storage import load_job

    job_id = row.get("job_id") or ""
    if not job_id:
        return
    manifest = load_job(job_id) or row
    input_path = _input_path_for_resume(manifest, row)
    if not input_path or not Path(str(input_path)).is_file():
        write_job_record(
            job_id,
            {
                "status": "failed",
                "error": "Interrupted: input file missing after restart.",
            },
        )
        counters["failed"] += 1
        return

    options = _options_for_resume(manifest, row)
    engines = _engines_for_resume(options, manifest)
    _mark_running_as_queued(job_id, manifest, row)
    result = enqueue_media_job(
        str(input_path),
        _enqueue_options_for_resume(options, engines, manifest, row),
        job_id=job_id,
        already_archived=True,
    )
    _tally_resume_result(result, counters)


def resume_interrupted_jobs() -> dict[str, int]:
    """Re-queue jobs left in queued/running after process restart."""
    from backend.jobs_db import list_resumable_jobs

    counters = {"resumed": 0, "failed": 0, "skipped": 0}
    for row in list_resumable_jobs():
        _resume_one_job(row, counters)
    logger.info("Job resume after startup: %s", counters)
    return counters


__all__ = [
    "EnqueueBatchResult",
    "EnqueueOptions",
    "EnqueuedJob",
    "enqueue_media_batch",
    "enqueue_media_job",
    "resume_interrupted_jobs",
]
