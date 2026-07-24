"""Local Transcript App - Gradio UI for local-only ASR and diarization."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import importlib.machinery
import logging
import os
import sys
import threading
import time
import types
import warnings
from dataclasses import dataclass

from backend.dotenv_load import load_dotenv_safe
from backend.paths import app_root, ensure_bundle_on_path, resolve_path

ensure_bundle_on_path()
os.chdir(app_root())

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Prevent Gradio from making outbound requests (messaging/version/IP).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TELEMETRY_ENABLED", "False")

_install_root = app_root()
load_dotenv_safe(_install_root / ".env")
load_dotenv_safe(_install_root / ".env.production", override=False)

from backend.cpu_limits import apply_cpu_thread_limits

apply_cpu_thread_limits()

_model_root = os.getenv("APP_MODEL_ROOT") or str(_install_root / "models")
if not os.path.isabs(_model_root):
    _model_root = str(resolve_path(_model_root))

from engines.model_cache import (
    apply_runtime_cache_env_defaults,
    configure_project_cache_paths,
    consolidate_misplaced_hub_caches,
)

configure_project_cache_paths(_install_root, model_root=_model_root)
apply_runtime_cache_env_defaults()
consolidate_misplaced_hub_caches(_install_root)


def _install_torchcodec_stub() -> None:
    """Avoid torchcodec native DLL failures on systems using librosa input."""
    if "torchcodec" in sys.modules:
        return
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
    torchcodec.AudioDecoder = type("AudioDecoder", (), {})
    torchcodec.AudioSamples = type("AudioSamples", (), {})
    torchcodec.AudioStreamMetadata = type("AudioStreamMetadata", (), {})
    for name in ["decoders", "encoders", "samplers", "transforms"]:
        module = types.ModuleType(f"torchcodec.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(f"torchcodec.{name}", None)
        if name == "decoders":
            setattr(module, "AudioDecoder", type("AudioDecoder", (), {}))
            setattr(module, "AudioSamples", type("AudioSamples", (), {}))
            setattr(module, "AudioStreamMetadata", type("AudioStreamMetadata", (), {}))
        setattr(torchcodec, name, module)
        sys.modules[f"torchcodec.{name}"] = module
    sys.modules["torchcodec"] = torchcodec


_install_torchcodec_stub()
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec.*|.*libtorchcodec.*|.*FFmpeg.*version.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*|.*TF32.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logging.captureWarnings(True)

import gradio as gr
from gradio.themes import Soft as _SoftTheme


def _patch_gradio_schema_parser() -> None:
    """Prevent gradio_client crash on bool JSON-schema additionalProperties."""
    try:
        import gradio_client.utils as _gcu
    except (ImportError, RuntimeError):
        return

    original = getattr(_gcu, "_json_schema_to_python_type", None)
    if not callable(original) or getattr(_gcu, "_local_transcript_schema_patch", False):
        return

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        if schema is None:
            return "Any"
        return original(schema, defs)

    _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
    _gcu._local_transcript_schema_patch = True


_patch_gradio_schema_parser()

from backend.auth_users import (
    get_user_by_username,
    gradio_auth_credentials,
    init_user_db,
    register_user,
)
from backend.client_identity import client_ip_from_request
from backend.job_api import build_api_routes
from backend.job_cancel import cancel_tab_job
from backend.job_queue import get_job_progress as get_api_job_progress
from backend.pipeline import JobMeta, active_job_count, run_transcription_job
from backend.progress import JobProgress, get_job_progress
from backend.ui_session import (
    clear_active_job,
    fresh_cancel_event,
    init_tab_instance_id,
    is_job_running,
    resolve_runtime,
    set_active_job,
)
from backend.services.asr_local import (
    ALL_ENGINES,
    ENGINE_AUTO,
    ENGINE_PATHUMMA,
    ENGINE_TYPHOON,
    LANGUAGES,
    UI_ENGINE_CHOICES,
    clear_accelerator_cache,
    default_asr_engines,
    engine_for_preload,
    is_auto_engine,
    load_model,
    resolve_asr_engines,
    switch_asr_engine,
)
from backend.storage import ensure_app_dirs, list_jobs, load_job
from backend.services.hardware_policy import detect_hardware, hardware_summary
from backend.ui_limits import (
    display_transcript_text as _display_transcript_text,
    format_media_info as _format_media_info,
    media_too_large_for_browser as _media_too_large_for_browser,
)
from backend.services.media_pipeline import clear_prejob_caches


LABEL_ELAPSED = "Elapsed Time"
LABEL_DOWNLOAD = "Download .txt"
_CANCELLED = "(cancelled)"
_MSG_CANCELLED = "Cancelled."
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".ts"}

APP_CSS = """
.config-landscape textarea { min-height: 42px !important; }
.tall-audio-preview { min-height: 220px !important; }
.tall-audio-preview audio { min-height: 100px !important; }
.tall-audio-preview .wrap { min-height: 200px !important; }
.tall-audio-preview .audio-container { min-height: 200px !important; }
.tall-audio-preview .waveform-container { min-height: 120px !important; }
.tall-video-preview { min-height: 360px !important; }
.tall-video-preview video { min-height: 320px !important; }
.live-status {
    padding: 14px 18px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    margin: 8px 0;
    border: 1px solid transparent;
    min-height: 52px;
}
.live-status.idle    { background: #f4f4f5; color: #52525b; border-color: #e4e4e7; }
.live-status.running { background: #fef3c7; color: #92400e; border-color: #fde68a;
                       animation: live-pulse 1.4s ease-in-out infinite; }
.live-status.done    { background: #dcfce7; color: #166534; border-color: #bbf7d0; }
.live-status.error   { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
.live-status .spinner { display: inline-block; margin-right: 8px; }
.job-progress-panel {
    padding: 14px 18px;
    border-radius: 10px;
    background: #fafafa;
    border: 1px solid #e4e4e7;
    margin: 8px 0 12px 0;
    min-height: 96px;
}
.job-progress-panel.active { border-color: #fde68a; background: #fffbeb; }
.job-progress-panel.done { border-color: #bbf7d0; background: #f0fdf4; }
.job-progress-panel.error { border-color: #fecaca; background: #fef2f2; }
.job-progress-panel.idle { opacity: 0.85; }
.job-status-label { font-weight: 600; font-size: 14px; margin-bottom: 8px; color: #3f3f46; }
#elapsed-timer {
    font-size: 20px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #18181b;
    margin-bottom: 6px;
}
.progress-message { font-size: 13px; color: #71717a; }
@keyframes live-pulse {
    0%, 100% { opacity: 1.0; }
    50%      { opacity: 0.65; }
}
@media (max-width: 900px) {
    .config-landscape { flex-wrap: wrap; }
}
#tab-instance-id { display: none !important; }
"""

# sessionStorage is per browser tab — isolates cancel/progress from other tabs/users.
TAB_INSTANCE_SCRIPT = """
<script>
(function initTabInstanceId() {
  const KEY = 'lta_tab_instance_id';
  function apply() {
    const host = document.getElementById('tab-instance-id');
    if (!host) return false;
    const input = host.querySelector('textarea, input');
    if (!input) return false;
    let id = sessionStorage.getItem(KEY);
    if (!id && input.value && input.value.trim()) {
      id = input.value.trim();
      sessionStorage.setItem(KEY, id);
    }
    if (!id) {
      id = (typeof crypto !== 'undefined' && crypto.randomUUID)
        ? crypto.randomUUID()
        : ('tab_' + Date.now() + '_' + Math.random().toString(36).slice(2));
      sessionStorage.setItem(KEY, id);
    }
    if (input.value !== id) {
      input.value = id;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    return true;
  }
  if (!apply()) {
    const obs = new MutationObserver(function() { if (apply()) obs.disconnect(); });
    obs.observe(document.body, { childList: true, subtree: true });
    setTimeout(function() { obs.disconnect(); }, 15000);
  }
})();
</script>
"""

_models_ready = threading.Event()
_load_status = dict.fromkeys(ALL_ENGINES, "pending")
_last_load_status_text: str | None = None
_last_ready_state: bool | None = None


def _gradio_transcribe_concurrency() -> int:
    """How many transcribe streams may run at once (tabs/users); independent of GPU slots."""
    raw = os.getenv("UI_GRADIO_TRANSCRIBE_CONCURRENCY")
    if raw is None:
        raw = os.getenv("UI_MAX_CONCURRENT_JOBS", "8")
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _gradio_queue_max_size() -> int:
    try:
        return max(1, int(os.getenv("UI_QUEUE_MAX_SIZE", "16")))
    except ValueError:
        return 16


def _history_per_client_ip() -> bool:
    return _env_bool("UI_HISTORY_PER_CLIENT_IP", True)


def _is_eager_preload_mode(mode: str | None = None) -> bool:
    if mode is None:
        mode = os.getenv("ASR_PRELOAD_MODE", "eager").strip().lower()
    return mode in {"eager", "preload", "true", "1"}


def _resolve_preload_engines() -> list[str]:
    configured = os.getenv("ASR_PRELOAD_ENGINES", "").strip()
    if configured:
        return [part.strip() for part in configured.split(",") if part.strip()]
    # When running cache-first (kept resident) on OpenVINO/CPU, preload both engines
    # so switching is instant.
    try:
        hw = detect_hardware()
    except Exception:  # pylint: disable=broad-exception-caught
        hw = {}
    keep_preloaded = os.getenv("ASR_KEEP_PRELOADED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if keep_preloaded and hw.get("backend") == "openvino":
        return [ENGINE_TYPHOON, ENGINE_PATHUMMA]
    return default_asr_engines()


def _mark_engines_available(engines: list[str]) -> None:
    for engine in engines:
        _load_status[engine] = "available"


def _preload_asr_engine(engine: str) -> bool:
    try:
        _load_status[engine] = "loading..."
        load_model(engine)
        _load_status[engine] = "ready"
        logger.info("%s loaded.", engine)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _load_status[engine] = f"FAILED: {exc}"
        logger.exception("%s load failed: %s", engine, exc)
        return False


def _try_preload_diarization() -> None:
    mode = os.getenv("DIARIZATION_PRELOAD_MODE", "eager").strip().lower()
    if not _is_eager_preload_mode(mode):
        return
    try:
        from engines.diarization import load_model as load_diarization_model
        from engines.model_cache import configured_diarization_model_id, has_cached_pipeline

        diarization_model = configured_diarization_model_id()
        if not has_cached_pipeline(diarization_model):
            logger.warning(
                "Skipping diarization preload; model %s is not cached yet.",
                diarization_model,
            )
            return
        logger.info("Preloading pyannote diarization model...")
        load_diarization_model()
        logger.info("Pyannote diarization loaded.")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Pyannote diarization preload skipped: %s", exc)


def _preload_models() -> None:
    """Preload the configured ASR model at startup by default."""
    if not _is_eager_preload_mode():
        _mark_engines_available(ALL_ENGINES)
        _models_ready.set()
        logger.info("ASR preload skipped; models are available on demand.")
        return

    preload_engines = _resolve_preload_engines()
    warmed_engines = [engine_for_preload(engine) for engine in preload_engines]
    skipped = [e for e in ALL_ENGINES if e not in warmed_engines]
    _mark_engines_available(skipped)
    if skipped:
        logger.info(
            "ASR eager preload limited to %s; others available on demand.",
            ", ".join(warmed_engines),
        )

    asr_ready = False
    preload_failed = False
    for engine in preload_engines:
        if _preload_asr_engine(engine_for_preload(engine)):
            asr_ready = True
        else:
            preload_failed = True

    if not asr_ready:
        logger.error("No ASR engine preloaded; upload remains disabled until model cache is fixed.")
        return

    _try_preload_diarization()

    if preload_failed:
        logger.warning("Some optional models failed preload; continuing with available engines.")
    _models_ready.set()
    logger.info("Model preload finished.")


def _get_load_status() -> str:
    if _models_ready.is_set():
        lines = ["### All Local Models Ready"]
    else:
        lines = ["### Loading Local Models"]
    for engine in ALL_ENGINES:
        lines.append(f"- {engine}: **{_load_status[engine]}**")
    if not _models_ready.is_set():
        lines.append("\n*Upload will be enabled after model preload completes.*")
    return "\n".join(lines)


def _status_html(state: str, message: str) -> str:
    """Render a status banner shown ABOVE the engine output tabs.

    state: 'idle' | 'running' | 'done' | 'error'
    """
    icon = {
        "idle":    "\u23F8",   # pause
        "running": "\u23F3",   # hourglass
        "done":    "\u2705",   # check
        "error":   "\u26A0\uFE0F",  # warning
    }.get(state, "\u2139\uFE0F")
    return (
        f'<div class="live-status {state}">'
        f'<span class="spinner">{icon}</span>{message}'
        f"</div>"
    )


STATUS_IDLE = _status_html("idle", "Idle. Upload media and click Transcribe.")


def _format_elapsed_hms(elapsed_s: float) -> str:
    elapsed = max(0.0, float(elapsed_s))
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _job_status_html(snapshot: dict) -> str:
    """Render stopwatch + phase message for the active job."""
    phase = snapshot.get("phase", "idle")
    elapsed_txt = _format_elapsed_hms(snapshot.get("elapsed_s", 0))
    active_cls = "active" if snapshot.get("active") else phase
    return (
        f'<div class="job-progress-panel {active_cls}" data-phase="{phase}">'
        f'<div class="job-status-label">Job Status</div>'
        f'<div id="elapsed-timer">Elapsed Time: {elapsed_txt}</div>'
        f'<div class="progress-message">{snapshot.get("message", "")}</div>'
        f"</div>"
    )


PROGRESS_IDLE = _job_status_html({
    "phase": "idle",
    "active": False,
    "elapsed_s": 0,
    "message": "Waiting for a transcription job.",
})


def _request_username(request: gr.Request | None) -> str:
    if request is None:
        return ""
    return (getattr(request, "username", None) or "").strip()


def _request_user_meta(request: gr.Request | None) -> tuple[int, str]:
    username = _request_username(request)
    if not username:
        return 0, ""
    user = get_user_by_username(username)
    if user is None:
        return 0, username
    return user.id, user.username


def _history_dropdown_update(
    client_ip: str | None = None,
    *,
    username: str | None = None,
    user_id: int | None = None,
):
    filter_ip = None
    if not username and not user_id:
        filter_ip = client_ip if (_history_per_client_ip() and client_ip) else None
    choices = []
    for row in list_jobs(50, client_ip=filter_ip, username=username, user_id=user_id):
        name = row.get("display_name") or row.get("source_filename") or row["job_id"]
        engines = ", ".join(row.get("selected_engines") or [])
        created = (row.get("created_at") or "")[:16]
        user_tag = ""
        if row.get("username"):
            user_tag = f" | @{row['username']}"
        elif not filter_ip and row.get("client_ip"):
            user_tag = f" | {row['client_ip']}"
        label = f"{created} | {name} | {engines} | {row.get('status', '')}{user_tag}"
        choices.append((label, row["job_id"]))
    return gr.update(choices=choices)


def _manifest_to_job_result(job: dict) -> dict:
    job_id = job.get("job_id", "")
    return {
        "job_id": job_id,
        "manifest_path": f"storage/jobs/{job_id}.json",
        "audio_duration_s": job.get("audio_duration_s", 0.0),
        "total_elapsed_s": job.get("total_elapsed_s", 0.0),
        "target_elapsed_s": job.get("target_elapsed_s", 0.0),
        "target_met": job.get("target_met", False),
        "results": job.get("results") or {},
    }


def _default_output_names(media_path: str | None) -> str:
    if not media_path:
        return ""
    return os.path.splitext(os.path.basename(media_path))[0]


def _transcribe_btn_running() -> dict:
    return gr.update(interactive=False)


def _transcribe_btn_ready() -> dict:
    return gr.update(interactive=_models_ready.is_set())


def _empty_outputs(
    message: str,
    status_state: str = "idle",
    status_message: str = "Idle.",
    tracker: JobProgress | None = None,
    *,
    refresh_history: bool = True,
    client_ip: str | None = None,
    username: str | None = None,
    user_id: int | None = None,
) -> tuple:
    no_download = gr.update(value=None, interactive=False)
    snap = (tracker or JobProgress()).snapshot()
    progress = _job_status_html({
        "phase": status_state if status_state in {"error"} else "idle",
        "active": status_state == "running",
        "elapsed_s": snap["elapsed_s"],
        "message": status_message,
    })
    history = (
        _history_dropdown_update(
            client_ip, username=username, user_id=user_id,
        ) if refresh_history else gr.update()
    )
    return (
        message, "", gr.update(), no_download,
        "",
        _status_html(status_state, status_message),
        progress,
        _transcribe_btn_ready(),
        history,
        gr.update(value=None, interactive=False),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_outputs(
    job_result: dict,
    selected_engines: list[str],
    tracker: JobProgress,
    *,
    client_ip: str | None = None,
    username: str | None = None,
    user_id: int | None = None,
) -> tuple:
    engine = selected_engines[0] if selected_engines else default_asr_engines()[0]
    result = job_result["results"].get(engine)
    if result:
        path = result.get("download_path")
        raw_text = result.get("text", "")
        text = _display_transcript_text(raw_text)
        outputs = [
            text,
            f"{result.get('elapsed', 0.0):.2f}s",
            gr.update(),
            gr.update(value=path, interactive=path is not None),
        ]
    elif engine in selected_engines:
        outputs = [
            "(failed)",
            "",
            gr.update(),
            gr.update(value=None, interactive=False),
        ]
    else:
        outputs = [
            "",
            "(not selected)",
            gr.update(),
            gr.update(value=None, interactive=False),
        ]
    manifest = job_result.get("manifest_path", "")
    audio_duration = job_result.get("audio_duration_s", 0.0)
    total_elapsed = job_result.get("total_elapsed_s", 0.0)
    target_elapsed = job_result.get("target_elapsed_s", 0.0)
    target_state = "met" if job_result.get("target_met") else "not met"
    performance = (
        f"\nAudio: {audio_duration:.2f}s | Total elapsed: {total_elapsed:.2f}s | "
        f"Target: {target_elapsed:.2f}s ({target_state})"
        if target_elapsed
        else ""
    )
    outputs.append(f"Job ID: {job_result.get('job_id', '')}\nManifest: {manifest}{performance}")
    notes = [
        f"{engine}: {result.get('note')}"
        for engine, result in job_result.get("results", {}).items()
        if result.get("note")
    ]
    if notes:
        outputs[-1] = f"{outputs[-1]}\n" + "\n".join(notes)
    target_met = bool(job_result.get("target_met"))
    if target_elapsed:
        target_state = "met \u2705" if target_met else "over target"
        perf_text = (
            f"Done in {total_elapsed:.1f}s (audio {audio_duration:.1f}s, "
            f"target {target_elapsed:.1f}s, {target_state})"
        )
    else:
        perf_text = f"Done in {total_elapsed:.1f}s."
    job_note = (
        f"Done. Job ID {job_result.get('job_id', '')} — safe to close this page; "
        "retrieve later under Previous transcripts or GET /api/jobs/{id}."
    )
    outputs.append(_status_html("done", f"{perf_text} {job_note}"))
    outputs.append(_job_status_html(tracker.snapshot()))
    outputs.append(_transcribe_btn_ready())
    outputs.append(
        _history_dropdown_update(client_ip, username=username, user_id=user_id)
    )
    outputs.append(gr.update(value=None, interactive=False))
    return tuple(outputs)


def _reset_ui_outputs(tab_id: str, request: gr.Request | None = None) -> tuple:
    """Signal cancellation for this browser tab only; free GPU cache for queued jobs."""
    runtime, _ = resolve_runtime(tab_id)
    tracker = runtime["progress"]
    cancel_tab_job(runtime, tracker=tracker, message=_MSG_CANCELLED)
    client_ip = client_ip_from_request(request)
    user_id, username = _request_user_meta(request)
    no_dl = gr.update(value=None, interactive=False)
    return (
        _CANCELLED, "", gr.update(), no_dl,
        "Cancelled by user.",
        _status_html("error", "Cancelled by user."),
        PROGRESS_IDLE,
        _transcribe_btn_ready(),
        _history_dropdown_update(
            client_ip, username=username or None, user_id=user_id or None,
        ),
        gr.update(value=None, interactive=False),
    )


def _selected_engines_for_job(selected_engines, language: str) -> list[str]:
    return resolve_asr_engines(language, selected_engines or default_asr_engines())


def _diarize_kwargs_for_job(
    diarization: bool,
    max_speakers: int,
    diar_override_defaults: bool,
    diar_short_clip_preset: bool,
    diar_seg_threshold,
    diar_min_off,
    diar_clust_threshold,
    diar_clust_min_size,
) -> dict | None:
    if not diarization:
        return None
    kwargs: dict = {}
    if max_speakers >= 2:
        kwargs["num_speakers"] = int(max_speakers)
    if diar_override_defaults or diar_short_clip_preset:
        kwargs.update({
            "seg_threshold": float(diar_seg_threshold),
            "seg_min_duration_off": float(diar_min_off),
            "clust_threshold": float(diar_clust_threshold),
            "clust_min_size": int(diar_clust_min_size),
        })
    return kwargs


def _running_transcript_outputs(snap: dict, engines_label: str, no_dl) -> tuple:
    return (
        "", "", gr.update(), no_dl,
        "",
        _status_html("running", snap.get("message", f"Transcribing with {engines_label}\u2026")),
        _job_status_html(snap),
        _transcribe_btn_running(),
        gr.update(),
        gr.update(value=None, interactive=False),
    )


def _transcription_progress_signature(snap: dict) -> tuple:
    return (
        snap.get("phase"),
        round(float(snap.get("percent", 0)), 1),
        int(float(snap.get("elapsed_s", 0))),
        snap.get("message"),
    )


def _cancelled_transcription_outputs(tracker: JobProgress) -> tuple:
    tracker.fail(_MSG_CANCELLED)
    return _empty_outputs(_CANCELLED, "error", _MSG_CANCELLED, tracker=tracker)


def _transcription_error_outputs(tracker: JobProgress, exc: Exception) -> tuple:
    if isinstance(exc, RuntimeError) and "cancelled" in str(exc).lower():
        return _cancelled_transcription_outputs(tracker)
    logger.exception("Transcription job failed: %s", exc)
    tracker.fail(str(exc))
    return _empty_outputs(f"ERROR: {exc}", "error", f"Error: {exc}", tracker=tracker)


@dataclass
class _TranscribeRequest:
    media_path: str
    selected_engines: str
    language: str
    diarization: bool
    max_speakers: int
    enhance: bool
    diar_short_clip_preset: bool
    diar_override_defaults: bool
    diar_seg_threshold: float
    diar_min_off: float
    diar_clust_threshold: float
    diar_clust_min_size: int
    output_name: str
    tab_id: str


def _parse_transcribe_request(inputs: tuple) -> _TranscribeRequest:
    return _TranscribeRequest(*inputs)


def _progress_poll_interval() -> float:
    try:
        return max(0.25, float(os.getenv("UI_PROGRESS_POLL_S", "1.0")))
    except ValueError:
        return 1.0


def _transcribe_blocked_output(runtime, tracker, media_path):
    if is_job_running(runtime):
        return _empty_outputs(
            "Job still running — wait or click Cancel.",
            "running",
            "Job still running — wait or click Cancel.",
            tracker=tracker,
        )
    if not media_path:
        tracker.reset()
        return _empty_outputs("(no media provided)", "error", "No media uploaded.", tracker=tracker)
    if not _models_ready.is_set():
        return _empty_outputs(
            "Models are still loading, please wait...",
            "running",
            "Models are still loading\u2026",
            tracker=tracker,
        )
    return None


def _resolve_job_names(
    media_path: str,
    output_name_field: str,
) -> tuple[str, str, str]:
    output_name = output_name_field.strip() if output_name_field and output_name_field.strip() else ""
    source_filename = os.path.basename(media_path)
    display_name = output_name or os.path.splitext(source_filename)[0]
    return output_name, display_name, source_filename


def _poll_transcription_worker(
    worker: threading.Thread,
    cancel_event: threading.Event,
    tracker: JobProgress,
    runtime: dict,
    engines_label: str,
    no_dl: dict,
    poll_s: float,
):
    last_sig = None
    while worker.is_alive():
        if cancel_event.is_set():
            cancel_tab_job(runtime)
            yield _cancelled_transcription_outputs(tracker)
            return
        snap = tracker.snapshot()
        if snap.get("job_id") and runtime.get("active_job_id") != snap["job_id"]:
            set_active_job(runtime, snap["job_id"], worker)
        sig = _transcription_progress_signature(snap)
        if sig != last_sig or last_sig is None:
            last_sig = sig
        yield _running_transcript_outputs(snap, engines_label, no_dl)
        time.sleep(poll_s)


def _stream_worker_progress(
    worker: threading.Thread,
    tracker: JobProgress,
    no_dl: dict,
    poll_s: float,
):
    while worker.is_alive():
        snap = tracker.snapshot()
        yield _running_transcript_outputs(snap, "ASR", no_dl)
        time.sleep(poll_s)


def _recover_manifest_or_idle(tab_id: str, tracker: JobProgress, no_dl: dict):
    for row in list_jobs(20):
        if row.get("tab_id") != tab_id or row.get("status") != "running":
            continue
        job_id = row["job_id"]
        while True:
            job = load_job(job_id)
            if not job or job.get("status") != "running":
                break
            prog = job.get("progress") or {}
            snap = {
                "phase": prog.get("phase", "running"),
                "message": prog.get("message", "Finishing in background\u2026"),
                "elapsed_s": prog.get("elapsed_s", 0),
                "active": True,
            }
            yield _running_transcript_outputs(snap, "ASR", no_dl)
            time.sleep(1.0)
        yield _empty_outputs(
            "",
            "idle",
            "Previous job finished — open Previous transcripts to load results.",
            tracker=tracker,
        )
        return
    yield _empty_outputs("", "idle", "Idle. Upload media and click Transcribe.", tracker=tracker)


def transcribe(*inputs, request: gr.Request | None = None):
    """Gradio callback — per browser-tab isolation; stopwatch via gr.Timer."""
    req = _parse_transcribe_request(inputs)
    client_ip = client_ip_from_request(request)
    user_id, username = _request_user_meta(request)
    runtime, tid = resolve_runtime(req.tab_id)
    tracker = runtime["progress"]
    blocked = _transcribe_blocked_output(runtime, tracker, req.media_path)
    if blocked is not None:
        yield blocked
        return

    worker = runtime.get("worker")
    if worker is not None and worker.is_alive():
        cancel_tab_job(runtime)

    cancel_event = fresh_cancel_event(runtime, cancel_previous=False)
    selected = _selected_engines_for_job(req.selected_engines, req.language)
    diarize_kwargs = _diarize_kwargs_for_job(
        req.diarization,
        int(req.max_speakers),
        req.diar_override_defaults,
        req.diar_short_clip_preset,
        req.diar_seg_threshold,
        req.diar_min_off,
        req.diar_clust_threshold,
        req.diar_clust_min_size,
    )

    no_dl = gr.update(value=None, interactive=False)
    auto_selected = is_auto_engine(req.selected_engines)
    engines_label = f"Auto → {selected[0]}" if auto_selected else selected[0]
    runtime["selected_asr_engine"] = req.selected_engines if auto_selected else selected[0]
    output_name, display_name, source_filename = _resolve_job_names(
        req.media_path,
        req.output_name,
    )

    holder: dict = {}
    error_holder: dict = {}

    def _run_job() -> None:
        try:
            holder["result"] = run_transcription_job(
                media_path=req.media_path,
                selected_engines=selected,
                language=req.language,
                diarization=req.diarization,
                max_speakers=int(req.max_speakers),
                enhance=req.enhance,
                diarize_kwargs=diarize_kwargs,
                cancel_event=cancel_event,
                progress=tracker,
                meta=JobMeta(
                    tab_id=tid,
                    display_name=display_name,
                    source_filename=source_filename,
                    output_name=output_name or None,
                    client_ip=client_ip,
                    user_id=user_id,
                    username=username,
                ),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_holder["error"] = exc
        finally:
            clear_active_job(runtime)

    tracker.reset()
    tracker.start()
    worker = threading.Thread(target=_run_job, daemon=True)
    set_active_job(runtime, "", worker)
    worker.start()

    yield _running_transcript_outputs(tracker.snapshot(), engines_label, no_dl)

    poll_s = _progress_poll_interval()
    yield from _poll_transcription_worker(
        worker, cancel_event, tracker, runtime, engines_label, no_dl, poll_s,
    )
    if cancel_event.is_set():
        return

    worker.join()

    if error_holder.get("error"):
        yield _transcription_error_outputs(tracker, error_holder["error"])
        return

    yield _build_outputs(
        holder["result"],
        selected,
        tracker,
        client_ip=client_ip,
        username=username or None,
        user_id=user_id or None,
    )


def load_history(request: gr.Request | None = None):
    client_ip = client_ip_from_request(request)
    user_id, username = _request_user_meta(request)
    return _history_dropdown_update(
        client_ip, username=username or None, user_id=user_id or None,
    )


def _job_access_denied(job: dict, request: gr.Request | None) -> bool:
    user_id, username = _request_user_meta(request)
    if user_id or username:
        if int(job.get("user_id") or 0) == int(user_id):
            return False
        job_user = (job.get("username") or "").strip().lower()
        if job_user and job_user == username.lower():
            return False
        # Legacy jobs without username: allow owner via IP when enabled.
        if not job_user and not int(job.get("user_id") or 0):
            client_ip = client_ip_from_request(request)
            if (
                _history_per_client_ip()
                and client_ip
                and (job.get("client_ip") or "") not in {"", client_ip}
            ):
                return True
            return False
        return True
    client_ip = client_ip_from_request(request)
    return bool(
        _history_per_client_ip()
        and client_ip
        and (job.get("client_ip") or "") not in {"", client_ip}
    )


def load_selected_job(job_id: str, request: gr.Request | None = None):
    client_ip = client_ip_from_request(request)
    user_id, username = _request_user_meta(request)
    if not job_id:
        return _empty_outputs(
            "Select a job from Previous transcripts.",
            refresh_history=False,
            client_ip=client_ip,
            username=username or None,
            user_id=user_id or None,
        )
    job = load_job(job_id)
    if not job:
        return _empty_outputs(
            f"Job not found: {job_id}",
            "error",
            "Job not found.",
            client_ip=client_ip,
            username=username or None,
            user_id=user_id or None,
        )
    if _job_access_denied(job, request):
        return _empty_outputs(
            "Job not found for this account.",
            "error",
            "Access denied.",
            client_ip=client_ip,
            username=username or None,
            user_id=user_id or None,
        )
    selected = job.get("selected_engines") or default_asr_engines()
    return _build_outputs(
        _manifest_to_job_result(job),
        selected,
        JobProgress(),
        client_ip=client_ip,
        username=username or None,
        user_id=user_id or None,
    )


def download_selected_job(job_id: str, request: gr.Request | None = None):
    if not job_id:
        return gr.update(value=None, interactive=False)
    job = load_job(job_id)
    if not job:
        return gr.update(value=None, interactive=False)
    if _job_access_denied(job, request):
        return gr.update(value=None, interactive=False)
    for result in (job.get("results") or {}).values():
        path = result.get("download_path")
        if path:
            return gr.update(value=path, interactive=True)
    return gr.update(value=None, interactive=False)


def register_account(username: str, password: str) -> str:
    try:
        user = register_user(username or "", password or "")
    except ValueError as exc:
        return f"Registration failed: {exc}"
    return (
        f"Registered @{user.username}. Log in with these credentials "
        "(Gradio login / API POST /api/auth/login)."
    )


def recover_session(tab_id: str):
    """On page load, reattach to an in-flight worker or poll a running manifest."""
    runtime, tid = resolve_runtime(tab_id)
    tracker = runtime["progress"]
    worker = runtime.get("worker")
    no_dl = gr.update(value=None, interactive=False)

    if worker and worker.is_alive():
        yield from _stream_worker_progress(worker, tracker, no_dl, _progress_poll_interval())
        return

    yield from _recover_manifest_or_idle(tid, tracker, no_dl)


def _apply_short_clip_preset(enabled):
    if not enabled:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return True, 0.42, 0.10, 0.60, 6


def _on_engine_change(new_engine: str, tab_id: str):
    """Swap ASR weights only when the user picks a different engine in the UI."""
    runtime, _ = resolve_runtime(tab_id)
    if is_job_running(runtime):
        return gr.update()
    prev = runtime.get("selected_asr_engine") or default_asr_engines()[0]
    if new_engine == prev:
        runtime["selected_asr_engine"] = new_engine
        return gr.update()
    try:
        switch_asr_engine(new_engine, language="Thai")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("ASR engine switch failed: %s", exc)
        return gr.update()
    runtime["selected_asr_engine"] = new_engine
    logger.info("User switched ASR engine: %s -> %s", prev, new_engine)
    return gr.update()


def _on_media_upload(path, tab_id):
    """Update preview; cancel only this tab's in-flight job when the file changes."""
    runtime, _ = resolve_runtime(tab_id)
    path_changed = bool(path) and path != runtime.get("last_upload_path")
    runtime["last_upload_path"] = path
    if path_changed:
        if runtime["progress"].snapshot().get("active"):
            runtime["cancel_event"].set()
            runtime["progress"].reset()
        if active_job_count() == 0:
            clear_prejob_caches()
    if not path:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "No file selected.",
        )
    too_large, _ = _media_too_large_for_browser(path)
    info = _format_media_info(path)
    if too_large:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            info,
        )
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return gr.update(value=path, visible=True), gr.update(value=None, visible=False), info
    return gr.update(value=None, visible=False), gr.update(value=path, visible=True), info


def _apply_ready_state():
    """Update load status and input interactivity only when values change."""
    global _last_load_status_text, _last_ready_state
    ready = _models_ready.is_set()
    current_text = _get_load_status()

    if _last_load_status_text is None or current_text != _last_load_status_text:
        load_update = current_text
        _last_load_status_text = current_text
    else:
        load_update = gr.update()

    if _last_ready_state is None or ready != _last_ready_state:
        media_update = gr.update(interactive=ready)
        btn_update = gr.update(interactive=ready)
        _last_ready_state = ready
    else:
        media_update = gr.update()
        btn_update = gr.update()

    return load_update, media_update, btn_update


def build_ui() -> gr.Blocks:
    """Build and wire the Gradio application UI."""
    hw_md = hardware_summary()
    models_ready = _models_ready.is_set()

    with gr.Blocks(title="Local Transcript App", theme=_SoftTheme(), css=APP_CSS) as demo:
        gr.Markdown("# Local Transcript App")
        gr.Markdown("Upload audio or video, then transcribe locally with open-source models.")

        load_status = gr.Markdown(_get_load_status())
        tab_instance_id = gr.Textbox(
            value="",
            visible=False,
            elem_id="tab-instance-id",
            label="",
            interactive=True,
        )
        gr.HTML(TAB_INSTANCE_SCRIPT)

        media_input = gr.File(
            label="Audio or Video File",
            file_types=["audio", "video"],
            type="filepath",
            interactive=models_ready,
            elem_id="media-input",
        )

        with gr.Row(elem_classes=["config-landscape"]):
            with gr.Column(scale=1, min_width=160):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Thai",
                    label="Language",
                )
            with gr.Column(scale=2, min_width=260):
                engine_selector = gr.Radio(
                    choices=UI_ENGINE_CHOICES,
                    value=default_asr_engines()[0],
                    label="Local ASR Engine",
                    info=(
                        "Auto picks the best engine for the selected language "
                        "(Typhoon for Thai/English quality; Pathumma for Thai when "
                        "ASR_AUTO_POLICY=fast)."
                    ),
                )
            with gr.Column(scale=1, min_width=180):
                enhance = gr.Checkbox(
                    label="Audio Enhancement",
                    value=_env_bool("AUDIO_ENHANCE_DEFAULT", False),
                    elem_id="enhance-checkbox",
                    info="Recommended for diarization — denoise and normalize loudness.",
                )
                diarization = gr.Checkbox(
                    label="Speaker Diarization",
                    value=False,
                    elem_id="diarization-checkbox",
                )
            with gr.Column(scale=1, min_width=180, visible=False) as speakers_row:
                max_speakers = gr.Slider(1, 10, step=1, value=6, label="Max Speakers")
            with gr.Column(scale=1, min_width=180):
                transcribe_btn = gr.Button(
                    "Transcribe", variant="primary", interactive=models_ready, elem_id="transcribe-btn",
                )
                cancel_btn = gr.Button("Cancel & Reset", variant="stop", interactive=True)

        # Diarization advanced config — shown when Speaker Diarization is enabled.
        with gr.Group(visible=False) as diarize_config_group:
            with gr.Accordion("Advanced Diarization Settings", open=False):
                gr.Markdown(
                    "Short clips (&lt; 90 s) with 2–3 speakers use **automatic adaptive tuning** "
                    "when overrides are off. Enable the preset below only if you still get too "
                    "few speakers after trying **Audio Enhancement** and raising **Max Speakers**."
                )
                diar_short_clip_preset = gr.Checkbox(
                    value=False,
                    label="Short clip / multi-speaker preset",
                    info="Fills the sliders below for aggressive multi-speaker detection and enables overrides.",
                )
                diar_override_defaults = gr.Checkbox(  # noqa: F841  (wired via inputs= below)
                    value=False,
                    label="Override model-tuned defaults with the sliders below",
                    info="Unchecked (recommended) = adaptive short-clip tuning + community-1 defaults.",
                )
                with gr.Row():
                    diar_seg_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_SEGMENTATION_THRESHOLD", "0.42")),
                        label="Segmentation Threshold",
                        info="Lower = catches quieter / shorter speaker turns",
                    )
                    diar_min_off = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.10")),
                        label="Min Silence Gap (s)",
                        info="Min silence before splitting a turn",
                    )
                with gr.Row():
                    diar_clust_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_CLUSTERING_THRESHOLD", "0.60")),
                        label="Clustering Threshold",
                        info="Lower = more speakers kept separate",
                    )
                    diar_clust_min_size = gr.Slider(
                        minimum=1, maximum=30, step=1,
                        value=int(os.getenv("DIARIZATION_MIN_CLUSTER_SIZE", "6")),
                        label="Min Cluster Size",
                        info="Min segments to form a speaker cluster",
                    )

        diar_short_clip_preset.change(  # pylint: disable=no-member
            fn=_apply_short_clip_preset,
            inputs=[diar_short_clip_preset],
            outputs=[
                diar_override_defaults,
                diar_seg_threshold,
                diar_min_off,
                diar_clust_threshold,
                diar_clust_min_size,
            ],
        )

        diarization.change(  # pylint: disable=no-member
            fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)),
            inputs=[diarization],
            outputs=[speakers_row, diarize_config_group],
        )

        engine_selector.change(  # pylint: disable=no-member
            fn=_on_engine_change,
            inputs=[engine_selector, tab_instance_id],
            outputs=[load_status],
        )

        with gr.Accordion("Media preview (short files only)", open=False):
            media_info = gr.Markdown("No file selected.")
            with gr.Row():
                original_video = gr.Video(
                    label="Video",
                    interactive=False,
                    visible=False,
                    elem_classes=["tall-video-preview"],
                )
                original_audio_preview = gr.Audio(
                    label="Audio",
                    interactive=False,
                    visible=False,
                    type="filepath",
                    elem_classes=["tall-audio-preview"],
                )

        media_input.change(  # pylint: disable=no-member
            fn=_on_media_upload,
            inputs=[media_input, tab_instance_id],
            outputs=[original_video, original_audio_preview, media_info],
        )

        live_status = gr.HTML(
            value=STATUS_IDLE,
            elem_id="live-status",
            label="Status",
        )
        job_progress = gr.HTML(
            value=PROGRESS_IDLE,
            elem_id="job-progress",
            label="Progress",
        )

        with gr.TabItem("Output"):
            output_text = gr.Textbox(
                label="Transcript",
                lines=20,
                max_lines=200,
                interactive=False,
                elem_id="output-transcript",
            )
            with gr.Row():
                output_time = gr.Textbox(
                    label=LABEL_ELAPSED,
                    interactive=False,
                    max_lines=1,
                    scale=2,
                    elem_id="output-elapsed",
                )
                output_name = gr.Textbox(
                    label="Output name",
                    interactive=True,
                    scale=2,
                    placeholder="Download filename (without .txt)",
                )
                output_dl = gr.DownloadButton(
                    label=LABEL_DOWNLOAD,
                    value=None,
                    scale=1,
                    interactive=False,
                )

        with gr.Accordion("Job Info", open=False):
            job_info = gr.Textbox(label="", lines=2, interactive=False, elem_id="job-info", show_label=False)

        with gr.Accordion("Invite another user", open=False):
            gr.Markdown(
                "Create an extra account (or send others to the public page "
                "**[/register](/register)** before login). Same store as "
                "`POST /api/auth/register`."
            )
            reg_user = gr.Textbox(label="New username", max_lines=1)
            reg_pass = gr.Textbox(label="Password", type="password", max_lines=1)
            reg_btn = gr.Button("Create account")
            reg_status = gr.Markdown("")
            reg_btn.click(  # pylint: disable=no-member
                fn=register_account,
                inputs=[reg_user, reg_pass],
                outputs=[reg_status],
            )

        gr.Markdown(
            "### Previous transcripts\n"
            "Filtered by your logged-in account (fallback: client IP when "
            "`UI_HISTORY_PER_CLIENT_IP=true`). Jobs keep running after you close "
            "the page — use this list or `GET /api/jobs/{id}`. Cancel stops your "
            "job and frees GPU cache for the next queued user."
        )
        with gr.Row():
            history_dropdown = gr.Dropdown(
                label="Past jobs",
                choices=[],
                value=None,
                interactive=True,
                scale=4,
            )
            history_refresh_btn = gr.Button("Refresh list", scale=1)
        with gr.Row():
            history_load_btn = gr.Button("Load into editor", variant="secondary")
            history_download_btn = gr.DownloadButton("Download selected", value=None, interactive=False)

        gr.Markdown(hw_md)

        transcribe_outputs = [
            output_text, output_time, output_name, output_dl,
            job_info, live_status, job_progress, transcribe_btn,
            history_dropdown, history_download_btn,
        ]

        transcribe_btn.click(  # pylint: disable=no-member
            fn=transcribe,
            inputs=[
                media_input,
                engine_selector,
                language,
                diarization,
                max_speakers,
                enhance,
                diar_short_clip_preset,
                diar_override_defaults,
                diar_seg_threshold,
                diar_min_off,
                diar_clust_threshold,
                diar_clust_min_size,
                output_name,
                tab_instance_id,
            ],
            outputs=transcribe_outputs,
            concurrency_limit=_gradio_transcribe_concurrency(),
        )

        cancel_btn.click(  # pylint: disable=no-member
            fn=_reset_ui_outputs,
            inputs=[tab_instance_id],
            outputs=transcribe_outputs,
        )

        history_refresh_btn.click(  # pylint: disable=no-member
            fn=load_history,
            outputs=[history_dropdown],
        )
        history_load_btn.click(  # pylint: disable=no-member
            fn=load_selected_job,
            inputs=[history_dropdown],
            outputs=transcribe_outputs,
        )
        history_download_btn.click(  # pylint: disable=no-member
            fn=download_selected_job,
            inputs=[history_dropdown],
            outputs=[history_download_btn],
        )

        media_input.change(  # pylint: disable=no-member
            fn=_default_output_names,
            inputs=[media_input],
            outputs=[output_name],
        )

        demo.load(  # pylint: disable=no-member
            fn=_apply_ready_state,
            outputs=[load_status, media_input, transcribe_btn],
        )
        demo.load(  # pylint: disable=no-member
            fn=init_tab_instance_id,
            inputs=[tab_instance_id],
            outputs=[tab_instance_id],
        )
        demo.load(  # pylint: disable=no-member
            fn=recover_session,
            inputs=[tab_instance_id],
            outputs=transcribe_outputs,
        )
        demo.queue(
            default_concurrency_limit=_gradio_transcribe_concurrency() + 4,
            max_size=_gradio_queue_max_size(),
        )

    return demo


def _mount_custom_routes(app) -> None:
    """Attach REST + register + progress routes to a freshly built Gradio app."""
    if getattr(app, "_lta_routes_mounted", False):
        return
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    def progress_api(request: Request):
        job_id = (request.query_params.get("job_id") or "").strip()
        if job_id:
            tracker = get_api_job_progress(job_id)
            if tracker is not None:
                return JSONResponse(tracker.snapshot())
            job = load_job(job_id)
            if job and isinstance(job.get("progress"), dict):
                return JSONResponse(job["progress"])
            return JSONResponse({"phase": "unknown", "active": False, "job_id": job_id})
        tab_id = request.query_params.get("tab_id")
        if tab_id:
            runtime, _ = resolve_runtime(tab_id)
            return JSONResponse(runtime["progress"].snapshot())
        return JSONResponse(get_job_progress().snapshot())

    # Gradio launch() rebuilds the FastAPI app — mount after create_app.
    for route in reversed(build_api_routes()):
        app.routes.insert(0, route)
    app.routes.insert(0, Route("/job/progress", progress_api, methods=["GET"]))
    app._lta_routes_mounted = True  # type: ignore[attr-defined]


def _patch_gradio_create_app_for_custom_routes() -> None:
    """Ensure our routes survive Gradio's App.create_app() inside launch()."""
    from gradio.routes import App

    if getattr(App, "_lta_custom_routes_patched", False):
        return
    original = App.create_app

    def create_app_with_routes(*args, **kwargs):
        app = original(*args, **kwargs)
        _mount_custom_routes(app)
        logger.info("Mounted /register, /api/*, and /job/progress on Gradio app.")
        return app

    App.create_app = staticmethod(create_app_with_routes)  # type: ignore[method-assign]
    App._lta_custom_routes_patched = True  # type: ignore[attr-defined]


def main() -> None:
    """Start the Gradio server (CLI, launcher subprocess, or PyInstaller --app-server)."""
    from backend.asr_quality import apply_quality_profile

    ensure_app_dirs()
    init_user_db()
    apply_cpu_thread_limits()
    apply_quality_profile()
    hardware = detect_hardware()
    logger.info("Selected backend: %s / %s", hardware["backend"], hardware["selected_device"])
    public_base = os.getenv("APP_PUBLIC_BASE_URL", "").strip()
    if public_base:
        logger.info("Public base URL hint: %s", public_base)
    _preload_models()
    _patch_gradio_create_app_for_custom_routes()
    application = build_ui()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7896"))
    launch_kwargs: dict = {
        "server_name": server_name,
        "server_port": server_port,
        "max_threads": 8,
        "show_error": True,
        "share": _env_bool("GRADIO_SHARE", False),
    }
    root_path = os.getenv("GRADIO_ROOT_PATH", "").strip()
    if root_path:
        launch_kwargs["root_path"] = root_path
        logger.info("Gradio root_path=%s (reverse proxy)", root_path)
    max_file = os.getenv("GRADIO_MAX_FILE_SIZE", "").strip()
    if max_file:
        launch_kwargs["max_file_size"] = max_file
    auth_user = os.getenv("GRADIO_AUTH_USER", "").strip()
    auth_password = os.getenv("GRADIO_AUTH_PASSWORD", "").strip()
    # Prefer the shared SQLite user store (seed + register) for UI + API.
    if _env_bool("APP_AUTH_ENABLED", True):
        launch_kwargs["auth"] = gradio_auth_credentials
        launch_kwargs["auth_message"] = (
            'New user? <a href="/register" target="_self">'
            "Create an account</a> first, then sign in here."
        )
        logger.info("Gradio auth enabled via user database (APP_AUTH_ENABLED).")
    elif auth_user and auth_password:
        launch_kwargs["auth"] = (auth_user, auth_password)
        logger.info("Gradio basic auth enabled for user=%s", auth_user)
    elif auth_user or auth_password:
        logger.warning(
            "GRADIO_AUTH_USER/PASSWORD incomplete; basic auth disabled "
            "(set both, or APP_AUTH_ENABLED=true with seeded users)."
        )
    try:
        application.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc).lower():
            raise
        logger.warning("Localhost probe failed; retrying Gradio launch with share=True.")
        launch_kwargs["share"] = True
        application.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
