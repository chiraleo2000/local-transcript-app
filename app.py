"""Local Transcript App - Gradio UI for local-only ASR and diarization."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import importlib.machinery
import logging
import os
import re
import sys
import threading
import time
import types
import warnings

from dotenv import load_dotenv

from backend.paths import app_root, ensure_bundle_on_path, resolve_path

ensure_bundle_on_path()
os.chdir(app_root())

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_install_root = app_root()
load_dotenv(_install_root / ".env")
load_dotenv(_install_root / ".env.production", override=False)

_model_root = os.getenv("APP_MODEL_ROOT") or str(_install_root / "models")
if not os.path.isabs(_model_root):
    _model_root = str(resolve_path(_model_root))
_HF_HOME = os.path.join(_model_root, "hf_cache")
os.environ.setdefault("APP_MODEL_ROOT", _model_root)
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(_model_root, "torch"))
os.environ.setdefault("OV_CACHE_DIR", os.path.join(_model_root, "ov_cache"))

for _cache_dir in [
    os.environ["APP_MODEL_ROOT"],
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["TORCH_HOME"],
    os.environ["OV_CACHE_DIR"],
]:
    os.makedirs(_cache_dir, exist_ok=True)


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

from backend.pipeline import run_transcription_job
from backend.progress import get_job_progress
from backend.services.asr_local import (
    ALL_ENGINES,
    ENGINE_PATHUMMA,
    ENGINE_TYPHOON,
    LANGUAGES,
    clear_accelerator_cache,
    default_asr_engines,
    load_model,
)
from backend.services.correction_local import correct_with_local_llm
from backend.services.hardware_policy import detect_hardware, hardware_summary
from backend.services.media_pipeline import enhance_audio
from backend.storage import ensure_app_dirs, save_transcript


LABEL_ELAPSED = "Elapsed Time"
LABEL_DOWNLOAD = "Download .txt"
_CANCELLED = "(cancelled)"
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
.correction-button { margin-top: 4px; }
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
.progress-label { font-weight: 600; font-size: 14px; margin-bottom: 8px; color: #3f3f46; }
.progress-track {
    height: 14px;
    border-radius: 999px;
    background: #e4e4e7;
    overflow: hidden;
    margin-bottom: 10px;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    transition: width 0.35s ease;
    min-width: 2%;
}
.job-progress-panel.done .progress-fill {
    background: linear-gradient(90deg, #16a34a, #22c55e);
}
.progress-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 24px;
    font-size: 14px;
    color: #52525b;
    margin-bottom: 6px;
}
.progress-meta strong { color: #18181b; font-variant-numeric: tabular-nums; }
.progress-pct { margin-left: auto; font-weight: 700; color: #2563eb; }
.progress-message { font-size: 13px; color: #71717a; }
@keyframes live-pulse {
    0%, 100% { opacity: 1.0; }
    50%      { opacity: 0.65; }
}
@media (max-width: 900px) {
    .config-landscape { flex-wrap: wrap; }
}
"""

_models_ready = threading.Event()
_cancel_event = threading.Event()
_load_status = dict.fromkeys(ALL_ENGINES, "pending")
_last_load_status_text: str | None = None
_last_ready_state: bool | None = None


def _preload_models() -> None:
    """Preload the configured ASR model at startup by default."""
    preload_mode = os.getenv("ASR_PRELOAD_MODE", "eager").strip().lower()
    if preload_mode not in {"eager", "preload", "true", "1"}:
        for engine in ALL_ENGINES:
            _load_status[engine] = "available"
        _models_ready.set()
        logger.info("ASR preload skipped; models are available on demand.")
        return

    preload_engines = default_asr_engines()
    skipped = [e for e in ALL_ENGINES if e not in preload_engines]
    for engine in skipped:
        _load_status[engine] = "available"
    if skipped:
        logger.info(
            "ASR eager preload limited to %s; others available on demand.",
            ", ".join(preload_engines),
        )

    preload_failed = False

    def _load(engine: str) -> None:
        nonlocal preload_failed
        try:
            _load_status[engine] = "loading..."
            load_model(engine)
            _load_status[engine] = "ready"
            logger.info("%s loaded.", engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            preload_failed = True
            _load_status[engine] = f"FAILED: {exc}"
            logger.exception("%s load failed: %s", engine, exc)

    for engine in preload_engines:
        _load(engine)

    diarization_preload_mode = os.getenv("DIARIZATION_PRELOAD_MODE", "eager").strip().lower()
    if diarization_preload_mode in {"eager", "preload", "true", "1"}:
        try:
            from engines.diarization import load_model as load_diarization_model

            logger.info("Preloading pyannote diarization model...")
            load_diarization_model()
            logger.info("Pyannote diarization loaded.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            preload_failed = True
            logger.exception("Pyannote diarization load failed: %s", exc)

    if preload_failed:
        logger.error("Model preload failed; upload remains disabled until Docker cache/env is fixed.")
        return
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


def _progress_html(snapshot: dict | None = None) -> str:
    """Render progress bar + elapsed / remaining timer for the active job."""
    snap = snapshot if snapshot is not None else get_job_progress().snapshot()
    phase = snap.get("phase", "idle")
    pct = float(snap.get("percent", 0))
    elapsed = float(snap.get("elapsed_s", 0))
    remaining = snap.get("remaining_s")
    remaining_txt = "\u2014" if remaining is None else f"{float(remaining):.0f}s"
    active_cls = "active" if snap.get("active") else phase
    return (
        f'<div class="job-progress-panel {active_cls}" data-phase="{phase}" '
        f'data-percent="{pct:.1f}">'
        f'<div class="progress-label">Processing Progress</div>'
        f'<div class="progress-track">'
        f'<div class="progress-fill" style="width:{pct:.1f}%" data-progress="{pct:.1f}">'
        f"</div></div>"
        f'<div class="progress-meta">'
        f'<span class="progress-elapsed">Time elapsed: <strong>{elapsed:.1f}s</strong></span>'
        f'<span class="progress-remaining">Est. remaining: <strong>{remaining_txt}</strong></span>'
        f'<span class="progress-pct">{pct:.0f}%</span>'
        f"</div>"
        f'<div class="progress-message">{snap.get("message", "")}</div>'
        f"</div>"
    )


PROGRESS_IDLE = _progress_html({
    "phase": "idle",
    "percent": 0,
    "active": False,
    "elapsed_s": 0,
    "remaining_s": None,
    "message": "Waiting for a transcription job.",
    "job_id": "",
    "audio_duration_s": 0,
})


def _empty_outputs(message: str, status_state: str = "idle",
                   status_message: str = "Idle.") -> tuple:
    no_download = gr.update(value=None, interactive=False)
    no_correction = gr.update(interactive=False)
    progress = _progress_html({
        "phase": status_state if status_state in {"error"} else "idle",
        "percent": 0 if status_state != "error" else get_job_progress().snapshot()["percent"],
        "active": False,
        "elapsed_s": get_job_progress().snapshot()["elapsed_s"],
        "remaining_s": None,
        "message": status_message,
        "job_id": "",
        "audio_duration_s": 0,
    })
    return (
        message, "", no_download, no_correction,
        message, "", no_download, no_correction,
        "",
        _status_html(status_state, status_message),
        progress,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_outputs(job_result: dict, selected_engines: list[str]) -> tuple:
    outputs = []
    for engine in ALL_ENGINES:
        result = job_result["results"].get(engine)
        if result:
            path = result.get("download_path")
            text = result.get("text", "")
            can_correct = bool(text and not text.startswith(("(", "ERROR")))
            outputs.extend([
                text,
                f"{result.get('elapsed', 0.0):.2f}s",
                gr.update(value=path, interactive=path is not None),
                gr.update(interactive=can_correct),
            ])
        elif engine in selected_engines:
            outputs.extend([
                "(failed)",
                "",
                gr.update(value=None, interactive=False),
                gr.update(interactive=False),
            ])
        else:
            outputs.extend([
                "",
                "(not selected)",
                gr.update(value=None, interactive=False),
                gr.update(interactive=False),
            ])
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
    outputs.append(_status_html("done", perf_text))
    outputs.append(_progress_html(get_job_progress().snapshot()))
    return tuple(outputs)


def _reset_ui_outputs() -> tuple:
    """Signal cancellation and return a blank UI state for all transcript outputs."""
    _cancel_event.set()
    clear_accelerator_cache()  # immediately release GPU VRAM held by in-flight inference
    get_job_progress().reset()
    no_dl = gr.update(value=None, interactive=False)
    no_btn = gr.update(interactive=False)
    return (
        _CANCELLED, "", no_dl, no_btn,
        _CANCELLED, "", no_dl, no_btn,
        "Cancelled by user.",
        _status_html("error", "Cancelled by user."),
        PROGRESS_IDLE,
    )


def transcribe(
    media_path,
    selected_engines,
    language,
    diarization,
    max_speakers,
    enhance,
    diar_override_defaults,
    diar_seg_threshold,
    diar_min_off,
    diar_clust_threshold,
    diar_clust_min_size,
    _progress=gr.Progress(track_tqdm=True),
):
    """Gradio callback — generator yielding live progress bar + timer updates."""
    _cancel_event.clear()
    tracker = get_job_progress()
    if not media_path:
        tracker.reset()
        yield _empty_outputs("(no media provided)", "error", "No media uploaded.")
        return
    if not _models_ready.is_set():
        yield _empty_outputs(
            "Models are still loading, please wait...",
            "running",
            "Models are still loading\u2026",
        )
        return

    if isinstance(selected_engines, str):
        selected = [selected_engines]
    else:
        selected = list(selected_engines or default_asr_engines())[:1]

    diarize_kwargs: dict | None = None
    if diarization and diar_override_defaults:
        diarize_kwargs = {
            "seg_threshold":        float(diar_seg_threshold),
            "seg_min_duration_off": float(diar_min_off),
            "clust_threshold":      float(diar_clust_threshold),
            "clust_min_size":       int(diar_clust_min_size),
        }

    no_dl = gr.update(value=None, interactive=False)
    no_btn = gr.update(interactive=False)
    engines_label = selected[0]
    holder: dict = {}
    error_holder: dict = {}

    def _run_job() -> None:
        try:
            holder["result"] = run_transcription_job(
                media_path=media_path,
                selected_engines=selected,
                language=language,
                diarization=diarization,
                max_speakers=int(max_speakers),
                enhance=enhance,
                local_correction=False,
                diarize_kwargs=diarize_kwargs,
                cancel_event=_cancel_event,
                progress=tracker,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_holder["error"] = exc

    tracker.reset()
    tracker.start()
    worker = threading.Thread(target=_run_job, daemon=True)
    worker.start()

    def _running_outputs(snap: dict) -> tuple:
        return (
            "", "", no_dl, no_btn,
            "", "", no_dl, no_btn,
            "Transcription in progress\u2026",
            _status_html("running", snap.get("message", f"Transcribing with {engines_label}\u2026")),
            _progress_html(snap),
        )

    def _progress_signature(snap: dict) -> tuple:
        return (
            snap.get("phase"),
            round(float(snap.get("percent", 0)), 1),
            int(float(snap.get("elapsed_s", 0))),
            snap.get("message"),
        )

    last_sig = None
    snap = tracker.snapshot()
    last_sig = _progress_signature(snap)
    yield _running_outputs(snap)

    while worker.is_alive():
        if _cancel_event.is_set():
            tracker.fail("Cancelled.")
            yield _empty_outputs(_CANCELLED, "error", "Cancelled.")
            return
        snap = tracker.snapshot()
        sig = _progress_signature(snap)
        if sig != last_sig:
            last_sig = sig
            yield _running_outputs(snap)
        time.sleep(0.25)

    worker.join()

    if error_holder.get("error"):
        exc = error_holder["error"]
        if isinstance(exc, RuntimeError) and "cancelled" in str(exc).lower():
            tracker.fail("Cancelled.")
            yield _empty_outputs(_CANCELLED, "error", "Cancelled.")
            return
        logger.exception("Transcription job failed: %s", exc)
        tracker.fail(str(exc))
        yield _empty_outputs(f"ERROR: {exc}", "error", f"Error: {exc}")
        return

    yield _build_outputs(holder["result"], selected)


def _job_id_from_info(job_info: str) -> str | None:
    match = re.search(r"^Job ID:\s*(\S+)", job_info or "", flags=re.MULTILINE)
    return match.group(1) if match else None


def correct_transcript(engine_name: str, text: str, elapsed: str, job_info: str) -> tuple:
    """Run optional local correction only after a transcript exists."""
    if not text or text.startswith(("(", "ERROR")):
        return text, elapsed, gr.update(), job_info

    corrected, correction_elapsed, note = correct_with_local_llm(text)
    job_id = _job_id_from_info(job_info) or "manual_correction"
    transcript_path = save_transcript(job_id, f"{engine_name}_corrected", corrected)
    elapsed_note = (
        f"{elapsed} + LLM {correction_elapsed:.2f}s"
        if elapsed
        else f"LLM {correction_elapsed:.2f}s"
    )
    info = f"{job_info}\n{engine_name}: {note}".strip()
    return (
        corrected,
        elapsed_note,
        gr.update(value=transcript_path, interactive=transcript_path is not None),
        info,
    )


def build_ui() -> gr.Blocks:
    """Build and wire the Gradio application UI."""
    hw_md = hardware_summary()
    models_ready = _models_ready.is_set()

    with gr.Blocks(title="Local Transcript App", theme=_SoftTheme(), css=APP_CSS) as demo:
        gr.Markdown("# Local Transcript App")
        gr.Markdown("Upload audio or video, then transcribe locally with open-source models.")

        load_status = gr.Markdown(_get_load_status())

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
                    choices=ALL_ENGINES,
                    value=default_asr_engines()[0],
                    label="Local ASR Engine",
                )
            with gr.Column(scale=1, min_width=180):
                enhance = gr.Checkbox(
                    label="Audio Enhancement",
                    value=_env_bool("AUDIO_ENHANCE_DEFAULT", True),
                    elem_id="enhance-checkbox",
                )
                diarization = gr.Checkbox(
                    label="Speaker Diarization",
                    value=False,
                    elem_id="diarization-checkbox",
                )
            with gr.Column(scale=1, min_width=180, visible=False) as speakers_row:
                max_speakers = gr.Slider(1, 10, step=1, value=3, label="Max Speakers")
            with gr.Column(scale=1, min_width=180):
                transcribe_btn = gr.Button(
                    "Transcribe", variant="primary", interactive=models_ready, elem_id="transcribe-btn",
                )
                cancel_btn = gr.Button("Cancel & Reset", variant="stop", interactive=True)

        # Diarization advanced config — shown when Speaker Diarization is enabled.
        with gr.Group(visible=False) as diarize_config_group:
            with gr.Accordion("Advanced Diarization Settings", open=False):
                gr.Markdown(
                    "Tune pyannote speaker detection. **Leave at defaults for best accuracy** — "
                    "community-1 ships with training-tuned hyperparameters. "
                    "Only adjust if you have characterised your specific audio domain."
                )
                diar_override_defaults = gr.Checkbox(  # noqa: F841  (wired via inputs= below)
                    value=False,
                    label="Override model-tuned defaults with the sliders below",
                    info="Unchecked (recommended) = use community-1's own tuned hyperparameters.",
                )
                with gr.Row():
                    diar_seg_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_SEGMENTATION_THRESHOLD", "0.5")),
                        label="Segmentation Threshold",
                        info="Lower = catches quieter / shorter speaker turns (community-1 default ~0.5)",
                    )
                    diar_min_off = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.0")),
                        label="Min Silence Gap (s)",
                        info="Min silence before splitting a turn (default 0.0)",
                    )
                with gr.Row():
                    diar_clust_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_CLUSTERING_THRESHOLD", "0.7")),
                        label="Clustering Threshold",
                        info="Lower = more speakers kept separate (community-1 default ~0.7)",
                    )
                    diar_clust_min_size = gr.Slider(
                        minimum=1, maximum=30, step=1,
                        value=int(os.getenv("DIARIZATION_MIN_CLUSTER_SIZE", "12")),
                        label="Min Cluster Size",
                        info="Min segments to form a speaker cluster (default 12)",
                    )

        diarization.change(  # pylint: disable=no-member
            fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)),
            inputs=[diarization],
            outputs=[speakers_row, diarize_config_group],
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Original Media Preview")
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
            with gr.Column():
                gr.Markdown("#### Enhanced Audio Preview")
                enhanced_audio = gr.Audio(
                    label="Enhanced",
                    interactive=False,
                    type="filepath",
                    elem_classes=["tall-audio-preview"],
                )

        def _route_media_preview(path):
            """Route uploaded media to the video or audio preview component."""
            if not path:
                return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                return gr.update(value=path, visible=True), gr.update(value=None, visible=False)
            return gr.update(value=None, visible=False), gr.update(value=path, visible=True)

        media_input.change(  # pylint: disable=no-member
            fn=_route_media_preview,
            inputs=[media_input],
            outputs=[original_video, original_audio_preview],
        )

        def _run_enhance(media_path, do_enhance):
            """Generate enhanced audio only when enhancement is enabled."""
            if not media_path or not do_enhance:
                return None
            return enhance_audio(media_path)

        enhance.change(  # pylint: disable=no-member
            fn=_run_enhance,
            inputs=[media_input, enhance],
            outputs=[enhanced_audio],
        )
        media_input.change(  # pylint: disable=no-member
            fn=lambda _path: None,
            inputs=[media_input],
            outputs=[enhanced_audio],
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

        with gr.Tabs():
            with gr.TabItem(ENGINE_TYPHOON):
                typhoon_text = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=200,
                    interactive=False,
                    elem_id="typhoon-transcript",
                )
                with gr.Row():
                    typhoon_time = gr.Textbox(
                        label=LABEL_ELAPSED,
                        interactive=False,
                        max_lines=1,
                        scale=3,
                        elem_id="typhoon-elapsed",
                    )
                    typhoon_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD,
                        value=None,
                        scale=1,
                        interactive=False,
                    )
                typhoon_correct = gr.Button(
                    "Run Local LLM Correction",
                    interactive=False,
                    elem_classes=["correction-button"],
                )

            with gr.TabItem(ENGINE_PATHUMMA):
                pathumma_text = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=200,
                    interactive=False,
                )
                with gr.Row():
                    pathumma_time = gr.Textbox(
                        label=LABEL_ELAPSED,
                        interactive=False,
                        max_lines=1,
                        scale=3,
                    )
                    pathumma_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD,
                        value=None,
                        scale=1,
                        interactive=False,
                    )
                pathumma_correct = gr.Button(
                    "Run Local LLM Correction",
                    interactive=False,
                    elem_classes=["correction-button"],
                )

        job_info = gr.Textbox(label="Job Info", lines=3, interactive=False, elem_id="job-info")

        gr.Markdown(hw_md)

        transcribe_event = transcribe_btn.click(  # pylint: disable=no-member
            fn=transcribe,
            inputs=[
                media_input,
                engine_selector,
                language,
                diarization,
                max_speakers,
                enhance,
                diar_override_defaults,
                diar_seg_threshold,
                diar_min_off,
                diar_clust_threshold,
                diar_clust_min_size,
            ],
            outputs=[
                typhoon_text, typhoon_time, typhoon_dl, typhoon_correct,
                pathumma_text, pathumma_time, pathumma_dl, pathumma_correct,
                job_info,
                live_status,
                job_progress,
            ],
        )

        cancel_btn.click(  # pylint: disable=no-member
            fn=_reset_ui_outputs,
            outputs=[
                typhoon_text, typhoon_time, typhoon_dl, typhoon_correct,
                pathumma_text, pathumma_time, pathumma_dl, pathumma_correct,
                job_info,
                live_status,
                job_progress,
            ],
            cancels=[transcribe_event],
        )

        typhoon_correct.click(  # pylint: disable=no-member
            fn=lambda text, elapsed, info: correct_transcript(ENGINE_TYPHOON, text, elapsed, info),
            inputs=[typhoon_text, typhoon_time, job_info],
            outputs=[typhoon_text, typhoon_time, typhoon_dl, job_info],
        )
        pathumma_correct.click(  # pylint: disable=no-member
            fn=lambda text, elapsed, info: correct_transcript(
                ENGINE_PATHUMMA,
                text,
                elapsed,
                info,
            ),
            inputs=[pathumma_text, pathumma_time, job_info],
            outputs=[pathumma_text, pathumma_time, pathumma_dl, job_info],
        )

        def apply_ready_state():
            """Only update the page when readiness or load text actually changes.

            This avoids continuous DOM replacements and layout jank by returning
            no-op updates when nothing has changed.
            """
            global _last_load_status_text, _last_ready_state
            ready = _models_ready.is_set()

            # Compute current status text only when needed.
            current_text = _get_load_status()

            # Decide whether to update the Markdown text.
            if _last_load_status_text is None or current_text != _last_load_status_text:
                load_update = current_text
                _last_load_status_text = current_text
            else:
                load_update = gr.update()

            # Decide whether to update interactive state for inputs/buttons.
            if _last_ready_state is None or ready != _last_ready_state:
                media_update = gr.update(interactive=ready)
                btn_update = gr.update(interactive=ready)
                _last_ready_state = ready
            else:
                media_update = gr.update()
                btn_update = gr.update()

            return load_update, media_update, btn_update

        demo.load(  # pylint: disable=no-member
            fn=apply_ready_state,
            outputs=[load_status, media_input, transcribe_btn],
        )
        demo.queue(default_concurrency_limit=4)

    return demo


def _register_progress_api(demo: gr.Blocks) -> None:
    """Expose GET /job/progress for frontend polling (avoid Gradio /api POST namespace)."""
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def progress_api(_request):
        return JSONResponse(get_job_progress().snapshot())

    demo.app.routes.insert(0, Route("/job/progress", progress_api, methods=["GET"]))


def main() -> None:
    """Start the Gradio server (CLI, launcher subprocess, or PyInstaller --app-server)."""
    ensure_app_dirs()
    hardware = detect_hardware()
    logger.info("Selected backend: %s / %s", hardware["backend"], hardware["selected_device"])
    _preload_models()
    application = build_ui()
    _register_progress_api(application)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7896"))
    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "max_threads": 40,
        "show_error": True,
        "share": _env_bool("GRADIO_SHARE", False),
    }
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
