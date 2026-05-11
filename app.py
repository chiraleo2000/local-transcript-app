"""Local Transcript App - Gradio UI for local-only ASR and diarization."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import importlib.machinery
import logging
import os
import re
import sys
import threading
import types
import warnings

from dotenv import load_dotenv

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_MODEL_ROOT = os.getenv("APP_MODEL_ROOT") or os.path.join(os.getcwd(), "models")
_HF_HOME = os.path.join(_MODEL_ROOT, "hf_cache")
os.environ.setdefault("APP_MODEL_ROOT", _MODEL_ROOT)
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(_MODEL_ROOT, "torch"))
os.environ.setdefault("OV_CACHE_DIR", os.path.join(_MODEL_ROOT, "ov_cache"))

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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

import gradio as gr
from gradio.themes import Soft as _SoftTheme

from backend.pipeline import run_transcription_job
from backend.services.asr_local import (
    ALL_ENGINES,
    ENGINE_THONBURIAN,
    ENGINE_TYPHOON,
    LANGUAGES,
    default_asr_engines,
    load_model,
    strict_memory_mode_active,
)
from backend.services.correction_local import correct_with_local_llm
from backend.services.hardware_policy import detect_hardware, hardware_summary
from backend.services.media_pipeline import enhance_audio
from backend.storage import ensure_app_dirs, save_transcript


LABEL_ELAPSED = "Elapsed Time"
LABEL_DOWNLOAD = "Download .txt"
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
@media (max-width: 900px) {
    .config-landscape { flex-wrap: wrap; }
}
"""

_models_ready = threading.Event()
_load_status = dict.fromkeys(ALL_ENGINES, "pending")


def _preload_models() -> None:
    """Preload local ASR models only when explicitly requested.

    The default lazy mode keeps 8 GB GPUs from holding both large Whisper models
    in VRAM before a job starts. Set ASR_PRELOAD_MODE=eager to restore startup
    preloading when the machine has enough memory.
    """
    preload_mode = os.getenv("ASR_PRELOAD_MODE", "lazy").strip().lower()
    if preload_mode not in {"eager", "preload", "true", "1"}:
        for engine in ALL_ENGINES:
            _load_status[engine] = "available"
        _models_ready.set()
        logger.info("ASR preload skipped; models are available on demand.")
        return

    preload_engines = ALL_ENGINES

    def _load(engine: str) -> None:
        try:
            _load_status[engine] = "loading..."
            load_model(engine)
            _load_status[engine] = "ready"
            logger.info("%s loaded.", engine)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _load_status[engine] = f"FAILED: {exc}"
            logger.error("%s load failed: %s", engine, exc, exc_info=True)

    def _worker() -> None:
        for engine in preload_engines:
            _load(engine)
        _models_ready.set()
        logger.info("Model preload finished.")

    import threading
    threading.Thread(target=_worker, daemon=True).start()


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


def _empty_outputs(message: str) -> tuple:
    no_download = gr.update(value=None, interactive=False)
    no_correction = gr.update(interactive=False)
    return (
        message, "", no_download, no_correction,
        message, "", no_download, no_correction,
        "",
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
    return tuple(outputs)


def transcribe(
    media_path,
    selected_engines,
    language,
    diarization,
    max_speakers,
    enhance,
    diar_seg_threshold,
    diar_min_off,
    diar_clust_threshold,
    diar_clust_min_size,
):
    """Gradio callback: run local backend pipeline."""
    if not media_path:
        return _empty_outputs("(no media provided)")
    if not _models_ready.is_set():
        return _empty_outputs("Models are still loading, please wait...")
    try:
        diarize_kwargs: dict | None = None
        if diarization:
            diarize_kwargs = {
                "seg_threshold":       float(diar_seg_threshold),
                "seg_min_duration_off": float(diar_min_off),
                "clust_threshold":     float(diar_clust_threshold),
                "clust_min_size":      int(diar_clust_min_size),
            }
        result = run_transcription_job(
            media_path=media_path,
            selected_engines=selected_engines or default_asr_engines(),
            language=language,
            diarization=diarization,
            min_speakers=1,
            max_speakers=int(max_speakers),
            enhance=enhance,
            local_correction=False,
            diarize_kwargs=diarize_kwargs,
        )
        return _build_outputs(result, selected_engines or default_asr_engines())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Transcription job failed: %s", exc, exc_info=True)
        return _empty_outputs(f"ERROR: {exc}")


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
    hw_md = hardware_summary()

    with gr.Blocks(title="Local Transcript App") as demo:
        gr.Markdown("# Local Transcript App")
        gr.Markdown("Upload audio or video, then transcribe locally with open-source models.")

        load_status = gr.Markdown(_get_load_status())

        media_input = gr.File(
            label="Audio or Video File",
            file_types=["audio", "video"],
            type="filepath",
            interactive=False,
        )

        with gr.Row(elem_classes=["config-landscape"]):
            with gr.Column(scale=1, min_width=160):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Thai",
                    label="Language",
                )
            with gr.Column(scale=2, min_width=260):
                engine_selector = gr.CheckboxGroup(
                    choices=ALL_ENGINES,
                    value=default_asr_engines(),
                    label="Local ASR Engines",
                )
            with gr.Column(scale=1, min_width=180):
                enhance = gr.Checkbox(
                    label="Audio Enhancement",
                    value=_env_bool("AUDIO_ENHANCE_DEFAULT", True),
                )
                diarization = gr.Checkbox(label="Speaker Diarization", value=False)
            with gr.Column(scale=1, min_width=180, visible=False) as speakers_row:
                max_speakers = gr.Slider(1, 10, step=1, value=3, label="Max Speakers")
            with gr.Column(scale=1, min_width=180):
                transcribe_btn = gr.Button("Transcribe", variant="primary", interactive=False)

        # Diarization advanced config — shown when Speaker Diarization is enabled.
        with gr.Group(visible=False) as diarize_config_group:
            with gr.Accordion("Advanced Diarization Settings", open=False):
                gr.Markdown(
                    "Tune pyannote speaker detection accuracy. "
                    "Defaults are loaded from `.env`. Changes apply only to the current run."
                )
                with gr.Row():
                    diar_seg_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_SEGMENTATION_THRESHOLD", "0.42")),
                        label="Segmentation Threshold",
                        info="Lower = catches quieter / shorter speaker turns (default 0.42)",
                    )
                    diar_min_off = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=float(os.getenv("DIARIZATION_MIN_DURATION_OFF", "0.10")),
                        label="Min Silence Gap (s)",
                        info="Min silence before splitting a turn (default 0.10)",
                    )
                with gr.Row():
                    diar_clust_threshold = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.01,
                        value=float(os.getenv("DIARIZATION_CLUSTERING_THRESHOLD", "0.60")),
                        label="Clustering Threshold",
                        info="Lower = more speakers kept separate (default 0.60)",
                    )
                    diar_clust_min_size = gr.Slider(
                        minimum=1, maximum=30, step=1,
                        value=int(os.getenv("DIARIZATION_MIN_CLUSTER_SIZE", "6")),
                        label="Min Cluster Size",
                        info="Min segments to form a speaker cluster (default 6)",
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

        with gr.Tabs():
            with gr.TabItem(ENGINE_TYPHOON):
                typhoon_text = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=200,
                    buttons=["copy"],
                    interactive=False,
                )
                with gr.Row():
                    typhoon_time = gr.Textbox(
                        label=LABEL_ELAPSED,
                        interactive=False,
                        max_lines=1,
                        scale=3,
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

            with gr.TabItem(ENGINE_THONBURIAN):
                thonburian_text = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=200,
                    buttons=["copy"],
                    interactive=False,
                )
                with gr.Row():
                    thonburian_time = gr.Textbox(
                        label=LABEL_ELAPSED,
                        interactive=False,
                        max_lines=1,
                        scale=3,
                    )
                    thonburian_dl = gr.DownloadButton(
                        label=LABEL_DOWNLOAD,
                        value=None,
                        scale=1,
                        interactive=False,
                    )
                thonburian_correct = gr.Button(
                    "Run Local LLM Correction",
                    interactive=False,
                    elem_classes=["correction-button"],
                )

        job_info = gr.Textbox(label="Job Info", lines=3, interactive=False)

        gr.Markdown(hw_md)

        transcribe_btn.click(  # pylint: disable=no-member
            fn=transcribe,
            inputs=[
                media_input,
                engine_selector,
                language,
                diarization,
                max_speakers,
                enhance,
                diar_seg_threshold,
                diar_min_off,
                diar_clust_threshold,
                diar_clust_min_size,
            ],
            outputs=[
                typhoon_text, typhoon_time, typhoon_dl, typhoon_correct,
                thonburian_text, thonburian_time, thonburian_dl, thonburian_correct,
                job_info,
            ],
        )

        typhoon_correct.click(  # pylint: disable=no-member
            fn=lambda text, elapsed, info: correct_transcript(ENGINE_TYPHOON, text, elapsed, info),
            inputs=[typhoon_text, typhoon_time, job_info],
            outputs=[typhoon_text, typhoon_time, typhoon_dl, job_info],
        )
        thonburian_correct.click(  # pylint: disable=no-member
            fn=lambda text, elapsed, info: correct_transcript(
                ENGINE_THONBURIAN,
                text,
                elapsed,
                info,
            ),
            inputs=[thonburian_text, thonburian_time, job_info],
            outputs=[thonburian_text, thonburian_time, thonburian_dl, job_info],
        )

        timer = gr.Timer(value=2)

        def check_ready():
            """Refresh model readiness and enable upload controls when ready."""
            ready = _models_ready.is_set()
            return _get_load_status(), gr.update(interactive=ready), gr.update(interactive=ready)

        timer.tick(  # pylint: disable=no-member
            fn=check_ready,
            outputs=[load_status, media_input, transcribe_btn],
        )
        demo.queue()

    return demo


if __name__ == "__main__":
    ensure_app_dirs()
    hardware = detect_hardware()
    logger.info("Selected backend: %s / %s", hardware["backend"], hardware["selected_device"])
    _preload_models()
    application = build_ui()
    application.launch(
        server_name="0.0.0.0",
        server_port=7896,
        max_threads=40,
        theme=_SoftTheme(),
        css=APP_CSS,
    )
