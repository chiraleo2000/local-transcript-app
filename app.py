"""Local Transcript App - Gradio UI for local-only ASR and diarization."""

from __future__ import annotations

import importlib.machinery
import logging
import os
import sys
import threading
import types
import warnings

from dotenv import load_dotenv


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
    load_model,
)
from backend.services.hardware_policy import detect_hardware, hardware_summary
from backend.services.media_pipeline import enhance_audio
from backend.storage import ensure_app_dirs


LABEL_ELAPSED = "Elapsed Time"
LABEL_DOWNLOAD = "Download .txt"

_models_ready = threading.Event()
_load_status = dict.fromkeys(ALL_ENGINES, "pending")


def _preload_models() -> None:
    """Preload local ASR models sequentially to avoid sharded checkpoint races."""
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
        for engine in ALL_ENGINES:
            _load(engine)
        _models_ready.set()
        logger.info("Model preload finished.")

    threading.Thread(target=_worker, daemon=True).start()


def _get_load_status() -> str:
    if _models_ready.is_set() and all(status == "ready" for status in _load_status.values()):
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
    return (
        message, "", no_download,
        message, "", no_download,
        "",
    )


def _build_outputs(job_result: dict, selected_engines: list[str]) -> tuple:
    outputs = []
    for engine in ALL_ENGINES:
        result = job_result["results"].get(engine)
        if result:
            path = result.get("download_path")
            outputs.extend([
                result.get("text", ""),
                f"{result.get('elapsed', 0.0):.2f}s",
                gr.update(value=path, interactive=path is not None),
            ])
        elif engine in selected_engines:
            outputs.extend(["(failed)", "", gr.update(value=None, interactive=False)])
        else:
            outputs.extend(["", "(not selected)", gr.update(value=None, interactive=False)])
    manifest = job_result.get("manifest_path", "")
    outputs.append(f"Job ID: {job_result.get('job_id', '')}\nManifest: {manifest}")
    return tuple(outputs)


def transcribe(
    media_path,
    selected_engines,
    language,
    diarization,
    max_speakers,
    enhance,
    local_correction,
):
    """Gradio callback: run local backend pipeline."""
    if not media_path:
        return _empty_outputs("(no media provided)")
    if not _models_ready.is_set():
        return _empty_outputs("Models are still loading, please wait...")
    try:
        result = run_transcription_job(
            media_path=media_path,
            selected_engines=selected_engines or list(ALL_ENGINES),
            language=language,
            diarization=diarization,
            min_speakers=1,
            max_speakers=int(max_speakers),
            enhance=enhance,
            local_correction=local_correction,
        )
        return _build_outputs(result, selected_engines or list(ALL_ENGINES))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Transcription job failed: %s", exc, exc_info=True)
        return _empty_outputs(f"ERROR: {exc}")


def build_ui() -> gr.Blocks:
    hw_md = hardware_summary()

    with gr.Blocks(title="Local Transcript App") as demo:
        gr.Markdown("# Local Transcript App")
        gr.Markdown("Upload audio or video, then transcribe locally with open-source models.")

        load_status = gr.Markdown(_get_load_status())

        with gr.Row():
            with gr.Column(scale=2):
                media_input = gr.File(
                    label="Audio or Video File",
                    file_types=["audio", "video"],
                    type="filepath",
                    interactive=False,
                )
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="Thai",
                    label="Language",
                )
                engine_selector = gr.CheckboxGroup(
                    choices=ALL_ENGINES,
                    value=ALL_ENGINES,
                    label="Local ASR Engines",
                )
                enhance = gr.Checkbox(label="Audio Enhancement", value=False)
                diarization = gr.Checkbox(label="Speaker Diarization", value=False)
                local_correction = gr.Checkbox(
                    label="Optional Local LLM Correction",
                    value=False,
                )
                with gr.Row(visible=False) as speakers_row:
                    max_speakers = gr.Slider(1, 10, step=1, value=3, label="Max Speakers")
                transcribe_btn = gr.Button("Transcribe", variant="primary", interactive=False)

        diarization.change(  # pylint: disable=no-member
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[diarization],
            outputs=[speakers_row],
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Original Media Preview")
                original_video = gr.Video(
                    label="Video",
                    interactive=False,
                    visible=False,
                )
                original_audio_preview = gr.Audio(
                    label="Audio",
                    interactive=False,
                    visible=False,
                    type="filepath",
                )
            with gr.Column():
                gr.Markdown("#### Enhanced Audio Preview")
                enhanced_audio = gr.Audio(label="Enhanced", interactive=False, type="filepath")

        _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".ts"}

        def _route_media_preview(path):
            if not path:
                return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
            ext = os.path.splitext(path)[1].lower()
            if ext in _VIDEO_EXTS:
                return gr.update(value=path, visible=True), gr.update(value=None, visible=False)
            return gr.update(value=None, visible=False), gr.update(value=path, visible=True)

        media_input.change(  # pylint: disable=no-member
            fn=_route_media_preview,
            inputs=[media_input],
            outputs=[original_video, original_audio_preview],
        )

        def _run_enhance(media_path, do_enhance):
            if not media_path or not do_enhance:
                return None
            return enhance_audio(media_path)

        enhance.change(  # pylint: disable=no-member
            fn=_run_enhance,
            inputs=[media_input, enhance],
            outputs=[enhanced_audio],
        )
        media_input.change(  # pylint: disable=no-member
            fn=_run_enhance,
            inputs=[media_input, enhance],
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
                    typhoon_time = gr.Textbox(label=LABEL_ELAPSED, interactive=False, max_lines=1, scale=3)
                    typhoon_dl = gr.DownloadButton(label=LABEL_DOWNLOAD, value=None, scale=1, interactive=False)

            with gr.TabItem(ENGINE_THONBURIAN):
                thonburian_text = gr.Textbox(
                    label="Transcript",
                    lines=20,
                    max_lines=200,
                    buttons=["copy"],
                    interactive=False,
                )
                with gr.Row():
                    thonburian_time = gr.Textbox(label=LABEL_ELAPSED, interactive=False, max_lines=1, scale=3)
                    thonburian_dl = gr.DownloadButton(label=LABEL_DOWNLOAD, value=None, scale=1, interactive=False)

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
                local_correction,
            ],
            outputs=[
                typhoon_text, typhoon_time, typhoon_dl,
                thonburian_text, thonburian_time, thonburian_dl,
                job_info,
            ],
        )

        timer = gr.Timer(value=2)

        def check_ready():
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
    application.launch(server_name="0.0.0.0", server_port=7896, max_threads=40, theme=_SoftTheme())
