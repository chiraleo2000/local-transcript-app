"""CTranslate2 / faster-whisper adapter for Typhoon ASR on Pascal GPUs.

Transformers Whisper large-v3 on Tesla P4 is ~2.1x RT (beam=1). Long meetings
need ~0.67x RT for the 2/3-duration budget ╬ô├ç├╢ CT2 is the practical path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_CACHE: list[Any] = []


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def faster_whisper_enabled() -> bool:
    return _env_bool("ASR_USE_FASTER_WHISPER", False)


def clear_faster_whisper_cache() -> None:
    _MODEL_CACHE.clear()


def default_ct2_model_dir() -> Path:
    root = Path(os.getenv("APP_MODEL_ROOT", "./models")).expanduser()
    configured = os.getenv("ASR_CT2_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    model_id = (
        os.getenv("TYPHOON_MODEL_ID") or "typhoon-ai/typhoon-whisper-large-v3"
    ).replace("/", "__")
    return root / "ct2" / model_id


def ct2_model_ready(model_dir: Path | None = None) -> bool:
    """True when a usable CT2 model.bin exists (rejects truncated conversions)."""
    path = model_dir or default_ct2_model_dir()
    if not path.is_dir():
        return False
    model_bin = path / "model.bin"
    if not model_bin.is_file():
        bins = sorted(path.glob("*.bin"))
        model_bin = bins[0] if bins else None
    if model_bin is None or not model_bin.is_file():
        return False
    # Whisper large-v3 CT2: int8 ╬ô├½├¬1.5GB, float16 ╬ô├½├¬3.0GB. Truncated files fail mid-load.
    min_bytes = int(os.getenv("ASR_CT2_MIN_MODEL_BYTES", str(1_400_000_000)))
    try:
        size = model_bin.stat().st_size
    except OSError:
        return False
    if size < min_bytes:
        logger.warning(
            "CT2 model.bin too small (%s bytes < %s); treat as incomplete: %s",
            size,
            min_bytes,
            model_bin,
        )
        return False
    marker = path / ".ct2_ok"
    if marker.is_file():
        try:
            expected = int(marker.read_text(encoding="utf-8").strip().splitlines()[0])
            if expected != size:
                logger.warning(
                    "CT2 integrity marker mismatch (marker=%s file=%s); treat as incomplete",
                    expected,
                    size,
                )
                return False
        except (OSError, ValueError):
            pass
    return True


def write_ct2_ok_marker(model_dir: Path | None = None) -> None:
    path = model_dir or default_ct2_model_dir()
    model_bin = path / "model.bin"
    if not model_bin.is_file():
        return
    size = model_bin.stat().st_size
    (path / ".ct2_ok").write_text(
        f"{size}\nsha_probe=size_only\n",
        encoding="utf-8",
    )


def verify_ct2_model_load(model_dir: Path | None = None, *, device: str | None = None) -> None:
    """Open CT2 weights end-to-end; raise if file is truncated/corrupt."""
    path = model_dir or default_ct2_model_dir()
    if not ct2_model_ready(path):
        raise FileNotFoundError(
            f"CT2 model missing/incomplete at {path}. "
            "Run: python3 scripts/convert_typhoon_ct2.py"
        )
    from faster_whisper import WhisperModel

    use_device = device or ("cuda" if _env_bool("ASR_CT2_USE_CUDA", True) else "cpu")
    # CPU int8 is enough to force a full weight read without needing GPU.
    probe_device = "cpu"
    probe_compute = "int8"
    logger.info("CT2 integrity probe: %s (device=%s)", path, probe_device)
    try:
        WhisperModel(str(path), device=probe_device, compute_type=probe_compute)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"CT2 model.bin is corrupt/incomplete at {path}: {exc}. "
            "Delete that folder and re-run: python3 scripts/convert_typhoon_ct2.py"
        ) from exc
    write_ct2_ok_marker(path)
    logger.info("CT2 integrity OK (%s)", use_device)


def _compute_type() -> str:
    # Tesla P4 (Pascal): no efficient CT2 float16; int8_float32 fits ╬ô├½├▒6 GB shared GPU.
    return (os.getenv("ASR_CT2_COMPUTE_TYPE") or "int8_float32").strip() or "int8_float32"


def load_faster_whisper_model():
    """Load or return cached faster-whisper WhisperModel."""
    if _MODEL_CACHE:
        return _MODEL_CACHE[0]
    model_dir = default_ct2_model_dir()
    # Require .ct2_ok so a half-written model.bin cannot be used after a failed convert.
    require_ok = _env_bool("ASR_CT2_REQUIRE_OK_MARKER", True)
    marker = model_dir / ".ct2_ok"
    if require_ok and not marker.is_file():
        raise FileNotFoundError(
            f"CT2 integrity marker missing at {marker}. "
            "Run: python3 scripts/convert_typhoon_ct2.py "
            "(or python3 scripts/verify_ct2_load.py)"
        )
    if not ct2_model_ready(model_dir):
        raise FileNotFoundError(
            f"faster-whisper CT2 model missing/incomplete at {model_dir}. "
            "Run: python3 scripts/convert_typhoon_ct2.py"
        )
    from faster_whisper import WhisperModel

    device = "cuda" if _env_bool("ASR_CT2_USE_CUDA", True) else "cpu"
    compute = _compute_type()
    logger.info(
        "Loading faster-whisper CT2 model from %s (device=%s compute=%s)",
        model_dir,
        device,
        compute,
    )
    try:
        model = WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute,
            cpu_threads=_env_int("APP_CPU_THREADS", 6),
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "incomplete" in msg or "failed to read" in msg:
            raise RuntimeError(
                f"CT2 model.bin corrupt at {model_dir}: {exc}. "
                "Delete models/ct2/typhoon-ai__typhoon-whisper-large-v3 and run "
                "python3 scripts/convert_typhoon_ct2.py"
            ) from exc
        raise
    _MODEL_CACHE.append(model)
    return model


def _normalize_audio_array(audio_input) -> "np.ndarray":
    import numpy as np

    if isinstance(audio_input, dict):
        audio = np.asarray(audio_input["raw"], dtype=np.float32)
        sr = int(audio_input.get("sampling_rate") or 16000)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio
    return np.asarray(audio_input, dtype=np.float32)


def _resolve_language(generate_kwargs: dict) -> str:
    language = generate_kwargs.get("language") or "th"
    if str(language).lower() in {"thai", "th-th"}:
        return "th"
    return str(language)


def _resolve_beams(generate_kwargs: dict) -> int:
    return max(1, int(generate_kwargs.get("num_beams") or _env_int("ASR_NUM_BEAMS", 1)))


def _resolve_temperature(generate_kwargs: dict) -> float:
    temperature = generate_kwargs.get("temperature", 0.0)
    if isinstance(temperature, (list, tuple)):
        return float(temperature[0]) if temperature else 0.0
    return float(temperature or 0.0)


def _threshold(generate_kwargs: dict, key: str, env_name: str, default: str) -> float:
    return float(generate_kwargs.get(key) or os.getenv(env_name, default))


def _ct2_transcription_options(
    *,
    tokenizer,
    beams: int,
    temperature: float,
    generate_kwargs: dict,
    clip_count: int,
):
    from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens

    return TranscriptionOptions(
        beam_size=beams,
        best_of=beams,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=float(os.getenv("ASR_REPETITION_PENALTY", "1.0") or 1.0),
        no_repeat_ngram_size=_env_int("ASR_NO_REPEAT_NGRAM_SIZE", 0),
        log_prob_threshold=_threshold(
            generate_kwargs, "logprob_threshold", "ASR_LOGPROB_THRESHOLD", "-1.0",
        ),
        no_speech_threshold=_threshold(
            generate_kwargs, "no_speech_threshold", "ASR_NO_SPEECH_THRESHOLD", "0.6",
        ),
        compression_ratio_threshold=_threshold(
            generate_kwargs, "compression_ratio_threshold",
            "ASR_COMPRESSION_RATIO_THRESHOLD", "2.4",
        ),
        temperatures=[temperature],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        prepend_punctuations="\"'ΓÇ£┬┐([{-",
        append_punctuations="\"'.πÇé,∩╝î!∩╝ü?∩╝ƒ:∩╝ÜΓÇ¥)]}πÇü",
        max_new_tokens=None,
        hotwords=None,
        word_timestamps=False,
        hallucination_silence_threshold=None,
        condition_on_previous_text=False,
        clip_timestamps=[{"start": 0, "end": 0}] * clip_count,
        prompt_reset_on_temperature=0.5,
        multilingual=False,
        without_timestamps=True,
        max_initial_timestamp=0.0,
    )


def _is_cuda_oom_message(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("out of memory", "oom", "cuda"))


def _joined_segment_text(segments: list[dict]) -> str:
    texts = [
        (segment.get("text") or "").strip()
        for segment in segments
        if (segment.get("text") or "").strip()
    ]
    return " ".join(texts).strip()


class FasterWhisperPipeline:
    """Duck-types enough of the HF ASR pipeline for whisper_runtime / typhoon."""

    def __init__(self, model=None):
        self.model = model or load_faster_whisper_model()
        self._batched = None

    def supports_turn_batch(self) -> bool:
        return True

    def _batched_pipeline(self):
        if self._batched is None:
            from faster_whisper import BatchedInferencePipeline

            self._batched = BatchedInferencePipeline(model=self.model)
        return self._batched

    def __call__(
        self,
        audio_input,
        *,
        batch_size: int = 1,
        generate_kwargs: dict | None = None,
        return_timestamps=True,
        chunk_length_s: int | None = None,  # noqa: ARG002 ΓÇö HF API parity
        **_extra,
    ) -> dict:
        generate_kwargs = generate_kwargs or {}
        audio = _normalize_audio_array(audio_input)
        language = _resolve_language(generate_kwargs)
        beams = _resolve_beams(generate_kwargs)
        temperature = _resolve_temperature(generate_kwargs)
        vad = _env_bool("ASR_CT2_VAD_FILTER", False)
        word_ts = return_timestamps == "word"
        duration_s = float(audio.shape[0]) / 16000.0
        use_batched = (
            batch_size > 1
            and duration_s >= max(35.0, _env_int("ASR_CT2_BATCHED_MIN_DURATION_S", 35))
            and not word_ts
        )
        common = {
            "language": language,
            "task": "transcribe",
            "beam_size": beams,
            "best_of": beams,
            "temperature": temperature,
            "condition_on_previous_text": _env_bool(
                "ASR_CONDITION_ON_PREVIOUS_TEXT", False,
            ),
            "vad_filter": vad,
            "word_timestamps": word_ts,
            "compression_ratio_threshold": _threshold(
                generate_kwargs, "compression_ratio_threshold",
                "ASR_COMPRESSION_RATIO_THRESHOLD", "2.4",
            ),
            "log_prob_threshold": _threshold(
                generate_kwargs, "logprob_threshold",
                "ASR_LOGPROB_THRESHOLD", "-1.0",
            ),
            "no_speech_threshold": _threshold(
                generate_kwargs, "no_speech_threshold",
                "ASR_NO_SPEECH_THRESHOLD", "0.6",
            ),
        }
        if use_batched:
            logger.info(
                "CT2 BatchedInferencePipeline: audio=%.1fs batch_size=%d",
                duration_s,
                batch_size,
            )
            segments_iter, _info = self._batched_pipeline().transcribe(
                audio,
                batch_size=max(1, int(batch_size)),
                without_timestamps=False,
                **common,
            )
        else:
            segments_iter, _info = self.model.transcribe(audio, **common)

        chunks: list[dict] = []
        texts: list[str] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            texts.append(text)
            if return_timestamps:
                chunks.append({
                    "text": text,
                    "timestamp": (float(seg.start), float(seg.end)),
                })
        joined = " ".join(texts).strip()
        result: dict[str, Any] = {"text": joined}
        if return_timestamps:
            result["chunks"] = chunks
        return result

    def transcribe_batch(
        self,
        audio_inputs: list,
        *,
        batch_size: int = 8,
        generate_kwargs: dict | None = None,
    ) -> list[dict]:
        """Transcribe many short clips in shared GPU encode/decode batches.

        Turn-guided diarization ASR calls this so N speaker turns share one CT2
        forward pass instead of N serial launches that leave VRAM half idle.
        """
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer

        if not audio_inputs:
            return []
        generate_kwargs = generate_kwargs or {}
        language = _resolve_language(generate_kwargs)
        beams = _resolve_beams(generate_kwargs)
        temperature = _resolve_temperature(generate_kwargs)
        batch_size = max(1, int(batch_size))

        audios = [_normalize_audio_array(item) for item in audio_inputs]
        fe = self.model.feature_extractor
        sampling_rate = fe.sampling_rate
        features = []
        chunks_metadata = []
        for audio in audios:
            duration = float(audio.shape[0]) / float(sampling_rate)
            feat = fe(audio)[..., :-1]
            features.append(pad_or_trim(feat))
            chunks_metadata.append({
                "offset": 0.0,
                "duration": duration,
                "segments": [{"start": 0, "end": int(audio.shape[0])}],
            })
        feature_batch = np.stack(features)

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        options = _ct2_transcription_options(
            tokenizer=tokenizer,
            beams=beams,
            temperature=temperature,
            generate_kwargs=generate_kwargs,
            clip_count=len(audios),
        )

        batched = self._batched_pipeline()
        results: list[dict] = [{"text": ""} for _ in audios]
        logger.info(
            "CT2 turn batch: clips=%d batch_size=%d beams=%d",
            len(audios),
            batch_size,
            beams,
        )
        for start in range(0, len(audios), batch_size):
            end = min(start + batch_size, len(audios))
            try:
                segmented = batched.forward(
                    feature_batch[start:end],
                    tokenizer,
                    chunks_metadata[start:end],
                    options,
                )
            except Exception as exc:  # noqa: BLE001
                if end - start > 1 and _is_cuda_oom_message(exc):
                    logger.warning(
                        "CT2 turn batch OOM at size=%d; splitting: %s",
                        end - start,
                        exc,
                    )
                    mid = start + max(1, (end - start) // 2)
                    split_batch = max(1, batch_size // 2)
                    left = self.transcribe_batch(
                        audio_inputs[start:mid],
                        batch_size=split_batch,
                        generate_kwargs=generate_kwargs,
                    )
                    right = self.transcribe_batch(
                        audio_inputs[mid:end],
                        batch_size=split_batch,
                        generate_kwargs=generate_kwargs,
                    )
                    for offset, item in enumerate(left + right):
                        results[start + offset] = item
                    continue
                raise
            for offset, segments in enumerate(segmented):
                results[start + offset] = {"text": _joined_segment_text(segments)}
        return results


def get_faster_whisper_pipeline() -> FasterWhisperPipeline:
    return FasterWhisperPipeline()
