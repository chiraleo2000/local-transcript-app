"""Three-stage audio preprocessing for improved ASR accuracy.

Stage 1 — FFmpeg:
  speech bandpass → optional presence boost → loudnorm → 16 kHz mono WAV

Stage 2 — noisereduce (spectral gating):
  Non-stationary spectral gating; optional leading-noise profile.

Stage 3 — pedalboard (Spotify):
  Highpass → NoiseGate → Compressor → Limiter → speech-RMS peak gain
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Locate ffmpeg — works even when PATH does not include the ffmpeg bin dir
# (common on Windows when installed via winget into a user-local folder)
# ---------------------------------------------------------------------------

_FFMPEG_EXE_NAME = "ffmpeg.exe"


def _locate_ffmpeg_in_registry() -> str | None:
    """Search Windows registry PATH for ffmpeg."""
    import winreg  # pylint: disable=import-outside-toplevel

    paths = []
    for hive, subkey in [
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    ]:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                val, _ = winreg.QueryValueEx(key, "Path")
                paths.append(val)
        except FileNotFoundError:
            pass

    registry_path = os.pathsep.join(paths)
    found = shutil.which("ffmpeg", path=registry_path)
    if found:
        os.environ["PATH"] = registry_path + os.pathsep + os.environ.get("PATH", "")
        logger.info("ffmpeg found via registry PATH: %s", found)
    return found


def _locate_ffmpeg_via_scan() -> str | None:
    """Scan common Windows install locations for ffmpeg."""
    candidates = [
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages"),
        r"C:\ProgramData\chocolatey\bin",
        os.path.expanduser(r"~\scoop\shims"),
        r"C:\ffmpeg\bin",
        r"C:\tools\ffmpeg\bin",
    ]
    for base in candidates:
        for pattern in [
            os.path.join(base, _FFMPEG_EXE_NAME),
            os.path.join(base, "*", "bin", _FFMPEG_EXE_NAME),
            os.path.join(base, "*", "*", "bin", _FFMPEG_EXE_NAME),
        ]:
            matches = glob.glob(pattern)
            if matches:
                found = matches[0]
                bin_dir = os.path.dirname(found)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                logger.info("ffmpeg found via scan: %s", found)
                return found
    return None


def _locate_ffmpeg() -> str | None:
    """Return absolute path to ffmpeg executable, or None if not found."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    if sys.platform == "win32":
        try:
            found = _locate_ffmpeg_in_registry()
            if found:
                return found
        except OSError as exc:
            logger.debug("Registry PATH scan failed: %s", exc)

        found = _locate_ffmpeg_via_scan()
        if found:
            return found

    return None


# Resolve once at import time so subsequent calls are instant
_FFMPEG_EXE: str | None = _locate_ffmpeg()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %.2f.", name, value, default)
        return default


def _db_to_amp(db_value: float) -> float:
    return float(10 ** (db_value / 20.0))


def _ffmpeg_filter_chain() -> str:
    """Speech-focused bandpass + optional presence EQ + loudnorm before NR/DSP."""
    loudnorm_i = _env_float("AUDIO_ENHANCE_LOUDNORM_I", -14.0)
    highpass_hz = max(40.0, _env_float("AUDIO_ENHANCE_HIGHPASS_HZ", 100.0))
    lowpass_hz = min(8000.0, max(highpass_hz + 500.0, _env_float("AUDIO_ENHANCE_LOWPASS_HZ", 7500.0)))
    filters = [
        f"highpass=f={highpass_hz:.0f}",
        f"lowpass=f={lowpass_hz:.0f}",
    ]
    # Mild FFT denoise before loudnorm when speech-focus is on (cuts hum/hiss).
    if _env_bool("AUDIO_ENHANCE_SPEECH_FOCUS", True):
        afftdn_nr = max(0.0, min(30.0, _env_float("AUDIO_ENHANCE_AFFTDN_NR", 12.0)))
        if afftdn_nr > 0.0:
            filters.append(f"afftdn=nr={afftdn_nr:.1f}:nf=-25")
        presence_db = _env_float("AUDIO_ENHANCE_SPEECH_PRESENCE_DB", 4.0)
        if abs(presence_db) >= 0.5:
            # Presence shelf around speech intelligibility band (~2.5 kHz).
            filters.append(
                f"equalizer=f=2500:t=q:w=1.0:g={presence_db:.1f}"
            )
    atempo = _env_float("AUDIO_ENHANCE_ATEMPO", 1.0)
    if 0.5 <= atempo <= 2.0 and abs(atempo - 1.0) > 0.01:
        filters.append(f"atempo={atempo:.3f}")
    # Narrower LRA + hotter integrated loudness keeps speech more consistent.
    lra = max(3.0, min(18.0, _env_float("AUDIO_ENHANCE_LOUDNORM_LRA", 7.0)))
    filters.append(f"loudnorm=I={loudnorm_i}:TP=-1.0:LRA={lra:.0f}")
    return ",".join(filters)


def _ffmpeg_stage(audio_path: str, out_path: str) -> bool:
    """Run FFmpeg bandpass + loudnorm + format conversion. Returns True on success."""
    if not _FFMPEG_EXE:
        logger.error("ffmpeg not found — cannot run Stage 1.")
        return False

    filters = _ffmpeg_filter_chain()
    threads = max(1, int(os.getenv("APP_CPU_THREADS", "0") or "0") or (os.cpu_count() or 4))
    cmd = [
        _FFMPEG_EXE, "-y",
        "-threads", str(threads),
        "-i", audio_path,
        "-af", filters,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        out_path,
    ]
    logger.info("Preprocessing [Stage 1 — FFmpeg]: %s", filters)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            logger.error("ffmpeg stage failed (rc=%d): %s", result.returncode, result.stderr[-600:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg stage timed out.")
        return False
    except (subprocess.SubprocessError, OSError):
        logger.exception("ffmpeg stage error")
        return False


# ---------------------------------------------------------------------------
# Stage 2: noisereduce — spectral gating
# ---------------------------------------------------------------------------

def _noise_profile_seconds() -> float:
    return max(0.25, _env_float("AUDIO_ENHANCE_NOISE_PROFILE_SECONDS", 0.75))


def _noisereduce_stage(wav_path: str, out_path: str) -> bool:
    """Apply non-stationary spectral gating noise reduction. Returns True on success."""
    try:
        import librosa  # pylint: disable=import-outside-toplevel
        import soundfile as sf  # pylint: disable=import-outside-toplevel
        import noisereduce as nr  # pylint: disable=import-outside-toplevel

        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        prop_decrease = _env_float("AUDIO_ENHANCE_NOISE_REDUCTION", 0.92)
        profile_mode = os.getenv("AUDIO_ENHANCE_NOISE_PROFILE", "leading").strip().lower()
        speech_focus = _env_bool("AUDIO_ENHANCE_SPEECH_FOCUS", True)
        # Smoother masks preserve consonants better when NR is strong.
        nr_kwargs: dict = {
            "y": y,
            "sr": sr,
            "prop_decrease": prop_decrease,
            "n_fft": 2048,
            "freq_mask_smooth_hz": 500 if speech_focus else 700,
            "time_mask_smooth_ms": 64 if speech_focus else 80,
        }
        if profile_mode == "leading":
            profile_samples = int(_noise_profile_seconds() * sr)
            if profile_samples > 0 and len(y) > profile_samples * 2:
                nr_kwargs["y_noise"] = y[:profile_samples]
                nr_kwargs["stationary"] = True
                logger.info(
                    "noisereduce leading-noise profile: first %.2fs, prop=%.2f",
                    profile_samples / sr,
                    prop_decrease,
                )
            else:
                nr_kwargs["stationary"] = False
        else:
            nr_kwargs["stationary"] = False

        reduced = nr.reduce_noise(**nr_kwargs)
        sf.write(out_path, reduced, sr, subtype="PCM_16")
        return True
    except (OSError, ValueError, RuntimeError):
        logger.exception("noisereduce stage error")
        return False


# ---------------------------------------------------------------------------
# Stage 3: pedalboard DSP chain
# ---------------------------------------------------------------------------

def _pedalboard_stage(wav_path: str, out_path: str) -> bool:
    """Apply highpass → gate → compress → limit, then controlled peak gain."""
    try:
        from pedalboard import (  # pylint: disable=import-outside-toplevel
            Pedalboard,
            NoiseGate,
            Compressor,
            Limiter,
            HighpassFilter,
        )
        from pedalboard.io import AudioFile  # pylint: disable=import-outside-toplevel

        gate_db = _env_float("AUDIO_ENHANCE_GATE_THRESHOLD_DB", -42.0)
        gate_ratio = _env_float("AUDIO_ENHANCE_GATE_RATIO", 6.0)
        comp_db = _env_float("AUDIO_ENHANCE_COMPRESSOR_THRESHOLD_DB", -18.0)
        comp_ratio = _env_float("AUDIO_ENHANCE_COMPRESSOR_RATIO", 4.0)
        limiter_db = _env_float("AUDIO_ENHANCE_LIMITER_THRESHOLD_DB", -0.5)
        highpass_hz = max(40.0, _env_float("AUDIO_ENHANCE_HIGHPASS_HZ", 100.0))
        speech_focus = _env_bool("AUDIO_ENHANCE_SPEECH_FOCUS", True)

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=highpass_hz),
            NoiseGate(
                threshold_db=gate_db,
                ratio=gate_ratio,
                # Faster attack / longer release keeps speech onsets, cuts noise tails.
                attack_ms=2.0 if speech_focus else 3.0,
                release_ms=220.0 if speech_focus else 150.0,
            ),
            Compressor(
                threshold_db=comp_db,
                ratio=comp_ratio,
                attack_ms=4.0 if speech_focus else 5.0,
                release_ms=160.0 if speech_focus else 120.0,
            ),
            Limiter(threshold_db=limiter_db, release_ms=40.0),
        ])

        with AudioFile(wav_path) as f:  # pylint: disable=not-context-manager
            audio = f.read(f.frames)
            sr = f.samplerate

        processed = board(audio, sr)

        # Lift speech level toward target peak. With speech-focus, scale from the
        # RMS of louder frames (speaker parts) so quiet talkers are boosted more
        # than a single loud transient would allow.
        target_peak = _db_to_amp(_env_float("AUDIO_ENHANCE_TARGET_PEAK_DB", -1.5))
        max_gain = _db_to_amp(_env_float("AUDIO_ENHANCE_MAX_GAIN_DB", 18.0))
        peak = float(np.max(np.abs(processed)))
        if peak > 0.001:
            if speech_focus:
                flat = np.asarray(processed, dtype=np.float32).reshape(-1)
                # Frames above ~gate threshold approximate active speech energy.
                speech_floor = _db_to_amp(gate_db + 6.0)
                speech = flat[np.abs(flat) >= speech_floor]
                if speech.size >= int(0.05 * sr):
                    speech_rms = float(np.sqrt(np.mean(np.square(speech))))
                    # Map speech RMS toward ~-12 dBFS equivalent before peak cap.
                    target_rms = _db_to_amp(_env_float("AUDIO_ENHANCE_SPEECH_TARGET_RMS_DB", -12.0))
                    rms_gain = target_rms / max(speech_rms, 1e-6)
                    peak_gain = target_peak / peak
                    gain = min(max_gain, max(1.0, min(rms_gain, peak_gain * 1.35)))
                else:
                    gain = min(max_gain, target_peak / peak)
            else:
                gain = min(max_gain, target_peak / peak)
            if gain > 1.0:
                processed = processed * gain
                logger.info("Audio enhancement gain applied: %.2fx (speech_focus=%s)", gain, speech_focus)

        processed = np.clip(processed, -0.98, 0.98)

        with AudioFile(out_path, "w", samplerate=sr, num_channels=processed.shape[0]) as f:  # pylint: disable=not-context-manager
            f.write(processed)

        return True
    except (OSError, ValueError, RuntimeError):
        logger.exception("pedalboard stage error")
        return False


def preprocess_audio(audio_path: str) -> str:
    """Three-stage audio enhancement for ASR.

    Stage 1 (FFmpeg): speech bandpass + optional denoise/presence + loudnorm.
    Stage 2 (noisereduce): spectral gating (optional leading-noise profile).
    Stage 3 (pedalboard): highpass → gate → compress → limit → speech-RMS gain.

    Returns path to enhanced WAV, or original path on failure.
    """
    if not _FFMPEG_EXE:
        logger.warning("ffmpeg not found — skipping audio preprocessing.")
        return audio_path

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    work_dir = tempfile.mkdtemp(prefix="asr_preprocess_")
    stage1_path = os.path.join(work_dir, f"{stem}_stage1.wav")
    stage2_path = os.path.join(work_dir, f"{stem}_stage2.wav")
    final_path  = os.path.join(work_dir, f"{stem}_enhanced.wav")

    t0 = time.perf_counter()
    logger.info("Preprocessing [Stage 1 — FFmpeg bandpass + loudnorm]: %s", audio_path)

    if not _ffmpeg_stage(audio_path, stage1_path):
        logger.warning("Stage 1 failed — using original audio.")
        return audio_path

    logger.info("Preprocessing [Stage 2 — noisereduce spectral gating]")

    if not _noisereduce_stage(stage1_path, stage2_path):
        logger.warning("Stage 2 (noisereduce) failed — skipping to Stage 3.")
        stage2_path = stage1_path

    logger.info("Preprocessing [Stage 3 — pedalboard: gate + compress + limit]")

    if not _pedalboard_stage(stage2_path, final_path):
        logger.warning("Stage 3 failed — using Stage 2 output.")
        final_path = stage2_path

    elapsed = time.perf_counter() - t0
    in_kb  = os.path.getsize(audio_path) // 1024
    out_kb = os.path.getsize(final_path) // 1024
    logger.info("Preprocessing done in %.2fs  (%d KB → %d KB)", elapsed, in_kb, out_kb)
    return final_path


def enhance_audio_for_asr(audio_path: str) -> str:
    """Enhance audio for the ASR stage when diarization keeps the raw path."""
    return preprocess_audio(audio_path)
