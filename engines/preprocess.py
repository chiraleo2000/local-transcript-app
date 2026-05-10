"""Three-stage audio preprocessing for improved ASR accuracy.

Stage 1 — FFmpeg:
  highpass/lowpass bandpass → 16 kHz mono WAV (format conversion only)

Stage 2 — noisereduce (spectral gating):
  Non-stationary spectral gating for adaptive noise reduction.
  Much better than ffmpeg afftdn for speech with varying background noise.

Stage 3 — pedalboard (Spotify):
  NoiseGate (silence background) → Compressor (gentle) → Limiter →
  peak-normalise only if very quiet
"""

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


# ---------------------------------------------------------------------------
# Locate ffmpeg — works even when PATH does not include the ffmpeg bin dir
# (common on Windows when installed via winget into a user-local folder)
# ---------------------------------------------------------------------------

_FFMPEG_EXE_NAME = "ffmpeg.exe"


def _locate_ffmpeg_in_registry() -> str | None:
    """Search Windows registry PATH for ffmpeg."""
    import winreg

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


# ---------------------------------------------------------------------------
# Stage 1: FFmpeg — format conversion + bandpass only
# ---------------------------------------------------------------------------
# Only strip rumble (<80 Hz) and hiss (>8 kHz), convert to 16 kHz mono WAV.
# Actual noise reduction is handled by noisereduce in Stage 2.

_FILTERS = ",".join([
    "highpass=f=80",
    "lowpass=f=8000",
])


def _ffmpeg_stage(audio_path: str, out_path: str) -> bool:
    """Run FFmpeg bandpass + format conversion. Returns True on success."""
    if not _FFMPEG_EXE:
        logger.error("ffmpeg not found — cannot run Stage 1.")
        return False

    cmd = [
        _FFMPEG_EXE, "-y",
        "-i", audio_path,
        "-af", _FILTERS,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        out_path,
    ]
    logger.debug("ffmpeg stage: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            logger.error("ffmpeg stage failed (rc=%d): %s", result.returncode, result.stderr[-600:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg stage timed out.")
        return False
    except (subprocess.SubprocessError, OSError) as exc:
        logger.error("ffmpeg stage error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Stage 2: noisereduce — spectral gating
# ---------------------------------------------------------------------------

def _noisereduce_stage(wav_path: str, out_path: str) -> bool:
    """Apply non-stationary spectral gating noise reduction. Returns True on success."""
    try:
        import librosa
        import soundfile as sf
        import noisereduce as nr

        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        reduced = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,   # adaptive — tracks changing noise
            prop_decrease=0.75, # 75% noise reduction strength
            n_fft=2048,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )
        sf.write(out_path, reduced, sr, subtype="PCM_16")
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("noisereduce stage error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Stage 3: pedalboard DSP chain
# ---------------------------------------------------------------------------

def _pedalboard_stage(wav_path: str, out_path: str) -> bool:
    """Apply NoiseGate → Compressor → Limiter, then peak-normalise. Returns True on success."""
    try:
        from pedalboard import Pedalboard, NoiseGate, Compressor, Limiter
        from pedalboard.io import AudioFile

        board = Pedalboard([
            # Very soft gate — only kills true silence below -50dB.
            # -30dB was cutting soft speech (e.g. quieter speakers, pauses mid-sentence).
            NoiseGate(threshold_db=-50, ratio=4.0, attack_ms=5.0, release_ms=200.0),
            # Gentle compression only
            Compressor(threshold_db=-18, ratio=2.5, attack_ms=10.0, release_ms=200.0),
            # Hard limit — no clipping
            Limiter(threshold_db=-1.0, release_ms=50.0),
        ])

        with AudioFile(wav_path) as f:  # pylint: disable=not-context-manager
            audio = f.read(f.frames)
            sr = f.samplerate

        processed = board(audio, sr)

        # Only normalise if the signal is very quiet (under -20dBFS peak).
        # Do NOT normalise loud audio — that would boost any residual noise.
        peak = np.max(np.abs(processed))
        if 0.001 < peak < 0.1:  # only boost genuinely quiet recordings
            processed = processed * (0.1 / peak)

        with AudioFile(out_path, "w", samplerate=sr, num_channels=processed.shape[0]) as f:  # pylint: disable=not-context-manager
            f.write(processed)

        return True
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("pedalboard stage error: %s", exc)
        return False


def preprocess_audio(audio_path: str) -> str:
    """Three-stage audio enhancement for ASR.

    Stage 1 (FFmpeg): bandpass → 16 kHz mono WAV.
    Stage 2 (noisereduce): non-stationary spectral gating.
    Stage 3 (pedalboard): noise gate → compressor → limiter → peak normalise.

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
    logger.info("Preprocessing [Stage 1 — FFmpeg bandpass]: %s", audio_path)

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
