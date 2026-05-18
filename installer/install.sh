#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Local Transcript App — Linux installer (GUI launcher, no Docker required)
# ---------------------------------------------------------------------------
# Installs the app under $HOME/.local/share/local-transcript-app, sets up a
# venv, installs Python deps, registers a Desktop entry that launches the
# pywebview GUI, and creates a `local-transcript-app` shim in ~/.local/bin.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh                     # default install
#   ./install.sh --prefix /opt/lta   # custom prefix
#   ./install.sh --uninstall
# ---------------------------------------------------------------------------
set -euo pipefail

APP_ID="local-transcript-app"
APP_NAME="Local Transcript App"
APP_VERSION="1.2.0"
DEFAULT_PREFIX="${HOME}/.local/share/${APP_ID}"
PREFIX="${DEFAULT_PREFIX}"
DO_UNINSTALL=0
SKIP_DEPS=0
WITH_ROCM=0

print() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)     PREFIX="$2"; shift 2 ;;
    --uninstall)  DO_UNINSTALL=1; shift ;;
    --skip-deps)  SKIP_DEPS=1; shift ;;
    --with-rocm)  WITH_ROCM=1; shift ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

DESKTOP_FILE="${HOME}/.local/share/applications/${APP_ID}.desktop"
BIN_LINK="${HOME}/.local/bin/${APP_ID}"

if [[ "${DO_UNINSTALL}" -eq 1 ]]; then
  print "Uninstalling ${APP_NAME} from ${PREFIX}"
  rm -rf "${PREFIX}"
  rm -f "${DESKTOP_FILE}" "${BIN_LINK}"
  command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "${HOME}/.local/share/applications" >/dev/null 2>&1 || true
  print "Uninstall complete."
  exit 0
fi

# --- prerequisites ---------------------------------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+ first."
PY_MAJ=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MIN=$(python3 -c 'import sys; print(sys.version_info[1])')
if [[ "${PY_MAJ}" -lt 3 || ( "${PY_MAJ}" -eq 3 && "${PY_MIN}" -lt 10 ) ]]; then
  die "Python 3.10+ required (found ${PY_MAJ}.${PY_MIN})."
fi
command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg not found — audio decoding will fail. Install via your package manager (e.g. 'sudo apt install ffmpeg')."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

print "Installing ${APP_NAME} ${APP_VERSION}"
print "  source : ${SRC_DIR}"
print "  prefix : ${PREFIX}"

mkdir -p "${PREFIX}"
# Copy source tree (skip large/local-only dirs).
rsync -a --delete \
  --exclude 'venv/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'build/' --exclude 'dist/' \
  --exclude 'storage/jobs/' --exclude 'storage/transcripts/' --exclude 'storage/logs/' --exclude 'storage/audio/' \
  --exclude '.git/' \
  --exclude '.env' \
  "${SRC_DIR}/" "${PREFIX}/"

# Ship the no-token runtime config as the active .env.
if [[ -f "${PREFIX}/.env.production" && ! -f "${PREFIX}/.env" ]]; then
  cp "${PREFIX}/.env.production" "${PREFIX}/.env"
fi

# --- venv + deps ------------------------------------------------------------
if [[ "${SKIP_DEPS}" -eq 0 ]]; then
  print "Creating virtualenv"
  python3 -m venv "${PREFIX}/venv"
  # shellcheck disable=SC1091
  source "${PREFIX}/venv/bin/activate"
  pip install --upgrade pip wheel
  print "Installing Python dependencies (this can take several minutes)"
  pip install -r "${PREFIX}/requirements.txt"
  if [[ "${WITH_ROCM}" -eq 1 ]]; then
    print "Installing ROCm PyTorch (AMD GPU)"
    pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision torchaudio
  fi
  deactivate
else
  warn "--skip-deps set; venv not created. You must provide a working Python env at ${PREFIX}/venv."
fi

# --- launcher shim ---------------------------------------------------------
mkdir -p "$(dirname "${BIN_LINK}")"
cat > "${BIN_LINK}" <<EOF
#!/usr/bin/env bash
export APP_FORCE_DIRECT=1
cd "${PREFIX}"
exec "${PREFIX}/venv/bin/python" "${PREFIX}/launcher.py" "\$@"
EOF
chmod +x "${BIN_LINK}"

# --- desktop entry ---------------------------------------------------------
mkdir -p "$(dirname "${DESKTOP_FILE}")"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Local Thai speech-to-text with diarization (offline)
Exec=${BIN_LINK}
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;Utility;
StartupNotify=true
Version=${APP_VERSION}
EOF

command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "${HOME}/.local/share/applications" >/dev/null 2>&1 || true

print "Installation complete."
print "  Launch from your application menu: ${APP_NAME}"
print "  Or run from terminal: ${APP_ID}"
print ""
print "First-run notes:"
print "  - Gated models (Typhoon Whisper, pyannote community-1) must already be"
print "    present under ${PREFIX}/models/hf_cache/. If absent, set HF_TOKEN and"
print "    run: ${PREFIX}/venv/bin/python ${PREFIX}/scripts/bootstrap_models.py"
print "  - To force a backend (cuda | rocm | openvino | directml | cpu) export"
print "    APP_FORCE_BACKEND=<name> before launching."
