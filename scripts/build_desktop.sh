#!/usr/bin/env bash
# Build LocalTranscriptApp native desktop bundle (PyInstaller onedir).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "venv not found. Run ./setup.sh first, then: pip install 'pyinstaller>=6.10'" >&2
  exit 1
fi

echo "[build] Using $PYTHON"
"$PYTHON" -m pip install "pyinstaller>=6.10" -q

echo "[build] PyInstaller onedir -> dist/LocalTranscriptApp/"
"$PYTHON" -m PyInstaller --noconfirm --clean LocalTranscriptApp.spec

DIST="${ROOT}/dist/LocalTranscriptApp"
mkdir -p "${DIST}/models" "${DIST}/storage" "${DIST}/config"
cp -f "${ROOT}/.env.production" "${DIST}/.env"

# Copy optional vendor binaries into dist/LocalTranscriptApp/bin (ffmpeg, etc.)
if [[ -d "${ROOT}/vendor" ]]; then
  mkdir -p "${DIST}/bin"
  echo "[build] Copying vendor binaries to ${DIST}/bin"
  cp -a "${ROOT}/vendor/." "${DIST}/bin/"
fi

echo "[build] Done: ${DIST}/LocalTranscriptApp"
echo "[build] Copy pre-cached models into ${DIST}/models before distributing."
echo "[build] For full offline bundle: python scripts/stage_model_pack.py --pack <path-to-model-pack-models> --target ${DIST}"
