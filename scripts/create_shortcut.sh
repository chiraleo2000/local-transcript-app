#!/usr/bin/env bash
# create_shortcut.sh — create a Linux .desktop launcher for Local Transcript App
#
# Usage:
#   ./scripts/create_shortcut.sh            # Desktop shortcut only
#   ./scripts/create_shortcut.sh --apps     # Also install to ~/.local/share/applications
#
# After running, look for "Local Transcript App" in your application menu or
# double-click the .desktop file on your desktop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LAUNCHER_PY="$APP_ROOT/launcher.py"
VENV_PYTHON="$APP_ROOT/venv/bin/python"
ICON_FILE="$APP_ROOT/assets/icon.png"
DESKTOP_DIR="$HOME/Desktop"
APPS_DIR="$HOME/.local/share/applications"
SHORTCUT_NAME="local-transcript-app.desktop"

INSTALL_APPS=false
for arg in "$@"; do
    [[ "$arg" == "--apps" ]] && INSTALL_APPS=true
done

# ── Resolve Python ────────────────────────────────────────────────────────
if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "[ERROR] Python 3 not found. Run ./setup.sh first."
    exit 1
fi

# ── Resolve icon ──────────────────────────────────────────────────────────
if [[ ! -f "$ICON_FILE" ]]; then
    # Fall back to a built-in themed icon when the app asset is missing.
    ICON_FILE="utilities-terminal"
fi

# ── Build .desktop content ────────────────────────────────────────────────
DESKTOP_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=Local Transcript App
GenericName=Transcription Tool
Comment=GPU-accelerated local audio/video transcription with speaker diarization
Exec=$PYTHON $LAUNCHER_PY
Icon=$ICON_FILE
Terminal=false
StartupNotify=true
StartupWMClass=LocalTranscriptApp
Categories=AudioVideo;Audio;Utility;
Keywords=transcription;speech;diarization;whisper;
"

# ── Write Desktop shortcut ────────────────────────────────────────────────
if [[ -d "$DESKTOP_DIR" ]]; then
    DESKTOP_FILE="$DESKTOP_DIR/$SHORTCUT_NAME"
    printf '%s' "$DESKTOP_CONTENT" > "$DESKTOP_FILE"
    chmod +x "$DESKTOP_FILE"
    # Mark trusted on GNOME so it can be double-clicked without security prompt
    gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
    echo "[shortcut] Desktop launcher created: $DESKTOP_FILE"
else
    echo "[shortcut] Desktop directory not found — skipping Desktop shortcut."
fi

# ── Optionally install to application menu ───────────────────────────────
if [[ "$INSTALL_APPS" == true ]]; then
    mkdir -p "$APPS_DIR"
    APPS_FILE="$APPS_DIR/$SHORTCUT_NAME"
    printf '%s' "$DESKTOP_CONTENT" > "$APPS_FILE"
    chmod +x "$APPS_FILE"
    # Refresh desktop database
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
    echo "[shortcut] Application menu entry installed: $APPS_FILE"
fi

echo ""
echo "Done! Double-click 'Local Transcript App' on your Desktop to launch."
