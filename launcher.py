"""LocalTranscriptApp launcher — double-click to start the app.

This script is bundled by PyInstaller into LocalTranscriptApp.exe.
It locates the venv (or system Python), then launches app.py in this folder.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
VENV_PYTHON = os.path.join(HERE, "venv", "Scripts", "python.exe")
APP_PY = os.path.join(HERE, "app.py")


def _find_python() -> str:
    if os.path.isfile(VENV_PYTHON):
        return VENV_PYTHON
    return sys.executable  # fallback to bundled Python


def main() -> None:
    python = _find_python()
    if not os.path.isfile(APP_PY):
        print(f"ERROR: app.py not found at {APP_PY}")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"Starting Local Transcript App...")
    print(f"Open your browser at: http://localhost:7896")
    print("Press Ctrl+C in this window to stop.\n")

    # Open browser after a short delay
    try:
        import threading
        import time
        def _open():
            time.sleep(4)
            webbrowser.open("http://localhost:7896")
        threading.Thread(target=_open, daemon=True).start()
    except Exception:
        pass

    proc = subprocess.run(
        [python, APP_PY],
        cwd=HERE,
        check=False,
    )
    if proc.returncode != 0:
        print(f"\nApp exited with code {proc.returncode}.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
