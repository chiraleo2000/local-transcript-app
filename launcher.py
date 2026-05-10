"""LocalTranscriptApp launcher — double-click to start the app.

This script is bundled by PyInstaller into LocalTranscriptApp.exe.
It locates the venv (or system Python), launches app.py, then opens the UI
inside a native desktop window via pywebview (no separate browser needed).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
VENV_PYTHON = os.path.join(HERE, "venv", "Scripts", "python.exe")
APP_PY = os.path.join(HERE, "app.py")
APP_URL = "http://localhost:7896"
READY_URL = f"{APP_URL}/gradio_api/startup-events"


def _find_python() -> str:
    if os.path.isfile(VENV_PYTHON):
        return VENV_PYTHON
    return sys.executable  # fallback to bundled Python


def _wait_for_server(timeout: float = 120.0) -> bool:
    """Poll until the Gradio server responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(READY_URL, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _open_native_window() -> None:
    """Open the app in a native desktop window using pywebview.

    Falls back to the system browser if pywebview is unavailable (e.g. on a
    headless server) so the launcher degrades gracefully.
    """
    try:
        import webview  # pywebview package
        webview.create_window(
            "Local Transcript App",
            APP_URL,
            width=1400,
            height=900,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start(debug=False)
    except ImportError:
        import webbrowser
        webbrowser.open(APP_URL)
    except Exception as exc:
        print(f"[launcher] pywebview error ({exc}); falling back to browser.")
        import webbrowser
        webbrowser.open(APP_URL)


def main() -> None:
    python = _find_python()
    if not os.path.isfile(APP_PY):
        print(f"ERROR: app.py not found at {APP_PY}")
        input("Press Enter to exit...")
        sys.exit(1)

    print("Starting Local Transcript App...")
    proc = subprocess.Popen(
        [python, APP_PY],
        cwd=HERE,
    )

    print("Waiting for server to be ready...")
    ready = _wait_for_server()
    if not ready:
        print(f"[WARNING] Server did not respond within 120 s — opening {APP_URL} anyway.")

    _open_native_window()

    # When the window is closed, stop the backend server.
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
