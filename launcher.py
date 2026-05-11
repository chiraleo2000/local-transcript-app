"""LocalTranscriptApp launcher — double-click to start the app.

Modes
-----
  python launcher.py            — auto-detect: Docker if available, else venv
  python launcher.py --docker   — force Docker Compose (GPU) mode
  python launcher.py --direct   — force direct venv / Python mode

When bundled by PyInstaller into LocalTranscriptApp.exe, double-clicking
starts Docker Compose (or the venv), waits for the Gradio server, then
opens the UI inside a native desktop window via pywebview — no browser needed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(
    os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)
)
VENV_PYTHON_WIN = os.path.join(HERE, "venv", "Scripts", "python.exe")
VENV_PYTHON_NIX = os.path.join(HERE, "venv", "bin", "python")
APP_PY = os.path.join(HERE, "app.py")
COMPOSE_GPU = os.path.join(HERE, "docker-compose.gpu.yml")
COMPOSE_CPU = os.path.join(HERE, "docker-compose.yml")
APP_URL = "http://localhost:7896"
READY_URL = f"{APP_URL}/gradio_api/startup-events"

_SPLASH_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif;
         display:flex; align-items:center; justify-content:center; height:100vh; }}
  .card {{ text-align:center; padding:48px 64px; background:#1e293b;
           border-radius:16px; box-shadow:0 8px 32px rgba(0,0,0,.5); }}
  h1 {{ font-size:2rem; font-weight:700; margin-bottom:12px; color:#f8fafc; }}
  p  {{ color:#94a3b8; margin-bottom:32px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
          background:#6366f1; margin:0 4px;
          animation: bounce 1.2s infinite ease-in-out; }}
  .dot:nth-child(2) {{ animation-delay:.15s; }}
  .dot:nth-child(3) {{ animation-delay:.30s; }}
  @keyframes bounce {{ 0%,80%,100%{{transform:scale(0)}} 40%{{transform:scale(1)}} }}
</style></head>
<body><div class="card">
  <h1>Local Transcript App</h1>
  <p>Starting models and GPU backend&hellip;</p>
  <span class="dot"></span><span class="dot"></span><span class="dot"></span>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_python() -> str:
    for path in (VENV_PYTHON_WIN, VENV_PYTHON_NIX):
        if os.path.isfile(path):
            return path
    return sys.executable


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=8, check=False
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _gpu_in_docker() -> bool:
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:13.0.0-runtime-ubuntu24.04", "nvidia-smi"],
            capture_output=True, timeout=30, check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _wait_for_server(timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(READY_URL, timeout=3) as r:
                if r.status < 500:
                    return True
        except OSError:
            pass
        time.sleep(2)
    return False


def _open_native_window() -> None:
    """Show the app in a native pywebview desktop window.

    Degrades to system browser when pywebview is unavailable (CI / headless).
    Shows a loading splash first, then navigates to the live app URL.
    """
    try:
        webview = importlib.import_module("webview")
        window = webview.create_window(
            "Local Transcript App",
            html=_SPLASH_HTML,
            width=1400,
            height=900,
            resizable=True,
            min_size=(900, 600),
        )

        def _navigate():
            ready = _wait_for_server()
            if not ready:
                print(f"[launcher] Server did not respond in 180 s — loading {APP_URL} anyway.")
            window.load_url(APP_URL)

        import threading
        threading.Thread(target=_navigate, daemon=True).start()
        webview.start(debug=False)

    except ModuleNotFoundError:
        print("[launcher] pywebview not installed — opening system browser.")
        _wait_for_server()
        webbrowser.open(APP_URL)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"[launcher] pywebview error ({exc}) — opening system browser.")
        _wait_for_server()
        webbrowser.open(APP_URL)


# ---------------------------------------------------------------------------
# Start modes
# ---------------------------------------------------------------------------

def _start_docker() -> subprocess.Popen | None:
    compose_file = COMPOSE_GPU if _gpu_in_docker() else COMPOSE_CPU
    print(f"[launcher] Docker mode: {os.path.basename(compose_file)}")
    try:
        proc = subprocess.Popen(
            ["docker", "compose", "-f", compose_file, "up", "-d"],
            cwd=HERE,
        )
        proc.wait()
        return None  # Docker manages the server process; nothing to terminate
    except (FileNotFoundError, OSError) as exc:
        print(f"[launcher] docker compose failed: {exc}")
        return None


def _start_direct() -> subprocess.Popen:
    python = _find_python()
    if not os.path.isfile(APP_PY):
        print(f"[launcher] ERROR: app.py not found at {APP_PY}")
        input("Press Enter to exit...")
        sys.exit(1)

    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{HERE}{os.pathsep}{pp}" if pp else HERE

    print(f"[launcher] Direct mode: {python} app.py")
    return subprocess.Popen([python, APP_PY], cwd=HERE, env=env)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--direct", action="store_true")
    args, _ = parser.parse_known_args()

    print("=" * 56)
    print("  Local Transcript App")
    print("=" * 56)

    backend_proc: subprocess.Popen | None = None

    if args.docker or (not args.direct and _docker_available()):
        _start_docker()
    else:
        backend_proc = _start_direct()

    _open_native_window()

    # Shut down the direct backend when the window closes.
    if backend_proc is not None:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_proc.kill()


if __name__ == "__main__":
    main()
