"""LocalTranscriptApp launcher — double-click to start the app.

Modes
-----
  python launcher.py            — auto-detect: direct venv by default
  python launcher.py --docker   — force Docker Compose (GPU) mode (dev only)
  python launcher.py --direct   — force direct venv / Python mode
  LocalTranscriptApp.exe --app-server  — internal Gradio backend (PyInstaller)

When bundled by PyInstaller into LocalTranscriptApp.exe the launcher ALWAYS
uses direct mode — Docker is treated as a developer-only convenience and is
never invoked from the installed application. Set APP_FORCE_DIRECT=1 in any
run to opt out of the Docker auto-detection path.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser


def _early_app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _bootstrap_sys_path() -> str:
    root = _early_app_root()
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and meipass not in sys.path:
            sys.path.insert(0, meipass)
        internal = os.path.join(root, "_internal")
        if os.path.isdir(internal) and internal not in sys.path:
            sys.path.insert(0, internal)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


# PyInstaller: run Gradio backend when invoked with --app-server.
if "--app-server" in sys.argv:
    _root = _bootstrap_sys_path()
    os.chdir(_root)
    from backend.dotenv_load import load_dotenv_safe

    load_dotenv_safe(os.path.join(_root, ".env"))
    load_dotenv_safe(os.path.join(_root, ".env.production"), override=False)
    from backend.asr_quality import apply_quality_profile

    apply_quality_profile()
    import app as _app_module

    _app_module.main()
    raise SystemExit(0)

from backend.paths import app_root

HERE = str(app_root())
VENV_PYTHON_WIN = os.path.join(HERE, "venv", "Scripts", "python.exe")
VENV_PYTHON_NIX = os.path.join(HERE, "venv", "bin", "python")
APP_PY = os.path.join(HERE, "app.py")
COMPOSE_GPU = os.path.join(HERE, "docker-compose.gpu.yml")
COMPOSE_OPENVINO = os.path.join(HERE, "docker-compose.openvino.yml")
COMPOSE_CPU = COMPOSE_OPENVINO
APP_URL = "http://127.0.0.1:7896"
READY_URL = f"{APP_URL}/startup-events"

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
    except (subprocess.TimeoutExpired, OSError):
        return False


def _gpu_in_docker() -> bool:
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:13.0.0-runtime-ubuntu24.04", "nvidia-smi"],
            capture_output=True, timeout=30, check=False,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
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
    """Show the app in a native pywebview desktop window."""
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


def _select_compose_file() -> str:
    """Pick Docker compose file: NVIDIA GPU, OpenVINO/CPU AI, or generic CPU."""
    if _gpu_in_docker():
        return COMPOSE_GPU
    if os.path.isfile(COMPOSE_OPENVINO):
        return COMPOSE_OPENVINO
    return COMPOSE_CPU


def _start_docker() -> subprocess.Popen | None:
    compose_file = _select_compose_file()
    print(f"[launcher] Docker mode: {os.path.basename(compose_file)}")
    try:
        proc = subprocess.Popen(
            ["docker", "compose", "-f", compose_file, "up", "-d"],
            cwd=HERE,
        )
        proc.wait()
        return None
    except OSError as exc:
        print(f"[launcher] docker compose failed: {exc}")
        return None


def _app_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("APP_FORCE_DIRECT", "1")
    env.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{HERE}{os.pathsep}{pp}" if pp else HERE
    # When bundled, prepend a local `bin/` directory so bundled native
    # executables (ffmpeg, etc.) are discoverable by subprocesses.
    bin_dir = os.path.join(HERE, "bin")
    if os.path.isdir(bin_dir):
        path = env.get("PATH", "")
        env["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
    return env


def _start_direct() -> subprocess.Popen:
    env = _app_env()
    is_frozen = bool(getattr(sys, "frozen", False))

    if is_frozen:
        print("[launcher] Direct mode: bundled Gradio backend")
        return subprocess.Popen(
            [sys.executable, "--app-server"],
            cwd=HERE,
            env=env,
        )

    python = _find_python()
    if not os.path.isfile(APP_PY):
        print(f"[launcher] ERROR: app.py not found at {APP_PY}")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"[launcher] Direct mode: {python} app.py")
    return subprocess.Popen([python, APP_PY], cwd=HERE, env=env)


def main() -> None:
    """Parse CLI flags and start the chosen backend, then open the desktop window."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--direct", action="store_true")
    args, _ = parser.parse_known_args()

    print("=" * 56)
    print("  Local Transcript App")
    print("=" * 56)

    is_frozen = bool(getattr(sys, "frozen", False))
    force_direct = is_frozen or os.getenv("APP_FORCE_DIRECT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    backend_proc: subprocess.Popen | None = None

    if args.docker and not force_direct:
        _start_docker()
    elif force_direct or args.direct or not _docker_available():
        if args.docker and force_direct:
            print("[launcher] --docker ignored: packaged/forced direct mode active.")
        backend_proc = _start_direct()
    else:
        _start_docker()

    _open_native_window()

    if backend_proc is not None:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend_proc.kill()


if __name__ == "__main__":
    main()
