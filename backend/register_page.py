"""Public HTML registration page (reachable before Gradio login)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

_REGISTER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Create account — Local Transcript</title>
  <style>
    :root {
      --bg0: #0f1419;
      --bg1: #1a2332;
      --ink: #e8eef6;
      --muted: #9aabbd;
      --accent: #3d9cf0;
      --accent2: #2a7fc4;
      --err: #f07178;
      --ok: #7fd99a;
      --border: #2c3a4d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #243b2e 0%, transparent 50%),
        linear-gradient(160deg, var(--bg0), var(--bg1));
      display: grid; place-items: center; padding: 1.5rem;
    }
    main {
      width: min(420px, 100%);
      background: rgba(15, 20, 25, 0.82);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.75rem 1.5rem 1.5rem;
      box-shadow: 0 18px 50px rgba(0,0,0,0.35);
      backdrop-filter: blur(8px);
    }
    h1 { margin: 0 0 0.35rem; font-size: 1.45rem; font-weight: 650; }
    p.lead { margin: 0 0 1.25rem; color: var(--muted); font-size: 0.95rem; line-height: 1.45; }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin: 0.85rem 0 0.35rem; }
    input {
      width: 100%; padding: 0.7rem 0.8rem; border-radius: 8px;
      border: 1px solid var(--border); background: #0c1118; color: var(--ink);
      font-size: 1rem;
    }
    input:focus { outline: 2px solid rgba(61,156,240,0.45); border-color: var(--accent); }
    button {
      margin-top: 1.25rem; width: 100%; padding: 0.8rem;
      border: 0; border-radius: 8px; cursor: pointer;
      background: linear-gradient(180deg, var(--accent), var(--accent2));
      color: #fff; font-weight: 600; font-size: 1rem;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .msg { margin-top: 0.9rem; min-height: 1.25rem; font-size: 0.9rem; }
    .msg.err { color: var(--err); }
    .msg.ok { color: var(--ok); }
    .foot { margin-top: 1.2rem; font-size: 0.9rem; color: var(--muted); }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <main>
    <h1>Create account</h1>
    <p class="lead">Register once, then sign in on the Gradio login screen to transcribe. Jobs keep running after you close the browser.</p>
    <form id="reg" autocomplete="on">
      <label for="username">Username</label>
      <input id="username" name="username" required minlength="3" maxlength="64"
             pattern="[A-Za-z0-9_.\\-]{3,64}" placeholder="e.g. alex" />
      <label for="password">Password (min 6 characters)</label>
      <input id="password" name="password" type="password" required minlength="6" />
      <label for="password2">Confirm password</label>
      <input id="password2" name="password2" type="password" required minlength="6" />
      <button type="submit" id="btn">Create account</button>
      <div class="msg" id="msg" aria-live="polite"></div>
    </form>
    <p class="foot">Already registered? <a href="/">Go to login</a></p>
  </main>
  <script>
    const form = document.getElementById("reg");
    const msg = document.getElementById("msg");
    const btn = document.getElementById("btn");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      msg.className = "msg";
      msg.textContent = "";
      const username = document.getElementById("username").value.trim();
      const password = document.getElementById("password").value;
      const password2 = document.getElementById("password2").value;
      if (password !== password2) {
        msg.className = "msg err";
        msg.textContent = "Passwords do not match.";
        return;
      }
      btn.disabled = true;
      try {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          msg.className = "msg err";
          msg.textContent = data.error || ("Registration failed (" + res.status + ")");
          btn.disabled = false;
          return;
        }
        msg.className = "msg ok";
        msg.textContent = "Account created. Redirecting to login…";
        setTimeout(() => { window.location.href = "/"; }, 900);
      } catch (err) {
        msg.className = "msg err";
        msg.textContent = "Network error — is the server running?";
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def register_page(_request: Request) -> HTMLResponse:
    return HTMLResponse(_REGISTER_HTML)


def build_register_routes() -> list[Route]:
    return [
        Route("/register", register_page, methods=["GET"]),
        Route("/register/", register_page, methods=["GET"]),
    ]
