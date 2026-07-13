# Public / LAN deployment (Windows + Docker GPU)

Primary path on this stack: **NVIDIA GPU Docker** on port **7988**.

```powershell
docker compose -f docker-compose.gpu.yml up -d --build
# App: http://localhost:7988
```

Put **nginx** or **Windows IIS** in front for LAN/public HTTPS. Upstream is always `http://127.0.0.1:7988`.

`models/ov_cache` is for **OpenVINO only** — not required for GPU Docker.

Keep `UI_MAX_CONCURRENT_JOBS=1` on 8 GB VRAM. Up to **4 users** can submit via `UI_GRADIO_TRANSCRIBE_CONCURRENCY=4`; they **queue** (true 2–4 parallel GPU jobs will OOM). History is filtered by client IP (`UI_HISTORY_PER_CLIENT_IP`). Cancel frees GPU cache for the next queued job.

---

## Option A — nginx

1. Install nginx for Windows (or WSL).
2. Copy [`nginx/local-transcript.conf`](nginx/local-transcript.conf) into your nginx `conf.d` (or `include` it from `nginx.conf`).
3. Edit `server_name` and TLS certificate paths.
4. Optional basic auth: set `GRADIO_AUTH_USER` / `GRADIO_AUTH_PASSWORD` on the container, **or** use nginx `auth_basic`.
5. Reload nginx, open Windows Firewall for 80/443.

Subpath example: set `GRADIO_ROOT_PATH=/transcript` in compose/.env and use the `/transcript/` location block in the sample config.

---

## Option B — Windows IIS + ARR

1. Install **IIS**, **URL Rewrite**, and **Application Request Routing (ARR)**.
2. In ARR → Server Proxy Settings → enable proxy.
3. Enable **WebSocket Protocol** for the site (IIS feature).
4. Create a site (or application) and drop [`iis/web.config`](iis/web.config) at the site root (or merge rules).
5. Bind HTTPS with your certificate; open Firewall 80/443.
6. Confirm `http://127.0.0.1:7988` works before testing the IIS hostname.

Details: [`iis/README.md`](iis/README.md).

---

## Optional Gradio env (compose or `.env`)

| Variable | Purpose |
|----------|---------|
| `GRADIO_ROOT_PATH` | Subpath (e.g. `/transcript`) |
| `GRADIO_AUTH_USER` / `GRADIO_AUTH_PASSWORD` | Gradio basic auth |
| `GRADIO_MAX_FILE_SIZE` | Upload limit (default `512mb`) |
| `APP_PUBLIC_BASE_URL` | Logged public URL hint |
| `UI_GRADIO_TRANSCRIBE_CONCURRENCY` | Queued handlers (default **4**) |
| `UI_MAX_CONCURRENT_JOBS` | GPU slots (keep **1** on 8 GB) |
| `UI_HISTORY_PER_CLIENT_IP` | Per-user history via IP |
| `UI_CANCEL_FREES_GPU_FOR_QUEUE` | Cancel clears cache for next job |

Example:

```powershell
$env:GRADIO_AUTH_USER = "admin"
$env:GRADIO_AUTH_PASSWORD = "change-me"
$env:APP_PUBLIC_BASE_URL = "https://transcript.example.com"
docker compose -f docker-compose.gpu.yml up -d
```
