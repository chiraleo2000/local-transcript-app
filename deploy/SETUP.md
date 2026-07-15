# Setup guide: public hostname `asrservice.demotoday.th`

Step-by-step instructions for Docker, nginx, Cloudflare Tunnel, DNS, auth, and firewall.
Hostname stays the same at home or when you travel. Deploy scripts do **not** change WiFi adapter, DNS, gateway, or routes on this PC.

---

## 1. Requirements (before anything else)

| Requirement | Why |
|-------------|-----|
| Windows 10/11 + **Docker Desktop** with **NVIDIA GPU** | Runs the Gradio app (`docker-compose.gpu.yml`) |
| Models under `./models/` and a working `.env` | Offline ASR/diarization |
| Domain **demotoday.th** DNS you can edit | Point `asrservice.demotoday.th` at Cloudflare or your home WAN IP |
| Cloudflare account (free is enough) | Travel mode tunnel + HTTPS |
| (Home mode only) Router admin access | Port-forward 80/443 |

**Choose a mode:**

| Mode | When | Needs router port-forward? | Same name while traveling? |
|------|------|----------------------------|----------------------------|
| **A — Travel (Cloudflare Tunnel)** | Hotels, cafes, new offices | No | Yes |
| **B — Home (nginx)** | You control one router | Yes | No (breaks when you move) |
| **C — LAN only** | Same WiFi/LAN, no internet | No | N/A |

For a laptop that moves, use **Mode A**. Mode B is optional for a fixed desk.

Shared URL for A/B: **https://asrservice.demotoday.th**

---

## 2. Prepare the app (both modes)

From the repo root:

```powershell
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
copy .env.example .env   # only if .env does not exist yet
```

Edit `.env` and set at least:

```env
APP_PUBLIC_BASE_URL=https://asrservice.demotoday.th
GRADIO_AUTH_USER=admin
GRADIO_AUTH_PASSWORD=choose-a-strong-password
```

Optional later (travel):

```env
CLOUDFLARE_TUNNEL_TOKEN=   # paste after Cloudflare setup below
```

Confirm GPU Docker works on **loopback only** (WiFi-safe):

```powershell
docker compose -f docker-compose.gpu.yml -f deploy/docker-compose.proxy-override.yml up -d
# Wait until healthy, then:
Invoke-WebRequest http://127.0.0.1:7988/startup-events -UseBasicParsing
docker port transcription-service
# Expect: 7896/tcp -> 127.0.0.1:7988
# Must NOT show 0.0.0.0:7988
```

Do **not** publish `7988` on all interfaces; that would expose Gradio on WiFi.

---

## 3. Mode A — Cloudflare Tunnel (travel / any WiFi)

### 3.1 Why this works

Browsers hit Cloudflare → Cloudflare reaches your PC through an **outbound** tunnel started by `cloudflared`.  
Hotels do not need port-forward. Your public IP can change; the hostname stays `asrservice.demotoday.th`.

```text
Internet users
    -> https://asrservice.demotoday.th  (Cloudflare HTTPS)
    -> cloudflared container on this PC
    -> http://host.docker.internal:7988
    -> Docker Gradio (127.0.0.1:7988)
```

### 3.2 Add the domain to Cloudflare (one-time)

1. Sign in at [https://dash.cloudflare.com/](https://dash.cloudflare.com/)
2. **Add a site** → enter `demotoday.th` (if not already there)
3. Choose a plan (Free is fine)
4. Cloudflare shows nameservers (e.g. `ada.ns.cloudflare.com`). At your domain registrar, set those NS records and wait until the domain status is **Active**

### 3.3 Create a tunnel (one-time)

1. Open [Cloudflare Zero Trust](https://one.dash.cloudflare.com/)
2. Go to **Networks** → **Tunnels** → **Create a tunnel**
3. Select **Cloudflared** → name it e.g. `local-transcript` → **Save tunnel**
4. On the install page, copy the **token** (long string starting with `eyJ...`)
5. Paste into project `.env`:

```env
CLOUDFLARE_TUNNEL_TOKEN=eyJ...
APP_PUBLIC_BASE_URL=https://asrservice.demotoday.th
```

### 3.4 Public hostname on the tunnel (one-time)

Still in the tunnel wizard / **Published application routes**:

| Field | Value |
|-------|--------|
| Subdomain / hostname | `asrservice` (or full `asrservice.demotoday.th`) |
| Domain | `demotoday.th` |
| Type / Service | **HTTP** |
| URL | `http://host.docker.internal:7988` |

Save. Cloudflare usually creates a **proxied CNAME** for you:

- `asrservice.demotoday.th` → `<tunnel-id>.cfargotunnel.com` (**Proxied** / orange cloud)

If DNS was not auto-created, add it under **DNS** → **Records**:

| Type | Name | Target | Proxy |
|------|------|--------|-------|
| CNAME | `asrservice` | `<tunnel-id>.cfargotunnel.com` | Proxied (orange) |

### 3.5 Start on this PC (every location)

Connect to WiFi (complete any captive portal login first), then:

```powershell
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
.\deploy\scripts\Setup-TravelTunnel.ps1
```

What this starts:

- GPU app: `docker-compose.gpu.yml` + [`docker-compose.proxy-override.yml`](docker-compose.proxy-override.yml)
- Tunnel: [`docker-compose.tunnel.yml`](docker-compose.tunnel.yml) (`cloudflare/cloudflared`)

Manual equivalent:

```powershell
docker compose -f docker-compose.gpu.yml -f deploy/docker-compose.proxy-override.yml -f deploy/docker-compose.tunnel.yml up -d
```

### 3.6 Verify travel mode

```powershell
docker ps --filter "name=transcription-service" --filter "name=local-transcript-tunnel"
# From any browser (phone data, another PC):
#   https://asrservice.demotoday.th
# Login with GRADIO_AUTH_USER / GRADIO_AUTH_PASSWORD
```

Check tunnel logs if the page fails:

```powershell
docker logs local-transcript-tunnel --tail 50
```

### 3.7 Travel checklist (each new place)

1. Join WiFi / finish guest login  
2. Start Docker Desktop  
3. Run `.\deploy\scripts\Setup-TravelTunnel.ps1`  
4. Open `https://asrservice.demotoday.th`  
5. No router port-forward, no WiFi IP change on the PC  

---

## 4. Mode B — nginx + router (fixed home/office only)

Use when you **control the router** and the PC stays on that network. Not suitable for travel.

### 4.1 Architecture

```text
Internet users
    -> router WAN (DNS A record)
    -> port-forward 80/443 to PC WiFi/LAN IP
    -> nginx on PC (:80/:443)
    -> http://127.0.0.1:7988 Docker Gradio
```

### 4.2 DNS for home mode (one-time per location)

In Cloudflare DNS (or your DNS host), for a **fixed** home WAN IP:

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| A | `asrservice` | your home public WAN IP | DNS only (grey) **or** Proxied if you terminate TLS at Cloudflare |

If you use this script’s self-signed nginx TLS, prefer **DNS only** (grey cloud) so browsers talk to your nginx cert, or put Cloudflare in Flexible/Full carefully. Simplest for the repo script: grey cloud → your WAN IP.

Find WAN IP: open [https://ifconfig.me](https://ifconfig.me) from the home PC.

### 4.3 Run nginx setup on the PC

```powershell
cd C:\Users\chira\Documents\visualstudiocode\Work\local-transcript-app
.\deploy\scripts\Setup-PublicAccess.ps1 -Proxy nginx -Audience internet -PublicHost asrservice.demotoday.th -EnableTls
```

This will:

1. Write auth + `APP_PUBLIC_BASE_URL` into `.env`  
2. Start Docker on `127.0.0.1:7988` only  
3. Install/start nginx (repo prefix `deploy/nginx/runtime`) using [`nginx/asrservice.demotoday.th.conf`](nginx/asrservice.demotoday.th.conf)  
4. Create self-signed TLS under `deploy/nginx/certs/` (copied to `deploy/nginx/runtime/certs/`)  
5. Try to open Windows Firewall for TCP **80** and **443** only  

If firewall needs admin:

```powershell
# Run PowerShell as Administrator
.\deploy\scripts\Open-PublicFirewall.ps1
```

### 4.4 Router port-forward (home)

In the router admin UI:

| External port | Internal IP | Internal port | Protocol |
|---------------|-------------|---------------|----------|
| 80 | PC WiFi IP (e.g. `172.16.100.110`) | 80 | TCP |
| 443 | same | 443 | TCP |

Tips:

- Set a **DHCP reservation** for this PC so the internal IP does not change  
- Do **not** forward port **7988**  
- Re-check the PC WiFi IP after reconnect: `Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi"`

### 4.5 What nginx is doing

Config: [`deploy/nginx/asrservice.demotoday.th.conf`](nginx/asrservice.demotoday.th.conf)

- `:80` → redirect to HTTPS (ACME path reserved for later real certs)  
- `:443` → proxy to `127.0.0.1:7988` with WebSockets, 512 MB uploads, 3600s timeouts  
- TLS files: `deploy/nginx/runtime/certs/fullchain.pem` + `privkey.pem`  

Self-signed certs show a browser warning until you replace them with Let’s Encrypt / a Cloudflare Origin cert.

### 4.6 Verify home mode

```powershell
# Local upstream
Invoke-WebRequest http://127.0.0.1:7988/startup-events -UseBasicParsing

# Local HTTPS via nginx (Host header)
# Browser: https://asrservice.demotoday.th  (after DNS + forward)
```

Credentials: `deploy/.public-credentials.txt` (gitignored).

### 4.7 When you leave home

Stop relying on Mode B. Switch to **Mode A (tunnel)**:

1. In Cloudflare DNS, change `asrservice` from A→WAN to **CNAME→tunnel** (proxied), or use the hostname route already on the tunnel  
2. Run `.\deploy\scripts\Setup-TravelTunnel.ps1`  
3. You can stop nginx if you want: `Get-Process nginx | Stop-Process`

Do not try to reconfigure every hotel router.

---

## 5. Mode C — LAN only (no public internet)

```powershell
.\deploy\scripts\Setup-PublicAccess.ps1 -Audience lan
```

Others on the same WiFi use `http://<your-lan-ip>/` (script prints the IP). No Cloudflare required.

---

## 6. Auth and security checklist

| Item | Action |
|------|--------|
| Gradio login | Always set `GRADIO_AUTH_USER` / `GRADIO_AUTH_PASSWORD` in `.env` |
| Upstream port | Keep `127.0.0.1:7988` only ([`docker-compose.proxy-override.yml`](docker-compose.proxy-override.yml)) |
| Firewall | Allow 80/443 only in home mode; never open 7988 |
| Secrets | Do not commit `.env`, `deploy/.public-credentials.txt`, or `deploy/nginx/certs/` |
| Tunnel token | Treat `CLOUDFLARE_TUNNEL_TOKEN` like a password |

---

## 7. Common commands

```powershell
# Status
docker ps --filter "name=transcription" --filter "name=local-transcript-tunnel"
Get-Process nginx -ErrorAction SilentlyContinue

# Logs
docker compose -f docker-compose.gpu.yml logs -f --tail 100
docker logs local-transcript-tunnel -f

# Stop travel tunnel only
docker stop local-transcript-tunnel

# Stop GPU app
docker compose -f docker-compose.gpu.yml -f deploy/docker-compose.proxy-override.yml stop

# Restart travel stack
.\deploy\scripts\Setup-TravelTunnel.ps1
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Page never loads while traveling | Tunnel not running / captive portal | Finish WiFi login; run `Setup-TravelTunnel.ps1`; check `docker logs local-transcript-tunnel` |
| `CLOUDFLARE_TUNNEL_TOKEN is missing` | Token not in `.env` | Complete §3.3 |
| 502 from Cloudflare | App not on 7988 / wrong tunnel URL | Ensure `docker port` shows `127.0.0.1:7988`; service URL must be `http://host.docker.internal:7988` |
| Works at home, dead at hotel | Still using nginx + A record | Switch to tunnel (Mode A) |
| Browser SSL warning (home nginx) | Self-signed cert | Expected until real cert; or use Cloudflare Tunnel (Cloudflare terminates HTTPS) |
| WiFi broke after deploy | Something else changed adapters | These scripts must not edit WiFi; restore DHCP on Wi-Fi adapter if needed |
| Auth prompt missing | Empty `GRADIO_AUTH_*` | Set user/password in `.env` and recreate container |

---

## 9. File map

| Path | Role |
|------|------|
| [`../Deploy-Docker.bat`](../Deploy-Docker.bat) | One-click GPU/OpenVINO Docker deploy |
| [`scripts/Deploy-Docker.ps1`](scripts/Deploy-Docker.ps1) | Same deploy logic (PowerShell) |
| [`docker-compose.gpu.yml`](../docker-compose.gpu.yml) | GPU app image/service |
| [`docker-compose.openvino.yml`](../docker-compose.openvino.yml) | OpenVINO / CPU app image/service |
| [`docker-compose.proxy-override.yml`](docker-compose.proxy-override.yml) | Bind `127.0.0.1:7988` only |
| [`docker-compose.tunnel.yml`](docker-compose.tunnel.yml) | `cloudflared` travel tunnel |
| [`nginx/asrservice.demotoday.th.conf`](nginx/asrservice.demotoday.th.conf) | Home nginx vhost |
| [`scripts/Setup-TravelTunnel.ps1`](scripts/Setup-TravelTunnel.ps1) | Travel one-command start |
| [`scripts/Setup-PublicAccess.ps1`](scripts/Setup-PublicAccess.ps1) | Home nginx + auth + firewall |
| [`scripts/Open-PublicFirewall.ps1`](scripts/Open-PublicFirewall.ps1) | Admin: allow 80/443 only |
| `.env` | Auth, public URL, `DEPLOY_BACKEND`, tunnel token |

---

## 10. Recommended path for this project

1. Complete **§2** (app + auth on loopback)  
2. Complete **§3** Cloudflare Tunnel once  
3. Day to day / travel: `.\deploy\scripts\Setup-TravelTunnel.ps1`  
4. Optional fixed desk: **§4** nginx + router  

Public name always: **https://asrservice.demotoday.th**
