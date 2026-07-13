# Windows IIS reverse proxy for Local Transcript App

Upstream: GPU Docker UI at `http://127.0.0.1:7988`  
(`docker compose -f docker-compose.gpu.yml up -d`)

## Install features

1. **IIS** (Internet Information Services)
2. **URL Rewrite** module  
   https://www.iis.net/downloads/microsoft/url-rewrite
3. **Application Request Routing (ARR) 3.x**  
   https://www.iis.net/downloads/microsoft/application-request-routing
4. Enable **WebSocket Protocol**:  
   Server Manager → Add Roles → Web Server → Application Development → WebSocket Protocol

## Enable ARR proxy

1. IIS Manager → select the **server** node (not the site)
2. Open **Application Request Routing Cache**
3. Right-hand **Server Proxy Settings** → check **Enable proxy** → Apply
4. Optionally raise timeout (e.g. 3600s) for long transcriptions

## Site setup

1. Create a site (or use Default Web Site) with your public hostname
2. Bind HTTPS with your certificate
3. Place [`web.config`](web.config) in the site physical path (or merge the rewrite rule)
4. If ARR blocks server variables, unlock them:  
   IIS → URL Rewrite → View Server Variables → allow `HTTP_X_FORWARDED_PROTO`, `HTTP_X_FORWARDED_HOST`
5. Recycle the app pool

## Verify

```powershell
# App must answer locally first
Invoke-WebRequest http://127.0.0.1:7988/startup-events -UseBasicParsing

# Then browse https://your-hostname/
```

## Auth and subpaths

- Prefer Gradio auth: set `GRADIO_AUTH_USER` / `GRADIO_AUTH_PASSWORD` on the container
- For a virtual directory `/transcript`, also set `GRADIO_ROOT_PATH=/transcript` and adjust the rewrite rule accordingly

## Firewall

Allow inbound TCP **80** and **443** on the Windows host. Do not expose 7988 publicly if IIS terminates TLS.
