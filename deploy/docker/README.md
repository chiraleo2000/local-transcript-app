# Docker stacks (reorganized)

| Stack | Path | Host CUDA | UI |
|-------|------|-----------|-----|
| **latest** (recommended) | [`latest/`](latest/) | Driver CUDA >= 12.4; image CUDA **13.3** + cu130 | :7988 |
| **cuda126** | [`cuda126/`](cuda126/) | CUDA **12.6+** mid path + cu126 | :7988 |
| **cuda124** | [`cuda124/`](cuda124/) | Minimum CUDA **12.4** + cu124 | :7988 |
| **openvino** | [`openvino/`](openvino/) | No NVIDIA (Intel/AMD/ARM CPU/iGPU) | :7987 |

## Deploy

From **repo root**:

```bat
Deploy-Docker.bat gpu
Deploy-Docker.bat gpu -CudaStack cuda126 -Build
Deploy-Docker.bat gpu -CudaStack cuda124 -Build
Deploy-Docker.bat openvino -Build
```

Or:

```powershell
docker compose -f deploy/docker/latest/compose.yml up -d --build
docker compose -f deploy/docker/cuda126/compose.yml up -d --build
docker compose -f deploy/docker/cuda124/compose.yml up -d --build
docker compose -f deploy/docker/openvino/compose.yml up -d --build
```

Shared GPU diarization/ASR policy: [`gpu-app.env`](gpu-app.env)  
OpenVINO policy: [`openvino-app.env`](openvino-app.env)

Root `docker-compose.gpu.yml` / `Dockerfile*` are compatibility shims → these folders.
