# NVIDIA Docker stacks (canonical: deploy/docker/)
#
# | Stack   | Image CUDA | PyTorch | Compose |
# |---------|------------|---------|---------|
# | latest  | 13.3       | cu130   | deploy/docker/latest/compose.yml |
# | cuda126 | 12.6.3     | cu126   | deploy/docker/cuda126/compose.yml |
# | cuda124 | 12.4.1     | cu124   | deploy/docker/cuda124/compose.yml |
# | openvino| n/a        | cpu     | deploy/docker/openvino/compose.yml |
#
# Host driver must support at least CUDA 12.4 for any NVIDIA stack.
# Check: nvidia-smi
#
# Deploy:
#   Deploy-Docker.bat gpu -Build
#   Deploy-Docker.bat gpu -CudaStack cuda126 -Build
#   Deploy-Docker.bat gpu -CudaStack cuda124 -Build
#   Deploy-Docker.bat openvino -Build
#
# Shared GPU diarization/ASR: deploy/docker/gpu-app.env
