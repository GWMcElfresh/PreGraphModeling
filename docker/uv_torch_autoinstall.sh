#!/usr/bin/env sh
set -eu

# Runtime torch auto-installer for "single image" CPU-in-CI / GPU-at-runtime setups.
#
# Why this exists:
# - GitHub Actions runners often don't have enough disk for CUDA-enabled torch wheels.
# - GPU machines do, and we want the same image to "upgrade" itself when GPUs exist.
#
# Behavior:
# - If NVIDIA GPUs appear to be available inside the container AND the installed torch
#   is CPU-only (torch.version.cuda is None), we install a CUDA wheel via uv.
# - Otherwise we do nothing and run the given command.
#
# Controls:
# - PREGRAPHMODELING_TORCH_AUTO=0 disables auto-install.
# - PREGRAPHMODELING_TORCH_GPU_INDEX_URL selects the CUDA wheel index (default cu121).
# - PREGRAPHMODELING_UV_INDEX_STRATEGY controls uv's index strategy (default unsafe-best-match).
# - PREGRAPHMODELING_TORCH_GPU_PACKAGES controls which packages to upgrade
#   (default: "torch torchvision torchaudio").

AUTO="${PREGRAPHMODELING_TORCH_AUTO:-1}"
GPU_INDEX_URL="${PREGRAPHMODELING_TORCH_GPU_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
UV_INDEX_STRATEGY="${PREGRAPHMODELING_UV_INDEX_STRATEGY:-unsafe-best-match}"
GPU_PACKAGES="${PREGRAPHMODELING_TORCH_GPU_PACKAGES:-torch torchvision torchaudio}"

has_nvidia_gpu() {
  # These checks are intentionally lightweight and do not require nvidia-smi.
  # When running with NVIDIA Container Toolkit, /dev/nvidia* devices are typically present.
  [ -e /dev/nvidiactl ] || [ -e /dev/nvidia0 ] || [ -d /proc/driver/nvidia ]
}

torch_is_cpu_only() {
  # Prints "cpu" or "cuda". If torch isn't importable yet, treat as cpu-only.
  python - <<'PY' 2>/dev/null || echo cpu
try:
    import torch
    print('cuda' if getattr(torch.version, 'cuda', None) else 'cpu')
except Exception:
    print('cpu')
PY
}

maybe_install_cuda_torch() {
  if [ "$AUTO" = "0" ]; then
    return 0
  fi

  if ! has_nvidia_gpu; then
    return 0
  fi

  if [ "$(torch_is_cpu_only)" != "cpu" ]; then
    return 0
  fi

  echo "[entrypoint] NVIDIA GPU detected; upgrading torch to CUDA wheel from: $GPU_INDEX_URL" 1>&2

  # Install/upgrade torch (and optionally torchvision/torchaudio) from the CUDA wheel index.
  # This keeps the CI-built image small (CPU wheels), but enables GPU acceleration when
  # users run the container on a GPU machine with `--gpus all`.
  uv pip install --system \
    --index-url "$GPU_INDEX_URL" \
    --extra-index-url "https://pypi.org/simple" \
    --index-strategy "$UV_INDEX_STRATEGY" \
    --upgrade \
    $GPU_PACKAGES
}

maybe_install_cuda_torch

exec "$@"
