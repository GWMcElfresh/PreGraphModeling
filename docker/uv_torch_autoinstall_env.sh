#!/usr/bin/env sh
# Sourced by bash via BASH_ENV to enable torch auto-install even when
# container entrypoints are bypassed (e.g., `apptainer exec ... bash script.sh`).
#
# Intentionally avoids `set -e` / `set -u` to prevent changing caller semantics.

AUTO="${PREGRAPHMODELING_TORCH_AUTO:-1}"
GPU_INDEX_URL="${PREGRAPHMODELING_TORCH_GPU_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
UV_INDEX_STRATEGY="${PREGRAPHMODELING_UV_INDEX_STRATEGY:-unsafe-best-match}"
GPU_PACKAGES="${PREGRAPHMODELING_TORCH_GPU_PACKAGES:-torch torchvision torchaudio}"

has_nvidia_gpu() {
  [ -e /dev/nvidiactl ] || [ -e /dev/nvidia0 ] || [ -d /proc/driver/nvidia ]
}

torch_is_cpu_only() {
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

  echo "[torch-autoinstall] NVIDIA GPU detected; upgrading torch to CUDA wheel from: $GPU_INDEX_URL" 1>&2

  uv pip install --system \
    --index-url "$GPU_INDEX_URL" \
    --extra-index-url "https://pypi.org/simple" \
    --index-strategy "$UV_INDEX_STRATEGY" \
    --upgrade \
    $GPU_PACKAGES
}

maybe_install_cuda_torch
