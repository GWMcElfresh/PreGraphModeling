#!/usr/bin/env bash
set -euo pipefail

# Apptainer/Singularity-friendly runner for the Python ZINB graphical model.
#
# Key facts about this repo's Dockerfile.python:
# - Dependencies are installed into the *system* Python in the image via:
#     uv pip install --system -r /app/requirements.txt
#   and then:
#     uv pip install --system -e /app --no-deps
# - The code lives at /app inside the container.
#
# This script intentionally creates a *writable* venv in your current directory
# (e.g. on a bind-mounted host path), installs requirements into it, then runs.
#
# Why create a venv at all?
# - In Apptainer, the container filesystem is often read-only.
# - A per-run venv in $PWD avoids writing into the image.
#
# Usage example (recommended):
#   apptainer exec --nv -B "$PWD":/work <image.sif> bash docker/run_zinb_apptainer_uv.sh \
#     --counts /work/python/synthetic_counts.csv \
#     --method mcmc --num-chains 16 --num-samples 1000 --warmup-steps 400 \
#     --device cuda --batch-size 512

VENV_DIR="${VENV_DIR:-.venv}"
PROJECT_DIR="${PROJECT_DIR:-}"

# If you set UV_PYTHON=3.12, uv will try to use that interpreter.
# IMPORTANT: Dockerfile.python currently uses python:3.11-slim by default,
# so python 3.12 may not be available inside the container unless you rebuild
# with `--build-arg BASE_IMAGE=python:3.12-slim`.
UV_PYTHON="${UV_PYTHON:-}"

# Torch CUDA wheel index used when GPUs exist.
GPU_INDEX_URL="${PREGRAPHMODELING_TORCH_GPU_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
UV_INDEX_STRATEGY="${PREGRAPHMODELING_UV_INDEX_STRATEGY:-unsafe-best-match}"
GPU_PACKAGES="${PREGRAPHMODELING_TORCH_GPU_PACKAGES:-torch torchvision torchaudio}"

log() { printf '%s\n' "$*" 1>&2; }

die() {
  log "[run_zinb] ERROR: $*"
  exit 2
}

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

maybe_upgrade_torch_to_cuda() {
  # Upgrade torch *inside the venv* if a GPU is available and torch is CPU-only.
  if ! has_nvidia_gpu; then
    return 0
  fi

  if [ "$(torch_is_cpu_only)" != "cpu" ]; then
    return 0
  fi

  log "[run_zinb] NVIDIA GPU detected; upgrading torch in venv from: $GPU_INDEX_URL"
  uv pip install \
    --index-url "$GPU_INDEX_URL" \
    --extra-index-url "https://pypi.org/simple" \
    --index-strategy "$UV_INDEX_STRATEGY" \
    --upgrade \
    $GPU_PACKAGES
}

resolve_project_dir() {
  if [ -n "$PROJECT_DIR" ]; then
    echo "$PROJECT_DIR"
    return 0
  fi

  # Prefer bind-mounted repo layout if present
  if [ -f "./python/fit_zinb_graphical_model.py" ]; then
    echo "./python"
    return 0
  fi

  # Otherwise fall back to the in-image copy
  if [ -f "/app/fit_zinb_graphical_model.py" ]; then
    echo "/app"
    return 0
  fi

  die "Could not find fit_zinb_graphical_model.py (looked in ./python and /app). Set PROJECT_DIR explicitly."
}

ensure_uv_present() {
  command -v uv >/dev/null 2>&1 || die "uv not found in PATH inside container."
}

ensure_venv() {
  if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
    return 0
  fi

  ensure_uv_present

  if [ -n "$UV_PYTHON" ]; then
    log "[run_zinb] Creating venv at $VENV_DIR with Python: $UV_PYTHON"
    uv venv --python "$UV_PYTHON" "$VENV_DIR"
  else
    log "[run_zinb] Creating venv at $VENV_DIR (default interpreter)"
    uv venv "$VENV_DIR"
  fi
}

install_deps_once() {
  local proj_dir="$1"
  local marker="$VENV_DIR/.pregraphmodeling_deps_installed"

  if [ -f "$marker" ]; then
    return 0
  fi

  if [ ! -f "$proj_dir/requirements.txt" ]; then
    die "requirements.txt not found at $proj_dir/requirements.txt"
  fi

  log "[run_zinb] Installing requirements into venv from $proj_dir/requirements.txt"
  uv pip install -r "$proj_dir/requirements.txt"

  # Install the local package in editable mode if possible.
  if [ -f "$proj_dir/pyproject.toml" ]; then
    log "[run_zinb] Installing project editable ($proj_dir)"
    uv pip install -e "$proj_dir" --no-deps
  fi

  touch "$marker"
}

main() {
  local proj_dir
  proj_dir="$(resolve_project_dir)"

  ensure_venv

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  # Re-install only once per venv
  install_deps_once "$proj_dir"

  # If we're on a GPU host, ensure venv torch is CUDA-enabled.
  maybe_upgrade_torch_to_cuda

  # Run from the project directory so relative paths work naturally.
  cd "$proj_dir"

  log "[run_zinb] Running: python fit_zinb_graphical_model.py $*"
  uv run --active python fit_zinb_graphical_model.py "$@"
}

main "$@"
