# ZINB Graphical Model

A PyTorch/Pyro implementation of a Zero-Inflated Negative Binomial (ZINB) graphical model using pseudo-likelihood inference with NUTS/HMC or SVI.

## Features

- **Symmetric interaction matrix**: Parameterizes conditional dependencies via unconstrained matrix A mapped to symmetric Ω with unit diagonal
- **ZINB distribution**: Models count data with zero-inflation (μ, φ, π parameters)
- **Pseudo-likelihood inference**: Avoids partition functions and truncated sums
- **NUTS/HMC sampling**: Joint inference of all parameters using Pyro's NUTS kernel
- **SVI with minibatching**: Stochastic variational inference for large datasets
- **GPU support**: Runs on CUDA when available

## Installation

```bash
cd python
pip install -e .
```

## Usage

### Load count matrix from disk

```python
from zinb_graphical_model import load_count_matrix

# Supports .npy, .npz, .csv, .tsv formats
X = load_count_matrix("counts.npy", device="cuda")
```

### Create and run inference

```python
from zinb_graphical_model import ZINBPseudoLikelihoodGraphicalModel, run_inference

# Initialize model
n_features = X.shape[1]
model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cuda")

# Run NUTS inference
results = run_inference(
    model,
    X,
    num_samples=1000,
    warmup_steps=500,
    num_chains=1,
    target_accept_prob=0.8,
)

# Access posterior samples
A_tril_samples = results["samples"]["A_tril"]  # Unconstrained interaction parameters
omega_samples = results["omega_samples"]       # Interaction matrix
mu_samples = results["samples"]["mu"]          # ZINB mean parameters
phi_samples = results["samples"]["phi"]        # ZINB dispersion parameters
pi_samples = results["samples"]["pi_zero"]     # Zero-inflation probabilities

# Summary statistics
summary = results["summary"]
print(f"Omega posterior mean:\n{summary['omega']['mean']}")
print(f"Mu posterior mean: {summary['mu']['mean']}")
```

### Run SVI inference (for large datasets)

For large datasets that don't fit in GPU memory with NUTS, use SVI with minibatching:

```python
from zinb_graphical_model import ZINBPseudoLikelihoodGraphicalModel, run_svi_inference

# Initialize model
n_features = X.shape[1]
model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cuda")

# Run SVI with minibatching - keep X on CPU, batches moved to GPU automatically
results = run_svi_inference(
    model,
    X.cpu(),  # Keep full data on CPU
    batch_size=256,
    num_epochs=200,
    learning_rate=1e-2,
    num_posterior_samples=200,
)

# Access posterior samples (same interface as NUTS)
omega_samples = results["omega_samples"]
losses = results["losses"]  # Training loss curve
```

## Model Details

### Interaction Matrix

The interaction matrix Ω is parameterized via an unconstrained lower-triangular matrix A:
- Off-diagonal: Ω_ij = A_ij for i ≠ j (direct mapping)
- Diagonal: Ω_ii = 1 (fixed for identifiability)

This ensures Ω is symmetric with a unit diagonal.

### Pseudo-Likelihood

Instead of computing the intractable partition function, we use pseudo-likelihood:

P(X) ≈ ∏_j P(X_j | X_{-j})

where each conditional is a ZINB distribution with mean adjusted by the interaction effects from other features.

### Priors

- A_tril ~ Normal(0, 1) - Unconstrained interaction parameters (lower triangular)
- μ ~ LogNormal(0, 1) - Mean parameter for each feature
- φ ~ LogNormal(0, 1) - Dispersion parameter for each feature
- π ~ Beta(1, 1) - Zero-inflation probability for each feature

## Requirements

- Python >= 3.9
- PyTorch >= 2.6.0
- Pyro-PPL >= 1.8.0
- NumPy >= 1.20.0

## Testing

```bash
cd python
pytest tests/ -v
```

To skip slow inference tests:

```bash
pytest tests/ -v -m "not slow"
```

## Docker Usage

The package includes a multi-stage Dockerfile at `/Dockerfile.python` for containerized deployment.

### Building the Image

```bash
# Build with CPU-only PyTorch (default, smaller image for CI)
docker build -f Dockerfile.python -t zinb-graphical-model .

# Build with CUDA support baked in (larger image)
docker build -f Dockerfile.python \
  --build-arg UV_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t zinb-graphical-model:cuda .
```

### Running with Docker

```bash
# Run tests
docker run --rm zinb-graphical-model

# Run with GPU support (auto-upgrades torch to CUDA wheels)
docker run --rm --gpus all zinb-graphical-model

# Run inference on your data
docker run --rm --gpus all \
  -v "$PWD/data:/data" \
  zinb-graphical-model \
  python fit_zinb_graphical_model.py \
    --counts /data/counts.csv \
    --out /data/outputs \
    --method svi \
    --batch-size 256
```

## Apptainer/Singularity Usage (HPC)

For HPC clusters using Apptainer (formerly Singularity), the `docker/run_zinb_apptainer_uv.sh` script handles:

- Creating a writable venv on bind-mounted storage (container filesystem is read-only)
- Auto-detecting GPUs and upgrading torch to CUDA wheels
- Managing UV cache in a writable location

### Converting Docker Image to SIF

```bash
# Build and export Docker image, then convert
docker build -f Dockerfile.python -t zinb-graphical-model .
docker save zinb-graphical-model -o zinb-graphical-model.tar
apptainer build zinb-graphical-model.sif docker-archive://zinb-graphical-model.tar
```

### Running on HPC

```bash
# Bind your working directory to /work inside the container
apptainer exec --nv -B "$PWD":/work \
  zinb-graphical-model.sif \
  bash /docker/run_zinb_apptainer_uv.sh \
    --counts /work/data/counts.csv \
    --method svi \
    --batch-size 512 \
    --num-epochs 200 \
    --device cuda
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VENV_DIR` | `/work/.venv` | Location for the writable venv |
| `UV_CACHE_DIR` | `/work/.uv-cache` | UV package cache location |
| `PREGRAPHMODELING_CLEAN_VENV` | `1` | Remove venv each run (avoids stale state) |
| `PREGRAPHMODELING_TORCH_GPU_INDEX_URL` | `cu121` | CUDA wheel index for GPU upgrade |
| `PREGRAPHMODELING_TORCH_AUTO` | `0` | Enable auto torch upgrade in entrypoint |
