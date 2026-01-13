# ZINB Graphical Model

A PyTorch/Pyro implementation of a Zero-Inflated Negative Binomial (ZINB) graphical model using pseudo-likelihood inference with NUTS/HMC.

## Features

- **Symmetric interaction matrix**: Parameterizes conditional dependencies via unconstrained matrix A mapped to symmetric Ω with unit diagonal
- **ZINB distribution**: Models count data with zero-inflation (μ, φ, π parameters)
- **Pseudo-likelihood inference**: Avoids partition functions and truncated sums
- **NUTS/HMC sampling**: Joint inference of all parameters using Pyro's NUTS kernel
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
