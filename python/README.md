# ZINB Graphical Model

A PyTorch/Pyro implementation of a Zero-Inflated Negative Binomial (ZINB) graphical model using pseudo-likelihood inference with NUTS/HMC.

## Features

- **Learnable transform**: Applies F(X) = atan(X)^θ to stabilize dependence
- **Symmetric negative interactions**: Parameterizes interactions via unconstrained matrix A mapped to negative, symmetric Ω using -|atan(A)|^α
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
from zinb_graphical_model import ZINBGraphicalModel, run_inference

# Initialize model
n_features = X.shape[1]
model = ZINBGraphicalModel(n_features=n_features, device="cuda")

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
theta_samples = results["samples"]["theta"]  # Transform parameter
omega_samples = results["omega_samples"]     # Interaction matrix
mu_samples = results["samples"]["mu"]        # ZINB mean parameters
phi_samples = results["samples"]["phi"]      # ZINB dispersion parameters
pi_samples = results["samples"]["pi_zero"]   # Zero-inflation probabilities

# Summary statistics
summary = results["summary"]
print(f"Theta posterior mean: {summary['theta']['mean']:.3f}")
print(f"Omega posterior mean:\n{summary['omega']['mean']}")
```

## Model Details

### Transform Function

The learnable transform F(X) = atan(X)^θ where θ > 0 is inferred from the data. This transform stabilizes dependence in the count data.

### Interaction Matrix

The interaction matrix Ω is parameterized via an unconstrained matrix A:
- Off-diagonal: Ω_ij = -|atan(A_ij)|^α for i ≠ j (negative interactions)
- Diagonal: Ω_ii = 1 (fixed for identifiability)

This ensures Ω is symmetric and negative off-diagonal.

### Pseudo-Likelihood

Instead of computing the intractable partition function, we use pseudo-likelihood:

P(X) ≈ ∏_j P(X_j | X_{-j})

where each conditional is a ZINB distribution with mean adjusted by the interaction effects from other features.

### Priors

- θ ~ LogNormal(0, 0.5)
- α ~ LogNormal(0, 0.5)
- A_ij ~ Normal(0, 1)
- μ_j ~ LogNormal(0, 1)
- φ_j ~ LogNormal(0, 1)
- π_j ~ Beta(1, 1)

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
