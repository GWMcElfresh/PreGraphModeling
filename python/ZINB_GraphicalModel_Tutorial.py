"""
ZINB Graphical Model Tutorial script
"""

# ============================================================
# 1. Installation & Setup
# ============================================================
# Install the package from the local path and configure the compute device.

# Install the zinb_graphical_model package from local source
# Run this once to install
# get path to local source
import os
os.path.dirname(os.path.abspath("."))
# %pip install -e . --break-system-packages

# Additional dependencies for visualization and network analysis
# %pip install matplotlib seaborn networkx pandas --break-system-packages

# Install CUDA-enabled PyTorch (RTX 3060)
# %pip uninstall -y torch torchvision torchaudio

# Preferred: CUDA 12.1 wheels (most common for RTX 30-series)
# %pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --break-system-packages

# If the above fails due to Python version mismatch, try CUDA 11.8 wheels
# %pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --break-system-packages

import warnings
import math
import numpy as np
import torch
import pyro

# Optional (tutorial/visualization) dependencies
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None
from pathlib import Path

# ============================================================
# Inputs (edit these)
# ============================================================
# You can either:
#   (A) point COUNT_MATRIX_PATH at a real CSV/TSV/NPY/NPZ count matrix, OR
#   (B) set USE_SYNTHETIC_DATA=True to generate synthetic_counts.csv
#
# CSV/TSV expectations:
#   - first row is a header with feature (gene) names
#   - rows are cells/samples, columns are genes/features

COUNT_MATRIX_PATH = os.getenv("COUNT_MATRIX_PATH", "./synthetic_counts.csv")
USE_SYNTHETIC_DATA = os.getenv("USE_SYNTHETIC_DATA", "1").lower() in {"1", "true", "yes"}

# Inference mode:
#   - "mcmc" uses NUTS/HMC and requires the full matrix on the compute device
#   - "svi" uses minibatching and only moves batches to the device (recommended for big data)
INFERENCE_METHOD = os.getenv("INFERENCE_METHOD", "svi").lower()  # "mcmc" or "svi"

# SVI/minibatching settings
SVI_BATCH_SIZE = int(os.getenv("SVI_BATCH_SIZE", "256"))
SVI_EPOCHS = int(os.getenv("SVI_EPOCHS", "200"))
SVI_LR = float(os.getenv("SVI_LR", "0.01"))
SVI_POSTERIOR_SAMPLES = int(os.getenv("SVI_POSTERIOR_SAMPLES", "200"))

# -------------------------------
# Output/verbosity configuration
# -------------------------------
QUIET = False  # Set True to silence console output
OUTPUT_DIR = Path("./outputs")
FORCE_DEVICE = "cuda"  # Set to "cuda", "mps", or "cpu" to override auto-detect

if QUIET:
    def print(*args, **kwargs):  # noqa: A001
        return


def save_plot(filename: str):
    if plt is None:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# Import our package
from zinb_graphical_model import (
    ZINBPseudoLikelihoodGraphicalModel,
    run_inference,
    run_svi_inference,
    load_count_matrix,
)

# Set plot style (if plotting stack is available)
if sns is not None:
    sns.set_style("whitegrid")
if plt is not None:
    plt.rcParams['figure.figsize'] = (10, 6)

print(f"PyTorch version: {torch.__version__}")
print(f"Pyro version: {pyro.__version__}")
print(f"zinb_graphical_model loaded successfully!")

# ------------------------------------------------------------
# Device Selection: CUDA (NVIDIA), MPS (Mac Silicon), or CPU
# ------------------------------------------------------------
# On Windows with an NVIDIA RTX 3060, PyTorch should report CUDA
# availability once the CUDA-enabled build is installed.
# We'll attempt CUDA first, then MPS (Mac Silicon), with automatic
# fallback to CPU if issues arise.


def get_best_device():
    """
    Detect the best available compute device.
    Priority: CUDA > MPS (Mac Silicon) > CPU
    """
    # --- Debugging Start ---
    print("--- GPU Debugging ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.version, 'cuda'):
        print(f"PyTorch built with CUDA: {torch.version.cuda}")
    else:
        print("PyTorch built with CUDA: Not available (CPU-only build?)")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    print("---------------------\n")
    # --- Debugging End ---

    if FORCE_DEVICE:
        force = FORCE_DEVICE.lower()
        if force == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("FORCE_DEVICE='cuda' but CUDA is not available.")
            torch.cuda.set_device(0)
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA forced: {device_name}")
            return device
        if force == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("FORCE_DEVICE='mps' but MPS is not available.")
            print("✓ MPS forced")
            return "mps"
        if force == "cpu":
            print("✓ CPU forced")
            return "cpu"
        raise ValueError("FORCE_DEVICE must be one of: 'cuda', 'mps', 'cpu', or None")

    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA available: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✓ MPS (Mac Silicon GPU) available")
        print("  Note: Pyro's NUTS sampler may have limited MPS optimization.")
        print("  If inference fails, we'll fall back to CPU.")
    else:
        device = "cpu"
        print("ℹ Using CPU (no GPU detected)")
    return device


# Detect device
DEVICE = get_best_device()
print(f"\nSelected device: {DEVICE}")
print(torch.cuda.is_available())

# ============================================================
# 2. Background & Theory
# ============================================================
# Zero-Inflated Negative Binomial (ZINB) Distribution
# The ZINB distribution models zero-inflation and overdispersion:
#
# P(X = x | μ, φ, π) =
#   π + (1-π) * NB(0 | μ, φ)   if x = 0
#   (1-π) * NB(x | μ, φ)       if x > 0
#
# Parameters:
# - μ (mu): Mean of the negative binomial component
# - φ (phi): Dispersion parameter (larger = less overdispersion)
# - π (pi): Zero-inflation probability
#
# Graphical Model & Interaction Matrix Ω
# Ω[i,j] encodes conditional dependency between gene i and gene j
#
# Pseudo-Likelihood Inference
# P(X) ≈ ∏_j P(X_j | X_-j)

# ============================================================
# 3. Data Preparation
# ============================================================
# Generate Synthetic scRNA-seq Data


def generate_synthetic_counts(
    n_cells: int = 200,
    n_genes: int = 10,
    true_interactions: list = None,
    base_mean: float = 5.0,
    base_dispersion: float = 2.0,
    base_zero_inflation: float = 0.2,
    gamma_mu: float = 1.0,
    gamma_phi: float = 0.5,
    gamma_pi: float = 0.5,
    seed: int = 42,
):
    """
    Generate synthetic scRNA-seq-like count data where interactions affect all parameters.

    Shared latent structure:
    latent ~ MVN(0, Omega^-1)
    mu = base_mean * exp(gamma_mu * latent)
    phi = base_dispersion * exp(gamma_phi * latent)
    logit(pi) = logit(base_pi) + gamma_pi * latent
    """
    np.random.seed(seed)

    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]

    # Build true interaction matrix
    true_omega = np.eye(n_genes)

    if true_interactions is None:
        true_interactions = [
            (0, 1, 0.6),   # strong positive
            (2, 3, -0.5),  # strong negative
            (4, 5, 0.4),   # moderate positive
            (0, 4, 0.3),   # weak positive
            (6, 7, -0.7),  # very strong negative
        ]

    for i, j, strength in true_interactions:
        if i < n_genes and j < n_genes:
            true_omega[i, j] = strength
            true_omega[j, i] = strength

    # Generate correlated latent variables
    try:
        # Use Cholesky of precision matrix inverse (covariance) equivalent
        # For simulation, we can just use Cholesky of a covariance matrix derived from Omega
        # Here we approximate: latent structure induced by Omega
        # We'll use L from Cholesky(Omega + jitter) to generate correlated data
        L = np.linalg.cholesky(true_omega + np.eye(n_genes) * 0.1)
    except np.linalg.LinAlgError:
        L = np.eye(n_genes)

    z = np.random.randn(n_cells, n_genes)
    latent = z @ L.T  # Correlated latent factors

    # --- Apply Shared Latent Structure to Parameters ---

    # 1. Mean (μ): Log-link
    # mu = base * exp(gamma_mu * latent)
    mu_log = np.log(base_mean) + gamma_mu * latent
    mu = np.exp(mu_log)
    mu = np.clip(mu, 0.1, 1e4)

    # 2. Dispersion (φ): Log-link
    # phi = base * exp(gamma_phi * latent)
    phi_log = np.log(base_dispersion) + gamma_phi * latent
    phi = np.exp(phi_log)
    phi = np.clip(phi, 0.1, 1e4)

    # 3. Zero-Inflation (π): Logit-link
    # logit(pi) = logit(base) + gamma_pi * latent
    def logit(p):
        return np.log(p / (1 - p))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pi_logit = logit(base_zero_inflation) + gamma_pi * latent
    pi_prob = sigmoid(pi_logit)

    # Generate ZINB counts
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)

    for i in range(n_cells):
        for j in range(n_genes):
            # Bernoulli trial for zero-inflation
            if np.random.rand() < pi_prob[i, j]:
                counts[i, j] = 0
            else:
                # Negative Binomial trial
                current_phi = phi[i, j]
                current_mu = mu[i, j]
                p = current_phi / (current_phi + current_mu)
                counts[i, j] = np.random.negative_binomial(current_phi, p)

    return counts, gene_names, true_omega


def main():
    # Generate synthetic data with multi-parameter interactions
    # NOTE: Full MCMC scales poorly with number of genes.
    # Use env vars to override defaults for larger runs.
    N_CELLS = int(os.getenv("N_CELLS", "300"))
    N_GENES = int(os.getenv("N_GENES", "20"))
    PLOT_MAX_GENES = int(os.getenv("PLOT_MAX_GENES", "20"))

    # Define expected scaling factors (ground truth)
    TRUE_GAMMA_MU = 1.0    # Interactions strongly affect mean
    TRUE_GAMMA_PHI = 0.5   # Interactions moderately affect dispersion
    TRUE_GAMMA_PI = -0.5   # Interactions negatively affect dropout (higher interaction -> lower dropout)

    if USE_SYNTHETIC_DATA:
        counts, gene_names, true_omega = generate_synthetic_counts(
            n_cells=N_CELLS,
            n_genes=N_GENES,
            base_mean=5.0,
            base_dispersion=2.0,
            base_zero_inflation=0.2,
            gamma_mu=TRUE_GAMMA_MU,
            gamma_phi=TRUE_GAMMA_PHI,
            gamma_pi=TRUE_GAMMA_PI,
        )

        print(f"Generated count matrix: {counts.shape[0]} cells × {counts.shape[1]} genes")
        print(f"Gene names: {gene_names}")
        print(f"\nCount statistics:")
        print(f"  Total counts: {counts.sum():,}")
        print(f"  Sparsity (% zeros): {100 * (counts == 0).mean():.1f}%")
        print(f"  Mean count: {counts.mean():.2f}")
        print(f"  Max count: {counts.max()}")

    # Visualize the true interaction matrix
    if USE_SYNTHETIC_DATA:
        if plt is not None and sns is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                true_omega,
                annot=False,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                xticklabels=gene_names if len(gene_names) <= 30 else False,
                yticklabels=gene_names if len(gene_names) <= 30 else False,
                ax=ax,
                vmin=-1,
                vmax=1,
            )
            ax.set_title("Ground Truth Interaction Matrix (Ω)")
            plt.tight_layout()
            save_plot("true_interaction_matrix.png")
        else:
            print("Skipping plots: install matplotlib+seaborn to enable visualization.")

    # Save and Load Data as CSV
    # The package supports CSV, TSV, NPY, and NPZ formats.

    # Save to CSV (you can replace this file with real scRNA-seq data later)
    DATA_PATH = Path(COUNT_MATRIX_PATH)

    if USE_SYNTHETIC_DATA:
        # Write CSV with header row (works without pandas)
        header = ",".join(gene_names)
        np.savetxt(DATA_PATH, counts, delimiter=",", header=header, comments="", fmt="%d")

        print(f"Saved count matrix to: {DATA_PATH.absolute()}")
        print(f"\nFile preview:")
        if pd is not None:
            df = pd.read_csv(DATA_PATH)
            print(df.head())
        else:
            print(counts[:5, :min(5, counts.shape[1])])

    # Load the count matrix using our package's loader.
    # Always load to CPU first; for SVI we keep it on CPU and only move minibatches.
    X = load_count_matrix(str(DATA_PATH), device="cpu")

    print(f"Loaded tensor shape: {X.shape}")
    print(f"Tensor dtype: {X.dtype}")
    print(f"Device: {X.device}")

    # Initialize gene_names and counts for visualization (if not already set by synthetic data generation)
    if not USE_SYNTHETIC_DATA:
        # Try to read gene names from CSV/TSV header
        suffix = DATA_PATH.suffix.lower()
        if suffix in {".csv", ".tsv", ".txt"}:
            delim = "," if suffix == ".csv" else "\t"
            with DATA_PATH.open("r", encoding="utf-8") as f:
                header = f.readline().strip()
            if header:
                gene_names = [h.strip() for h in header.split(delim) if h.strip()]
            else:
                gene_names = [f"Gene{i+1}" for i in range(X.shape[1])]
        else:
            # For NPY/NPZ files, generate generic gene names
            gene_names = [f"Gene{i+1}" for i in range(X.shape[1])]
        
        # Convert tensor to numpy for visualization compatibility
        counts = X.cpu().numpy()

    # Visualize count distributions (subset for readability)
    n_genes = len(gene_names)
    n_plot = min(PLOT_MAX_GENES, n_genes)
    if plt is not None:
        n_cols = min(4, n_plot) if n_plot > 0 else 1
        n_rows = math.ceil(n_plot / n_cols) if n_plot > 0 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i in range(n_plot):
            gene = gene_names[i]
            ax = axes[i]
            gene_counts = counts[:, i]
            ax.hist(gene_counts, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(gene_counts.mean(), color='red', linestyle='--', label=f'mean={gene_counts.mean():.1f}')
            ax.set_xlabel('Count')
            ax.set_ylabel('Frequency')
            ax.set_title(gene)
            ax.legend(fontsize=8)

        # Hide any unused subplots
        for j in range(n_plot, len(axes)):
            axes[j].axis('off')

        plt.suptitle('Count Distributions per Gene', fontsize=14)
        plt.tight_layout()
        save_plot("count_distributions_per_gene.png")

    # ============================================================
    # 4. Model Setup
    # ============================================================
    model = ZINBPseudoLikelihoodGraphicalModel(n_features=int(X.shape[1]), device=DEVICE)
    print(f"Model initialized:")
    print(f"  - Number of features (genes): {model.n_features}")
    print(f"  - Number of interaction parameters: {model.n_interaction_params}")
    print(f"  - Device: {model.device}")
    print(f"  - Data shape: {tuple(X.shape)}")

    print(f"\nPriors (from model implementation):")
    print(f"  - A_tril (interaction params): Normal(0, 0.1)")
    print(f"  - γ_μ (mean scaling): Normal(1, 0.5)")
    print(f"  - γ_φ (dispersion scaling): Normal(0, 0.5)")
    print(f"  - γ_π (zero-inflation scaling): Normal(0, 0.5)")
    print(f"  - μ (mean): LogNormal(0, 1)")
    print(f"  - φ (dispersion): LogNormal(0, 1)")
    print(f"  - π (zero-inflation): Beta(1, 1)")

    # ============================================================
    # 5. Inference
    # ============================================================
    if INFERENCE_METHOD not in {"mcmc", "svi"}:
        raise ValueError("INFERENCE_METHOD must be 'mcmc' or 'svi'")

    if INFERENCE_METHOD == "mcmc":
        # Keep defaults small for a runnable tutorial. Override with env vars for real runs.
        NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "100"))
        WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "50"))
        NUM_CHAINS = int(os.getenv("NUM_CHAINS", "1"))
        TARGET_ACCEPT = float(os.getenv("TARGET_ACCEPT", "0.8"))
        MAX_TREE_DEPTH = int(os.getenv("MAX_TREE_DEPTH", "10"))
        JIT_COMPILE = os.getenv("JIT_COMPILE", "0").lower() in {"1", "true", "yes"}

        print("\nInference mode: MCMC (NUTS/HMC)")
        print("Inference settings:")
        print(f"  - Posterior samples: {NUM_SAMPLES}")
        print(f"  - Warmup steps: {WARMUP_STEPS}")
        print(f"  - Chains: {NUM_CHAINS}")
        print(f"  - Target acceptance: {TARGET_ACCEPT}")
        print(f"  - Max tree depth: {MAX_TREE_DEPTH}")
        print(f"  - JIT compile: {JIT_COMPILE}")
        if N_GENES >= 50:
            print("\n  N_GENES is large; MCMC may take a long time.")

        X_device = X.to(model.device)
        print("=" * 60)
        print("Starting MCMC Inference")
        print("=" * 60 + "\n")
        results, actual_device = run_inference_with_fallback(
            model,
            X_device,
            num_samples=NUM_SAMPLES,
            warmup_steps=WARMUP_STEPS,
            num_chains=NUM_CHAINS,
            target_accept_prob=TARGET_ACCEPT,
            max_tree_depth=MAX_TREE_DEPTH,
            jit_compile=JIT_COMPILE,
        )
        print(f"\nInference ran on: {actual_device}")
    else:
        print("\nInference mode: SVI (minibatching)")
        print("SVI settings:")
        print(f"  - Batch size: {SVI_BATCH_SIZE}")
        print(f"  - Epochs: {SVI_EPOCHS}")
        print(f"  - Learning rate: {SVI_LR}")
        print(f"  - Posterior samples (approx): {SVI_POSTERIOR_SAMPLES}")
        print("=" * 60)
        print("Starting SVI (minibatched) Inference")
        print("=" * 60 + "\n")

        # Keep the full X on CPU. Only minibatches are moved to model.device.
        results = run_svi_inference(
            model,
            X,
            batch_size=SVI_BATCH_SIZE,
            num_epochs=SVI_EPOCHS,
            learning_rate=SVI_LR,
            num_posterior_samples=SVI_POSTERIOR_SAMPLES,
        )

    # ============================================================
    # 6. Posterior Analysis
    # ============================================================
    samples = results["samples"]
    omega_samples = results["omega_samples"]
    summary = results["summary"]

    print("\nPosterior samples available:")
    for key, val in samples.items():
        print(f"  - {key}: shape {tuple(val.shape)}")
    print(f"  - omega_samples: shape {tuple(omega_samples.shape)}")

    print("\n" + "=" * 60)
    print("POSTERIOR SUMMARY")
    print("=" * 60)

    n_show = min(10, len(gene_names))
    if len(gene_names) > n_show:
        print(f"(Showing first {n_show} genes; set N_GENES/PLOT_MAX_GENES as needed)\n")

    mu_mean = summary["mu"]["mean"]
    mu_std = summary["mu"]["std"]
    print("μ (Mean) parameters:")
    for i in range(n_show):
        print(f"  {gene_names[i]}: {mu_mean[i]:.2f} ± {mu_std[i]:.2f}")

    phi_mean = summary["phi"]["mean"]
    phi_std = summary["phi"]["std"]
    print("\nφ (Dispersion) parameters:")
    for i in range(n_show):
        print(f"  {gene_names[i]}: {phi_mean[i]:.2f} ± {phi_std[i]:.2f}")

    pi_mean = summary["pi_zero"]["mean"]
    pi_std = summary["pi_zero"]["std"]
    print("\nπ (Zero-inflation) parameters:")
    for i in range(n_show):
        print(f"  {gene_names[i]}: {pi_mean[i]:.3f} ± {pi_std[i]:.3f}")

    # Posterior distributions (plot a subset to keep figures readable)
    if plt is not None:
        n_plot = min(PLOT_MAX_GENES, len(gene_names))
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        ax = axes[0]
        mu_samples = samples["mu"].cpu().numpy()
        for i in range(n_plot):
            ax.hist(mu_samples[:, i], bins=20, alpha=0.5, label=gene_names[i])
        ax.set_xlabel("μ (Mean)")
        ax.set_ylabel("Frequency")
        ax.set_title("Posterior Distribution of Mean Parameters (μ)")
        if n_plot <= 12:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        ax = axes[1]
        phi_samples = samples["phi"].cpu().numpy()
        for i in range(n_plot):
            ax.hist(phi_samples[:, i], bins=20, alpha=0.5, label=gene_names[i])
        ax.set_xlabel("φ (Dispersion)")
        ax.set_ylabel("Frequency")
        ax.set_title("Posterior Distribution of Dispersion Parameters (φ)")
        if n_plot <= 12:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        ax = axes[2]
        pi_samples = samples["pi_zero"].cpu().numpy()
        for i in range(n_plot):
            ax.hist(pi_samples[:, i], bins=20, alpha=0.5, label=gene_names[i])
        ax.set_xlabel("π (Zero-inflation probability)")
        ax.set_ylabel("Frequency")
        ax.set_title("Posterior Distribution of Zero-Inflation Parameters (π)")
        if n_plot <= 12:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        save_plot("posterior_zinb_parameters.png")

    gamma_mu = samples["gamma_mu"].cpu().numpy().flatten()
    gamma_phi = samples["gamma_phi"].cpu().numpy().flatten()
    gamma_pi = samples["gamma_pi"].cpu().numpy().flatten()

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.kdeplot(gamma_mu, fill=True, label=f"γ_μ: {gamma_mu.mean():.2f} ± {gamma_mu.std():.2f}", ax=ax)
            sns.kdeplot(gamma_phi, fill=True, label=f"γ_φ: {gamma_phi.mean():.2f} ± {gamma_phi.std():.2f}", ax=ax)
            sns.kdeplot(gamma_pi, fill=True, label=f"γ_π: {gamma_pi.mean():.2f} ± {gamma_pi.std():.2f}", ax=ax)
        else:
            ax.hist(gamma_mu, bins=30, alpha=0.5, label="γ_μ")
            ax.hist(gamma_phi, bins=30, alpha=0.5, label="γ_φ")
            ax.hist(gamma_pi, bins=30, alpha=0.5, label="γ_π")

        ax.axvline(TRUE_GAMMA_MU, color="gray", linestyle="--", alpha=0.5, label=f"True γ_μ = {TRUE_GAMMA_MU}")
        ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5, label="γ = 0 (independence)")
        ax.set_title("Posterior Distribution of Scaling Factors (γ)")
        ax.set_xlabel("Value")
        ax.legend()
        plt.tight_layout()
        save_plot("gamma_scaling_factors.png")

    print("\nRecovered Scaling Factors:")
    print(f"  γ_μ (True={TRUE_GAMMA_MU}): {gamma_mu.mean():.3f}")
    print(f"  γ_φ (True={TRUE_GAMMA_PHI}): {gamma_phi.mean():.3f}")
    print(f"  γ_π (True={TRUE_GAMMA_PI}): {gamma_pi.mean():.3f}")

    omega_mean = summary["omega"]["mean"]
    omega_std = summary["omega"]["std"]
    show_ticks = len(gene_names) <= 30

    if plt is not None and sns is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.heatmap(
            true_omega,
            annot=False,
            cmap="RdBu_r",
            center=0,
            xticklabels=gene_names if show_ticks else False,
            yticklabels=gene_names if show_ticks else False,
            ax=axes[0],
            vmin=-1,
            vmax=1,
        )
        axes[0].set_title("Ground Truth Ω")

        sns.heatmap(
            omega_mean,
            annot=False,
            cmap="RdBu_r",
            center=0,
            xticklabels=gene_names if show_ticks else False,
            yticklabels=gene_names if show_ticks else False,
            ax=axes[1],
            vmin=-1,
            vmax=1,
        )
        axes[1].set_title("Posterior Mean Ω")

        sns.heatmap(
            omega_std,
            annot=False,
            cmap="Reds",
            xticklabels=gene_names if show_ticks else False,
            yticklabels=gene_names if show_ticks else False,
            ax=axes[2],
        )
        axes[2].set_title("Posterior Std Ω")

        plt.tight_layout()
        save_plot("omega_comparison.png")

    # ============================================================
    # 7-8. Network inference + analysis
    # ============================================================
    edges, adjacency = extract_network_edges(
        omega_mean,
        omega_std,
        gene_names,
        threshold=0.05,
        use_credible_interval=True,
        alpha=0.05,
    )

    print(f"\nExtracted {len(edges)} significant edges from the network.")
    if edges:
        edges_sorted = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)
        if pd is not None:
            edges_df = pd.DataFrame(edges_sorted)
            print(edges_df.head(20).to_string(index=False))
        else:
            for e in edges_sorted[:20]:
                print(f"  {e['gene_i']} -- {e['gene_j']}: {e['weight']:.3f} (std={e['std']:.3f})")
    else:
        print("No edges met the significance criteria.")
        print("Try lowering threshold, increasing samples, or reducing N_GENES.")

    # Ground truth non-zero off-diagonal edges
    true_edges = []
    for i in range(N_GENES):
        for j in range(i + 1, N_GENES):
            if true_omega[i, j] != 0:
                true_edges.append((gene_names[i], gene_names[j], true_omega[i, j]))

    print(f"\nGround truth has {len(true_edges)} non-zero interactions.")

    # Export inferred edge list (works without networkx/pandas)
    if edges:
        import csv

        edge_list_path = Path("./gene_network_edges.csv")
        rows = []
        for e in edges:
            rows.append({
                "source": e["gene_i"],
                "target": e["gene_j"],
                "weight": float(e["weight"]),
                "abs_weight": float(abs(e["weight"])),
                "type": e["type"],
                "std": float(e["std"]),
            })
        with edge_list_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nExported edge list to: {edge_list_path.absolute()}")

    # Optional: network-level analysis + visualization
    if nx is None:
        print("\nSkipping NetworkX analysis: install networkx to enable graph stats/plots.")
        return

    G = build_gene_network(omega_mean, gene_names, threshold=0.05)
    print("\nNetwork Statistics:")
    print(f"  Nodes (genes): {G.number_of_nodes()}")
    print(f"  Edges (interactions): {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    if G.number_of_edges() > 0:
        print(f"  Connected components: {nx.number_connected_components(G)}")

    degree_cent = {}
    if G.number_of_edges() > 0:
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            eigen_cent = {n: 0 for n in G.nodes()}

        if pd is not None:
            centrality_df = pd.DataFrame({
                "Gene": list(degree_cent.keys()),
                "Degree": [G.degree(n) for n in degree_cent.keys()],
                "Degree_Centrality": list(degree_cent.values()),
                "Betweenness": list(betweenness_cent.values()),
                "Eigenvector": list(eigen_cent.values()),
            }).sort_values("Degree", ascending=False)
            print("\nTop genes by degree:")
            print(centrality_df.head(10).to_string(index=False))

    if plt is None:
        print("Skipping network plot: install matplotlib to enable visualization.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=2)
        edge_colors = ["#d62728" if G[u][v]["weight"] < 0 else "#1f77b4" for u, v in G.edges()]
        edge_weights = [abs(G[u][v]["weight"]) * 5 for u, v in G.edges()]
        node_sizes = [300 + 500 * degree_cent.get(n, 0.0) for n in G.nodes()]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color="lightblue",
            edgecolors="black",
            linewidths=1.0,
            ax=ax,
        )
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7, ax=ax)

        if G.number_of_nodes() <= 30:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title("Gene-Gene Interaction Network\n(Blue: positive, Red: negative)", fontsize=14)
    else:
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue", ax=ax)
        if G.number_of_nodes() <= 30:
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        ax.set_title("Gene Network (No significant interactions detected)", fontsize=14)

    ax.axis("off")
    plt.tight_layout()
    save_plot("gene_network.png")

# ============================================================
# 4. Model Setup
# ============================================================
# Create the ZINB Pseudo-Likelihood Graphical Model


def setup_model_and_data(X, device):
    """
    Setup model and move data to the specified device.
    Returns model and data tensor on the device.
    """
    n_features = X.shape[1]

    # Create model
    model = ZINBPseudoLikelihoodGraphicalModel(
        n_features=n_features,
        device=device
    )

    # Move data to device
    X_device = X.to(model.device)

    print(f"Model initialized:")
    print(f"  - Number of features (genes): {model.n_features}")
    print(f"  - Number of interaction parameters: {model.n_interaction_params}")
    print(f"  - Device: {model.device}")
    print(f"  - Data shape: {X_device.shape}")
    print(f"  - Data device: {X_device.device}")

    return model, X_device


def run_inference_with_fallback(model, X_device, **kwargs):
    """
    Run inference, falling back to CPU if MPS/CUDA fails.
    """
    try:
        print(f"Running inference on {model.device}...")
        results = run_inference(model, X_device, **kwargs)
        print(f"✓ Inference completed successfully on {model.device}!")
        return results, model.device

    except Exception as e:
        if model.device.type != "cpu":
            print(f"\n⚠️  Inference failed on {model.device}: {e}")
            print("Falling back to CPU...\n")

            # Recreate model and data on CPU
            cpu_model = ZINBPseudoLikelihoodGraphicalModel(
                n_features=model.n_features,
                device="cpu"
            )
            X_cpu = X_device.cpu()

            print(f"Running inference on CPU...")
            results = run_inference(cpu_model, X_cpu, **kwargs)
            print(f"✓ Inference completed successfully on CPU!")
            return results, cpu_model.device
        else:
            raise

# ============================================================
# 7. Network Inference from Ω
# ============================================================


def extract_network_edges(
    omega_mean: np.ndarray,
    omega_std: np.ndarray,
    gene_names: list,
    threshold: float = 0.1,
    use_credible_interval: bool = True,
    alpha: float = 0.05,
):
    """
    Extract network edges from the posterior Omega matrix.

    Parameters:
    -----------
    omega_mean : np.ndarray
        Posterior mean of Omega matrix
    omega_std : np.ndarray
        Posterior standard deviation of Omega matrix
    gene_names : list
        Names of genes
    threshold : float
        Minimum absolute value for an edge to be included
    use_credible_interval : bool
        If True, only include edges where credible interval excludes zero
    alpha : float
        Significance level for credible interval (uses normal approximation)

    Returns:
    --------
    edges : list of tuples
        List of (gene_i, gene_j, weight, significant) tuples
    adjacency : np.ndarray
        Binary adjacency matrix
    """
    from statistics import NormalDist

    n_genes = len(gene_names)
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)  # Two-tailed

    edges = []
    adjacency = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        for j in range(i+1, n_genes):  # Upper triangle only (symmetric)
            weight = omega_mean[i, j]
            std = omega_std[i, j]

            # Check threshold
            if abs(weight) < threshold:
                continue

            # Check credible interval
            if use_credible_interval:
                lower = weight - z_crit * std
                upper = weight + z_crit * std
                significant = (lower > 0) or (upper < 0)  # CI excludes zero
            else:
                significant = True

            if significant:
                edges.append({
                    'gene_i': gene_names[i],
                    'gene_j': gene_names[j],
                    'weight': weight,
                    'std': std,
                    'type': 'positive' if weight > 0 else 'negative',
                })
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    return edges, adjacency

# ============================================================
# 8. Network Analysis with NetworkX
# ============================================================


def build_gene_network(omega_mean: np.ndarray, gene_names: list, threshold: float = 0.05):
    """
    Build a NetworkX graph from the Omega interaction matrix.

    Parameters:
    -----------
    omega_mean : np.ndarray
        Posterior mean of Omega matrix
    gene_names : list
        Names of genes
    threshold : float
        Minimum absolute weight for edge inclusion

    Returns:
    --------
    G : nx.Graph
        NetworkX graph with weighted edges
    """
    if nx is None:
        raise RuntimeError("networkx is required for build_gene_network().")

    G = nx.Graph()

    # Add all genes as nodes
    G.add_nodes_from(gene_names)

    # Add edges from Omega
    n_genes = len(gene_names)
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            weight = omega_mean[i, j]
            if abs(weight) >= threshold:
                G.add_edge(
                    gene_names[i],
                    gene_names[j],
                    weight=weight,
                    abs_weight=abs(weight),
                    edge_type='positive' if weight > 0 else 'negative',
                )

    return G

# ============================================================
# Summary
# ============================================================
# This script demonstrated:
# 1. Installation of the zinb_graphical_model package from local source
# 2. Device detection for MPS/CUDA/CPU with automatic fallback
# 3. Data preparation with synthetic scRNA-seq-like counts saved to CSV
# 4. Model setup with the ZINBPseudoLikelihoodGraphicalModel
# 5. MCMC inference using NUTS for joint parameter estimation
# 6. Posterior analysis of μ, φ, π, and Ω parameters
# 7. Network inference from the Ω interaction matrix
# 8. Network analysis with NetworkX (centrality, visualization)
#
# Next Steps:
# - Replace synthetic_counts.csv with real scRNA-seq data
# - Increase num_samples and warmup_steps for production runs
# - Export to R for advanced igraph analysis
# - Integrate with the R PreGraphModeling package for RBM modeling

    # Clean up generated files (optional)
    # Uncomment to remove temporary files
    # import os
    # if DATA_PATH.exists():
    #     os.remove(DATA_PATH)
    #     print(f"Removed {DATA_PATH}")
    #
    # if edge_list_path.exists():
    #     os.remove(edge_list_path)
    #     print(f"Removed {edge_list_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
