"""
Inference utilities for the ZINB graphical model.

Implements NUTS/HMC sampling for joint inference of all parameters:
    Ω (Omega): Symmetric interaction matrix with unit diagonal
    μ (mu): ZINB mean parameters per feature
    φ (phi): ZINB dispersion parameters per feature
    π (pi_zero): Zero-inflation probabilities per feature
"""

from __future__ import annotations

from typing import Any

import torch
import pyro
import numpy as np
import dask.array as da
from pyro.infer import MCMC, NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from .model import ZINBPseudoLikelihoodGraphicalModel


# =============================================================================
# DASK-BASED Ω (OMEGA) MATRIX COMPUTATION
# =============================================================================
# These functions implement a memory-efficient pipeline for computing posterior
# statistics of the Ω matrix using Dask's lazy evaluation and chunked processing.
# This avoids OOM errors when n_samples × p × p exceeds available RAM.
# =============================================================================

def _batched_build_omega(A_tril: torch.Tensor, n_features: int) -> torch.Tensor:
    """
    Batched construction of symmetric Ω matrices from unconstrained parameters.
    
    This is the GPU-accelerated worker that processes a batch of A_tril vectors
    and returns the corresponding Ω matrices.
    
    Args:
        A_tril: Batch of lower triangular parameters (batch_size, n_params).
        n_features: Number of features (nodes) in the graph.
        
    Returns:
        Omega: Batch of symmetric matrices (batch_size, n_features, n_features)
               with unit diagonal.
    """
    batch_size = A_tril.shape[0]
    device = A_tril.device
    
    # Create empty batch of matrices
    Omega = torch.zeros(batch_size, n_features, n_features, device=device)
    
    # -------------------------------------------------------------------------
    # Vectorized Index Assignment:
    # -------------------------------------------------------------------------
    # Get indices for lower triangle (excluding diagonal) and assign values
    # to both lower and upper triangles simultaneously.
    # -------------------------------------------------------------------------
    tril_indices = torch.tril_indices(row=n_features, col=n_features, offset=-1, device=device)
    rows, cols = tril_indices[0], tril_indices[1]
    
    # Assign values to lower and upper triangles simultaneously using vectorization
    # A_tril shape: (B, N_params) -> broadcasts to indices
    Omega[:, rows, cols] = A_tril
    Omega[:, cols, rows] = A_tril
    
    # Set diagonal to 1.0 for identifiability
    diag_idx = torch.arange(n_features, device=device)
    Omega[:, diag_idx, diag_idx] = 1.0
    
    return Omega


def _compute_omega_chunk(A_tril_chunk: np.ndarray, n_features: int, device: str) -> np.ndarray:
    """
    Dask worker function to compute a chunk of Ω matrices on GPU.
    
    This function is called by Dask's map_blocks for each chunk of A_tril samples.
    It moves data to GPU, computes Ω, and immediately returns to CPU to free VRAM.
    
    Args:
        A_tril_chunk: Numpy array of shape (chunk_size, n_params).
        n_features: Number of features.
        device: Target device string (e.g. 'cuda', 'cpu').
        
    Returns:
        Omega_chunk: Numpy array of shape (chunk_size, n_features, n_features).
    """
    # Ensure input is standard numpy array (not Dask array)
    if isinstance(A_tril_chunk, da.Array):
        A_tril_chunk = A_tril_chunk.compute()
    
    # Move chunk to GPU/Device
    tensor_chunk = torch.as_tensor(A_tril_chunk, device=device)
    
    # Compute Ω batch
    omega_tensor = _batched_build_omega(tensor_chunk, n_features)
    
    # Move back to CPU immediately to free VRAM
    return omega_tensor.detach().cpu().numpy()


def _compute_omega_stats_dask(
    A_tril_samples: torch.Tensor,
    n_features: int,
    device: str,
    chunk_size: int = 100
) -> dict[str, np.ndarray]:
    """
    Compute mean and std of Ω using Dask for lazy, memory-efficient evaluation.
    
    This function implements the "lazy reduction" pattern from dask_plan.md:
    1. A_tril samples are chunked along the sample dimension
    2. Each chunk is transformed to Ω matrices via GPU-accelerated map_blocks
    3. Mean and std are computed via tree reduction without materializing full tensor
    
    Args:
        A_tril_samples: Tensor of posterior samples (n_samples, n_params).
        n_features: Number of features (p). Output will be (p, p) matrices.
        device: Device to use for matrix construction (e.g., 'cuda', 'cpu').
        chunk_size: Number of samples to process per chunk. Controls memory usage.
        
    Returns:
        Dict with 'mean' and 'std' of Ω, each of shape (n_features, n_features).
    """
    # -------------------------------------------------------------------------
    # Convert to Numpy for Dask:
    # -------------------------------------------------------------------------
    # Dask operates on numpy arrays. Move samples to CPU if they're on GPU.
    # -------------------------------------------------------------------------
    if torch.is_tensor(A_tril_samples):
        A_tril_np = A_tril_samples.detach().cpu().numpy()
    else:
        A_tril_np = A_tril_samples
        
    n_samples = A_tril_np.shape[0]
    
    # Create Dask array from numpy array
    # Chunk along the sample dimension (axis 0)
    dask_A = da.from_array(A_tril_np, chunks=(chunk_size, -1))
    
    # -------------------------------------------------------------------------
    # Lazy Reduction via Dask:
    # -------------------------------------------------------------------------
    # We use dask.array.map_blocks to transform the A_tril samples (n_samples, n_params)
    # into Ω matrices (n_samples, p, p) without ever materializing the full 3D tensor.
    # 
    # Input shape per chunk:  (chunk_size, n_params)
    # Output shape per chunk: (chunk_size, n_features, n_features)
    #
    # The `drop_axis=[1]` removes the n_params dimension.
    # The `new_axis=[1, 2]` inserts two new dimensions for the (p, p) matrix.
    # -------------------------------------------------------------------------
    omega_dask = dask_A.map_blocks(
        _compute_omega_chunk,
        n_features=n_features,
        device=device,
        chunks=(chunk_size, n_features, n_features),
        dtype=np.float32,
        drop_axis=[1],
        new_axis=[1, 2]
    )
    
    # -------------------------------------------------------------------------
    # Graph-Fused Computation of Mean and Standard Deviation:
    # -------------------------------------------------------------------------
    # By passing multiple delayed arrays to `da.compute()`, Dask builds a SINGLE
    # fused task graph. This means:
    #   1. The map_blocks transformation runs ONCE (not twice).
    #   2. Each chunk is loaded, transformed to Ω, and then both mean and
    #      std accumulations are updated before the chunk is discarded.
    #   3. Memory usage remains O(chunk_size × p²), NOT O(n_samples × p²).
    #
    # Internally, Dask uses tree reduction for aggregations like mean/std, so
    # intermediate partial results are combined hierarchically.
    # -------------------------------------------------------------------------
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    print(f"Computing Ω stats using Dask ({n_chunks} chunks of size {chunk_size})...")
    
    # Use Dask's built-in ProgressBar to display progress to stdout.
    # This shows a text-based progress bar as tasks complete.
    from dask.diagnostics import ProgressBar
    with ProgressBar():
        omega_mean, omega_std = da.compute(
            omega_dask.mean(axis=0),
            omega_dask.std(axis=0)
        )
    
    return {
        "mean": omega_mean,
        "std": omega_std
    }


# =============================================================================
# MCMC INFERENCE (NUTS/HMC)
# =============================================================================
# Full Bayesian inference using the No-U-Turn Sampler (NUTS), an adaptive
# variant of Hamiltonian Monte Carlo. Provides exact posterior samples but
# is computationally expensive for large datasets.
# =============================================================================

def run_inference(
    model: ZINBPseudoLikelihoodGraphicalModel,
    X: torch.Tensor,
    num_samples: int = 1000,
    warmup_steps: int = 500,
    num_chains: int = 1,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    jit_compile: bool = False,
) -> dict[str, Any]:
    """
    Run NUTS/HMC inference for the ZINB graphical model.

    Jointly infers all model parameters using the No-U-Turn Sampler (NUTS),
    an adaptive variant of Hamiltonian Monte Carlo.

    Args:
        model: ZINBPseudoLikelihoodGraphicalModel instance.
        X: Count matrix of shape (n_samples, n_features).
        num_samples: Number of MCMC samples to draw from posterior.
        warmup_steps: Number of warmup/burn-in steps for adaptation.
        num_chains: Number of parallel MCMC chains.
        target_accept_prob: Target acceptance probability for NUTS.
        max_tree_depth: Maximum tree depth for NUTS trajectory.
        jit_compile: Whether to JIT compile the model (can speed up inference).

    Returns:
        Dictionary containing:
            - 'samples': Dictionary of posterior samples:
                - 'A_tril': Unconstrained interaction parameter samples
                - 'mu' (μ): Mean parameter samples per feature
                - 'phi' (φ): Dispersion parameter samples per feature
                - 'pi_zero' (π): Zero-inflation probability samples
            - 'summary': Summary statistics (mean, std, quantiles) for all parameters.
            - 'mcmc': The Pyro MCMC object for diagnostics.
    """
    pyro.clear_param_store()

    # -------------------------------------------------------------------------
    # NUTS Kernel Configuration:
    # -------------------------------------------------------------------------
    # Use init_to_median for more robust initialization. This helps avoid
    # invalid initial parameter combinations that can cause divergences.
    # -------------------------------------------------------------------------
    from pyro.infer.autoguide.initialization import init_to_median
    
    nuts_kernel = NUTS(
        model.model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        jit_compile=jit_compile,
        init_strategy=init_to_median(num_samples=10),
    )

    # -------------------------------------------------------------------------
    # MCMC Sampler Setup:
    # -------------------------------------------------------------------------
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    # -------------------------------------------------------------------------
    # Run MCMC Sampling:
    # -------------------------------------------------------------------------
    mcmc.run(X)

    # Retrieve the posterior samples from all chains
    samples = mcmc.get_samples()

    # -------------------------------------------------------------------------
    # Dask-Based Ω Statistics:
    # -------------------------------------------------------------------------
    # Use Dask to compute Ω statistics (mean and std) from the A_tril samples.
    # This approach is memory-efficient as it avoids materializing the full
    # (n_samples, p, p) Ω tensor in memory.
    # -------------------------------------------------------------------------
    omega_stats = _compute_omega_stats_dask(
        samples["A_tril"],
        model.n_features,
        str(model.device),
        chunk_size=min(100, num_samples)
    )

    summary = compute_summary(samples, omega_stats=omega_stats)

    return {
        "samples": samples,
        "summary": summary,
        "mcmc": mcmc,
    }


# =============================================================================
# STOCHASTIC VARIATIONAL INFERENCE (SVI)
# =============================================================================
# Approximate Bayesian inference using variational methods with minibatching.
# Much faster than MCMC and scales to large datasets, but provides approximate
# posteriors (mean-field variational family).
# =============================================================================

def run_svi_inference(
    model: ZINBPseudoLikelihoodGraphicalModel,
    X: torch.Tensor,
    batch_size: int = 256,
    num_epochs: int = 200,
    learning_rate: float = 1e-2,
    num_particles: int = 1,
    seed: int | None = 0,
    shuffle: bool = True,
    clip_norm: float = 5.0,
    num_posterior_samples: int = 200,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Run stochastic variational inference (SVI) with optional minibatching.

    Notes:
        - NUTS/HMC (run_inference) does not support minibatching/subsampling.
        - This SVI path uses `pyro.plate("data", size=total_size, subsample=batch_indices)`
          and expects that `X` is already the batch slice corresponding to `batch_indices`.
        - Keep the full `X` on CPU if it does not fit GPU memory; batches are moved to model.device.

    Args:
        model: ZINBPseudoLikelihoodGraphicalModel instance.
        X: Full count matrix on CPU (recommended for large datasets), shape (n_samples, n_features).
        batch_size: Minibatch size.
        num_epochs: Number of passes over the dataset.
        learning_rate: Optimizer learning rate.
        num_particles: ELBO particles.
        seed: RNG seed.
        shuffle: Whether to shuffle per epoch.
        clip_norm: Gradient clipping norm.
        num_posterior_samples: Number of posterior samples to draw from the variational posterior.
        progress_every: Print a progress line every N epochs (0 disables).

    Returns:
        Dict compatible with run_inference():
            - 'samples': posterior samples dict (approximate)
            - 'summary': summary statistics
            - 'losses': list of mean epoch losses
            - 'guide': fitted AutoNormal guide
    """
    from torch.utils.data import DataLoader, TensorDataset

    pyro.clear_param_store()
    if seed is not None:
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)

    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor of shape (n_samples, n_features)")
    total_size = int(X.shape[0])

    # -------------------------------------------------------------------------
    # DataLoader Setup:
    # -------------------------------------------------------------------------
    # Dataset yields (row, index) pairs. Indices are needed for Pyro's
    # subsampling to correctly scale the ELBO.
    # -------------------------------------------------------------------------
    indices = torch.arange(total_size, dtype=torch.long)
    dataset = TensorDataset(X, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    # -------------------------------------------------------------------------
    # Variational Guide and Optimizer:
    # -------------------------------------------------------------------------
    # AutoNormal uses a mean-field Gaussian approximation for each latent variable.
    # ClippedAdam prevents gradient explosions during optimization.
    # -------------------------------------------------------------------------
    guide = AutoNormal(model.model)
    optim = ClippedAdam({"lr": learning_rate, "clip_norm": clip_norm})
    elbo = Trace_ELBO(num_particles=num_particles)
    svi = SVI(model.model, guide, optim, loss=elbo)

    # -------------------------------------------------------------------------
    # Training Loop:
    # -------------------------------------------------------------------------
    losses: list[float] = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_seen = 0
        for batch_X, batch_indices in dataloader:
            # Move only the batch to the compute device
            batch_X = batch_X.to(model.device)

            # Keep indices on CPU (Pyro subsample indices are metadata; the data is already sliced)
            batch_indices = batch_indices.to("cpu", non_blocking=True)

            loss = svi.step(batch_X, batch_indices=batch_indices, total_size=total_size)
            epoch_loss += float(loss)
            n_seen += int(batch_X.shape[0])

        mean_epoch_loss = epoch_loss / max(n_seen, 1)
        losses.append(mean_epoch_loss)
        if progress_every and ((epoch + 1) % progress_every == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"[SVI] epoch {epoch + 1:>4}/{num_epochs}  mean_loss_per_row={mean_epoch_loss:.4f}")

    # -------------------------------------------------------------------------
    # Posterior Sampling from Variational Guide:
    # -------------------------------------------------------------------------
    # Draw approximate posterior samples from the fitted guide.
    # We only need X to define plate sizes; use a tiny slice.
    # -------------------------------------------------------------------------
    if total_size == 0:
        raise ValueError("X has zero rows")
    x_shape_probe = X[:1].to(model.device)
    # Use a dummy batch_indices tensor to match the guide/model signature used during training.
    dummy_batch_indices = torch.arange(x_shape_probe.shape[0], device="cpu")

    sample_sites = ("A_tril", "mu", "phi", "pi_zero", "gamma_mu", "gamma_phi", "gamma_pi")
    samples: dict[str, torch.Tensor] = {name: [] for name in sample_sites}

    with torch.no_grad():
        for _ in range(num_posterior_samples):
            draw = guide(x_shape_probe, batch_indices=dummy_batch_indices, total_size=total_size)
            for name in sample_sites:
                samples[name].append(draw[name].detach().cpu())

    stacked: dict[str, torch.Tensor] = {}
    for name, draws in samples.items():
        stacked[name] = torch.stack(draws, dim=0)

    # -------------------------------------------------------------------------
    # Dask-Based Ω Statistics:
    # -------------------------------------------------------------------------
    # Using Dask for memory efficiency when computing Ω statistics.
    # -------------------------------------------------------------------------
    omega_stats = _compute_omega_stats_dask(
        stacked["A_tril"],
        model.n_features,
        str(model.device),
        chunk_size=min(100, num_posterior_samples)
    )

    summary = compute_summary(stacked, omega_stats=omega_stats)

    return {
        "samples": stacked,
        "summary": summary,
        "losses": losses,
        "guide": guide,
    }


# =============================================================================
# POSTERIOR SUMMARY STATISTICS
# =============================================================================
# Computes summary statistics (mean, std, median, quantiles) for all
# posterior samples, including the Ω matrix.
# =============================================================================

def compute_summary(
    samples: dict[str, torch.Tensor],
    omega_samples: torch.Tensor | None = None,
    omega_stats: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Compute summary statistics for posterior samples.

    Args:
        samples: Dictionary of posterior samples containing:
            - 'mu' (μ): Feature means
            - 'phi' (φ): Feature dispersions
            - 'pi_zero' (π): Zero-inflation probabilities
        omega_samples: Optional full tensor of Ω samples (n_posterior, p, p).
                       If provided, stats are computed from this.
        omega_stats: Optional pre-computed stats {'mean': ..., 'std': ...}.
                     Used if omega_samples is not provided (e.g. Dask computation).

    Returns:
        Dictionary of summary statistics with mean, std, median, and quantiles
        for each parameter, plus mean and std for the Ω matrix.
    """
    summary = {}

    # -------------------------------------------------------------------------
    # Per-Parameter Statistics:
    # -------------------------------------------------------------------------
    # For scalar parameters: compute mean, std, median, and 90% credible interval.
    # For vector/matrix parameters: compute element-wise mean and std.
    # -------------------------------------------------------------------------
    for name, tensor in samples.items():
        if tensor.ndim == 1:
            summary[name] = {
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "median": tensor.median().item(),
                "q05": tensor.quantile(0.05).item(),
                "q95": tensor.quantile(0.95).item(),
            }
        else:
            summary[name] = {
                "mean": tensor.mean(dim=0).cpu().numpy(),
                "std": tensor.std(dim=0).cpu().numpy(),
            }

    # -------------------------------------------------------------------------
    # Ω Statistics:
    # -------------------------------------------------------------------------
    # Use pre-computed Dask stats if available, otherwise compute from samples.
    # -------------------------------------------------------------------------
    if omega_stats is not None:
        summary["omega"] = omega_stats
    elif omega_samples is not None:
        omega_mean = omega_samples.mean(dim=0)
        omega_std = omega_samples.std(dim=0)
        summary["omega"] = {
            "mean": omega_mean.cpu().numpy(),
            "std": omega_std.cpu().numpy(),
        }
    else:
        # Fallback if no omega info is provided (should not happen in main path)
        summary["omega"] = {"mean": None, "std": None}

    return summary
