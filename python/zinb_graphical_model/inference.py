"""
Inference utilities for the ZINB graphical model.

Implements NUTS/HMC sampling for joint inference of all parameters:
    Ω (Omega): Symmetric interaction matrix with unit diagonal
    μ (mu): ZINB mean parameters per feature
    φ (phi): ZINB dispersion parameters per feature
    π (pi_zero): Zero-inflation probabilities per feature
"""

from typing import Any

import torch
import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from .model import ZINBPseudoLikelihoodGraphicalModel


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
            - 'omega_samples': Posterior samples of Ω matrix (n_samples, p, p).
            - 'summary': Summary statistics (mean, std, quantiles) for all parameters.
            - 'mcmc': The Pyro MCMC object for diagnostics.
    """
    pyro.clear_param_store()

    # Use init_to_median for more robust initialization
    # This helps avoid invalid initial parameter combinations
    from pyro.infer.autoguide.initialization import init_to_median
    
    nuts_kernel = NUTS(
        model.model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        jit_compile=jit_compile,
        init_strategy=init_to_median(num_samples=10),
    )

    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    mcmc.run(X)

    samples = mcmc.get_samples()

    A_tril_samples = samples["A_tril"]

    n_posterior_samples = A_tril_samples.shape[0]
    omega_samples = torch.zeros(
        n_posterior_samples,
        model.n_features,
        model.n_features,
        device=model.device,
    )

    for i in range(n_posterior_samples):
        omega_samples[i] = model.get_omega(A_tril_samples[i])

    summary = compute_summary(samples, omega_samples)

    return {
        "samples": samples,
        "omega_samples": omega_samples,
        "summary": summary,
        "mcmc": mcmc,
    }


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
            - 'omega_samples': sampled Omega matrices
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

    # Dataset yields (row, index)
    indices = torch.arange(total_size, dtype=torch.long)
    dataset = TensorDataset(X, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    guide = AutoNormal(model.model)
    optim = ClippedAdam({"lr": learning_rate, "clip_norm": clip_norm})
    elbo = Trace_ELBO(num_particles=num_particles)
    svi = SVI(model.model, guide, optim, loss=elbo)

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

    # Draw approximate posterior samples from the fitted guide.
    # We only need X to define plate sizes; use a tiny slice.
    if total_size == 0:
        raise ValueError("X has zero rows")
    x_shape_probe = X[:1].to(model.device)

    sample_sites = ("A_tril", "mu", "phi", "pi_zero", "gamma_mu", "gamma_phi", "gamma_pi")
    samples: dict[str, torch.Tensor] = {name: [] for name in sample_sites}

    with torch.no_grad():
        for _ in range(num_posterior_samples):
            draw = guide(x_shape_probe)
            for name in sample_sites:
                samples[name].append(draw[name].detach().cpu())

    stacked: dict[str, torch.Tensor] = {}
    for name, draws in samples.items():
        stacked[name] = torch.stack(draws, dim=0)

    # Build omega samples on model.device for speed, then bring back to CPU.
    omega_samples = torch.zeros(
        num_posterior_samples,
        model.n_features,
        model.n_features,
        device=model.device,
    )
    with torch.no_grad():
        for i in range(num_posterior_samples):
            omega_samples[i] = model.get_omega(stacked["A_tril"][i].to(model.device))

    summary = compute_summary(stacked, omega_samples)

    return {
        "samples": stacked,
        "omega_samples": omega_samples,
        "summary": summary,
        "losses": losses,
        "guide": guide,
    }


def compute_summary(
    samples: dict[str, torch.Tensor],
    omega_samples: torch.Tensor,
) -> dict[str, Any]:
    """
    Compute summary statistics for posterior samples.

    Args:
        samples: Dictionary of posterior samples containing:
            - 'mu' (μ): Feature means
            - 'phi' (φ): Feature dispersions
            - 'pi_zero' (π): Zero-inflation probabilities
        omega_samples: Posterior samples of Ω matrix of shape (n_posterior, p, p).

    Returns:
        Dictionary of summary statistics with mean, std, median, and quantiles
        for each parameter, plus mean and std for the Ω matrix.
    """
    summary = {}

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

    omega_mean = omega_samples.mean(dim=0)
    omega_std = omega_samples.std(dim=0)

    summary["omega"] = {
        "mean": omega_mean.cpu().numpy(),
        "std": omega_std.cpu().numpy(),
    }

    return summary
