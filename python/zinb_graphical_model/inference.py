"""
Inference utilities for the ZINB graphical model.

Implements NUTS/HMC sampling for joint inference of all parameters.
"""

from typing import Any

import torch
import pyro
from pyro.infer import MCMC, NUTS

from .model import ZINBGraphicalModel


def run_inference(
    model: ZINBGraphicalModel,
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

    Args:
        model: ZINBGraphicalModel instance.
        X: Count matrix of shape (n_samples, n_features).
        num_samples: Number of MCMC samples to draw.
        warmup_steps: Number of warmup/burn-in steps.
        num_chains: Number of parallel chains.
        target_accept_prob: Target acceptance probability for NUTS.
        max_tree_depth: Maximum tree depth for NUTS.
        jit_compile: Whether to JIT compile the model (can speed up inference).

    Returns:
        Dictionary containing:
            - 'samples': Dictionary of posterior samples for each parameter.
            - 'omega_samples': Posterior samples of the interaction matrix Omega.
            - 'summary': Summary statistics for the posterior.
    """
    pyro.clear_param_store()

    nuts_kernel = NUTS(
        model.model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        jit_compile=jit_compile,
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
    alpha_samples = samples["alpha"]

    n_posterior_samples = A_tril_samples.shape[0]
    omega_samples = torch.zeros(
        n_posterior_samples,
        model.n_features,
        model.n_features,
        device=model.device,
    )

    for i in range(n_posterior_samples):
        omega_samples[i] = model.get_omega(A_tril_samples[i], alpha_samples[i])

    summary = compute_summary(samples, omega_samples)

    return {
        "samples": samples,
        "omega_samples": omega_samples,
        "summary": summary,
        "mcmc": mcmc,
    }


def compute_summary(
    samples: dict[str, torch.Tensor],
    omega_samples: torch.Tensor,
) -> dict[str, Any]:
    """
    Compute summary statistics for posterior samples.

    Args:
        samples: Dictionary of posterior samples.
        omega_samples: Posterior samples of Omega matrix.

    Returns:
        Dictionary of summary statistics.
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
