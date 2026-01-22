"""
ZINB Graphical Model with pseudo-likelihood inference.

This module implements a Zero-Inflated Negative Binomial graphical model
parameterized via:
- Interaction matrix via unconstrained A mapped directly to symmetric Ω (Omega)
  with unit diagonal
- Scaling factors γ (gamma) that control how interactions affect each parameter
- Joint inference of θ, Ω, γ, and ZINB parameters (μ, φ, π) using NUTS/HMC

Greek Parameters:
    Ω (Omega): Symmetric interaction/precision matrix with off-diagonal
               entries and unit diagonal. Encodes conditional dependencies between
               features in the graphical model.
    γ_μ (gamma_mu): Scaling factor for how Ω affects the mean μ.
    γ_φ (gamma_phi): Scaling factor for how Ω affects the dispersion φ.
    γ_π (gamma_pi): Scaling factor for how Ω affects zero-inflation π.
    μ (mu): Mean parameter of the ZINB distribution for each feature. Must be positive.
    φ (phi): Dispersion parameter of the negative binomial component. Also known as
             'r' or 'size'. Larger values indicate less overdispersion. Must be positive.
    π (pi): Zero-inflation probability for each feature. Represents the probability
            of excess zeros beyond what the negative binomial predicts. Range [0, 1].
"""

from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from . import _check_gpu_availability


class ZINBPseudoLikelihoodGraphicalModel(PyroModule):
    """
    Zero-Inflated Negative Binomial (ZINB) Graphical Model with Pseudo-Likelihood Inference.

    The model uses raw counts X for dependence and parameterizes interactions via an unconstrained matrix A
    mapped directly to a symmetric precision matrix Ω.

    Parameters are jointly inferred using NUTS/HMC on GPU.

    Model Parameters:
        A: Unconstrained lower-triangular matrix mapped to Ω.
           Prior: Normal(0, 0.1) for each element.
        Ω (Omega): Resulting symmetric interaction matrix.
        γ_μ (gamma_mu): Scaling for μ interactions. Prior: Normal(1, 0.5).
        γ_φ (gamma_phi): Scaling for φ interactions. Prior: Normal(0, 0.5).
        γ_π (gamma_pi): Scaling for π interactions. Prior: Normal(0, 0.5).
        μ (mu): ZINB mean for each feature. Prior: LogNormal(0, 1).
        φ (phi): ZINB dispersion for each feature. Prior: LogNormal(0, 1).
        π (pi_zero): Zero-inflation probability per feature. Prior: Beta(1, 1).
    """

    def __init__(self, n_features: int, device: str = "cuda"):
        """
        Initialize the ZINB Pseudo-Likelihood Graphical Model.

        Args:
            n_features: Number of features (genes/columns) in the count matrix.
            device: Device for computation ('cuda' or 'cpu').
        """
        super().__init__()
        self.n_features = n_features
        
        # Check GPU availability and warn if in CPU-only mode
        if device == "cuda":
            _check_gpu_availability()
        
        available_device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(available_device)
        self.n_interaction_params = (n_features * (n_features - 1)) // 2

    def _build_omega(self, A_tril: torch.Tensor) -> torch.Tensor:
        """
        Build symmetric interaction matrix Ω from unconstrained A.

        Uses the mapping: Ω_ij = A_ij for i ≠ j.
        Diagonal is set to 1 for identifiability.

        Args:
            A_tril: Lower triangular unconstrained parameters (excluding diagonal).

        Returns:
            Ω (Omega): Symmetric interaction matrix of shape (n_features, n_features)
                       with unit diagonal.
        """
        n = self.n_features
        Omega = torch.zeros(n, n, device=self.device)

        idx = 0
        for i in range(n):
            for j in range(i):
                val = A_tril[idx]
                Omega[i, j] = val
                Omega[j, i] = val
                idx += 1

        diag_idx = torch.arange(n, device=self.device)
        Omega[diag_idx, diag_idx] = 1.0

        return Omega

    def _zinb_log_prob(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        pi_zero: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability under ZINB distribution.

        ZINB(x | μ, φ, π) = π · I(x=0) + (1-π) · NB(x | μ, φ)

        The ZINB distribution models count data with excess zeros by mixing
        a point mass at zero with a negative binomial distribution.

        Args:
            x: Observed counts.
            mu (μ): Mean parameter of the negative binomial component.
                    Must be positive. Represents expected count when not zero-inflated.
            phi (φ): Dispersion parameter of negative binomial, also known as 'r'
                     or 'size'. Must be positive. As φ → ∞, NB → Poisson.
                     Smaller φ indicates more overdispersion.
            pi_zero (π): Zero-inflation probability in range [0, 1].
                         Probability of excess zeros beyond NB prediction.

        Returns:
            Log probability log P(x | μ, φ, π) under ZINB.
        """
        # Ensure numerical stability with clamping
        mu_safe = torch.clamp(mu, min=1e-6, max=1e6)
        phi_safe = torch.clamp(phi, min=1e-6, max=1e6)
        
        # Compute logits with numerical stability
        # logits = log(mu / phi) = log(mu) - log(phi)
        logits = torch.log(mu_safe) - torch.log(phi_safe)
        logits = torch.clamp(logits, min=-20.0, max=20.0)  # Prevent extreme logits
        
        nb_dist = dist.NegativeBinomial(total_count=phi_safe, logits=logits)
        nb_log_prob = nb_dist.log_prob(x)
        
        # Clamp NB log prob to avoid -inf
        nb_log_prob = torch.clamp(nb_log_prob, min=-100.0)

        # Clamp pi_zero to valid range
        pi_safe = torch.clamp(pi_zero, min=1e-6, max=1.0 - 1e-6)
        log_pi = torch.log(pi_safe)
        log_one_minus_pi = torch.log(1 - pi_safe)

        log_prob = torch.where(
            x == 0,
            torch.logaddexp(log_pi, log_one_minus_pi + nb_log_prob),
            log_one_minus_pi + nb_log_prob,
        )

        return log_prob

    def _pseudo_log_likelihood(
        self,
        X: torch.Tensor,
        Omega: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        pi_zero: torch.Tensor,
        gamma_mu: torch.Tensor,
        gamma_phi: torch.Tensor,
        gamma_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pseudo-log-likelihood for the graphical model.

        Pseudo-likelihood avoids partition functions by conditioning on all
        other variables. For each feature j:
            effect_j = Ω_{j,-j} @ X_{-j}
            μ_j^cond = μ_j · exp(γ_μ · effect_j)
            φ_j^cond = φ_j · exp(γ_φ · effect_j)
            π_j^cond = σ(logit(π_j) + γ_π · effect_j)
            P(X_j | X_{-j}) = ZINB(X_j | μ_j^cond, φ_j^cond, π_j^cond)

        This approximation is computationally tractable and provides consistent
        parameter estimates for graphical models.

        Args:
            X: Count matrix of shape (n_samples, n_features).
            Omega (Ω): Interaction matrix of shape (n_features, n_features).
                       Encodes conditional dependencies between features.
            mu (μ): Base mean parameters of shape (n_features,) for each feature.
            phi (φ): Base dispersion parameters of shape (n_features,) for each feature.
            pi_zero (π): Base zero-inflation probabilities of shape (n_features,).
            gamma_mu (γ_μ): Scalar scaling factor for μ interactions.
            gamma_phi (γ_φ): Scalar scaling factor for φ interactions.
            gamma_pi (γ_π): Scalar scaling factor for π interactions.

        Returns:
            Pseudo-log-likelihood per sample (shape: (n_samples,)).
        """
        n_samples, n_features = X.shape

        total_log_prob = torch.zeros(n_samples, device=self.device)

        for j in range(n_features):
            mask = torch.ones(n_features, dtype=torch.bool, device=self.device)
            mask[j] = False

            X_minus_j = X[:, mask]
            Omega_j_minus_j = Omega[j, mask]

            # Compute the shared interaction effect
            effect = X_minus_j @ Omega_j_minus_j
            # Clamp the effect BEFORE scaling to prevent overflow
            effect = torch.clamp(effect, min=-10.0, max=10.0)
            
            # Conditional μ: log-link (exp transform)
            conditional_mu = mu[j] * torch.exp(gamma_mu * effect)
            conditional_mu = torch.clamp(conditional_mu, min=1e-6, max=1e6)
            
            # Conditional φ: log-link (exp transform)
            conditional_phi = phi[j] * torch.exp(gamma_phi * effect)
            conditional_phi = torch.clamp(conditional_phi, min=1e-6, max=1e6)
            
            # Conditional π: logit-link (sigmoid transform)
            # logit(pi) = log(pi / (1 - pi))
            pi_j_safe = torch.clamp(pi_zero[j], min=1e-6, max=1.0 - 1e-6)
            logit_pi = torch.log(pi_j_safe / (1 - pi_j_safe))
            conditional_logit_pi = logit_pi + gamma_pi * effect
            conditional_logit_pi = torch.clamp(conditional_logit_pi, min=-10.0, max=10.0)
            conditional_pi = torch.sigmoid(conditional_logit_pi)

            log_prob_j = self._zinb_log_prob(X[:, j], conditional_mu, conditional_phi, conditional_pi)
            total_log_prob = total_log_prob + log_prob_j

        return total_log_prob

    def model(
        self,
        X: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        total_size: int | None = None,
    ):
        """
        Pyro model definition for ZINB graphical model.

        Defines priors and likelihood using pseudo-likelihood approach.
        All parameters are jointly inferred via NUTS/HMC.

        Priors:
            A ~ Normal(0, 0.1): Unconstrained interaction parameters
            γ_μ (gamma_mu) ~ Normal(1, 0.5): Scaling for mean interactions
            γ_φ (gamma_phi) ~ Normal(0, 0.5): Scaling for dispersion interactions
            γ_π (gamma_pi) ~ Normal(0, 0.5): Scaling for zero-inflation interactions
            μ (mu) ~ LogNormal(0, 1): Feature means
            φ (phi) ~ LogNormal(0, 1): Feature dispersions
            π (pi_zero) ~ Beta(1, 1): Zero-inflation probabilities

        Args:
            X: Count matrix. If using minibatching, this should be the batch slice
               (shape: (batch_size, n_features)).
            batch_indices: Optional indices for subsampling within the full dataset.
                If provided, the model uses `pyro.plate("data", size=total_size, subsample=batch_indices)`
                so Pyro can scale the log-probability correctly.
            total_size: Total number of rows (cells/samples) in the full dataset.
                Required when batch_indices is provided.
        """
        n_samples, n_features = X.shape

        if batch_indices is not None and total_size is None:
            raise ValueError("total_size must be provided when batch_indices is used for subsampling")

        # Use tighter prior to prevent extreme initial interaction values
        # that could cause numerical overflow in exp(X @ Omega)
        A_tril = pyro.sample(
            "A_tril",
            dist.Normal(
                torch.zeros(self.n_interaction_params, device=self.device), 0.1
            ).to_event(1),
        )

        Omega = self._build_omega(A_tril)

        # Gamma scaling factors for multi-parameter interactions
        # gamma_mu centered at 1.0 (default behavior: interactions affect mu)
        gamma_mu = pyro.sample(
            "gamma_mu",
            dist.Normal(torch.tensor(1.0, device=self.device), 0.5),
        )
        
        # gamma_phi centered at 0.0 (default: no dispersion interactions)
        gamma_phi = pyro.sample(
            "gamma_phi",
            dist.Normal(torch.tensor(0.0, device=self.device), 0.5),
        )
        
        # gamma_pi centered at 0.0 (default: no zero-inflation interactions)
        gamma_pi = pyro.sample(
            "gamma_pi",
            dist.Normal(torch.tensor(0.0, device=self.device), 0.5),
        )

        with pyro.plate("features", n_features):
            mu = pyro.sample(
                "mu",
                dist.LogNormal(torch.zeros(n_features, device=self.device), 1.0),
            )

            phi = pyro.sample(
                "phi",
                dist.LogNormal(torch.zeros(n_features, device=self.device), 1.0),
            )

            pi_zero = pyro.sample(
                "pi_zero",
                dist.Beta(
                    torch.ones(n_features, device=self.device),
                    torch.ones(n_features, device=self.device),
                ),
            )

        plate_kwargs = {"size": n_samples} if batch_indices is None else {"size": total_size, "subsample": batch_indices}

        with pyro.plate("data", **plate_kwargs):
            pseudo_ll = self._pseudo_log_likelihood(
                X, Omega, mu, phi, pi_zero, gamma_mu, gamma_phi, gamma_pi
            )
            pyro.factor("pseudo_likelihood", pseudo_ll)

    def get_omega(
        self, A_tril: torch.Tensor
    ) -> torch.Tensor:
        """
        Public method to compute Ω from inferred parameters.

        Args:
            A_tril: Inferred lower triangular parameters from posterior.

        Returns:
            Ω (Omega): Symmetric interaction matrix with unit diagonal.
        """
        return self._build_omega(A_tril)
