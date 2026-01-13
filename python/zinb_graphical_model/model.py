"""
ZINB Graphical Model with pseudo-likelihood inference.

This module implements a Zero-Inflated Negative Binomial graphical model
parameterized via:
- Learnable transform F(X) = atan(X)^θ (theta > 0) to stabilize dependence
- Interaction matrix via unconstrained A mapped to negative, symmetric Ω (Omega)
  using -|atan(A)|^α (alpha)
- Joint inference of θ, Ω, and ZINB parameters (μ, φ, π) using NUTS/HMC

Greek Parameters:
    θ (theta): Transform exponent in F(X) = atan(X)^θ. Controls the nonlinearity
               of the variance-stabilizing transformation. Must be positive.
    α (alpha): Exponent in the interaction mapping Ω_ij = -|atan(A_ij)|^α.
               Controls the strength/sparsity of interactions. Must be positive.
    Ω (Omega): Symmetric interaction/precision matrix with negative off-diagonal
               entries and unit diagonal. Encodes conditional dependencies between
               features in the graphical model.
    μ (mu): Mean parameter of the ZINB distribution for each feature. Must be positive.
    φ (phi): Dispersion parameter of the negative binomial component. Also known as
             'r' or 'size'. Larger values indicate less overdispersion. Must be positive.
    π (pi): Zero-inflation probability for each feature. Represents the probability
            of excess zeros beyond what the negative binomial predicts. Range [0, 1].
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule


class ZINBGraphicalModel(PyroModule):
    """
    Zero-Inflated Negative Binomial Graphical Model with pseudo-likelihood.

    The model applies a learnable transform F(X) = atan(X)^θ to stabilize
    dependence and parameterizes interactions via an unconstrained matrix A
    mapped to a negative, symmetric precision matrix Ω using -|atan(A)|^α.

    Parameters are jointly inferred using NUTS/HMC on GPU.

    Model Parameters:
        θ (theta): Transform exponent, F(X) = atan(X)^θ. Controls nonlinearity
                   of variance stabilization. Prior: LogNormal(0, 0.5).
        α (alpha): Interaction exponent, Ω_ij = -|atan(A_ij)|^α. Controls
                   interaction strength. Prior: LogNormal(0, 0.5).
        A: Unconstrained lower-triangular matrix mapped to Ω.
           Prior: Normal(0, 1) for each element.
        Ω (Omega): Resulting negative symmetric interaction matrix.
        μ (mu): ZINB mean for each feature. Prior: LogNormal(0, 1).
        φ (phi): ZINB dispersion for each feature. Prior: LogNormal(0, 1).
        π (pi_zero): Zero-inflation probability per feature. Prior: Beta(1, 1).
    """

    def __init__(self, n_features: int, device: str = "cuda"):
        """
        Initialize the ZINB graphical model.

        Args:
            n_features: Number of features (genes/columns) in the count matrix.
            device: Device for computation ('cuda' or 'cpu').
        """
        super().__init__()
        self.n_features = n_features
        available_device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(available_device)
        self.n_interaction_params = (n_features * (n_features - 1)) // 2

    def _transform_counts(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable transform F(X) = atan(X)^θ.

        This variance-stabilizing transformation maps counts to a bounded range
        while preserving relative ordering. The exponent θ controls nonlinearity.

        Args:
            X: Count matrix of shape (n_samples, n_features).
            theta (θ): Positive transform exponent. Higher values increase
                       nonlinearity of the transformation.

        Returns:
            Transformed counts F(X) = atan(X)^θ.
        """
        return torch.atan(X).pow(theta)

    def _build_omega(self, A_tril: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Build negative, symmetric interaction matrix Ω from unconstrained A.

        Uses the mapping: Ω_ij = -|atan(A_ij)|^α for i ≠ j.
        Diagonal is set to 1 for identifiability.

        The atan transform bounds the interaction strengths, while the negative
        sign ensures repulsive/inhibitory interactions in the graphical model.

        Args:
            A_tril: Lower triangular unconstrained parameters (excluding diagonal).
            alpha (α): Positive exponent controlling interaction strength.
                       Higher α values create sparser, weaker interactions.

        Returns:
            Ω (Omega): Negative, symmetric interaction matrix of shape
                       (n_features, n_features) with unit diagonal.
        """
        n = self.n_features
        Omega = torch.zeros(n, n, device=self.device)

        idx = 0
        for i in range(n):
            for j in range(i):
                val = -torch.abs(torch.atan(A_tril[idx])).pow(alpha)
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
        nb_dist = dist.NegativeBinomial(total_count=phi, logits=torch.log(mu / (phi + 1e-10)))
        nb_log_prob = nb_dist.log_prob(x)

        is_zero = (x == 0).float()
        log_pi = torch.log(pi_zero + 1e-10)
        log_one_minus_pi = torch.log(1 - pi_zero + 1e-10)

        log_prob = torch.where(
            x == 0,
            torch.logaddexp(log_pi, log_one_minus_pi + nb_log_prob),
            log_one_minus_pi + nb_log_prob,
        )

        return log_prob

    def _pseudo_log_likelihood(
        self,
        X: torch.Tensor,
        theta: torch.Tensor,
        Omega: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        pi_zero: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pseudo-log-likelihood for the graphical model.

        Pseudo-likelihood avoids partition functions by conditioning on all
        other variables. For each feature j:
            P(X_j | X_{-j}) = ZINB(X_j | μ_j · exp(Ω_{j,-j} @ F(X_{-j})), φ_j, π_j)

        This approximation is computationally tractable and provides consistent
        parameter estimates for graphical models.

        Args:
            X: Count matrix of shape (n_samples, n_features).
            theta (θ): Transform parameter for F(X) = atan(X)^θ.
            Omega (Ω): Interaction matrix of shape (n_features, n_features).
                       Encodes conditional dependencies between features.
            mu (μ): Base mean parameters of shape (n_features,) for each feature.
            phi (φ): Dispersion parameters of shape (n_features,) for each feature.
            pi_zero (π): Zero-inflation probabilities of shape (n_features,).

        Returns:
            Total pseudo-log-likelihood (scalar) summed over all features and samples.
        """
        n_samples, n_features = X.shape
        F_X = self._transform_counts(X, theta)

        total_log_prob = torch.tensor(0.0, device=self.device)

        for j in range(n_features):
            mask = torch.ones(n_features, dtype=torch.bool, device=self.device)
            mask[j] = False

            F_X_minus_j = F_X[:, mask]
            Omega_j_minus_j = Omega[j, mask]

            conditional_effect = F_X_minus_j @ Omega_j_minus_j
            conditional_mu = (mu[j] + 1e-10) * torch.exp(conditional_effect)
            conditional_mu = torch.clamp(conditional_mu, min=1e-10)

            log_prob_j = self._zinb_log_prob(X[:, j], conditional_mu, phi[j], pi_zero[j])
            total_log_prob = total_log_prob + log_prob_j.sum()

        return total_log_prob

    def model(self, X: torch.Tensor):
        """
        Pyro model definition for ZINB graphical model.

        Defines priors and likelihood using pseudo-likelihood approach.
        All parameters are jointly inferred via NUTS/HMC.

        Priors:
            θ (theta) ~ LogNormal(0, 0.5): Transform exponent
            α (alpha) ~ LogNormal(0, 0.5): Interaction exponent
            A ~ Normal(0, 1): Unconstrained interaction parameters
            μ (mu) ~ LogNormal(0, 1): Feature means
            φ (phi) ~ LogNormal(0, 1): Feature dispersions
            π (pi_zero) ~ Beta(1, 1): Zero-inflation probabilities

        Args:
            X: Count matrix of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        theta = pyro.sample(
            "theta", dist.LogNormal(torch.tensor(0.0, device=self.device), 0.5)
        )

        alpha = pyro.sample(
            "alpha", dist.LogNormal(torch.tensor(0.0, device=self.device), 0.5)
        )

        A_tril = pyro.sample(
            "A_tril",
            dist.Normal(
                torch.zeros(self.n_interaction_params, device=self.device), 1.0
            ).to_event(1),
        )

        Omega = self._build_omega(A_tril, alpha)

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

        pseudo_ll = self._pseudo_log_likelihood(X, theta, Omega, mu, phi, pi_zero)
        pyro.factor("pseudo_likelihood", pseudo_ll)

    def get_omega(
        self, A_tril: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Public method to compute Ω from inferred parameters.

        Args:
            A_tril: Inferred lower triangular parameters from posterior.
            alpha (α): Inferred α parameter from posterior.

        Returns:
            Ω (Omega): Negative, symmetric interaction matrix with unit diagonal.
        """
        return self._build_omega(A_tril, alpha)
