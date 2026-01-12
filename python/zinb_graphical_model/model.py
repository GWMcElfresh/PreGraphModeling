"""
ZINB Graphical Model with pseudo-likelihood inference.

This module implements a Zero-Inflated Negative Binomial graphical model
parameterized via:
- Learnable transform F(X) = atan(X)^theta (theta > 0) to stabilize dependence
- Interaction matrix via unconstrained A mapped to negative, symmetric Omega
  using -|atan(A)|^alpha
- Joint inference of theta, Omega, and ZINB parameters using NUTS/HMC
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule


class ZINBGraphicalModel(PyroModule):
    """
    Zero-Inflated Negative Binomial Graphical Model with pseudo-likelihood.

    The model applies a learnable transform F(X) = atan(X)^theta to stabilize
    dependence and parameterizes interactions via an unconstrained matrix A
    mapped to a negative, symmetric precision matrix Omega using -|atan(A)|^alpha.

    Parameters are jointly inferred using NUTS/HMC on GPU.
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
        Apply learnable transform F(X) = atan(X)^theta.

        Args:
            X: Count matrix of shape (n_samples, n_features).
            theta: Positive transform parameter.

        Returns:
            Transformed counts.
        """
        return torch.atan(X).pow(theta)

    def _build_omega(self, A_tril: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Build negative, symmetric interaction matrix Omega from unconstrained A.

        Uses the mapping: Omega_ij = -|atan(A_ij)|^alpha for i != j.
        Diagonal is set to 1 for identifiability.

        Args:
            A_tril: Lower triangular unconstrained parameters (excluding diagonal).
            alpha: Positive exponent for the transform.

        Returns:
            Negative, symmetric interaction matrix Omega of shape (n_features, n_features).
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

        ZINB(x | mu, phi, pi) = pi * I(x=0) + (1-pi) * NB(x | mu, phi)

        Args:
            x: Observed counts.
            mu: Mean parameter (positive).
            phi: Dispersion parameter (positive), also known as 'r' or 'size'.
            pi_zero: Zero-inflation probability [0, 1].

        Returns:
            Log probability of x under ZINB.
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
            P(X_j | X_{-j}) = ZINB(X_j | mu_j + Omega_{j,-j} @ F(X_{-j}), phi_j, pi_j)

        Args:
            X: Count matrix of shape (n_samples, n_features).
            theta: Transform parameter for F(X) = atan(X)^theta.
            Omega: Interaction matrix of shape (n_features, n_features).
            mu: Base mean parameters of shape (n_features,).
            phi: Dispersion parameters of shape (n_features,).
            pi_zero: Zero-inflation probabilities of shape (n_features,).

        Returns:
            Total pseudo-log-likelihood (scalar).
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
        Public method to compute Omega from inferred parameters.

        Args:
            A_tril: Inferred lower triangular parameters.
            alpha: Inferred alpha parameter.

        Returns:
            Negative, symmetric interaction matrix Omega.
        """
        return self._build_omega(A_tril, alpha)
