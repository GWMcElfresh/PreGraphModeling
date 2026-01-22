"""
Tests for the ZINB graphical model package.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


class TestLoadCountMatrix:
    """Tests for data loading utilities."""

    def test_load_npy_file(self):
        """Test loading count matrix from .npy file."""
        from zinb_graphical_model.data import load_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "counts.npy"
            data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
            np.save(filepath, data)

            result = load_count_matrix(str(filepath), device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 3)
            assert result.device.type == "cpu"
            assert torch.allclose(result, torch.tensor(data))

    def test_load_npz_file(self):
        """Test loading count matrix from .npz file."""
        from zinb_graphical_model.data import load_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "counts.npz"
            data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            np.savez(filepath, counts=data)

            result = load_count_matrix(str(filepath), device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 2)

    def test_load_csv_file(self):
        """Test loading count matrix from .csv file."""
        from zinb_graphical_model.data import load_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "counts.csv"
            with open(filepath, "w") as f:
                f.write("col1,col2,col3\n")
                f.write("1,2,3\n")
                f.write("4,5,6\n")

            result = load_count_matrix(str(filepath), device="cpu")

            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 3)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from zinb_graphical_model.data import load_count_matrix

        with pytest.raises(FileNotFoundError):
            load_count_matrix("/nonexistent/path/to/file.npy", device="cpu")

    def test_unsupported_format(self):
        """Test that ValueError is raised for unsupported file format."""
        from zinb_graphical_model.data import load_count_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "counts.xyz"
            filepath.touch()

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_count_matrix(str(filepath), device="cpu")


class TestZINBPseudoLikelihoodGraphicalModel:
    """Tests for the ZINBPseudoLikelihoodGraphicalModel class."""

    def test_model_initialization(self):
        """Test model initialization."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        model = ZINBPseudoLikelihoodGraphicalModel(n_features=5, device="cpu")

        assert model.n_features == 5
        assert model.device.type == "cpu"
        assert model.n_interaction_params == 10



    def test_build_omega_symmetry(self):
        """Test that Omega is symmetric."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        n_features = 4
        model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

        n_params = (n_features * (n_features - 1)) // 2
        A_tril = torch.randn(n_params)
        Omega = model._build_omega(A_tril)

        assert Omega.shape == (n_features, n_features)
        assert torch.allclose(Omega, Omega.T)

    def test_build_omega_mixed_signs(self):
        """Test that Omega entries preserve the sign of A (mixed signs)."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        n_features = 3
        model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

        # A parameters for n=3: flattened lower tril.
        # Indices: (1,0), (2,0), (2,1) for 3 features.
        A_tril = torch.tensor([-1.0, 1.0, 0.0])

        Omega = model._build_omega(A_tril)

        # Omega[1,0] should be -1.0
        assert Omega[1, 0] == -1.0
        assert Omega[0, 1] == -1.0

        # Omega[2,0] should be 1.0
        assert Omega[2, 0] == 1.0
        assert Omega[0, 2] == 1.0

        # Omega[2,1] should be 0.0
        assert Omega[2, 1] == 0.0
        assert Omega[1, 2] == 0.0

    def test_build_omega_unit_diagonal(self):
        """Test that diagonal elements of Omega are 1."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        n_features = 5
        model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

        n_params = (n_features * (n_features - 1)) // 2
        A_tril = torch.randn(n_params)
        Omega = model._build_omega(A_tril)

        diagonal = torch.diag(Omega)
        assert torch.allclose(diagonal, torch.ones(n_features))

    def test_zinb_log_prob_zeros(self):
        """Test ZINB log prob computation for zero counts."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        model = ZINBPseudoLikelihoodGraphicalModel(n_features=1, device="cpu")

        x = torch.tensor([0.0, 0.0, 0.0])
        mu = torch.tensor(5.0)
        phi = torch.tensor(1.0)
        pi_zero = torch.tensor(0.5)

        log_prob = model._zinb_log_prob(x, mu, phi, pi_zero)

        assert log_prob.shape == x.shape
        assert torch.isfinite(log_prob).all()

    def test_zinb_log_prob_positive_counts(self):
        """Test ZINB log prob computation for positive counts."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        model = ZINBPseudoLikelihoodGraphicalModel(n_features=1, device="cpu")

        x = torch.tensor([1.0, 2.0, 5.0])
        mu = torch.tensor(3.0)
        phi = torch.tensor(2.0)
        pi_zero = torch.tensor(0.2)

        log_prob = model._zinb_log_prob(x, mu, phi, pi_zero)

        assert log_prob.shape == x.shape
        assert torch.isfinite(log_prob).all()
        assert (log_prob <= 0).all()

    def test_pseudo_log_likelihood_finite(self):
        """Test that pseudo-log-likelihood is finite."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel

        n_samples, n_features = 10, 3
        model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

        X = torch.randint(0, 10, (n_samples, n_features)).float()
        Omega = torch.eye(n_features)
        Omega[0, 1] = Omega[1, 0] = -0.1
        mu = torch.ones(n_features) * 5.0
        phi = torch.ones(n_features) * 2.0
        pi_zero = torch.ones(n_features) * 0.2

        gamma_mu = torch.tensor(1.0)
        gamma_phi = torch.tensor(0.0)
        gamma_pi = torch.tensor(0.0)

        pll = model._pseudo_log_likelihood(
            X, Omega, mu, phi, pi_zero, gamma_mu, gamma_phi, gamma_pi
        )

        assert torch.isfinite(pll).all()


class TestInference:
    """Tests for the inference module."""

    @pytest.mark.slow
    def test_run_inference_small(self):
        """Test running inference on a small dataset."""
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel
        from zinb_graphical_model.inference import run_inference

        torch.manual_seed(42)
        n_samples, n_features = 20, 3
        X = torch.randint(0, 5, (n_samples, n_features)).float()

        model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

        results = run_inference(
            model,
            X,
            num_samples=10,
            warmup_steps=5,
            num_chains=1,
        )

        assert "samples" in results
        assert "omega_samples" in results
        assert "summary" in results

        assert "mu" in results["samples"]
        assert "phi" in results["samples"]
        assert "pi_zero" in results["samples"]

        assert results["omega_samples"].shape == (10, n_features, n_features)

    def test_compute_summary(self):
        """Test computing summary statistics."""
        from zinb_graphical_model.inference import compute_summary

        samples = {
            "mu": torch.randn(4, 3),
        }
        omega_samples = torch.randn(4, 3, 3)

        summary = compute_summary(samples, omega_samples)

        assert "mu" in summary

        assert "omega" in summary
        assert summary["omega"]["mean"].shape == (3, 3)


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test the full pipeline from loading to inference."""
        from zinb_graphical_model.data import load_count_matrix
        from zinb_graphical_model.model import ZINBPseudoLikelihoodGraphicalModel
        from zinb_graphical_model.inference import run_inference

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "counts.npy"
            n_samples, n_features = 30, 4
            data = np.random.poisson(lam=3, size=(n_samples, n_features)).astype(
                np.float32
            )
            np.save(filepath, data)

            X = load_count_matrix(str(filepath), device="cpu")
            model = ZINBPseudoLikelihoodGraphicalModel(n_features=n_features, device="cpu")

            results = run_inference(
                model,
                X,
                num_samples=5,
                warmup_steps=5,
                num_chains=1,
            )

            assert results["omega_samples"].shape[0] == 5

            Omega_mean = results["summary"]["omega"]["mean"]
            assert Omega_mean.shape == (n_features, n_features)
            assert np.allclose(Omega_mean, Omega_mean.T, atol=1e-6)
