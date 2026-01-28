"""Runner to fit a ZINB graphical model from a count matrix.

Supports two inference modes:
- ``mcmc``: full Bayesian inference via MCMC (e.g., NUTS/HMC) over all parameters.
- ``svi``: stochastic variational inference, suitable for large datasets and minibatching.

Outputs:
- omega_mean.csv / omega_std.csv: posterior mean/std of Ω
- posterior_samples.npz: posterior samples for all parameters (MCMC or SVI)
- posterior_summary.json: full summary stats for all parameters (Ω omitted; see CSVs)
- checkpoint_svi_param_store.pt (SVI only): Pyro ParamStore checkpoint for reloading the variational posterior

Performance Considerations:
--------------------------
This model scales O(n_features²) per iteration due to the pairwise interaction matrix Ω.
For 300 genes, each iteration can take ~30 minutes. Below are parameters that significantly
affect runtime and their trade-offs.

PARAMETERS THAT SPEED UP FITTING:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Method Selection (--method):
    - ``svi`` (DEFAULT, RECOMMENDED FOR SPEED): Stochastic Variational Inference.
      Uses minibatching to process data in chunks, enabling GPU memory efficiency.
      *Pros*: 10-100x faster than MCMC; scales to large datasets; supports GPU minibatching.
      *Cons*: Provides approximate posterior (mean-field variational family); may underestimate
      uncertainty; may miss multimodal posteriors.

    - ``mcmc``: Full Bayesian NUTS/HMC sampling.
      *Pros*: Exact posterior samples; proper uncertainty quantification; gold standard.
      *Cons*: Extremely slow for large feature sets; cannot minibatch; entire dataset must fit
      in memory.

SVI-Specific Speed Parameters:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--batch-size (default: 256):
    Number of samples per minibatch during SVI.
    *Increasing* → Fewer gradient steps per epoch; more stable gradients; faster per-epoch.
    *Decreasing* → More gradient steps; noisier gradients but potentially better exploration;
                   uses less GPU memory per step.
    *Recommendation*: Start with 256-512. Increase to 1024+ if GPU memory allows for faster
    convergence. For very large datasets, larger batches = faster training.

--epochs (default: 200):
    Number of full passes over the dataset.
    *Decreasing* → Directly reduces runtime proportionally; may underfit.
    *Increasing* → Better convergence; diminishing returns after ELBO stabilizes.
    *Recommendation*: Monitor svi_losses.csv; stop early if ELBO plateaus. Use 50-100 for
    initial exploration, 200+ for final fits.

--lr (learning rate, default: 0.01):
    Adam optimizer learning rate.
    *Increasing* → Faster initial progress; risk of instability/divergence.
    *Decreasing* → More stable but slower convergence.
    *Recommendation*: 0.01-0.05 is usually stable. If loss diverges, reduce to 0.001.

--posterior-samples (default: 200):
    Number of samples drawn from the fitted variational posterior.
    *Decreasing* → Faster post-SVI sampling; noisier uncertainty estimates.
    *Increasing* → Better uncertainty quantification; more computation.
    *Recommendation*: 100 samples is often sufficient for point estimates (omega_mean.csv).
    Use 500+ if uncertainty quantification (omega_std.csv) is critical.

MCMC-Specific Speed Parameters:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--num-samples (default: 300):
    Number of posterior samples to draw after warmup.
    *Decreasing* → Directly reduces runtime; fewer effective samples; noisier estimates.
    *Increasing* → Better posterior characterization; diminishing returns.
    *Recommendation*: 100-200 for exploratory runs; 500-1000 for publication-quality inference.
    Check effective sample size (ESS) in diagnostics.

--warmup-steps (default: 150):
    Number of adaptation/burn-in steps for NUTS.
    *Decreasing* → Faster to start sampling; risk of poor adaptation leading to low acceptance
                   rates or divergences.
    *Increasing* → Better step-size tuning; safer but delays production samples.
    *Recommendation*: Rule of thumb: warmup ≈ 0.5-1x num-samples. For 300 samples, 150-300
    warmup is reasonable.

--max-tree-depth (default: 10):
    Maximum depth of NUTS trajectory tree.
    *Decreasing* (e.g., 6-8) → Faster per-step; may reduce effective sample size; risk of
                               poor exploration in complex posteriors.
    *Increasing* → Better exploration but exponentially more expensive (2^depth evaluations).
    *Recommendation*: Start at 8 for speed-critical runs. Use 10-12 for complex posteriors.
    If you see many "max treedepth" warnings, increase this value.

--target-accept (default: 0.8):
    Target acceptance probability for NUTS step-size adaptation.
    *Decreasing* (e.g., 0.65) → Larger steps; faster but may reduce ESS or cause divergences.
    *Increasing* (e.g., 0.9-0.95) → Smaller, safer steps; slower but more reliable.
    *Recommendation*: 0.8 is standard. Increase to 0.9 if you see divergences during warmup.

--num-chains (default: 1):
    Number of parallel MCMC chains.
    *Increasing* → Better convergence diagnostics (R-hat); parallelizable on multi-GPU.
    *Cost*: Linear increase in compute (unless parallelized across GPUs).
    *Recommendation*: Use 1 for speed. Use 4+ for final inference to check convergence.

--jit-compile:
    JIT-compile the model using PyTorch/Pyro tracing.
    *Enabled* → Significant speedup after initial compilation; not always compatible with
                all model structures.
    *Disabled* → More robust but slower.
    *Recommendation*: Try enabling for ~20-50% speedup. Disable if you encounter JIT errors.

GENERAL SPEED RECOMMENDATIONS:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Use SVI (--method svi) for exploration and initial fitting.
2. Use GPU (--device cuda) if available—provides 10-50x speedup over CPU.
3. Reduce feature count: Subset to top variable genes (e.g., 50-100) for initial tests.
4. For quick validation: --method svi --epochs 50 --batch-size 512 --posterior-samples 100
5. For production: Start with SVI to get good initialization, then refine with short MCMC if
   full Bayesian inference is required.

COMPLEXITY SCALING:
~~~~~~~~~~~~~~~~~~~
- Time per MCMC iteration: O(n_samples × n_features²) due to pseudo-likelihood loop.
- Time per SVI epoch: O(n_samples × n_features²), but batches reduce memory pressure.
- Ω matrix parameters: n_features × (n_features - 1) / 2 = 44,850 for 300 genes.

Example Commands:
-----------------
# Fast exploratory fit:
python fit_zinb_graphical_model.py --counts data.csv --out ./results --method svi --epochs 50 --batch-size 512

# Balanced SVI fit:
python fit_zinb_graphical_model.py --counts data.csv --out ./results --method svi --epochs 200 --batch-size 256

# Quick MCMC:
python fit_zinb_graphical_model.py --counts data.csv --out ./results --method mcmc --num-samples 100 \\
    --warmup-steps 50 --max-tree-depth 8

# Full Bayesian inference:
python fit_zinb_graphical_model.py --counts data.csv --out ./results --method mcmc --num-samples 500 \\
    --warmup-steps 250 --num-chains 4 --jit-compile

Notes:
------
- For large datasets, keep `--device cuda` but use `--method svi` so only batches go to GPU.
- CSV/TSV files are expected to include a header row with feature names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pyro

from zinb_graphical_model import (
    ZINBPseudoLikelihoodGraphicalModel,
    load_count_matrix,
    run_inference,
    run_svi_inference,
)


def _read_header_names(path: Path) -> list[str] | None:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"}:
        return None

    delim = "," if suffix == ".csv" else "\t"
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    if not header:
        return None
    return [h.strip() for h in header.split(delim) if h.strip()]


def _write_matrix_csv(path: Path, matrix: np.ndarray, colnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if colnames is None:
        np.savetxt(path, matrix, delimiter=",", fmt="%.6g")
        return

    header = ",".join([""] + colnames)
    rows = []
    for i, row in enumerate(matrix):
        rows.append(",".join([colnames[i]] + [f"{float(x):.6g}" for x in row]))
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _jsonify(x):
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
        }
    if torch.is_tensor(x):
        return _jsonify(x.detach().cpu().numpy())
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x

def _save_posterior_samples_npz(path: Path, samples: dict[str, torch.Tensor]):
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in samples.items():
        arrays[name] = _to_numpy(tensor)
    np.savez_compressed(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit ZINB pseudo-likelihood graphical model")
    p.add_argument("--counts", required=True, help="Path to count matrix (CSV/TSV/NPY/NPZ).")
    p.add_argument("--device", default="cuda", help="cuda|cpu|mps (model compute device)")
    p.add_argument("--method", choices=["mcmc", "svi"], default="svi")

    # SVI params
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--posterior-samples", type=int, default=200)

    # MCMC params
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--num-chains", type=int, default=1)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--jit-compile", action="store_true")

    # Memory/output options
    p.add_argument(
        "--supermassive-computer-destroying-samples-return",
        action="store_true",
        help=(
            "Return full omega_samples tensor (n_samples, p, p). "
            "WARNING: Can consume tens of terabytes for large models. "
            "Use only for small models where you need the full posterior samples."
        )
    )

    p.add_argument("--out", default="./outputs_fit", help="Output directory")
    args = p.parse_args(argv)

    counts_path = Path(args.counts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full matrix to CPU first; SVI will move only minibatches.
    X_cpu = load_count_matrix(str(counts_path), device="cpu")
    gene_names = _read_header_names(counts_path)

    model = ZINBPseudoLikelihoodGraphicalModel(n_features=int(X_cpu.shape[1]), device=args.device)

    if args.method == "mcmc":
        X_device = X_cpu.to(model.device)
        results = run_inference(
            model,
            X_device,
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            num_chains=args.num_chains,
            target_accept_prob=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            jit_compile=args.jit_compile,
            return_omega_samples=args.supermassive_computer_destroying_samples_return,
        )
    else:
        results = run_svi_inference(
            model,
            X_cpu,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            num_posterior_samples=args.posterior_samples,
            return_omega_samples=args.supermassive_computer_destroying_samples_return,
        )

    summary = results["summary"]
    omega_mean = summary["omega"]["mean"]
    omega_std = summary["omega"]["std"]

    _write_matrix_csv(out_dir / "omega_mean.csv", omega_mean, gene_names)
    _write_matrix_csv(out_dir / "omega_std.csv", omega_std, gene_names)

    # Save posterior samples for all parameters.
    _save_posterior_samples_npz(out_dir / "posterior_samples.npz", results["samples"])

    # Save a full (but compact) summary JSON.
    summary_no_omega = {k: v for k, v in summary.items() if k != "omega"}
    (out_dir / "posterior_summary.json").write_text(
        json.dumps(_jsonify(summary_no_omega), indent=2) + "\n",
        encoding="utf-8",
    )

    # Write a small human-friendly JSON summary.
    minimal = {
        "method": args.method,
        "device": str(model.device),
        "n_samples": int(X_cpu.shape[0]),
        "n_features": int(X_cpu.shape[1]),
        "gamma_mu": summary.get("gamma_mu", None),
        "gamma_phi": summary.get("gamma_phi", None),
        "gamma_pi": summary.get("gamma_pi", None),
    }

    (out_dir / "fit_summary.json").write_text(json.dumps(_jsonify(minimal), indent=2) + "\n", encoding="utf-8")

    if "losses" in results:
        np.savetxt(out_dir / "svi_losses.csv", np.asarray(results["losses"], dtype=float), delimiter=",")

    # SVI checkpointing: save the fitted variational posterior parameters.
    if args.method == "svi":
        ckpt_path = out_dir / "checkpoint_svi_param_store.pt"
        pyro.get_param_store().save(str(ckpt_path))
        meta = {
            "checkpoint": str(ckpt_path.name),
            "how_to_reload": (
                "Instantiate ZINBPseudoLikelihoodGraphicalModel with the same n_features, "
                "create AutoNormal(model.model), then call pyro.get_param_store().load(checkpoint_path)."
            ),
        }
        (out_dir / "checkpoint_svi_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote outputs to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
