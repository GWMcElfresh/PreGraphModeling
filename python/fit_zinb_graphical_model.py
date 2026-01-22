"""Runner to fit a ZINB graphical model from a count matrix.

Supports two inference modes:
Outputs:
- omega_mean.csv / omega_std.csv: posterior mean/std of Ω
- posterior_samples.npz: posterior samples for all parameters (MCMC or SVI)
- posterior_summary.json: full summary stats for all parameters (Ω omitted; see CSVs)
- checkpoint_svi_param_store.pt (SVI only): Pyro ParamStore checkpoint for reloading the variational posterior

Example:
  python fit_zinb_graphical_model.py --counts ./synthetic_counts.csv --method svi --device cuda --batch-size 512

Notes:
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
        )
    else:
        results = run_svi_inference(
            model,
            X_cpu,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            num_posterior_samples=args.posterior_samples,
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
