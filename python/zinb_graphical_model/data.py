"""
Data loading utilities for count matrices.
"""

import torch
import numpy as np
from pathlib import Path


def load_count_matrix(filepath: str, device: str = "cuda") -> torch.Tensor:
    """
    Load a count matrix from disk.

    Supports CSV, TSV, NPY, and NPZ formats.

    Args:
        filepath: Path to the count matrix file.
        device: Device to load the tensor on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Count matrix of shape (n_samples, n_features).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Count matrix file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".npy":
        data = np.load(filepath)
    elif suffix == ".npz":
        npz_data = np.load(filepath)
        keys = list(npz_data.keys())
        if len(keys) == 1:
            data = npz_data[keys[0]]
        elif "counts" in keys:
            data = npz_data["counts"]
        elif "data" in keys:
            data = npz_data["data"]
        else:
            data = npz_data[keys[0]]
    elif suffix == ".csv":
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    elif suffix in (".tsv", ".txt"):
        data = np.loadtxt(filepath, delimiter="\t", skiprows=1)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    available_device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA not available, using CPU instead.")

    return torch.tensor(data, dtype=torch.float32, device=available_device)
