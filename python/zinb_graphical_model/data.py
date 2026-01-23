"""
Data loading utilities for count matrices.
"""

import torch
import numpy as np
from pathlib import Path


def _load_delimited_with_row_names(filepath: Path, delimiter: str) -> np.ndarray:
    """
    Load a delimited file, detecting and skipping the first column if it contains row names.
    
    Args:
        filepath: Path to the file.
        delimiter: Field delimiter.
    
    Returns:
        np.ndarray: Data matrix (rows x columns).
    """
    # Read first line to count columns
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    n_cols = len(first_line.split(delimiter))
    
    # Try loading all columns except the first (assuming row names)
    # Use range to work with older NumPy versions
    try:
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1, usecols=range(1, n_cols))
        return data
    except (ValueError, IndexError):
        # If that fails, try loading all columns (no row names)
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
        return data


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
        # Load CSV, skipping header row and first column (row names)
        data = _load_delimited_with_row_names(filepath, delimiter=",")
    elif suffix in (".tsv", ".txt"):
        # Load TSV, skipping header row and first column (row names)
        data = _load_delimited_with_row_names(filepath, delimiter="\t")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    available_device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA not available, using CPU instead.")

    return torch.tensor(data, dtype=torch.float32, device=available_device)
