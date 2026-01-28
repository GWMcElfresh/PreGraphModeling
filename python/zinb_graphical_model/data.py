"""Data loading utilities for count matrices.

This module is intentionally conservative about file parsing, because
count matrices in the wild commonly vary in whether they include:
    - a header row of feature names, and/or
    - a first column of row names (e.g., cell barcodes).

The delimited loader (CSV/TSV/TXT) auto-detects these cases to avoid
silently dropping a real feature column.
"""

import torch
import numpy as np
from pathlib import Path


# =============================================================================
# DELIMITED FILE PARSING UTILITIES
# =============================================================================
# These helper functions handle the detection and parsing of CSV/TSV files,
# which may or may not contain headers and row name columns.
# =============================================================================

def _is_float_token(token: str) -> bool:
    """Check if a string token can be parsed as a float."""
    token = token.strip().strip('"').strip("'")
    if token == "":
        return False
    try:
        float(token)
        return True
    except Exception:
        return False


def _split_delimited_line(line: str, delimiter: str) -> list[str]:
    """Split a line by delimiter, stripping whitespace from each token."""
    # Keep it simple: our expected inputs are plain CSV/TSV without embedded delimiters.
    return [t.strip() for t in line.strip().split(delimiter)]


# -----------------------------------------------------------------------------
# Layout Detection:
# -----------------------------------------------------------------------------
# Heuristics for detecting file structure:
#   - A header row exists if the first non-empty line is NOT fully numeric.
#   - Row names exist if the first token of the first data row is non-numeric.
# -----------------------------------------------------------------------------

def _detect_delimited_layout(filepath: Path, delimiter: str) -> tuple[bool, bool, int]:
    """Detect (has_header, has_row_names, n_cols_data) for a delimited file.

    Returns:
      (has_header, has_row_names, n_cols_data)
    """
    with filepath.open("r", encoding="utf-8") as f:
        # First non-empty line
        first_line = ""
        while True:
            first_line = f.readline()
            if first_line == "":
                raise ValueError(f"Empty delimited file: {filepath}")
            if first_line.strip() != "":
                break

        first_tokens = _split_delimited_line(first_line, delimiter)
        has_header = not all(_is_float_token(tok) for tok in first_tokens)

        data_line = ""
        if has_header:
            # First data row is the next non-empty line.
            while True:
                data_line = f.readline()
                if data_line == "":
                    raise ValueError(f"Delimited file has header but no data rows: {filepath}")
                if data_line.strip() != "":
                    break
        else:
            data_line = first_line

        data_tokens = _split_delimited_line(data_line, delimiter)
        if len(data_tokens) == 0:
            raise ValueError(f"Could not parse delimited file: {filepath}")

        has_row_names = not _is_float_token(data_tokens[0])
        n_cols_data = len(data_tokens)
        return has_header, has_row_names, n_cols_data


# -----------------------------------------------------------------------------
# Delimited File Loading:
# -----------------------------------------------------------------------------
# Loads CSV/TSV/TXT files with automatic detection of headers and row names.
# Falls back to np.genfromtxt for messy files (e.g., missing values).
# -----------------------------------------------------------------------------

def _load_delimited_with_optional_row_names(filepath: Path, delimiter: str) -> np.ndarray:
    """Load a delimited file (CSV/TSV/TXT) with robust header/rowname detection."""

    has_header, has_row_names, n_cols_data = _detect_delimited_layout(filepath, delimiter)
    skiprows = 1 if has_header else 0
    start_col = 1 if has_row_names else 0
    usecols = range(start_col, n_cols_data)

    try:
        data = np.loadtxt(
            filepath,
            delimiter=delimiter,
            skiprows=skiprows,
            usecols=usecols,
        )
    except Exception:
        # Fallback for messy files (e.g., missing values). Keep behavior consistent.
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=skiprows,
            usecols=usecols,
        )

    return data


# =============================================================================
# COUNT MATRIX LOADING
# =============================================================================
# Main entry point for loading count matrices from various file formats.
# Returns a PyTorch tensor on the specified device.
# =============================================================================

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

    # -------------------------------------------------------------------------
    # Format-Specific Loading:
    # -------------------------------------------------------------------------
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
        data = _load_delimited_with_optional_row_names(filepath, delimiter=",")
    elif suffix in (".tsv", ".txt"):
        data = _load_delimited_with_optional_row_names(filepath, delimiter="\t")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # -------------------------------------------------------------------------
    # Shape Normalization:
    # -------------------------------------------------------------------------
    # Ensure 2D output even for single-feature datasets.
    # -------------------------------------------------------------------------
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Device Placement:
    # -------------------------------------------------------------------------
    available_device = device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA not available, using CPU instead.")

    return torch.tensor(data, dtype=torch.float32, device=available_device)
