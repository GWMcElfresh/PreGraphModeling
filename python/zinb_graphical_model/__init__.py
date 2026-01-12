"""
ZINB Graphical Model Implementation using PyTorch and Pyro.

This module implements a Zero-Inflated Negative Binomial (ZINB) graphical model
using pseudo-likelihood inference with NUTS/HMC on GPU.
"""

from .model import ZINBGraphicalModel
from .inference import run_inference
from .data import load_count_matrix

__all__ = ["ZINBGraphicalModel", "run_inference", "load_count_matrix"]
__version__ = "0.1.0"
