"""
ZINB Graphical Model Implementation using PyTorch and Pyro.

This module implements a Zero-Inflated Negative Binomial (ZINB) graphical model
using pseudo-likelihood inference with NUTS/HMC on GPU.
"""

import os
import warnings


# Define a custom warning category for GPU-related messages
class GPUNotAvailableWarning(UserWarning):
    """Warning issued when running in CPU-only mode."""
    pass


def _check_gpu_availability():
    """
    Check if running in CPU-only mode and issue a warning if needed.
    
    This function is called lazily when GPU functionality is actually used,
    rather than during module import.
    """
    if os.environ.get("PREGRAPHMODELING_DOCKER_MODE") == "cpu":
        warnings.warn(
            "\n"
            "================================================================================\n"
            "  NOTICE: Running in lightweight CPU-only Docker image.\n"
            "  GPU support is disabled to reduce image size.\n"
            "\n"
            "  To enable GPU support, you have two options:\n"
            "\n"
            "  1. (Recommended) Build a derived image with CUDA support:\n"
            "     \n"
            "     [Docker]\n"
            "     Create 'Dockerfile.gpu':\n"
            "       FROM ghcr.io/gwmcelfresh/pregraphmodeling:latest\n"
            "       RUN pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "     Build: docker build -t pregraphmodeling-gpu -f Dockerfile.gpu .\n"
            "     \n"
            "     [Apptainer]\n"
            "     Create 'PreGraphModeling.def':\n"
            "       Bootstrap: docker\n"
            "       From: ghcr.io/gwmcelfresh/pregraphmodeling:latest\n"
            "       %post\n"
            "           pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "     Build: apptainer build pregraphmodeling-gpu.sif PreGraphModeling.def\n"
            "\n"
            "  2. (Interactive) Reinstall inside the running container:\n"
            "     $ pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "\n"
            "  Finally, ensure your runtime command enables GPU access:\n"
            "     Docker:    docker run --gpus all ...\n"
            "     Apptainer: apptainer run --nv ...\n"
            "================================================================================\n",
            GPUNotAvailableWarning,
            stacklevel=2
        )

from .model import ZINBPseudoLikelihoodGraphicalModel
from .inference import run_inference
from .data import load_count_matrix

__all__ = [
    "ZINBPseudoLikelihoodGraphicalModel",
    "run_inference",
    "load_count_matrix",
    "GPUNotAvailableWarning",
    "_check_gpu_availability",
]
__version__ = "0.1.0"
