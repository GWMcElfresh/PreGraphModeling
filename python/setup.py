"""
Setup script for zinb_graphical_model package.
"""

from setuptools import setup, find_packages

setup(
    name="zinb_graphical_model",
    version="0.1.0",
    description="ZINB Graphical Model with PyTorch/Pyro using pseudo-likelihood inference",
    author="GW McElfresh",
    author_email="mcelfreshgw@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.6.0",
        "pyro-ppl>=1.8.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
