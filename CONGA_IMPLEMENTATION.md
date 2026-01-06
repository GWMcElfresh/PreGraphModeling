# CONGA Implementation Summary

## Overview

This document describes the implementation of the CONGA (Conditional Nonparametric Graphical Analysis) algorithm for the PreGraphModeling R package. CONGA is based on the work by Roy & Dunson for estimating conditional dependency graphs from count data using Bayesian MCMC methods.

## Algorithm Components

The CONGA algorithm (https://github.com/royarkaprava/CONGA) includes:

1. **Dirichlet Process Prior on Lambda**: Allows clustering of cells with similar Poisson intensity patterns
2. **Blocked Gibbs Sampling**: Updates entire rows/columns of the precision matrix at once
3. **Normalizing Constants**: Computation of partition functions for likelihood evaluation
4. **Horseshoe Priors**: Hierarchical spike-and-slab structure for sparsity

## Implementation Details

## Implementation Details

### 1. Lambda Updates (Poisson Intensities)

**Dirichlet Process Clustering with Auxiliary Variable Method**:
- Samples cluster assignments for each cell using auxiliary variables
- Updates concentration parameter M using Gamma distribution
- Full likelihood computations with normalizing constants
- Allows cells to share lambda values (clustering structure)
- Handles heterogeneity in Poisson intensities across cells

### 2. Beta Updates (Precision Matrix)

**Blocked Gibbs Sampling**:
- Updates entire row/column of beta matrix simultaneously
- Multivariate normal proposals based on conditional posterior
- Requires matrix inversions at each step for proposal construction
- Exploits conditional independence structure
- More efficient mixing than element-wise updates

### 3. Likelihood Computations

**Full Normalizing Constants**:
- Computes: Z = sum_k dpois(k, lambda) * exp(lambda + beta_sum * atan(k)^power)
- C++ implementation for computational efficiency
- Truncation at max_val=100 for practical computation
- Necessary for proper posterior inference

## Implementation Files

### R/conga_fit.R
- `FitCONGAModel()`: Core MCMC sampler with full Dirichlet Process and blocked Gibbs
- Implements Dirichlet Process clustering for lambda updates
- Implements blocked Gibbs sampling for beta updates
- Full likelihood computations with normalizing constants
- Extensive documentation with mathematical formulas
- ~600 lines with detailed comments

### R/conga_wrapper.R
- `FitCONGA()`: Seurat object wrapper
- `ExtractCONGAGraph()`: Post-processing to extract graph
- `ComputeCONGAROC()`: ROC curve computation for evaluation
- `print.CONGAfit()`, `print.CONGAgraph()`: S3 methods
- ~420 lines

### src/conga_helpers.cpp
- `SelectPowerParameter()`: C++ implementation for power parameter selection
- `ComputeAtanMean()`: Expected value under Poisson distribution
- `ComputeLogNormalizingConstant()`: Log partition function for likelihood computation
- ~160 lines with detailed comments

### tests/testthat/test-conga.R
- Unit tests for validation functions
- Input validation tests
- Mock object tests for graph extraction
- ~180 lines

## Mathematical Model

### Data Model
```
X[i,j] ~ Poisson(lambda[i,j])
```

With conditional dependencies modeled through:
```
log P(X[i,j] | X[i,-j], lambda, beta) ∝ 
  -lambda[i,j] + X[i,j] * log(lambda[i,j]) + 
  sum_k beta[j,k] * atan(X[i,j])^power * atan(X[i,k])^power
```

### Priors
```
lambda[,j] ~ DP(M, Gamma(alpha, beta))  [Dirichlet Process with Gamma base]
beta[k] ~ N(0, tau^2 * phi_k * psi_k)    [Horseshoe-like hierarchical prior]
```

### Power Transformation
The `atan(x)^power` transformation:
- Bounds the data to (-π/2, π/2)^power
- Handles extreme count values
- Power parameter selected to minimize: ||cov(atan(X)^power) - cov(X)||^2

## Usage Example

```r
library(PreGraphModeling)
library(Seurat)

# Load data and select features
pbmc <- FindVariableFeatures(pbmc, nfeatures = 50)
hvg <- VariableFeatures(pbmc)

# Fit CONGA model
result <- FitCONGA(
  seuratObject = pbmc,
  geneSubset = hvg,
  totalIterations = 1000,
  burnIn = 500,
  verbose = TRUE
)

# Extract graph
graph <- ExtractCONGAGraph(result, cutoff = 0.7)

# Visualize
library(igraph)
g <- graph_from_adjacency_matrix(graph$adjacency_matrix, mode = "undirected")
plot(g)
```

## Computational Complexity

- **Time**: O(iterations × cells × genes^2) for full algorithm
- **Space**: O(iterations × cells × genes) for MCMC samples
- **Typical Runtime**: 5-15 minutes for 1000 cells × 50 genes × 1000 iterations

## Validation Strategy

1. **Unit Tests**: Input validation, data structures, edge cases
2. **Simulation Studies**: Compare with known graph structures
3. **Acceptance Rates**: Monitor MCMC mixing (target 20-40%)
4. **Convergence Diagnostics**: Trace plots, Gelman-Rubin statistics

## Known Limitations

1. **Computational Intensity**: Full Dirichlet Process and blocked Gibbs are computationally expensive
2. **Feature Selection Recommended**: Computational cost grows as O(genes^2)
3. **No Automatic Tuning**: User must select appropriate MCMC parameters
4. **Matrix Inversions**: Numerical stability considerations in blocked Gibbs updates
5. **Practical Limits**: Best for small-medium graphs (~50-200 genes)

## Uncertainty Markers in Code

Throughout the code, comments marked with "UNCERTAINTY NOTE:" highlight:
- Heuristic methods (e.g., power parameter selection)
- Numerical stability concerns (e.g., matrix inversions in blocked Gibbs)
- Approximations (e.g., normalizing constant truncation at max_val=100)
- Convergence considerations for MCMC diagnostics

## Future Improvements

Potential enhancements (not implemented):
1. Adaptive MCMC proposals for better mixing
2. Parallel tempering for multimodal posteriors
3. Variational inference approximation for speed
4. Integration with existing graph structure learning packages
5. Automatic hyperparameter tuning via cross-validation

## References

- Roy, A. & Dunson, D. (2020). "Nonparametric graphical model for counts." 
  Journal of Machine Learning Research.
- Original implementation: https://github.com/royarkaprava/CONGA

## Contact

For questions or issues with the CONGA implementation:
- Package maintainer: GW McElfresh <mcelfreshgw@gmail.com>
- GitHub issues: https://github.com/GWMcElfresh/PreGraphModeling/issues

---

**Note**: This implementation follows the original CONGA algorithm by Roy & Dunson, including Dirichlet Process priors and blocked Gibbs sampling. The code emphasizes clarity with verbose variable names and extensive documentation of the mathematical formulation.
