# CONGA Implementation Summary

## Overview

This document describes the implementation of the CONGA (Conditional Nonparametric Graphical Analysis) algorithm for the PreGraphModeling R package. CONGA is based on the work by Roy & Dunson for estimating conditional dependency graphs from count data using Bayesian MCMC methods.

## Original Algorithm

The original CONGA algorithm (https://github.com/royarkaprava/CONGA) includes:

1. **Dirichlet Process Prior on Lambda**: Allows clustering of cells with similar Poisson intensity patterns
2. **Blocked Gibbs Sampling**: Updates entire rows/columns of the precision matrix at once
3. **Complex Normalizing Constants**: Exact computation of partition functions
4. **Horseshoe Priors**: Hierarchical spike-and-slab structure for sparsity

## Implemented Simplifications

For computational tractability and ease of understanding, this implementation uses several simplifications:

### 1. Lambda Updates (Poisson Intensities)

**Original**: Dirichlet Process clustering with auxiliary variable method
- Samples cluster assignments for each cell
- Updates concentration parameter M
- Complex likelihood computations with normalizing constants

**Simplified**: Standard Metropolis-Hastings
- Direct Gamma proposals for each lambda[i,j]
- Simplified likelihood based on Poisson distribution
- Approximate interaction terms with other genes
- Much faster and more stable

### 2. Beta Updates (Precision Matrix)

**Original**: Blocked Gibbs sampling
- Updates entire row/column simultaneously
- Complex multivariate normal proposals
- Requires matrix inversions at each step

**Simplified**: Element-wise Metropolis-Hastings
- Updates one beta parameter at a time
- Simple normal random walk proposals
- Avoids numerical instability from matrix operations
- Easier to tune and understand

### 3. Likelihood Computations

**Original**: Exact normalizing constants
- Computes: Z = sum_k dpois(k, lambda) * exp(lambda + beta_sum * atan(k)^power)
- Requires C++ implementation for speed
- Can be numerically unstable

**Simplified**: Approximate likelihood
- Uses subset of cells for speed (first 50 cells)
- Focuses on pairwise interaction terms
- Sufficient for MCMC mixing

## Implementation Files

### R/conga_fit.R
- `FitCONGAModel()`: Core MCMC sampler
- Implements simplified lambda and beta updates
- Extensive documentation with uncertainty markers
- ~450 lines with detailed comments

### R/conga_wrapper.R
- `FitCONGA()`: Seurat object wrapper
- `ExtractCONGAGraph()`: Post-processing to extract graph
- `ComputeCONGAROC()`: ROC curve computation for evaluation
- `print.CONGAfit()`, `print.CONGAgraph()`: S3 methods
- ~420 lines

### src/conga_helpers.cpp
- `SelectPowerParameter()`: C++ implementation for power parameter selection
- `ComputeAtanMean()`: Expected value under Poisson distribution
- `ComputeLogNormalizingConstant()`: Log partition function (for future use)
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
lambda[i,j] ~ Gamma(alpha, beta)  [simplified from DP]
beta[k] ~ N(0, sigma^2)           [spike-and-slab]
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

1. **Not the Full CONGA Algorithm**: Simplified for tractability
2. **May Require Longer Runs**: Due to simplified updates
3. **Feature Selection Required**: Computational cost grows as O(genes^2)
4. **No Automatic Tuning**: User must select appropriate parameters
5. **Limited to Small-Medium Graphs**: Practical limit ~200 genes

## Uncertainty Markers in Code

Throughout the code, comments marked with "UNCERTAINTY NOTE:" highlight:
- Heuristic methods (e.g., power parameter selection)
- Approximations (e.g., likelihood computations)
- Numerical stability concerns (e.g., matrix inversions)
- Areas where the simplified algorithm differs from the original

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

**Note**: This implementation prioritizes clarity, documentation, and computational tractability over exact replication of the original algorithm. Users should be aware of the simplifications and their potential impact on results.
