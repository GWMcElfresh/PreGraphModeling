# PreGraphModeling

R package for single-cell RNA-seq analysis with three complementary approaches:
1. **Per-Gene ZINB Modeling** - Estimate zero-inflated negative binomial distribution parameters for each gene across cell subsets
2. **Restricted Boltzmann Machine (RBM)** - Model relationships between gene expression features and cell metadata using partial correlations
3. **CONGA (Conditional Nonparametric Graphical Analysis)** - Estimate conditional dependency graphs using Bayesian MCMC methods

## Installation

```r
# Install from GitHub
devtools::install_github("GWMcElfresh/PreGraphModeling")

# Install optional dependencies for RBM visualization
BiocManager::install(c("ComplexHeatmap", "circlize"))
install.packages(c("progressr", "viridisLite"))

# Optional: graph visualizations
install.packages("igraph")

# Required for CONGA
install.packages(c("Rcpp", "RcppArmadillo", "mvtnorm", "combinat", "MCMCpack"))
```

---

## Approach 1: Per-Gene ZINB Modeling

This approach fits zero-inflated negative binomial models to each gene within cell subsets, estimating distribution parameters (mean, dispersion, zero-inflation) that can be compared across conditions.

### Key Functions
- `SubsetSeurat()` - Divide Seurat object by metadata columns
- `FitZeroInflatedModels()` - Fit ZINB models to expression data
- `AnalyzeWithZINB()` - Complete pipeline combining subsetting and modeling

### Quick Start

```r
library(PreGraphModeling)

# Complete analysis with parallel processing
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  parallel = TRUE,
  verbose = TRUE
)

# Access combined parameters with keys
head(result$combined_parameters)
#   gene   mu  phi   pi converged n_nonzero n_datapoints            key
# 1 Gene1 12.5 2.34 0.15     TRUE        18           20 TypeA_Control
```

### Features
- Subset by multiple metadata columns for fine-grained analysis
- Estimates mean (μ), dispersion (φ), and zero-inflation (π) per gene
- Optional residualization of size factors against cellular saturation using GAM
- Parallel processing with `future`/`future.apply`
- Key-based output for easy downstream joining

### Detailed Example

```r
# Subset by multiple columns
result <- AnalyzeWithZINB(
  seuratObject = pbmc,
  groupByColumns = c("CellType", "Treatment"),
  saturationColumn = "Saturation.RNA",  # Residualize size factors
  geneSubset = c("CD3D", "CD4", "CD8A", "MS4A1"),
  minNonZero = 5,
  parallel = TRUE,
  numWorkers = 4
)

# Filter to specific gene
cd3d_data <- result$combined_parameters[result$combined_parameters$gene == "CD3D", ]
```

> **See vignette:** `vignette("zinb_analysis")` for complete workflows

---

## Approach 2: Restricted Boltzmann Machine (RBM)

This approach uses an RBM architecture to model relationships between gene expression (visible layer) and cell metadata factors (hidden layers). It learns weight matrices connecting features to metadata using Contrastive Divergence training.

### Key Functions
- `EstimatePartialCorrelationsFromSeurat()` - Compute gene-gene partial correlations from Seurat object (recommended)
- `EstimatePartialCorrelations()` - Compute gene-gene partial correlations from expression matrix
- `FitRBM()` - Fit RBM connecting expression to metadata factors
- `predict.RBM()` - Predict hidden activations from new expression data
- `ReconstructRBM()` - Reconstruct expression from hidden layer values
- Heatmaps (safe-by-default): `PlotPartialCorrelationHeatmap()`, `PlotRBMWeightsHeatmap()`
- Graphs (safe-by-default): `PlotPartialCorrelationGraph()`, `PlotRBMLayerGraph()`
- Backward-compatible wrappers: `PlotRBMHeatmap()`, `PlotRBMWeights()`

### Quick Start (Recommended Workflow)

```r
library(PreGraphModeling)
library(ComplexHeatmap)

# Step 1: Pre-compute partial correlations (expensive, do once)
pcor <- EstimatePartialCorrelationsFromSeurat(
  seuratObject = pbmc,
  family = "zinb",
  parallel = TRUE
)

# Inspect the partial correlations
print(pcor)

# Step 2: Fit RBM with pre-computed correlations
rbm <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = "CellType",
  partialCorrelations = pcor,  # Reuse pre-computed!
  cd_k = 5,
  n_epochs = 20,
  parallel = TRUE
)

# Reuse for another RBM (no recomputation!)
rbm2 <- FitRBM(pbmc, hiddenFactors = c("CellType", "Treatment"), partialCorrelations = pcor)

# Visualize
draw(PlotPartialCorrelationHeatmap(rbm, nEdges = 50, color_palette = "RdBu"))

# RBM weight heatmap (selects ~nFeatures across layers)
draw(PlotRBMWeightsHeatmap(rbm, nFeatures = 50))

# Graph views
PlotPartialCorrelationGraph(rbm, nEdges = 50, nFeatures = 50)
PlotRBMLayerGraph(rbm, nFeatures = 50)
```

### Features
- Multiple hidden layer types with type-specific activations:
  - Binary factors → Bernoulli/sigmoid
  - Categorical factors → Softmax
  - Ordinal/Continuous factors → Gaussian/linear
- CD-k and Persistent Contrastive Divergence training
- Parallel processing for partial correlation estimation
- Visualization with ComplexHeatmap

### RBM Architecture

The RBM connects:
- **Visible layer**: Gene expression matrix (genes × cells), modeled with ZINB
- **Hidden layers**: One per metadata factor, type-specific activations
- **No hidden-to-hidden connections**: Layers are conditionally independent

### Detailed Example

```r
# Fit RBM with multiple hidden factors
rbm <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = c("CellType", "Treatment", "Batch"),
  family = "zinb",
  cd_k = 5,
  n_epochs = 20,
  learning_rate = 0.01,
  persistent = TRUE,  # Persistent CD
  parallel = TRUE,
  numWorkers = 4
)

# Predict on new data
predictions <- predict(rbm, newdata = new_expr_matrix)

# Reconstruct expression patterns
reconstructed <- ReconstructRBM(rbm)
```

> **See vignette:** `vignette("rbm_analysis")` for complete workflows

---

## Parallel Processing

Both approaches support parallel processing:

```r
# For ZINB pipeline
result <- AnalyzeWithZINB(
  seuratObject = pbmc,
  groupByColumns = "CellType",
  parallel = TRUE,
  numWorkers = 4,
  parallelPlan = "multisession"  # Options: multisession, multicore, cluster
)

# For RBM/partial correlations
rbm <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = "CellType",
  parallel = TRUE,
  numWorkers = 4
)
```

---

## Approach 3: CONGA (Conditional Graphical Models)

This approach implements a simplified version of the CONGA algorithm for estimating conditional dependency graphs between genes using Bayesian MCMC methods. Unlike correlation-based methods, CONGA models the full conditional probability structure and can handle count data directly.

### Key Functions
- `FitCONGA()` - Fit CONGA model to Seurat object (recommended)
- `FitCONGAModel()` - Fit CONGA model to expression matrix
- `ExtractCONGAGraph()` - Extract conditional dependency graph from MCMC samples
- `ComputeCONGAROC()` - Compute ROC curve for evaluation (if true graph is known)

### Quick Start

```r
library(PreGraphModeling)

# Recommended: Use feature selection for computational efficiency
library(Seurat)
pbmc <- FindVariableFeatures(pbmc, nfeatures = 50)
hvg <- VariableFeatures(pbmc)

# Fit CONGA model (this may take several minutes)
result <- FitCONGA(
  seuratObject = pbmc,
  geneSubset = hvg,
  totalIterations = 1000,
  burnIn = 500,
  verbose = TRUE
)

# Extract conditional dependency graph
graph <- ExtractCONGAGraph(result, cutoff = 0.7)

# View top edges
print(graph)

# Visualize adjacency matrix
library(ComplexHeatmap)
Heatmap(graph$adjacency_matrix, 
        name = "Edge",
        col = c("white", "black"))

# Convert to igraph for network analysis
library(igraph)
g <- graph_from_adjacency_matrix(graph$adjacency_matrix, 
                                  mode = "undirected")
plot(g, vertex.size = 10, vertex.label.cex = 0.8)
```

### Features
- Bayesian MCMC sampling for uncertainty quantification
- Handles count data directly (no normalization required)
- Sparse graph estimation via spike-and-slab priors
- Power transformation to handle non-Gaussian structure
- Posterior edge probabilities for graph construction

### Important Notes

**Computational Considerations:**
- CONGA is computationally intensive: O(iterations × cells × genes²)
- Recommended: Use feature selection (50-200 genes) for practical runtimes
- Typical runtime: 5-15 minutes for 1000 cells × 50 genes with 1000 iterations

**Implementation:**
This is a **simplified version** of the full CONGA algorithm for computational tractability:
- Uses standard Metropolis-Hastings instead of Dirichlet Process clustering
- Element-wise beta updates instead of blocked Gibbs sampling
- Approximate likelihood computations

These simplifications make the algorithm practical for real datasets but may require:
- Longer MCMC runs for convergence
- More careful tuning of parameters
- Thorough convergence diagnostics

### Detailed Example

```r
# Load data
library(Seurat)
data("pbmc_small")

# Select genes of interest (keep it small for faster computation)
genes <- c("CD3D", "CD4", "CD8A", "MS4A1", "CD14", 
           "LYZ", "GNLY", "NKG7", "FCGR3A", "IL7R")

# Fit CONGA
result <- FitCONGA(
  seuratObject = pbmc_small,
  geneSubset = genes,
  totalIterations = 2000,
  burnIn = 1000,
  lambdaShrinkage = 1,
  verbose = TRUE
)

# Check acceptance rates (target 20-40% for good mixing)
cat(sprintf("Lambda acceptance: %.1f%%\n", 
            result$acceptance_rate_lambda * 100))
cat(sprintf("Beta acceptance: %.1f%%\n", 
            result$acceptance_rate_beta * 100))

# Extract graph with different cutoffs
graph_sparse <- ExtractCONGAGraph(result, cutoff = 0.9)  # Very sparse
graph_medium <- ExtractCONGAGraph(result, cutoff = 0.7)  # Moderate
graph_dense <- ExtractCONGAGraph(result, cutoff = 0.5)   # Less sparse

# Compare edge counts
cat(sprintf("Sparse: %d edges\n", graph_sparse$n_edges))
cat(sprintf("Medium: %d edges\n", graph_medium$n_edges))
cat(sprintf("Dense: %d edges\n", graph_dense$n_edges))

# Examine specific edges
edges_df <- graph_medium$edge_list
edges_df <- edges_df[edges_df$included, ]
edges_df <- edges_df[order(-edges_df$posterior_prob), ]
head(edges_df)
```

> **See documentation:** `?FitCONGA` for complete details and mathematical formulation

---

## Function Reference

### ZINB Modeling
| Function | Description |
|----------|-------------|
| `SubsetSeurat()` | Subset Seurat objects by metadata columns |
| `FitZeroInflatedModels()` | Fit ZINB models to expression data |
| `AnalyzeWithZINB()` | Complete workflow: subsetting + modeling |

### RBM Functionality
| Function | Description |
|----------|-------------|
| `EstimatePartialCorrelationsFromSeurat()` | Compute partial correlations from Seurat object (recommended) |
| `EstimatePartialCorrelations()` | Compute partial correlations from expression matrix |
| `FitRBM()` | Fit RBM to Seurat object (accepts pre-computed correlations) |
| `predict.RBM()` | Predict hidden activations from expression |
| `ReconstructRBM()` | Reconstruct expression from hidden layer |
| `PlotPartialCorrelationHeatmap()` | Heatmap of strongest partial correlations (safe default) |
| `PlotRBMWeightsHeatmap()` | Heatmap of strongest RBM weights (safe default) |
| `PlotPartialCorrelationGraph()` | Graph view of partial correlations (safe default) |
| `PlotRBMLayerGraph()` | Bipartite graph: features → hidden layers (safe default) |
| `PlotRBMHeatmap()` | Backward-compatible wrapper for partial-correlation heatmap |
| `PlotRBMWeights()` | Backward-compatible wrapper for weights heatmap |

### CONGA Functionality
| Function | Description |
|----------|-------------|
| `FitCONGA()` | Fit CONGA model to Seurat object (recommended) |
| `FitCONGAModel()` | Fit CONGA model to expression matrix |
| `ExtractCONGAGraph()` | Extract conditional dependency graph |
| `ComputeCONGAROC()` | Compute ROC curve for evaluation |

---

## Requirements

### Required
- R >= 4.0.0
- SeuratObject, pscl, DESeq2, mgcv, future, future.apply
- Rcpp, RcppArmadillo, mvtnorm, combinat, MCMCpack (for CONGA)

### Optional
- Seurat (for full object support)
- ComplexHeatmap, circlize, viridisLite (for RBM heatmaps)
- igraph (for RBM and CONGA graph visualizations)
- progressr (for progress tracking)

---

## Naming Conventions

- **Functions**: PascalCase (e.g., `SubsetSeurat`)
- **Internal variables**: snake_case
- **External parameters**: camelCase (e.g., `groupByColumns`, `minNonZero`)

## Testing

```r
devtools::test()
devtools::check()
```

## License

GPL-3