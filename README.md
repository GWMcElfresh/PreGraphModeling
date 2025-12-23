# PreGraphModeling

R package for single-cell RNA-seq analysis with two complementary approaches:
1. **Per-Gene ZINB Modeling** - Estimate zero-inflated negative binomial distribution parameters for each gene across cell subsets
2. **Restricted Boltzmann Machine (RBM)** - Model relationships between gene expression features and cell metadata using partial correlations

## Installation

```r
# Install from GitHub
devtools::install_github("GWMcElfresh/PreGraphModeling")

# Install optional dependencies for RBM visualization
BiocManager::install(c("ComplexHeatmap", "circlize"))
install.packages(c("progressr", "viridisLite"))

# Optional: graph visualizations
install.packages("igraph")
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

---

## Requirements

### Required
- R >= 4.0.0
- SeuratObject, pscl, DESeq2, mgcv, future, future.apply

### Optional
- Seurat (for full object support)
- ComplexHeatmap, circlize, viridisLite (for RBM heatmaps)
- igraph (for RBM graph visualizations)
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