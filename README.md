# PreGraphModeling

R package for subsetting Seurat objects by metadata and fitting zero-inflated negative binomial (ZINB) models to estimate distribution parameters for each subset. Additionally includes **Restricted Boltzmann Machine (RBM)** functionality for modeling relationships between gene expression features and cell metadata using partial correlations via quasilikelihood.

## Installation

```r
# Install from GitHub
devtools::install_github("GWMcElfresh/PreGraphModeling")

# Install optional dependencies for RBM visualization
BiocManager::install(c("ComplexHeatmap", "circlize"))
install.packages(c("progressr", "viridisLite"))
```

## Features

- **Subsetting**: Divide single-cell data by metadata columns
  - Supports multiple metadata columns for fine-grained subsets
  - Works with Seurat and SeuratObject
  - Each unique combination of metadata values creates a separate subset
  - Optional extraction of cellular saturation values for technical covariate correction
  
- **Zero-Inflated Modeling**: Fit ZINB models using `pscl` package
  - Estimates mean (mu), dispersion (phi), and zero-inflation probability (pi)
  - Fits separate models to each subset
  - Handles sparse single-cell count data effectively
  - Returns total datapoints (n_datapoints) for each gene
  - Optional residualization of size factors against cellular saturation using GAM

- **Restricted Boltzmann Machine (RBM)**: Connect features to hidden layers
  - Estimate partial correlations via quasilikelihood for ZINB, NB, Poisson, or Gaussian families
  - Fit RBM connecting gene expression features to metadata factors
  - Parallel processing with `progressr` for progress tracking at scale
  - Visualization using ComplexHeatmap for partial correlations and weights
  - Predict hidden layer activations and reconstruct expression patterns

- **Key-based Results**: Easy joining and merging
  - Delimited keys (e.g., "CellType1|Treatment1") for each subset
  - `key_colnames` field indicates the join order
  - Combined data frame with all results merged

- **Parallel Processing**: Speed up analysis on large datasets
  - Defaults to `multisession` for memory safety across all platforms
  - Configurable parallel plan (multisession, multicore, cluster)
  - Configurable number of workers
  - Uses `future` and `future.apply` packages

- **Timing Information**: Track performance
  - Reports elapsed time for each step (subsetting, model fitting, merging)
  - Helps identify bottlenecks in analysis pipeline

## Usage

### Basic Subsetting

```r
library(PreGraphModeling)
library(Seurat)

# Subset by a single metadata column
result <- SubsetSeurat(seurat_obj, groupByColumns = "CellType")

# Access the subset expression matrices (one per cell type)
subset_matrices <- result$subset_matrices

# View metadata for subsets
print(result$group_metadata)
```

### Subsetting with Multiple Columns

```r
# Subset by multiple metadata columns
# Example: CellType=[TypeA, TypeB] + Condition=[Control, Treatment]
# Creates 4 subsets: TypeA_Control, TypeA_Treatment, TypeB_Control, TypeB_Treatment
result <- SubsetSeurat(
  seurat_obj, 
  groupByColumns = c("CellType", "Condition")
)

# Each subset contains only the cells matching that combination
names(result$subset_matrices)
# [1] "TypeA_Control"    "TypeA_Treatment"  "TypeB_Control"    "TypeB_Treatment"
```

### Fit ZINB Models to a Subset

```r
# Fit zero-inflated negative binomial models to one subset
zinb_params <- FitZeroInflatedModels(
  expressionMatrix = result$subset_matrices[[1]],
  minNonZero = 3,
  verbose = TRUE
)

# View estimated parameters
head(zinb_params)
#   gene      mu     phi       pi converged n_nonzero
# 1 Gene1  12.45   2.34    0.15      TRUE        20
# 2 Gene2   8.92   1.87    0.22      TRUE        18
```

### Combined Analysis

```r
# Perform subsetting and ZINB modeling for all subsets in one step
# By default, uses "Saturation.RNA" column to residualize size factors
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  minNonZero = 3,
  verbose = TRUE
)

# Use a different saturation column
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  saturationColumn = "percent.mito",  # or any other technical covariate
  minNonZero = 3,
  verbose = TRUE
)

# Disable saturation correction
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  saturationColumn = NULL,  # No residualization
  minNonZero = 3,
  verbose = TRUE
)

# Access results
subset_matrices <- result$subset_matrices  # List of expression matrices
group_metadata <- result$group_metadata    # Data frame with subset info
model_params <- result$model_parameters    # List of parameter data frames
combined_params <- result$combined_parameters  # Single merged data frame with keys
timing <- result$timing  # Timing information for each step

# View parameters for first subset
head(result$model_parameters[[1]])

# View parameters for a specific subset by name
head(result$model_parameters[["TypeA_Control"]])

# View combined parameters with keys
head(result$combined_parameters)
#   gene    mu   phi   pi converged n_nonzero n_datapoints            key    key_colnames
# 1 Gene1 12.5  2.34 0.15      TRUE        18           20 TypeA_Control CellType|Treatment
# 2 Gene2  8.9  1.87 0.22      TRUE        16           20 TypeA_Control CellType|Treatment

# Check timing information
print(result$timing)
#            step elapsed_seconds
# 1    Subsetting            0.15
# 2 Model Fitting            2.35
# 3  Data Merging            0.08
```

### Parallel Processing

For large datasets, enable parallel processing to speed up model fitting. The default uses `multisession` plan for memory safety:

```r
# Use parallel processing with auto-detected number of cores (default: multisession)
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  parallel = TRUE,
  verbose = TRUE
)

# Specify number of workers and parallel plan explicitly
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  parallel = TRUE,
  numWorkers = 4,
  parallelPlan = "multisession",  # Options: "multisession", "multicore", "cluster"
  verbose = TRUE
)
```

### Subset Analysis with Gene Selection

```r
# Analyze only specific genes across all subsets
genes_of_interest <- c("CD3D", "CD4", "CD8A", "CD19", "MS4A1")

result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  geneSubset = genes_of_interest
)

# Each subset will have models fit for only the genes of interest
```

## Restricted Boltzmann Machine (RBM) Analysis

### Basic RBM Workflow

```r
library(PreGraphModeling)
library(ComplexHeatmap)
library(progressr)

# Fit RBM connecting gene expression to cell type metadata
rbm <- FitRBM(
  seuratObject = pbmc,
  visibleFeatures = NULL,  # NULL = use all features
  hiddenFactors = "CellType",
  family = "zinb",  # Zero-inflated negative binomial
  minNonZero = 10,
  parallel = TRUE,
  progressr = TRUE,
  verbose = TRUE
)

# View RBM summary
print(rbm)
summary(rbm)
```

### Visualize Partial Correlations

```r
# Create heatmap of partial correlations between genes
heatmap_pcor <- PlotRBMHeatmap(
  rbmObject = rbm,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  color_palette = "RdBu",
  title = "Gene-Gene Partial Correlations"
)

# Draw the heatmap
draw(heatmap_pcor)

# Focus on specific marker genes
marker_genes <- c("CD3D", "CD8A", "CD4", "CD19", "MS4A1")
heatmap_markers <- PlotRBMHeatmap(
  rbmObject = rbm,
  features = marker_genes,
  title = "Marker Gene Correlations"
)
draw(heatmap_markers)
```

### Visualize RBM Weights

```r
# Create heatmap of weights connecting features to metadata
heatmap_weights <- PlotRBMWeights(
  rbmObject = rbm,
  cluster_rows = TRUE,
  cluster_columns = FALSE,
  title = "Feature-Metadata Connections"
)
draw(heatmap_weights)
```

### Make Predictions

```r
# Predict hidden layer (metadata) from new expression data
new_expr <- GetAssayData(pbmc_new, assay = "RNA", slot = "counts")
predictions <- predict(rbm, newdata = new_expr, type = "activation")

# Get probability-transformed predictions
probs <- predict(rbm, newdata = new_expr, type = "probability")
```

### Estimate Partial Correlations Directly

```r
# Compute partial correlations without full RBM fitting
expr_matrix <- GetAssayData(pbmc, assay = "RNA", slot = "counts")

pcor_result <- EstimatePartialCorrelations(
  expressionMatrix = expr_matrix,
  family = "zinb",  # Options: "zinb", "nb", "poisson", "gaussian"
  minNonZero = 10,
  parallel = TRUE,
  progressr = TRUE
)

# Access the partial correlation matrix
pcor_matrix <- pcor_result$partial_cor
```

### Multiple Hidden Factors

```r
# Fit RBM with multiple metadata factors
rbm_multi <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = c("CellType", "Treatment", "Batch"),
  family = "zinb",
  parallel = TRUE
)

# Visualize weights for each factor
heatmap_multi <- PlotRBMWeights(
  rbmObject = rbm_multi,
  features = marker_genes,
  factors = "CellType"
)
```

## Function Details

### Exported Functions

#### ZINB Modeling
- `SubsetSeurat()`: Subset Seurat objects by metadata columns
- `FitZeroInflatedModels()`: Fit ZINB models to expression data
- `AnalyzeWithZINB()`: Complete workflow combining subsetting and modeling for all subsets

#### RBM Functionality
- `EstimatePartialCorrelations()`: Compute partial correlations via quasilikelihood
- `FitRBM()`: Fit restricted Boltzmann machine to Seurat object
- `predict.RBM()`: Predict hidden layer activations from expression
- `ReconstructRBM()`: Reconstruct expression from hidden layer
- `PlotRBMHeatmap()`: Visualize partial correlations as heatmap
- `PlotRBMWeights()`: Visualize RBM weights as heatmap


### Naming Conventions

The package follows strict naming conventions:
- **Functions**: PascalCase (e.g., `SubsetSeurat`)
- **Internal variables**: snake_case
- **External parameters**: camelCase (e.g., `groupByColumns`, `minNonZero`)

## Parameters

### ZINB Model Parameters

The ZINB model estimates three key parameters for each gene in each subset:

- **mu (μ)**: Mean of the negative binomial distribution
- **phi (φ)**: Dispersion parameter (theta in NB parameterization)
- **pi (π)**: Probability of excess zeros from zero-inflation component

## Example Workflow

```r
# Example: Compare T cells vs B cells under different treatments
library(PreGraphModeling)

# Create subsets for each CellType + Treatment combination with parallel processing
result <- AnalyzeWithZINB(
  seuratObject = pbmc,
  groupByColumns = c("CellType", "Treatment"),
  geneSubset = c("CD3D", "CD4", "CD8A", "MS4A1"),
  minNonZero = 5,
  parallel = TRUE
)

# Use the combined_parameters data frame with keys for easy filtering
cd3d_data <- result$combined_parameters[result$combined_parameters$gene == "CD3D", ]
print(cd3d_data)
#   gene   mu  phi   pi converged n_nonzero n_datapoints               key       key_colnames
# 1 CD3D 15.2 3.1 0.12      TRUE        45           50  Tcell_Control CellType|Treatment
# 2 CD3D 18.7 3.4 0.10      TRUE        48           50 Tcell_Treatment CellType|Treatment
# 3 CD3D  2.1 1.2 0.45      TRUE        15           50  Bcell_Control CellType|Treatment
# 4 CD3D  2.3 1.3 0.43      TRUE        17           50 Bcell_Treatment CellType|Treatment

# Join with external data using the key
# The key_colnames field tells you the order: "CellType|Treatment"
external_data <- data.frame(
  key = c("Tcell_Control", "Tcell_Treatment", "Bcell_Control", "Bcell_Treatment"),
  condition_label = c("T-Ctrl", "T-Trt", "B-Ctrl", "B-Trt"),
  expected_expression = c("High", "High", "Low", "Low")
)

merged_results <- merge(cd3d_data, external_data, by = "key")
print(merged_results)
```

## Testing

The package includes comprehensive tests using `testthat`:

```r
# Run tests
devtools::test()

# Run R CMD check
devtools::check()
```

## Requirements

### Required
- R >= 4.0.0
- SeuratObject
- pscl
- Matrix
- methods
- parallel
- DESeq2
- mgcv
- stats

### Optional (for full functionality)
- future, future.apply (for parallel processing)
- HDF5Array, DelayedArray (for advanced data handling)
- progressr (for progress tracking in RBM)
- ComplexHeatmap, circlize (for RBM visualization)
- viridisLite (for color palettes)
- knitr, rmarkdown (for vignettes)

### Optional (for parallel processing)
- future
- future.apply

### Optional (for advanced data handling)
- HDF5Array
- DelayedArray

## License

GPL-3