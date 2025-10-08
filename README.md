# PreGraphModeling

R package for pseudobulking Seurat objects and fitting zero-inflated negative binomial (ZINB) models to estimate distribution parameters.

## Installation

```r
# Install from GitHub
devtools::install_github("GWMcElfresh/PreGraphModeling")
```

## Features

- **Pseudobulking**: Aggregate single-cell data by metadata columns
  - Supports multiple metadata columns for fine-grained grouping
  - Works with Seurat and SeuratObject
  
- **Zero-Inflated Modeling**: Fit ZINB models using `pscl` package
  - Estimates mean (mu), dispersion (phi), and zero-inflation probability (pi)
  - Handles sparse single-cell count data effectively

## Usage

### Basic Pseudobulking

```r
library(PreGraphModeling)
library(Seurat)

# Pseudobulk by a single metadata column
result <- PseudobulkSeurat(seurat_obj, groupByColumns = "CellType")

# Access the pseudobulked expression matrix
pseudobulk_expr <- result$pseudobulk_matrix

# View metadata for pseudobulk samples
print(result$group_metadata)
```

### Pseudobulking with Multiple Columns

```r
# Pseudobulk by multiple metadata columns
result <- PseudobulkSeurat(
  seurat_obj, 
  groupByColumns = c("CellType", "Sample", "Condition")
)
```

### Fit ZINB Models

```r
# Fit zero-inflated negative binomial models to pseudobulked data
zinb_params <- FitZeroInflatedModels(
  expressionMatrix = result$pseudobulk_matrix,
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
# Perform pseudobulking and ZINB modeling in one step
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Condition"),
  minNonZero = 3,
  verbose = TRUE
)

# Access results
pseudobulk_matrix <- result$pseudobulk_matrix
group_metadata <- result$group_metadata
model_params <- result$model_parameters
```

### Subset Analysis

```r
# Analyze only specific genes
genes_of_interest <- c("CD3D", "CD4", "CD8A", "CD19", "MS4A1")

result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = "CellType",
  geneSubset = genes_of_interest
)
```

## Function Details

### Exported Functions

- `PseudobulkSeurat()`: Pseudobulk Seurat objects by metadata columns
- `FitZeroInflatedModels()`: Fit ZINB models to expression data
- `AnalyzeWithZINB()`: Complete workflow combining pseudobulking and modeling

### Naming Conventions

The package follows strict naming conventions:
- **Functions**: PascalCase (e.g., `PseudobulkSeurat`)
- **Internal variables**: snake_case
- **External parameters**: camelCase (e.g., `groupByColumns`, `minNonZero`)

## Parameters

### ZINB Model Parameters

The ZINB model estimates three key parameters for each gene:

- **mu (μ)**: Mean of the negative binomial distribution
- **phi (φ)**: Dispersion parameter (theta in NB parameterization)
- **pi (π)**: Probability of excess zeros from zero-inflation component

## Testing

The package includes comprehensive tests using `testthat`:

```r
# Run tests
devtools::test()

# Run R CMD check
devtools::check()
```

## Requirements

- R >= 4.0.0
- SeuratObject
- pscl
- Matrix
- methods

## License

GPL-3