# PreGraphModeling

R package for subsetting Seurat objects by metadata and fitting zero-inflated negative binomial (ZINB) models to estimate distribution parameters for each subset.

## Installation

```r
# Install from GitHub
devtools::install_github("GWMcElfresh/PreGraphModeling")
```

## Features

- **Subsetting**: Divide single-cell data by metadata columns
  - Supports multiple metadata columns for fine-grained subsets
  - Works with Seurat and SeuratObject
  - Each unique combination of metadata values creates a separate subset
  
- **Zero-Inflated Modeling**: Fit ZINB models using `pscl` package
  - Estimates mean (mu), dispersion (phi), and zero-inflation probability (pi)
  - Fits separate models to each subset
  - Handles sparse single-cell count data effectively

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
result <- AnalyzeWithZINB(
  seuratObject = seurat_obj,
  groupByColumns = c("CellType", "Treatment"),
  minNonZero = 3,
  verbose = TRUE
)

# Access results
subset_matrices <- result$subset_matrices  # List of expression matrices
group_metadata <- result$group_metadata    # Data frame with subset info
model_params <- result$model_parameters    # List of parameter data frames

# View parameters for first subset
head(result$model_parameters[[1]])

# View parameters for a specific subset by name
head(result$model_parameters[["TypeA_Control"]])
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

## Function Details

### Exported Functions

- `SubsetSeurat()`: Subset Seurat objects by metadata columns
- `FitZeroInflatedModels()`: Fit ZINB models to expression data
- `AnalyzeWithZINB()`: Complete workflow combining subsetting and modeling for all subsets
- `PseudobulkSeurat()`: Alias for `SubsetSeurat()` (for backward compatibility)

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

# Create subsets for each CellType + Treatment combination
result <- AnalyzeWithZINB(
  seuratObject = pbmc,
  groupByColumns = c("CellType", "Treatment"),
  geneSubset = c("CD3D", "CD4", "CD8A", "MS4A1"),
  minNonZero = 5
)

# Compare CD3D expression parameters across subsets
cd3d_params <- lapply(result$model_parameters, function(df) {
  df[df$gene == "CD3D", ]
})

# Combine into one data frame for easy comparison
cd3d_comparison <- do.call(rbind, cd3d_params)
print(cd3d_comparison)
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

- R >= 4.0.0
- SeuratObject
- pscl
- Matrix
- methods

## License

GPL-3