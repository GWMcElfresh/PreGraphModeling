#' Subset and Fit ZINB Models in One Step
#'
#' This convenience function combines subsetting and zero-inflated negative
#' binomial model fitting into a single step. It subsets a Seurat object
#' according to specified metadata columns and then fits ZINB models to each
#' subset to estimate distribution parameters.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data
#' @param groupByColumns Character vector of metadata column names to group by.
#'   Multiple columns can be specified to create fine-grained subsets.
#' @param assay Character string specifying which assay to use (default: "RNA")
#' @param slot Character string specifying which slot to use (default: "counts")
#' @param geneSubset Optional character vector of gene names to fit models for.
#'   If NULL, models are fit for all genes (default: NULL).
#' @param minNonZero Minimum number of non-zero observations required to fit a model
#'   (default: 3).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return A list with three elements:
#'   \itemize{
#'     \item subset_matrices: List of expression matrices, one per subset
#'     \item group_metadata: Metadata for each subset
#'     \item model_parameters: List of data frames with estimated ZINB parameters for each subset
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Complete analysis with single metadata column
#' result <- AnalyzeWithZINB(seurat_obj, groupByColumns = "CellType")
#' 
#' # Complete analysis with multiple metadata columns
#' result <- AnalyzeWithZINB(seurat_obj, 
#'                          groupByColumns = c("CellType", "Sample"))
#' 
#' # Access results for first subset
#' head(result$model_parameters[[1]])
#' }
AnalyzeWithZINB <- function(seuratObject,
                           groupByColumns,
                           assay = "RNA",
                           slot = "counts",
                           geneSubset = NULL,
                           minNonZero = 3,
                           verbose = TRUE) {
  
  if (verbose) {
    message("Step 1: Subsetting Seurat object by metadata columns...")
  }
  
  # Perform subsetting
  subset_result <- SubsetSeurat(seuratObject = seuratObject,
                                groupByColumns = groupByColumns,
                                assay = assay,
                                slot = slot)
  
  if (verbose) {
    message(sprintf("  Created %d subsets", length(subset_result$subset_matrices)))
    message("\nStep 2: Fitting zero-inflated negative binomial models for each subset...")
  }
  
  # Fit ZINB models for each subset
  model_params_list <- list()
  
  for (i in seq_along(subset_result$subset_matrices)) {
    subset_name <- names(subset_result$subset_matrices)[i]
    
    if (verbose) {
      message(sprintf("  Fitting models for subset: %s", subset_name))
    }
    
    model_params_list[[subset_name]] <- FitZeroInflatedModels(
      expressionMatrix = subset_result$subset_matrices[[i]],
      geneSubset = geneSubset,
      minNonZero = minNonZero,
      verbose = FALSE
    )
    
    # Add subset identifier to results
    model_params_list[[subset_name]]$subset <- subset_name
  }
  
  if (verbose) {
    message("\nAnalysis complete!")
  }
  
  return(list(
    subset_matrices = subset_result$subset_matrices,
    group_metadata = subset_result$group_metadata,
    model_parameters = model_params_list
  ))
}
