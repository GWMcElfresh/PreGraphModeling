#' Pseudobulk and Fit ZINB Models in One Step
#'
#' This convenience function combines pseudobulking and zero-inflated negative
#' binomial model fitting into a single step. It pseudobulks a Seurat object
#' according to specified metadata columns and then fits ZINB models to estimate
#' distribution parameters.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data
#' @param groupByColumns Character vector of metadata column names to group by.
#'   Multiple columns can be specified to create fine-grained pseudobulk groups.
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
#'     \item pseudobulk_matrix: The pseudobulked expression matrix
#'     \item group_metadata: Metadata for each pseudobulk sample
#'     \item model_parameters: Data frame with estimated ZINB parameters for each gene
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
#' # Access results
#' head(result$model_parameters)
#' }
AnalyzeWithZINB <- function(seuratObject,
                           groupByColumns,
                           assay = "RNA",
                           slot = "counts",
                           geneSubset = NULL,
                           minNonZero = 3,
                           verbose = TRUE) {
  
  if (verbose) {
    message("Step 1: Pseudobulking Seurat object...")
  }
  
  # Perform pseudobulking
  pb_result <- PseudobulkSeurat(seuratObject = seuratObject,
                                groupByColumns = groupByColumns,
                                assay = assay,
                                slot = slot)
  
  if (verbose) {
    message(sprintf("  Created %d pseudobulk samples", ncol(pb_result$pseudobulk_matrix)))
    message("\nStep 2: Fitting zero-inflated negative binomial models...")
  }
  
  # Fit ZINB models
  model_params <- FitZeroInflatedModels(expressionMatrix = pb_result$pseudobulk_matrix,
                                        geneSubset = geneSubset,
                                        minNonZero = minNonZero,
                                        verbose = verbose)
  
  if (verbose) {
    message("\nAnalysis complete!")
  }
  
  return(list(
    pseudobulk_matrix = pb_result$pseudobulk_matrix,
    group_metadata = pb_result$group_metadata,
    model_parameters = model_params
  ))
}
