#' Subset and Fit ZINB Models in One Step
#'
#' This convenience function combines subsetting and zero-inflated negative
#' binomial model fitting into a single step. It subsets a Seurat object
#' according to specified metadata columns and then fits ZINB models to each
#' subset to estimate distribution parameters. Supports parallel processing.
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
#' @param parallel Logical indicating whether to use parallel processing (default: FALSE).
#' @param numWorkers Integer specifying number of workers for parallel processing.
#'   If NULL, uses detectCores()-1 (default: NULL).
#' @param parallelPlan Character string specifying the future plan to use for parallel processing.
#'   Options: "multisession" (default, memory-safe), "multicore" (Unix only, faster but not memory-safe),
#'   "cluster". If NULL, defaults to "multisession" (default: NULL).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return A list with four elements:
#'   \itemize{
#'     \item subset_matrices: List of expression matrices, one per subset
#'     \item group_metadata: Metadata for each subset
#'     \item model_parameters: List of data frames with estimated ZINB parameters for each subset
#'     \item combined_parameters: Single data frame with all parameters merged, including key and key_colnames
#'     \item timing: Data frame with elapsed time for each step
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Complete analysis with single metadata column
#' result <- AnalyzeWithZINB(seurat_obj, groupByColumns = "CellType")
#' 
#' # Complete analysis with multiple metadata columns and parallel processing
#' result <- AnalyzeWithZINB(seurat_obj, 
#'                          groupByColumns = c("CellType", "Sample"),
#'                          parallel = TRUE,
#'                          parallelPlan = "multisession")
#' 
#' # Access results for first subset
#' head(result$model_parameters[[1]])
#' 
#' # Access combined parameters with keys
#' head(result$combined_parameters)
#' }
AnalyzeWithZINB <- function(seuratObject,
                           groupByColumns,
                           assay = "RNA",
                           slot = "counts",
                           geneSubset = NULL,
                           minNonZero = 3,
                           parallel = FALSE,
                           numWorkers = NULL,
                           parallelPlan = NULL,
                           verbose = TRUE) {
  
  # ============================================================================
  # INITIALIZE TIMING TRACKING
  # ============================================================================
  timing_data <- data.frame(
    step = character(),
    elapsed_seconds = numeric(),
    stringsAsFactors = FALSE
  )
  
  # ============================================================================
  # STEP 1: SUBSET SEURAT OBJECT BY METADATA
  # ============================================================================
  step1_start <- Sys.time()
  if (verbose) {
    message("Step 1: Subsetting Seurat object by metadata columns...")
  }
  
  subset_result <- SubsetSeurat(seuratObject = seuratObject,
                                groupByColumns = groupByColumns,
                                assay = assay,
                                slot = slot)
  
  step1_elapsed <- as.numeric(difftime(Sys.time(), step1_start, units = "secs"))
  timing_data <- rbind(timing_data, data.frame(
    step = "Subsetting",
    elapsed_seconds = step1_elapsed,
    stringsAsFactors = FALSE
  ))
  
  if (verbose) {
    message(sprintf("  Created %d subsets (%.2f seconds)", 
                   length(subset_result$subset_matrices), step1_elapsed))
    message("\nStep 2: Fitting zero-inflated negative binomial models for each subset...")
  }
  
  # ============================================================================
  # STEP 2: FIT ZINB MODELS (PARALLEL OR SEQUENTIAL)
  # ============================================================================
  step2_start <- Sys.time()
  
  if (parallel) {
    # Check if future package is available
    if (!requireNamespace("future", quietly = TRUE) || 
        !requireNamespace("future.apply", quietly = TRUE)) {
      warning("Packages 'future' and 'future.apply' required for parallel processing. Falling back to sequential.")
      parallel <- FALSE
    }
  }
  
  if (parallel) {
    # Set up parallel processing
    if (is.null(numWorkers)) {
      numWorkers <- max(1, parallel::detectCores() - 1)
    }
    
    # Default to multisession for memory safety
    if (is.null(parallelPlan)) {
      parallelPlan <- "multisession"
    }
    
    if (verbose) {
      message(sprintf("  Using parallel processing with %d workers (plan: %s)", 
                     numWorkers, parallelPlan))
    }
    
    # Set future plan based on user specification
    if (parallelPlan == "multisession") {
      future::plan(future::multisession, workers = numWorkers)
    } else if (parallelPlan == "multicore") {
      future::plan(future::multicore, workers = numWorkers)
    } else if (parallelPlan == "cluster") {
      future::plan(future::cluster, workers = numWorkers)
    } else {
      stop("Invalid parallelPlan. Must be 'multisession', 'multicore', or 'cluster'")
    }
    
    # Fit models in parallel
    model_params_list <- future.apply::future_lapply(
      names(subset_result$subset_matrices),
      function(subset_name) {
        params <- FitZeroInflatedModels(
          expressionMatrix = subset_result$subset_matrices[[subset_name]],
          geneSubset = geneSubset,
          minNonZero = minNonZero,
          verbose = FALSE
        )
        params$subset <- subset_name
        return(params)
      },
      future.seed = TRUE
    )
    names(model_params_list) <- names(subset_result$subset_matrices)
    
    # Reset to sequential plan
    future::plan(future::sequential)
    
  } else {
    # Sequential processing
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
  }
  
  step2_elapsed <- as.numeric(difftime(Sys.time(), step2_start, units = "secs"))
  timing_data <- rbind(timing_data, data.frame(
    step = "Model Fitting",
    elapsed_seconds = step2_elapsed,
    stringsAsFactors = FALSE
  ))
  
  if (verbose) {
    message(sprintf("  Model fitting complete (%.2f seconds)", step2_elapsed))
    message("\nStep 3: Merging results and creating keys...")
  }
  
  # ============================================================================
  # STEP 3: MERGE RESULTS AND CREATE KEY-BASED OUTPUT
  # ============================================================================
  step3_start <- Sys.time()
  
  # Create key_colnames string
  key_colnames <- paste(groupByColumns, collapse = "|")
  
  # Combine all parameter data frames with keys
  combined_params <- do.call(rbind, lapply(names(model_params_list), function(subset_name) {
    params <- model_params_list[[subset_name]]
    params$key <- subset_name
    params$key_colnames <- key_colnames
    return(params)
  }))
  rownames(combined_params) <- NULL
  
  step3_elapsed <- as.numeric(difftime(Sys.time(), step3_start, units = "secs"))
  timing_data <- rbind(timing_data, data.frame(
    step = "Data Merging",
    elapsed_seconds = step3_elapsed,
    stringsAsFactors = FALSE
  ))
  
  if (verbose) {
    message(sprintf("  Results merged (%.2f seconds)", step3_elapsed))
    message(sprintf("\nAnalysis complete! Total time: %.2f seconds", sum(timing_data$elapsed_seconds)))
  }
  
  return(list(
    subset_matrices = subset_result$subset_matrices,
    group_metadata = subset_result$group_metadata,
    model_parameters = model_params_list,
    combined_parameters = combined_params,
    timing = timing_data
  ))
}
