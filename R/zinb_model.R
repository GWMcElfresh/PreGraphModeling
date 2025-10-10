#' Fit Zero-Inflated Negative Binomial Models to Gene Expression
#'
#' This function fits zero-inflated negative binomial (ZINB) models to gene
#' expression data and extracts the estimated parameters: mu (mean), 
#' phi (dispersion), and pi (probability of zero).
#'
#' @param expressionMatrix A numeric matrix where rows are genes and columns are cells.
#'   Can be a single matrix or one from the subset_matrices output from SubsetSeurat.
#' @param geneSubset Optional character vector of gene names to fit models for.
#'   If NULL, models are fit for all genes (default: NULL).
#' @param minNonZero Minimum number of non-zero observations required to fit a model.
#'   Genes with fewer non-zero values are skipped (default: 3).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return A data frame with one row per gene containing:
#'   \itemize{
#'     \item gene: Gene name
#'     \item mu: Estimated mean parameter from the count model
#'     \item phi: Estimated dispersion parameter (theta in negative binomial)
#'     \item pi: Estimated zero-inflation probability
#'     \item converged: Logical indicating if the model converged
#'     \item n_nonzero: Number of non-zero observations for the gene
#'     \item n_datapoints: Total number of data points (cells) for the gene
#'   }
#'
#' @export
#' @importFrom pscl zeroinfl
#' @importFrom stats coef
#' @examples
#' \dontrun{
#' # After subsetting
#' result <- SubsetSeurat(seurat_obj, groupByColumns = "CellType")
#' 
#' # Fit ZINB models to all genes in first subset
#' params <- FitZeroInflatedModels(result$subset_matrices[[1]])
#' 
#' # Fit models to subset of genes
#' params <- FitZeroInflatedModels(result$subset_matrices[[1]], 
#'                                  geneSubset = c("CD3D", "CD4", "CD8A"))
#' }
FitZeroInflatedModels <- function(expressionMatrix,
                                  geneSubset = NULL,
                                  minNonZero = 3,
                                  verbose = TRUE) {
  
  # Input validation
  if (!is.matrix(expressionMatrix) && !inherits(expressionMatrix, "Matrix")) {
    stop("expressionMatrix must be a matrix or Matrix object")
  }
  
  if (!is.numeric(minNonZero) || minNonZero < 1) {
    stop("minNonZero must be a positive integer")
  }
  
  # Determine genes to process
  all_genes <- rownames(expressionMatrix)
  if (is.null(all_genes)) {
    stop("expressionMatrix must have row names (gene names)")
  }
  
  if (!is.null(geneSubset)) {
    if (!is.character(geneSubset)) {
      stop("geneSubset must be a character vector of gene names")
    }
    missing_genes <- setdiff(geneSubset, all_genes)
    if (length(missing_genes) > 0) {
      warning(paste("Genes not found in matrix:", paste(missing_genes, collapse = ", ")))
    }
    genes_to_fit <- intersect(geneSubset, all_genes)
  } else {
    genes_to_fit <- all_genes
  }
  
  if (length(genes_to_fit) == 0) {
    stop("No valid genes to fit models for")
  }
  
  if (verbose) {
    message(sprintf("Fitting ZINB models for %d genes...", length(genes_to_fit)))
  }
  
  # Initialize results data frame
  results <- data.frame(
    gene = character(length(genes_to_fit)),
    mu = numeric(length(genes_to_fit)),
    phi = numeric(length(genes_to_fit)),
    pi = numeric(length(genes_to_fit)),
    converged = logical(length(genes_to_fit)),
    n_nonzero = integer(length(genes_to_fit)),
    n_datapoints = integer(length(genes_to_fit)),
    stringsAsFactors = FALSE
  )
  # ============================================================================
  # GENERATE SIZE FACTORS
  # ============================================================================

  #unsure how long this will take - it's probably chunkable if it takes a minute for per-cell size factors. 
  sizeFactors <- DESeq2::estimateSizeFactorsForMatrix(expressionMatrix) 
  
  
  # ============================================================================
  # FIT ZINB MODELS FOR EACH GENE
  # ============================================================================
  for (i in seq_along(genes_to_fit)) {
    gene_name <- genes_to_fit[i]
    
    if (verbose && i %% 100 == 0) {
      message(sprintf("  Processing gene %d of %d...", i, length(genes_to_fit)))
    }
    
    # --------------------------------------------------------------------------
    # Extract and validate gene expression data
    # --------------------------------------------------------------------------
    gene_expr <- as.numeric(expressionMatrix[gene_name, ])
    n_nonzero <- sum(gene_expr > 0)
    n_datapoints <- length(gene_expr)
    
    # Store gene name, n_nonzero, and n_datapoints
    results$gene[i] <- gene_name
    results$n_nonzero[i] <- n_nonzero
    results$n_datapoints[i] <- n_datapoints
    
    # Skip if not enough non-zero values
    if (n_nonzero < minNonZero) {
      results$mu[i] <- NA
      results$phi[i] <- NA
      results$pi[i] <- NA
      results$converged[i] <- FALSE
      next
    }
    
    # --------------------------------------------------------------------------
    # Fit ZINB model and extract parameters
    # --------------------------------------------------------------------------
    tryCatch({
      # Create data frame for modeling
      model_data <- data.frame(counts = gene_expr, 
                               log_sizeFactor = log(sizeFactors))
      
      # Fit zero-inflated negative binomial model
      fit <- pscl::zeroinfl(counts ~ offset(log_sizeFactor) | 1, 
                            data = model_data,
                            dist = "negbin",
                            link = "logit")
      
      # Extract parameters
      # Mu: mean of the negative binomial component
      mu_coef <- coef(fit, model = "count")
      results$mu[i] <- exp(mu_coef[1])  # exp to get actual mean from log link
      
      # Phi: dispersion parameter (theta)
      results$phi[i] <- fit$theta
      
      # Pi: probability of excess zeros
      pi_coef <- coef(fit, model = "zero")
      results$pi[i] <- 1 / (1 + exp(-pi_coef[1]))  # inverse logit
      
      # Check convergence
      results$converged[i] <- fit$converged
      
    }, error = function(e) {
      if (verbose) {
        message(sprintf("    Warning: Model failed for gene %s: %s", gene_name, e$message))
      }
      results$mu[i] <<- NA
      results$phi[i] <<- NA
      results$pi[i] <<- NA
      results$converged[i] <<- FALSE
    })
  }
  
  if (verbose) {
    n_converged <- sum(results$converged, na.rm = TRUE)
    message(sprintf("Completed: %d/%d models converged successfully", 
                   n_converged, length(genes_to_fit)))
  }
  
  return(results)
}
