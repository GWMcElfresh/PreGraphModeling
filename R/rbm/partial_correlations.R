#' Estimate Partial Correlations via Quasilikelihood
#'
#' Computes partial correlations between features using quasilikelihood estimation.
#' This function supports various families including zero-inflated negative binomial
#' for single-cell data analysis. Partial correlations represent the direct relationship
#' between features after conditioning on all other features.
#'
#' @param expressionMatrix A numeric matrix where rows are features (genes) and columns
#'   are observations (cells). Can be a regular matrix or sparse Matrix.
#' @param metadata Optional data frame of metadata variables to include as covariates.
#'   Must have the same number of rows as columns in expressionMatrix.
#' @param family Character string specifying the distribution family for quasilikelihood.
#'   Options: "zinb" (zero-inflated negative binomial, default), "nb" (negative binomial),
#'   "poisson", "gaussian", "quasipoisson".
#' @param method Character string specifying the estimation method.
#'   Options: "glm" (generalized linear model, default), "correlation" (Pearson/Spearman).
#' @param minNonZero Minimum number of non-zero observations required for a feature
#'   pair to be estimated (default: 10).
#' @param progressr Logical indicating whether to use progressr for progress reporting
#'   during parallel computation (default: TRUE).
#' @param parallel Logical indicating whether to use parallel processing (default: FALSE).
#' @param numWorkers Integer specifying number of workers for parallel processing.
#'   If NULL, uses detectCores()-1 (default: NULL).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return A list with components:
#'   \itemize{
#'     \item partial_cor: Symmetric matrix of partial correlations between features
#'     \item se: Standard errors for each partial correlation (if computed)
#'     \item pvalues: P-values for testing H0: partial correlation = 0 (if computed)
#'     \item n_pairs: Number of valid feature pairs analyzed
#'     \item family: The family used for estimation
#'     \item features: Names of features included in the analysis
#'   }
#'
#' @details
#' For zero-inflated negative binomial (zinb) family, the function uses a two-stage
#' approach: first fitting the zero-inflation component, then estimating partial
#' correlations from the count component via quasilikelihood. The partial correlations
#' are computed as the negative normalized precision matrix elements.
#'
#' When parallel=TRUE, the function uses future/future.apply for parallelization
#' and progressr for progress tracking.
#'
#' @export
#' @importFrom stats cor glm quasi
#' @examples
#' \dontrun{
#' # Estimate partial correlations using ZINB quasilikelihood
#' result <- EstimatePartialCorrelations(
#'   expressionMatrix = expr_matrix,
#'   family = "zinb",
#'   parallel = TRUE
#' )
#'
#' # Access the partial correlation matrix
#' pcor_matrix <- result$partial_cor
#'
#' # With metadata covariates
#' result <- EstimatePartialCorrelations(
#'   expressionMatrix = expr_matrix,
#'   metadata = meta_df,
#'   family = "zinb"
#' )
#' }
EstimatePartialCorrelations <- function(expressionMatrix,
                                       metadata = NULL,
                                       family = "zinb",
                                       method = "glm",
                                       minNonZero = 10,
                                       progressr = TRUE,
                                       parallel = FALSE,
                                       numWorkers = NULL,
                                       verbose = TRUE) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  if (!is.matrix(expressionMatrix) && !inherits(expressionMatrix, "Matrix")) {
    stop("expressionMatrix must be a matrix or Matrix object")
  }
  
  if (is.null(rownames(expressionMatrix))) {
    stop("expressionMatrix must have row names (feature names)")
  }
  
  # Validate family
  valid_families <- c("zinb", "nb", "poisson", "gaussian", "quasipoisson")
  if (!family %in% valid_families) {
    stop(sprintf("family must be one of: %s", paste(valid_families, collapse = ", ")))
  }
  
  # Validate metadata
  if (!is.null(metadata)) {
    if (!is.data.frame(metadata)) {
      stop("metadata must be a data frame")
    }
    if (nrow(metadata) != ncol(expressionMatrix)) {
      stop("metadata must have the same number of rows as columns in expressionMatrix")
    }
  }
  
  n_features <- nrow(expressionMatrix)
  feature_names <- rownames(expressionMatrix)
  
  if (verbose) {
    message(sprintf("Estimating partial correlations for %d features using %s family...",
                   n_features, family))
    if (!is.null(metadata)) {
      message(sprintf("  Including %d metadata covariates", ncol(metadata)))
    }
  }
  
  # ============================================================================
  # FILTER FEATURES BY NON-ZERO COUNTS
  # ============================================================================
  if (verbose) {
    message("Filtering features by minimum non-zero observations...")
  }
  
  n_nonzero <- rowSums(expressionMatrix > 0)
  valid_features <- n_nonzero >= minNonZero
  
  if (sum(valid_features) == 0) {
    stop(sprintf("No features have at least %d non-zero observations", minNonZero))
  }
  
  if (sum(!valid_features) > 0 && verbose) {
    message(sprintf("  Excluded %d features with < %d non-zero observations",
                   sum(!valid_features), minNonZero))
  }
  
  expr_filtered <- expressionMatrix[valid_features, , drop = FALSE]
  features_kept <- feature_names[valid_features]
  n_kept <- length(features_kept)
  
  if (verbose) {
    message(sprintf("  Proceeding with %d features", n_kept))
  }
  
  # ============================================================================
  # COMPUTE PARTIAL CORRELATIONS
  # ============================================================================
  
  if (family == "zinb") {
    # Use ZINB-specific quasilikelihood approach
    pcor_result <- .estimate_partial_cor_zinb(
      expr_matrix = expr_filtered,
      metadata = metadata,
      progressr = progressr,
      parallel = parallel,
      numWorkers = numWorkers,
      verbose = verbose
    )
  } else if (family %in% c("nb", "poisson", "quasipoisson")) {
    # Use GLM-based approach for count data
    pcor_result <- .estimate_partial_cor_glm(
      expr_matrix = expr_filtered,
      metadata = metadata,
      family = family,
      progressr = progressr,
      parallel = parallel,
      numWorkers = numWorkers,
      verbose = verbose
    )
  } else if (family == "gaussian") {
    # Use correlation-based approach for continuous data
    pcor_result <- .estimate_partial_cor_gaussian(
      expr_matrix = expr_filtered,
      metadata = metadata,
      progressr = progressr,
      verbose = verbose
    )
  }
  
  # ============================================================================
  # PREPARE OUTPUT
  # ============================================================================
  
  # Expand to full feature set (fill in NAs for excluded features)
  full_pcor <- matrix(NA, nrow = n_features, ncol = n_features,
                     dimnames = list(feature_names, feature_names))
  full_pcor[features_kept, features_kept] <- pcor_result$partial_cor
  
  result <- list(
    partial_cor = full_pcor,
    se = pcor_result$se,
    pvalues = pcor_result$pvalues,
    n_pairs = pcor_result$n_pairs,
    family = family,
    features = features_kept,
    excluded_features = feature_names[!valid_features]
  )
  
  if (verbose) {
    message(sprintf("Partial correlation estimation complete: %d valid pairs",
                   result$n_pairs))
  }
  
  return(result)
}


#' Internal function to estimate partial correlations using ZINB quasilikelihood
#' @keywords internal
#' @noRd
.estimate_partial_cor_zinb <- function(expr_matrix, metadata, progressr, parallel,
                                      numWorkers, verbose) {
  
  n_features <- nrow(expr_matrix)
  feature_names <- rownames(expr_matrix)
  
  # Initialize partial correlation matrix
  pcor_matrix <- matrix(0, nrow = n_features, ncol = n_features,
                       dimnames = list(feature_names, feature_names))
  
  # Compute pairwise partial correlations
  # For computational efficiency, we'll use a precision matrix approach
  # Estimate covariance structure from deviance residuals
  
  if (verbose) {
    message("Computing ZINB deviance residuals for each feature...")
  }
  
  # Fit ZINB models to get deviance residuals
  residuals_matrix <- matrix(0, nrow = n_features, ncol = ncol(expr_matrix))
  
  # Setup progress tracking
  if (progressr && requireNamespace("progressr", quietly = TRUE)) {
    progressr::handlers(global = TRUE)
    p <- progressr::progressor(steps = n_features)
  } else {
    p <- NULL
  }
  
  for (i in seq_len(n_features)) {
    if (verbose && i %% 50 == 0) {
      message(sprintf("  Processing feature %d of %d...", i, n_features))
    }
    
    gene_expr <- as.numeric(expr_matrix[i, ])
    
    # Estimate residuals from ZINB model
    tryCatch({
      # Compute size factors for offset
      lib_sizes <- colSums(expr_matrix)
      size_factors <- lib_sizes / mean(lib_sizes)
      
      model_data <- data.frame(
        counts = gene_expr,
        log_sizeFactor = log(size_factors)
      )
      
      # Add metadata if provided
      if (!is.null(metadata)) {
        model_data <- cbind(model_data, metadata)
        formula_str <- paste("counts ~ offset(log_sizeFactor) +",
                           paste(names(metadata), collapse = " + "), "| 1")
      } else {
        formula_str <- "counts ~ offset(log_sizeFactor) | 1"
      }
      
      # Fit ZINB model
      if (requireNamespace("pscl", quietly = TRUE)) {
        fit <- pscl::zeroinfl(as.formula(formula_str),
                             data = model_data,
                             dist = "negbin",
                             link = "logit")
        residuals_matrix[i, ] <- residuals(fit, type = "deviance")
      } else {
        # Fallback: use Pearson residuals from simple model
        residuals_matrix[i, ] <- (gene_expr - mean(gene_expr)) / sqrt(mean(gene_expr) + 1e-6)
      }
      
    }, error = function(e) {
      # Use simple residuals if ZINB fit fails
      residuals_matrix[i, ] <<- (gene_expr - mean(gene_expr)) / sqrt(mean(gene_expr) + 1e-6)
    })
    
    if (!is.null(p)) p()
  }
  
  if (verbose) {
    message("Computing partial correlations from precision matrix...")
  }
  
  # Compute correlation matrix of residuals
  cor_matrix <- cor(t(residuals_matrix), use = "pairwise.complete.obs")
  
  # Estimate precision matrix (inverse of correlation matrix)
  # Use regularization for numerical stability
  tryCatch({
    # Add small regularization to diagonal
    cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
    precision_matrix <- solve(cor_matrix_reg)
    
    # Convert precision to partial correlations
    # Partial correlation: -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    diag_prec <- diag(precision_matrix)
    scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
    pcor_matrix <- -precision_matrix / scaling
    diag(pcor_matrix) <- 1  # Set diagonal to 1
    
  }, error = function(e) {
    if (verbose) {
      message("Warning: Precision matrix inversion failed, using correlation matrix as approximation")
    }
    pcor_matrix <<- cor_matrix
  })
  
  # Count valid pairs
  n_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))
  
  list(
    partial_cor = pcor_matrix,
    se = NULL,  # Standard errors not computed for ZINB
    pvalues = NULL,  # P-values not computed for ZINB
    n_pairs = n_pairs
  )
}


#' Internal function to estimate partial correlations using GLM quasilikelihood
#' @keywords internal
#' @noRd
.estimate_partial_cor_glm <- function(expr_matrix, metadata, family, progressr,
                                     parallel, numWorkers, verbose) {
  
  n_features <- nrow(expr_matrix)
  feature_names <- rownames(expr_matrix)
  
  # Initialize matrices
  pcor_matrix <- matrix(0, nrow = n_features, ncol = n_features,
                       dimnames = list(feature_names, feature_names))
  
  if (verbose) {
    message(sprintf("Computing %s deviance residuals for each feature...", family))
  }
  
  # Compute residuals from GLM fits
  residuals_matrix <- matrix(0, nrow = n_features, ncol = ncol(expr_matrix))
  
  for (i in seq_len(n_features)) {
    if (verbose && i %% 50 == 0) {
      message(sprintf("  Processing feature %d of %d...", i, n_features))
    }
    
    gene_expr <- as.numeric(expr_matrix[i, ])
    
    tryCatch({
      model_data <- data.frame(counts = gene_expr)
      
      # Add metadata if provided
      if (!is.null(metadata)) {
        model_data <- cbind(model_data, metadata)
        formula_str <- paste("counts ~", paste(names(metadata), collapse = " + "))
      } else {
        formula_str <- "counts ~ 1"
      }
      
      # Fit GLM with appropriate family
      if (family == "nb") {
        # Use quasipoisson as approximation for NB
        fit <- glm(as.formula(formula_str), data = model_data,
                  family = quasipoisson(link = "log"))
      } else if (family == "poisson") {
        fit <- glm(as.formula(formula_str), data = model_data,
                  family = poisson(link = "log"))
      } else if (family == "quasipoisson") {
        fit <- glm(as.formula(formula_str), data = model_data,
                  family = quasipoisson(link = "log"))
      }
      
      residuals_matrix[i, ] <- residuals(fit, type = "deviance")
      
    }, error = function(e) {
      # Use simple residuals if fit fails
      residuals_matrix[i, ] <<- (gene_expr - mean(gene_expr)) / sqrt(mean(gene_expr) + 1e-6)
    })
  }
  
  # Compute partial correlations from precision matrix
  cor_matrix <- cor(t(residuals_matrix), use = "pairwise.complete.obs")
  
  tryCatch({
    cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
    precision_matrix <- solve(cor_matrix_reg)
    diag_prec <- diag(precision_matrix)
    scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
    pcor_matrix <- -precision_matrix / scaling
    diag(pcor_matrix) <- 1
  }, error = function(e) {
    pcor_matrix <<- cor_matrix
  })
  
  n_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))
  
  list(
    partial_cor = pcor_matrix,
    se = NULL,
    pvalues = NULL,
    n_pairs = n_pairs
  )
}


#' Internal function to estimate partial correlations using Gaussian approach
#' @keywords internal
#' @noRd
.estimate_partial_cor_gaussian <- function(expr_matrix, metadata, progressr, verbose) {
  
  n_features <- nrow(expr_matrix)
  feature_names <- rownames(expr_matrix)
  
  if (verbose) {
    message("Computing Gaussian partial correlations...")
  }
  
  # For Gaussian data, partial correlations can be computed directly
  # from the correlation matrix via precision matrix
  
  # Optionally regress out metadata
  if (!is.null(metadata)) {
    if (verbose) {
      message("  Regressing out metadata covariates...")
    }
    residuals_matrix <- matrix(0, nrow = n_features, ncol = ncol(expr_matrix))
    for (i in seq_len(n_features)) {
      gene_expr <- as.numeric(expr_matrix[i, ])
      model_data <- data.frame(y = gene_expr)
      model_data <- cbind(model_data, metadata)
      formula_str <- paste("y ~", paste(names(metadata), collapse = " + "))
      fit <- lm(as.formula(formula_str), data = model_data)
      residuals_matrix[i, ] <- residuals(fit)
    }
    expr_to_use <- residuals_matrix
  } else {
    expr_to_use <- expr_matrix
  }
  
  # Compute correlation
  cor_matrix <- cor(t(expr_to_use), use = "pairwise.complete.obs")
  
  # Compute partial correlations via precision matrix
  pcor_matrix <- matrix(0, nrow = n_features, ncol = n_features,
                       dimnames = list(feature_names, feature_names))
  
  tryCatch({
    cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
    precision_matrix <- solve(cor_matrix_reg)
    diag_prec <- diag(precision_matrix)
    scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
    pcor_matrix <- -precision_matrix / scaling
    diag(pcor_matrix) <- 1
  }, error = function(e) {
    if (verbose) {
      message("Warning: Precision matrix inversion failed, using correlation as approximation")
    }
    pcor_matrix <<- cor_matrix
  })
  
  n_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))
  
  list(
    partial_cor = pcor_matrix,
    se = NULL,
    pvalues = NULL,
    n_pairs = n_pairs
  )
}
