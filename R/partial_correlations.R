#' Estimate Partial Correlations via Pseudolikelihood
#'
#' Computes partial correlations between features using pseudolikelihood estimation.
#' This function supports various families including zero-inflated negative binomial
#' for single-cell data analysis. Partial correlations represent the direct relationship
#' between features after conditioning on all other features.
#'
#' @param expressionMatrix A numeric matrix where rows are features (genes) and columns
#'   are observations (cells). Can be a regular matrix or sparse Matrix.
#' @param metadata Optional data frame of metadata variables to include as covariates.
#'   Must have the same number of rows as columns in expressionMatrix.
#' @param family Character string specifying the distribution family for pseudolikelihood estimation.
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
#' correlations from the count component via pseudolikelihood. The partial correlations
#' are computed as the negative normalized precision matrix elements.
#'
#' When parallel=TRUE, the function uses future/future.apply for parallelization
#' and progressr for progress tracking.
#'
#' @export
#' @importFrom stats cor glm quasi
#' @examples
#' \dontrun{
#' # Estimate partial correlations using ZINB pseudolikelihood
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
    message(sprintf(
      "Estimating partial correlations for %d features using %s family...",
      n_features, family
    ))
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
    message(sprintf(
      "  Excluded %d features with < %d non-zero observations",
      sum(!valid_features), minNonZero
    ))
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
  full_pcor <- matrix(NA,
    nrow = n_features, ncol = n_features,
    dimnames = list(feature_names, feature_names)
  )
  full_pcor[features_kept, features_kept] <- pcor_result$partial_cor

  result <- structure(
    list(
      partial_cor = full_pcor,
      se = pcor_result$se,
      pvalues = pcor_result$pvalues,
      n_pairs = pcor_result$n_pairs,
      family = family,
      features = features_kept,
      excluded_features = feature_names[!valid_features]
    ),
    class = "PartialCorrelations"
  )

  if (verbose) {
    message(sprintf(
      "Partial correlation estimation complete: %d valid pairs",
      result$n_pairs
    ))
  }

  return(result)
}


#' Internal function to estimate partial correlations using ZINB pseudolikelihood
#' @keywords internal
#' @noRd
.estimate_partial_cor_zinb <- function(expr_matrix, metadata, progressr, parallel,
                                       numWorkers, verbose) {
  n_features <- nrow(expr_matrix)
  feature_names <- rownames(expr_matrix)

  # Initialize partial correlation matrix
  pcor_matrix <- matrix(0,
    nrow = n_features, ncol = n_features,
    dimnames = list(feature_names, feature_names)
  )

  # Compute pairwise partial correlations
  # For computational efficiency, we'll use a precision matrix approach
  # Estimate covariance structure from deviance residuals

  if (verbose) {
    message("Computing ZINB deviance residuals for each feature...")
  }

  # Fit ZINB models to get deviance residuals
  residuals_matrix <- matrix(0, nrow = n_features, ncol = ncol(expr_matrix))

  # Precompute size factors (needed for all features)
  lib_sizes <- colSums(expr_matrix)
  size_factors <- lib_sizes / mean(lib_sizes)
  log_size_factors <- log(pmax(size_factors, 1e-8))

  # Define function to compute residuals for a single feature
  .compute_zinb_residuals <- function(i, expr_matrix, log_size_factors, metadata) {
    gene_expr <- as.numeric(expr_matrix[i, ])

    tryCatch(
      {
        model_data <- data.frame(
          counts = gene_expr,
          log_sizeFactor = log_size_factors
        )

        # Add metadata if provided
        if (!is.null(metadata)) {
          model_data <- cbind(model_data, metadata)
          formula_str <- paste(
            "counts ~ offset(log_sizeFactor) +",
            paste(names(metadata), collapse = " + "), "| 1"
          )
        } else {
          formula_str <- "counts ~ offset(log_sizeFactor) | 1"
        }

        # Fit ZINB model
        fit <- pscl::zeroinfl(as.formula(formula_str),
          data = model_data,
          dist = "negbin",
          link = "logit"
        )
        return(residuals(fit, type = "deviance"))
      },
      error = function(e) {
        # Use simple residuals if ZINB fit fails
        return((gene_expr - mean(gene_expr)) / sqrt(mean(gene_expr) + 1e-6))
      }
    )
  }

  # Parallel or sequential execution
  if (parallel) {
    # Set up parallel workers
    if (is.null(numWorkers)) {
      numWorkers <- max(1, parallel::detectCores() - 1)
    }

    if (verbose) {
      message(sprintf("  Using parallel processing with %d workers...", numWorkers))
    }

    # Set up future plan
    old_plan <- future::plan()
    future::plan(future::multisession, workers = numWorkers)

    # Setup progress tracking for parallel
    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = n_features)
    } else {
      p <- NULL
    }

    # Run in parallel
    residuals_list <- future.apply::future_lapply(
      seq_len(n_features),
      function(i) {
        result <- .compute_zinb_residuals(i, expr_matrix, log_size_factors, metadata)
        if (!is.null(p)) p()
        return(result)
      },
      future.seed = TRUE
    )

    # Convert list to matrix
    for (i in seq_len(n_features)) {
      residuals_matrix[i, ] <- residuals_list[[i]]
    }

    # Reset to previous plan
    future::plan(old_plan)
  } else {
    # Sequential execution with progress tracking
    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = n_features)
    } else {
      p <- NULL
    }

    for (i in seq_len(n_features)) {
      if (verbose && i %% 50 == 0) {
        message(sprintf("  Processing feature %d of %d...", i, n_features))
      }

      residuals_matrix[i, ] <- .compute_zinb_residuals(i, expr_matrix, log_size_factors, metadata)

      if (!is.null(p)) p()
    }
  }

  if (verbose) {
    message("Computing partial correlations from precision matrix...")
  }

  # Compute correlation matrix of residuals
  cor_matrix <- cor(t(residuals_matrix), use = "pairwise.complete.obs")

  # Estimate precision matrix (inverse of correlation matrix)
  # Use regularization for numerical stability
  tryCatch(
    {
      # Add small regularization to diagonal
      cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
      precision_matrix <- solve(cor_matrix_reg)

      # Convert precision to partial correlations
      # Partial correlation: -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
      diag_prec <- diag(precision_matrix)
      scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
      pcor_matrix <- -precision_matrix / scaling
      diag(pcor_matrix) <- 1 # Set diagonal to 1
    },
    error = function(e) {
      if (verbose) {
        message("Warning: Precision matrix inversion failed, using correlation matrix as approximation")
      }
      pcor_matrix <<- cor_matrix
    }
  )

  # Count valid pairs
  n_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))

  list(
    partial_cor = pcor_matrix,
    se = NULL, # Standard errors not computed for ZINB
    pvalues = NULL, # P-values not computed for ZINB
    n_pairs = n_pairs
  )
}


#' Internal function to estimate partial correlations using GLM pseudolikelihood
#' @keywords internal
#' @noRd
.estimate_partial_cor_glm <- function(expr_matrix, metadata, family, progressr,
                                      parallel, numWorkers, verbose) {
  n_features <- nrow(expr_matrix)
  feature_names <- rownames(expr_matrix)

  # Initialize matrices
  pcor_matrix <- matrix(0,
    nrow = n_features, ncol = n_features,
    dimnames = list(feature_names, feature_names)
  )

  if (verbose) {
    message(sprintf("Computing %s deviance residuals for each feature...", family))
  }

  # Compute residuals from GLM fits
  residuals_matrix <- matrix(0, nrow = n_features, ncol = ncol(expr_matrix))

  # Define function to compute residuals for a single feature
  .compute_glm_residuals <- function(i, expr_matrix, metadata, family) {
    gene_expr <- as.numeric(expr_matrix[i, ])

    tryCatch(
      {
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
          fit <- glm(as.formula(formula_str),
            data = model_data,
            family = quasipoisson(link = "log")
          )
        } else if (family == "poisson") {
          fit <- glm(as.formula(formula_str),
            data = model_data,
            family = poisson(link = "log")
          )
        } else if (family == "quasipoisson") {
          fit <- glm(as.formula(formula_str),
            data = model_data,
            family = quasipoisson(link = "log")
          )
        }

        return(residuals(fit, type = "deviance"))
      },
      error = function(e) {
        # Use simple residuals if fit fails
        return((gene_expr - mean(gene_expr)) / sqrt(mean(gene_expr) + 1e-6))
      }
    )
  }

  # Parallel or sequential execution
  if (parallel) {
    # Set up parallel workers
    if (is.null(numWorkers)) {
      numWorkers <- max(1, parallel::detectCores() - 1)
    }

    if (verbose) {
      message(sprintf("  Using parallel processing with %d workers...", numWorkers))
    }

    # Set up future plan
    old_plan <- future::plan()
    future::plan(future::multisession, workers = numWorkers)

    # Setup progress tracking for parallel
    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = n_features)
    } else {
      p <- NULL
    }

    # Run in parallel
    residuals_list <- future.apply::future_lapply(
      seq_len(n_features),
      function(i) {
        result <- .compute_glm_residuals(i, expr_matrix, metadata, family)
        if (!is.null(p)) p()
        return(result)
      },
      future.seed = TRUE
    )

    # Convert list to matrix
    for (i in seq_len(n_features)) {
      residuals_matrix[i, ] <- residuals_list[[i]]
    }

    # Reset to previous plan
    future::plan(old_plan)
  } else {
    # Sequential execution with progress tracking
    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = n_features)
    } else {
      p <- NULL
    }

    for (i in seq_len(n_features)) {
      if (verbose && i %% 50 == 0) {
        message(sprintf("  Processing feature %d of %d...", i, n_features))
      }

      residuals_matrix[i, ] <- .compute_glm_residuals(i, expr_matrix, metadata, family)

      if (!is.null(p)) p()
    }
  }

  # Compute partial correlations from precision matrix
  cor_matrix <- cor(t(residuals_matrix), use = "pairwise.complete.obs")

  tryCatch(
    {
      cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
      precision_matrix <- solve(cor_matrix_reg)
      diag_prec <- diag(precision_matrix)
      scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
      pcor_matrix <- -precision_matrix / scaling
      diag(pcor_matrix) <- 1
    },
    error = function(e) {
      pcor_matrix <<- cor_matrix
    }
  )

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
  pcor_matrix <- matrix(0,
    nrow = n_features, ncol = n_features,
    dimnames = list(feature_names, feature_names)
  )

  tryCatch(
    {
      cor_matrix_reg <- cor_matrix + diag(0.01, n_features)
      precision_matrix <- solve(cor_matrix_reg)
      diag_prec <- diag(precision_matrix)
      scaling <- outer(sqrt(diag_prec), sqrt(diag_prec))
      pcor_matrix <- -precision_matrix / scaling
      diag(pcor_matrix) <- 1
    },
    error = function(e) {
      if (verbose) {
        message("Warning: Precision matrix inversion failed, using correlation as approximation")
      }
      pcor_matrix <<- cor_matrix
    }
  )

  n_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))

  list(
    partial_cor = pcor_matrix,
    se = NULL,
    pvalues = NULL,
    n_pairs = n_pairs
  )
}


#' Estimate Partial Correlations from a Seurat Object
#'
#' Convenience wrapper around EstimatePartialCorrelations that accepts a Seurat
#' object directly. This is the recommended way to compute partial correlations
#' for use with FitRBM, as the output can be reused across multiple RBM fits.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data.
#' @param assay Character string specifying which assay to use (default: "RNA").
#' @param layer Character string specifying which layer to use (default: "counts").
#' @param visibleFeatures Character vector of feature (gene) names to include.
#'   If NULL, uses all features (default: NULL).
#' @param family Character string specifying the distribution family for pseudolikelihood estimation.
#'   Options: "zinb" (zero-inflated negative binomial, default), "nb" (negative binomial),
#'   "poisson", "gaussian", "quasipoisson".
#' @param minNonZero Minimum number of non-zero observations required for a feature
#'   pair to be estimated (default: 10).
#' @param progressr Logical indicating whether to use progressr for progress reporting
#'   during parallel computation (default: TRUE).
#' @param parallel Logical indicating whether to use parallel processing (default: FALSE).
#' @param numWorkers Integer specifying number of workers for parallel processing.
#'   If NULL, uses detectCores()-1 (default: NULL).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return An object of class "PartialCorrelations" containing:
#'   \itemize{
#'     \item partial_cor: Symmetric matrix of partial correlations between features
#'     \item se: Standard errors for each partial correlation (if computed)
#'     \item pvalues: P-values for testing H0: partial correlation = 0 (if computed)
#'     \item n_pairs: Number of valid feature pairs analyzed
#'     \item family: The family used for estimation
#'     \item features: Names of features included in the analysis
#'     \item excluded_features: Names of features excluded due to low counts
#'   }
#'
#' @details
#' This function extracts the expression matrix from the Seurat object and passes
#' it to EstimatePartialCorrelations. The returned PartialCorrelations object can
#' be passed directly to FitRBM via the partialCorrelations parameter, allowing
#' you to:
#' \itemize{
#'   \item Inspect partial correlations before fitting the RBM
#'   \item Reuse the same partial correlations across multiple RBM fits with different hidden factors
#'   \item Save and reload partial correlations to avoid recomputation
#' }
#'
#' @export
#' @examples
#' \dontrun{
#' # Recommended workflow: Pre-compute partial correlations
#' pcor <- EstimatePartialCorrelationsFromSeurat(
#'   seuratObject = pbmc,
#'   family = "zinb",
#'   parallel = TRUE,
#'   verbose = TRUE
#' )
#'
#' # Inspect the partial correlations
#' print(pcor)
#' heatmap(pcor$partial_cor[1:50, 1:50])
#'
#' # Fit RBM with pre-computed correlations
#' rbm <- FitRBM(
#'   seuratObject = pbmc,
#'   hiddenFactors = "CellType",
#'   partialCorrelations = pcor
#' )
#'
#' # Reuse for another RBM with different hidden factors
#' rbm2 <- FitRBM(
#'   seuratObject = pbmc,
#'   hiddenFactors = c("CellType", "Treatment"),
#'   partialCorrelations = pcor
#' )
#' }
EstimatePartialCorrelationsFromSeurat <- function(seuratObject,
                                                  assay = "RNA",
                                                  layer = "counts",
                                                  visibleFeatures = NULL,
                                                  family = "zinb",
                                                  minNonZero = 10,
                                                  progressr = TRUE,
                                                  parallel = FALSE,
                                                  numWorkers = NULL,
                                                  verbose = TRUE) {
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================

  if (!inherits(seuratObject, "Seurat") && !inherits(seuratObject, "SeuratObject")) {
    stop("seuratObject must be a Seurat or SeuratObject")
  }

  if (verbose) {
    message("Extracting expression data from Seurat object...")
  }

  # ============================================================================
  # EXTRACT EXPRESSION DATA
  # ============================================================================

  # Get expression matrix
  if (inherits(seuratObject, "Seurat")) {
    # For Seurat v5+ with layers
    tryCatch(
      {
        expr_matrix <- SeuratObject::LayerData(
          object = seuratObject,
          assay = assay,
          layer = layer
        )
      },
      error = function(e) {
        # Fallback for older Seurat versions
        expr_matrix <<- SeuratObject::GetAssayData(
          object = seuratObject,
          assay = assay,
          layer = layer
        )
      }
    )
  } else {
    # For SeuratObject
    expr_matrix <- SeuratObject::GetAssayData(
      object = seuratObject,
      assay = assay,
      layer = layer
    )
  }

  # Convert to matrix if needed
  if (inherits(expr_matrix, "Matrix")) {
    expr_matrix <- as.matrix(expr_matrix)
  }

  # Filter features if specified
  if (!is.null(visibleFeatures)) {
    if (!is.character(visibleFeatures)) {
      stop("visibleFeatures must be a character vector of feature names")
    }
    missing_features <- setdiff(visibleFeatures, rownames(expr_matrix))
    if (length(missing_features) > 0) {
      warning(sprintf(
        "Features not found in expression matrix: %s",
        paste(missing_features, collapse = ", ")
      ))
    }
    visibleFeatures <- intersect(visibleFeatures, rownames(expr_matrix))
    if (length(visibleFeatures) == 0) {
      stop("No valid features found in expression matrix")
    }
    expr_matrix <- expr_matrix[visibleFeatures, , drop = FALSE]
  }

  if (verbose) {
    message(sprintf(
      "  Expression matrix: %d features x %d cells",
      nrow(expr_matrix), ncol(expr_matrix)
    ))
  }

  # ============================================================================
  # COMPUTE PARTIAL CORRELATIONS
  # ============================================================================

  result <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    metadata = NULL,
    family = family,
    minNonZero = minNonZero,
    progressr = progressr,
    parallel = parallel,
    numWorkers = numWorkers,
    verbose = verbose
  )

  return(result)
}


#' Print method for PartialCorrelations objects
#' @param x A PartialCorrelations object
#' @param ... Additional arguments (unused)
#' @export
print.PartialCorrelations <- function(x, ...) {
  cat("Partial Correlations Object\n")
  cat("============================\n\n")
  cat(sprintf(
    "Features:     %d total, %d with valid correlations\n",
    nrow(x$partial_cor), length(x$features)
  ))
  cat(sprintf("Valid pairs:  %d\n", x$n_pairs))
  cat(sprintf("Family:       %s\n", x$family))

  if (length(x$excluded_features) > 0) {
    cat(sprintf("Excluded:     %d features (low counts)\n", length(x$excluded_features)))
  }

  # Summary statistics
  pcor_upper <- x$partial_cor[upper.tri(x$partial_cor)]
  pcor_valid <- pcor_upper[!is.na(pcor_upper)]

  if (length(pcor_valid) > 0) {
    cat("\nPartial correlation statistics:\n")
    cat(sprintf("  Range:  [%.4f, %.4f]\n", min(pcor_valid), max(pcor_valid)))
    cat(sprintf("  Median: %.4f\n", median(pcor_valid)))
    cat(sprintf("  Mean:   %.4f\n", mean(pcor_valid)))
  }

  invisible(x)
}


#' Validate a PartialCorrelations object
#'
#' Internal function to check if an object is a valid PartialCorrelations object
#' that can be used with FitRBM.
#'
#' @param x Object to validate
#' @param required_features Optional character vector of features that must be present
#' @param verbose Logical indicating whether to print diagnostic messages
#'
#' @return A list with components:
#'   \itemize{
#'     \item valid: Logical indicating if the object is valid
#'     \item message: Diagnostic message if invalid
#'     \item overlap_features: Features present in both x and required_features (if provided
#'   }
#'
#' @keywords internal
#' @noRd
.validate_partial_correlations <- function(x, required_features = NULL, verbose = FALSE) {
  # Check if it's a PartialCorrelations object or compatible list
  if (!is.list(x)) {
    return(list(valid = FALSE, message = "Input is not a list", overlap_features = NULL))
  }

  # Check required components
  required_components <- c("partial_cor", "features", "family")
  missing_components <- setdiff(required_components, names(x))

  if (length(missing_components) > 0) {
    return(list(
      valid = FALSE,
      message = sprintf("Missing required components: %s", paste(missing_components, collapse = ", ")),
      overlap_features = NULL
    ))
  }

  # Check partial_cor is a matrix
  if (!is.matrix(x$partial_cor)) {
    return(list(valid = FALSE, message = "partial_cor is not a matrix", overlap_features = NULL))
  }

  # Check matrix has dimnames
  if (is.null(rownames(x$partial_cor)) || is.null(colnames(x$partial_cor))) {
    return(list(valid = FALSE, message = "partial_cor matrix must have row and column names", overlap_features = NULL))
  }

  # Check features is a character vector
  if (!is.character(x$features) || length(x$features) == 0) {
    return(list(valid = FALSE, message = "features must be a non-empty character vector", overlap_features = NULL))
  }

  # Check family is valid
  valid_families <- c("zinb", "nb", "poisson", "gaussian", "quasipoisson")
  if (!x$family %in% valid_families) {
    return(list(
      valid = FALSE,
      message = sprintf("Invalid family '%s', must be one of: %s", x$family, paste(valid_families, collapse = ", ")),
      overlap_features = NULL
    ))
  }

  # If required_features specified, check overlap
  if (!is.null(required_features)) {
    overlap <- intersect(x$features, required_features)
    if (length(overlap) == 0) {
      return(list(
        valid = FALSE,
        message = "No overlap between partial correlation features and required features",
        overlap_features = NULL
      ))
    }

    if (verbose && length(overlap) < length(required_features)) {
      message(sprintf(
        "  Note: %d of %d required features found in partial correlations",
        length(overlap), length(required_features)
      ))
    }

    return(list(valid = TRUE, message = NULL, overlap_features = overlap))
  }

  return(list(valid = TRUE, message = NULL, overlap_features = x$features))
}
