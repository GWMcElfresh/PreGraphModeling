#' Fit Restricted Boltzmann Machine
#'
#' Fits a Restricted Boltzmann Machine (RBM) to connect features from a Seurat object
#' to a hidden layer populated by metadata factors. The edges are estimated using
#' partial correlations via quasilikelihood for the specified family.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data.
#' @param visibleFeatures Character vector of feature (gene) names to use as the visible layer.
#'   If NULL, uses all features (default: NULL).
#' @param hiddenFactors Character vector of metadata column names to use as hidden units.
#'   These represent outcome or grouping variables.
#' @param assay Character string specifying which assay to use (default: "RNA").
#' @param layer Character string specifying which layer to use (default: "counts").
#' @param family Character string specifying the distribution family for quasilikelihood.
#'   Options: "zinb" (zero-inflated negative binomial, default), "nb", "poisson", "gaussian".
#' @param minNonZero Minimum number of non-zero observations required for a feature
#'   to be included (default: 10).
#' @param progressr Logical indicating whether to use progressr for progress reporting
#'   (default: TRUE).
#' @param parallel Logical indicating whether to use parallel processing (default: FALSE).
#' @param numWorkers Integer specifying number of workers for parallel processing.
#'   If NULL, uses detectCores()-1 (default: NULL).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return An object of class "RBM" containing:
#'   \itemize{
#'     \item weights: Matrix of edge weights (partial correlations) from visible to hidden layer
#'     \item visible_bias: Bias terms for visible units (features)
#'     \item hidden_bias: Bias terms for hidden units (metadata factors)
#'     \item partial_correlations: Full partial correlation matrix among visible features
#'     \item visible_features: Names of features in visible layer
#'     \item hidden_factors: Names of metadata factors in hidden layer
#'     \item family: Distribution family used for estimation
#'     \item metadata: Metadata used in fitting
#'     \item fit_info: Additional information about the fitting process
#'   }
#'
#' @details
#' The RBM is fit by:
#' 1. Extracting expression data for specified features
#' 2. Computing partial correlations among features using quasilikelihood
#' 3. Estimating connections from features to hidden metadata factors
#' 4. Computing bias terms based on feature means and metadata distributions
#'
#' The resulting model can be used for prediction, visualization, and understanding
#' relationships between gene expression and cell metadata.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM with cell type as hidden layer
#' rbm <- FitRBM(
#'   seuratObject = pbmc,
#'   visibleFeatures = c("CD3D", "CD8A", "CD4", "CD19"),
#'   hiddenFactors = "CellType",
#'   family = "zinb",
#'   parallel = TRUE
#' )
#'
#' # Fit RBM with multiple metadata factors
#' rbm <- FitRBM(
#'   seuratObject = pbmc,
#'   hiddenFactors = c("CellType", "Treatment", "Batch"),
#'   family = "zinb"
#' )
#'
#' # Access the weights
#' print(rbm$weights)
#' }
FitRBM <- function(seuratObject,
                   visibleFeatures = NULL,
                   hiddenFactors,
                   assay = "RNA",
                   layer = "counts",
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

  if (missing(hiddenFactors) || is.null(hiddenFactors)) {
    stop("hiddenFactors must be specified (metadata column names)")
  }

  if (!is.character(hiddenFactors)) {
    stop("hiddenFactors must be a character vector of metadata column names")
  }

  # Check that metadata columns exist
  metadata_obj <- seuratObject@meta.data
  missing_factors <- setdiff(hiddenFactors, colnames(metadata_obj))
  if (length(missing_factors) > 0) {
    stop(sprintf("Metadata columns not found: %s",
                paste(missing_factors, collapse = ", ")))
  }

  if (verbose) {
    message("Fitting Restricted Boltzmann Machine...")
    message(sprintf("  Hidden factors: %s", paste(hiddenFactors, collapse = ", ")))
  }

  # ============================================================================
  # EXTRACT EXPRESSION DATA
  # ============================================================================

  if (verbose) {
    message("Extracting expression data...")
  }

  # Get expression matrix
  if (inherits(seuratObject, "Seurat")) {
    # For Seurat v5+ with layers
    if (requireNamespace("SeuratObject", quietly = TRUE)) {
      tryCatch({
        expr_matrix <- SeuratObject::LayerData(
          object = seuratObject,
          assay = assay,
          layer = layer
        )
      }, error = function(e) {
        # Fallback for older Seurat versions
        expr_matrix <<- SeuratObject::GetAssayData(
          object = seuratObject,
          assay = assay,
          layer = layer
        )
      })
    } else {
      stop("SeuratObject package required but not available")
    }
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
      warning(sprintf("Features not found in expression matrix: %s",
                     paste(missing_features, collapse = ", ")))
    }
    visibleFeatures <- intersect(visibleFeatures, rownames(expr_matrix))
    if (length(visibleFeatures) == 0) {
      stop("No valid features found in expression matrix")
    }
    expr_matrix <- expr_matrix[visibleFeatures, , drop = FALSE]
  }

  if (verbose) {
    message(sprintf("  Expression matrix: %d features x %d cells",
                   nrow(expr_matrix), ncol(expr_matrix)))
  }

  # ============================================================================
  # EXTRACT METADATA
  # ============================================================================

  metadata_df <- metadata_obj[, hiddenFactors, drop = FALSE]

  # Convert factors to numeric for correlation computation
  metadata_numeric <- as.data.frame(lapply(metadata_df, function(col) {
    if (is.factor(col) || is.character(col)) {
      as.numeric(as.factor(col))
    } else {
      as.numeric(col)
    }
  }))
  colnames(metadata_numeric) <- hiddenFactors

  if (verbose) {
    message(sprintf("  Metadata: %d factors", length(hiddenFactors)))
  }

  # ============================================================================
  # COMPUTE PARTIAL CORRELATIONS AMONG VISIBLE FEATURES
  # ============================================================================

  if (verbose) {
    message("Computing partial correlations among visible features...")
  }

  pcor_result <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    metadata = NULL,  # Don't include metadata in feature-feature correlations
    family = family,
    minNonZero = minNonZero,
    progressr = progressr,
    parallel = parallel,
    numWorkers = numWorkers,
    verbose = verbose
  )

  # Get valid features (those that passed filtering)
  valid_features <- pcor_result$features
  
  # ============================================================================
  # PRUNE FEATURES WITH NA/NaN PARTIAL CORRELATIONS
  # ============================================================================
  
  # Check for features with NA/NaN in their partial correlation row/column
  pcor_matrix <- pcor_result$partial_cor[valid_features, valid_features, drop = FALSE]
  
  # Identify features with any NA/NaN in their correlations
  na_per_feature <- rowSums(is.na(pcor_matrix) | is.nan(pcor_matrix))
  features_with_na <- valid_features[na_per_feature > 0]
  
  if (length(features_with_na) > 0) {
    if (verbose) {
      message(sprintf("  Pruning %d features with NA/NaN partial correlations:", 
                     length(features_with_na)))
      if (length(features_with_na) <= 10) {
        message(sprintf("    %s", paste(features_with_na, collapse = ", ")))
      } else {
        message(sprintf("    %s ... (and %d more)", 
                       paste(head(features_with_na, 10), collapse = ", "),
                       length(features_with_na) - 10))
      }
    }
    
    # Remove features with NA/NaN from valid_features
    valid_features <- valid_features[na_per_feature == 0]
    
    # Update partial correlation matrix
    pcor_matrix <- pcor_result$partial_cor[valid_features, valid_features, drop = FALSE]
    
    if (length(valid_features) == 0) {
      stop("No valid features remain after pruning NA/NaN partial correlations")
    }
    
    if (verbose) {
      message(sprintf("  Retained %d features with valid partial correlations", 
                     length(valid_features)))
    }
  }

  # ============================================================================
  # COMPUTE FEATURE-METADATA CONNECTIONS (WEIGHTS)
  # ============================================================================

  if (verbose) {
    message("Computing connections from features to hidden factors...")
  }

  # Extract valid features from expression matrix
  expr_valid <- expr_matrix[valid_features, , drop = FALSE]

  # Compute feature-metadata correlations as initial weights
  weights_matrix <- matrix(0, nrow = length(valid_features), ncol = length(hiddenFactors),
                          dimnames = list(valid_features, hiddenFactors))

  # Setup progress tracking
  if (progressr && requireNamespace("progressr", quietly = TRUE)) {
    progressr::handlers(global = TRUE)
    p <- progressr::progressor(steps = length(valid_features))
  } else {
    p <- NULL
  }
  
  # Track features with NA weights
  features_with_na_weights <- character(0)

  for (i in seq_along(valid_features)) {
    feature_name <- valid_features[i]
    feature_expr <- as.numeric(expr_valid[i, ])

    for (j in seq_along(hiddenFactors)) {
      factor_name <- hiddenFactors[j]
      factor_values <- metadata_numeric[[factor_name]]

      # Compute correlation (edge weight)
      # Use Spearman for robustness to non-linearity
      weight_val <- cor(feature_expr, factor_values,
                       method = "spearman",
                       use = "pairwise.complete.obs")
      
      weights_matrix[i, j] <- weight_val
      
      # Check for NA/NaN weights
      if (is.na(weight_val) || is.nan(weight_val)) {
        if (!(feature_name %in% features_with_na_weights)) {
          features_with_na_weights <- c(features_with_na_weights, feature_name)
        }
      }
    }

    if (!is.null(p)) p()
  }
  
  # Prune features with NA/NaN weights
  if (length(features_with_na_weights) > 0) {
    if (verbose) {
      message(sprintf("  Pruning %d features with NA/NaN weights:", 
                     length(features_with_na_weights)))
      if (length(features_with_na_weights) <= 10) {
        message(sprintf("    %s", paste(features_with_na_weights, collapse = ", ")))
      } else {
        message(sprintf("    %s ... (and %d more)", 
                       paste(head(features_with_na_weights, 10), collapse = ", "),
                       length(features_with_na_weights) - 10))
      }
    }
    
    # Remove features with NA/NaN weights
    features_to_keep <- setdiff(valid_features, features_with_na_weights)
    
    if (length(features_to_keep) == 0) {
      stop("No valid features remain after pruning NA/NaN weights")
    }
    
    valid_features <- features_to_keep
    expr_valid <- expr_valid[valid_features, , drop = FALSE]
    weights_matrix <- weights_matrix[valid_features, , drop = FALSE]
    pcor_matrix <- pcor_matrix[valid_features, valid_features, drop = FALSE]
    
    if (verbose) {
      message(sprintf("  Final feature count: %d features with valid weights", 
                     length(valid_features)))
    }
  }

  # ============================================================================
  # COMPUTE BIAS TERMS
  # ============================================================================

  if (verbose) {
    message("Computing bias terms...")
  }

  # Visible bias: mean expression for each feature
  visible_bias <- rowMeans(expr_valid)
  names(visible_bias) <- valid_features

  # Hidden bias: mean value for each metadata factor
  hidden_bias <- colMeans(metadata_numeric)
  names(hidden_bias) <- hiddenFactors

  # ============================================================================
  # PREPARE OUTPUT
  # ============================================================================
  
  # Combine all excluded features
  all_excluded_features <- unique(c(
    pcor_result$excluded_features,
    features_with_na,
    features_with_na_weights
  ))
  
  # Count valid pairs from the pruned pcor_matrix
  n_valid_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))

  fit_info <- list(
    n_features = length(valid_features),
    n_hidden = length(hiddenFactors),
    n_cells = ncol(expr_matrix),
    family = family,
    minNonZero = minNonZero,
    n_pairs = n_valid_pairs,
    excluded_features = all_excluded_features,
    n_excluded_low_counts = length(pcor_result$excluded_features),
    n_excluded_na_pcor = length(features_with_na),
    n_excluded_na_weights = length(features_with_na_weights)
  )

  rbm <- structure(
    list(
      weights = weights_matrix,
      visible_bias = visible_bias,
      hidden_bias = hidden_bias,
      partial_correlations = pcor_matrix,
      visible_features = valid_features,
      hidden_factors = hiddenFactors,
      family = family,
      metadata = metadata_df,
      fit_info = fit_info
    ),
    class = "RBM"
  )

  if (verbose) {
    message("RBM fitting complete!")
    message(sprintf("  Visible layer: %d features", fit_info$n_features))
    message(sprintf("  Hidden layer: %d factors", fit_info$n_hidden))
    message(sprintf("  Total connections: %d", fit_info$n_features * fit_info$n_hidden))
  }

  return(rbm)
}


#' Print method for RBM objects
#' @param x An RBM object
#' @param ... Additional arguments (unused)
#' @export
print.RBM <- function(x, ...) {
  cat("Restricted Boltzmann Machine\n")
  cat("============================\n\n")
  cat(sprintf("Visible layer:  %d features\n", x$fit_info$n_features))
  cat(sprintf("Hidden layer:   %d factors (%s)\n",
              x$fit_info$n_hidden,
              paste(x$hidden_factors, collapse = ", ")))
  cat(sprintf("Total edges:    %d\n", x$fit_info$n_features * x$fit_info$n_hidden))
  cat(sprintf("Family:         %s\n", x$family))
  cat(sprintf("Observations:   %d cells\n", x$fit_info$n_cells))
  cat(sprintf("Valid feature pairs: %d\n", x$fit_info$n_pairs))

  if (length(x$fit_info$excluded_features) > 0) {
    cat(sprintf("\nExcluded features: %d total\n", length(x$fit_info$excluded_features)))
    if (!is.null(x$fit_info$n_excluded_low_counts) && x$fit_info$n_excluded_low_counts > 0) {
      cat(sprintf("  - %d features: < %d non-zero observations\n",
                  x$fit_info$n_excluded_low_counts, x$fit_info$minNonZero))
    }
    if (!is.null(x$fit_info$n_excluded_na_pcor) && x$fit_info$n_excluded_na_pcor > 0) {
      cat(sprintf("  - %d features: NA/NaN partial correlations\n",
                  x$fit_info$n_excluded_na_pcor))
    }
    if (!is.null(x$fit_info$n_excluded_na_weights) && x$fit_info$n_excluded_na_weights > 0) {
      cat(sprintf("  - %d features: NA/NaN weights\n",
                  x$fit_info$n_excluded_na_weights))
    }
  }

  invisible(x)
}


#' Summary method for RBM objects
#' @param object An RBM object
#' @param ... Additional arguments (unused)
#' @export
summary.RBM <- function(object, ...) {
  cat("Restricted Boltzmann Machine Summary\n")
  cat("====================================\n\n")

  print(object)

  cat("\nWeight statistics:\n")
  cat(sprintf("  Min:    %.4f\n", min(object$weights, na.rm = TRUE)))
  cat(sprintf("  Q1:     %.4f\n", quantile(object$weights, 0.25, na.rm = TRUE)))
  cat(sprintf("  Median: %.4f\n", median(object$weights, na.rm = TRUE)))
  cat(sprintf("  Mean:   %.4f\n", mean(object$weights, na.rm = TRUE)))
  cat(sprintf("  Q3:     %.4f\n", quantile(object$weights, 0.75, na.rm = TRUE)))
  cat(sprintf("  Max:    %.4f\n", max(object$weights, na.rm = TRUE)))

  cat("\nPartial correlation statistics:\n")
  # Get upper triangle (exclude diagonal)
  pcor_upper <- object$partial_correlations[upper.tri(object$partial_correlations)]
  pcor_valid <- pcor_upper[!is.na(pcor_upper)]

  if (length(pcor_valid) > 0) {
    cat(sprintf("  Min:    %.4f\n", min(pcor_valid)))
    cat(sprintf("  Q1:     %.4f\n", quantile(pcor_valid, 0.25)))
    cat(sprintf("  Median: %.4f\n", median(pcor_valid)))
    cat(sprintf("  Mean:   %.4f\n", mean(pcor_valid)))
    cat(sprintf("  Q3:     %.4f\n", quantile(pcor_valid, 0.75)))
    cat(sprintf("  Max:    %.4f\n", max(pcor_valid)))
  } else {
    cat("  No valid partial correlations computed\n")
  }

  invisible(object)
}
