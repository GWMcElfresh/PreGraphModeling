#' Fit Restricted Boltzmann Machine
#'
#' Fits a Restricted Boltzmann Machine (RBM) to connect features from a Seurat object
#' to a hidden layer populated by metadata factors. The edges are estimated using
#' partial correlations via quasilikelihood for the specified family, then trained
#' using Contrastive Divergence.
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
#' @param cd_k Number of Gibbs sampling steps for Contrastive Divergence (default: 1).
#'   Use cd_k=1 for CD-1, cd_k=5 for CD-5, etc.
#' @param learning_rate Learning rate for weight updates (default: 0.01).
#' @param n_epochs Number of training epochs for Contrastive Divergence (default: 10).
#' @param batch_size Batch size for mini-batch training (default: 100).
#' @param persistent Logical indicating whether to use Persistent Contrastive Divergence
#'   (default: FALSE). If TRUE, maintains chains between batches.
#' @param momentum Momentum coefficient for weight updates (default: 0.5).
#' @param weight_decay L2 regularization coefficient (default: 0.0001).
#' @param progressr Logical indicating whether to use progressr for progress reporting
#'   (default: TRUE).
#' @param parallel Logical indicating whether to use parallel processing (default: FALSE).
#' @param numWorkers Integer specifying number of workers for parallel processing.
#'   If NULL, uses detectCores()-1 (default: NULL).
#' @param verbose Logical indicating whether to print progress messages (default: TRUE).
#'
#' @return An object of class "RBM" containing:
#'   \itemize{
#'     \item weights_per_layer: List of weight matrices per hidden layer
#'     \item visible_bias: Bias terms for visible units (features)
#'     \item hidden_bias_per_layer: List of bias terms per hidden layer
#'     \item partial_correlations: Full partial correlation matrix among visible features
#'     \item visible_features: Names of features in visible layer
#'     \item hidden_layers_info: Information about each hidden layer (type, encoding, etc.)
#'     \item family: Distribution family used for estimation
#'     \item training_error: Reconstruction error per epoch during training
#'     \item fit_info: Additional information about the fitting process
#'   }
#'
#' @details
#' The RBM is fit by:
#' 1. Extracting expression data for specified features
#' 2. Computing partial correlations among features using quasilikelihood
#' 3. Initializing connections from features to hidden metadata factors (correlation-based)
#' 4. Training weights using Contrastive Divergence (CD-k) or Persistent CD
#' 5. Computing bias terms based on feature means and metadata distributions
#'
#' The resulting model can be used for prediction, visualization, and understanding
#' relationships between gene expression and cell metadata.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM with cell type as hidden layer, using CD-1
#' rbm <- FitRBM(
#'   seuratObject = pbmc,
#'   visibleFeatures = c("CD3D", "CD8A", "CD4", "CD19"),
#'   hiddenFactors = "CellType",
#'   family = "zinb",
#'   cd_k = 1,
#'   n_epochs = 10,
#'   parallel = TRUE
#' )
#'
#' # Fit RBM with multiple metadata factors using Persistent CD
#' rbm <- FitRBM(
#'   seuratObject = pbmc,
#'   hiddenFactors = c("CellType", "Treatment", "Batch"),
#'   family = "zinb",
#'   cd_k = 5,
#'   persistent = TRUE,
#'   n_epochs = 20
#' )
#'
#' # Access the weights and training error
#' print(rbm$weights_per_layer)
#' plot(rbm$training_error)
#' }
FitRBM <- function(seuratObject,
                   visibleFeatures = NULL,
                   hiddenFactors,
                   assay = "RNA",
                   layer = "counts",
                   family = "zinb",
                   minNonZero = 10,
                   cd_k = 1,
                   learning_rate = 0.01,
                   n_epochs = 10,
                   batch_size = 100,
                   persistent = FALSE,
                   momentum = 0.5,
                   weight_decay = 0.0001,
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
  # EXTRACT AND PROCESS METADATA - MULTI-LAYER ARCHITECTURE
  # ============================================================================

  if (verbose) {
    message("Processing metadata for multi-layer RBM...")
    message("  Detecting factor types...")
  }

  metadata_df <- metadata_obj[, hiddenFactors, drop = FALSE]
  
  # Detect type for each metadata factor
  hidden_layers_info <- list()
  
  for (factor_name in hiddenFactors) {
    factor_info <- .detect_factor_type(
      metadata_df[[factor_name]], 
      col_name = factor_name,
      verbose = verbose
    )
    
    hidden_layers_info[[factor_name]] <- factor_info
  }
  
  # Encode metadata factors appropriately
  if (verbose) {
    message("  Encoding metadata factors...")
  }
  
  hidden_layers_encoded <- list()
  hidden_layers_dim <- list()
  
  for (factor_name in hiddenFactors) {
    factor_info <- hidden_layers_info[[factor_name]]
    encoded <- .encode_factor(
      metadata_df[[factor_name]],
      factor_info,
      col_name = factor_name
    )
    
    hidden_layers_encoded[[factor_name]] <- encoded
    hidden_layers_dim[[factor_name]] <- ncol(encoded)
    
    if (verbose) {
      message(sprintf("    %s: %d hidden units (%s)", 
                     factor_name, ncol(encoded), factor_info$type))
    }
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
  # IDENTIFY FEATURES WITH NA/NaN PARTIAL CORRELATIONS
  # ============================================================================
  
  # Check for features with NA/NaN in their partial correlation row/column
  pcor_matrix <- pcor_result$partial_cor[valid_features, valid_features, drop = FALSE]
  
  # Identify features with any NA/NaN in their correlations
  na_per_feature <- rowSums(is.na(pcor_matrix) | is.nan(pcor_matrix))
  features_with_na_pcor <- valid_features[na_per_feature > 0]
  
  if (length(features_with_na_pcor) > 0) {
    if (verbose) {
      message(sprintf("  Identified %d features with NA/NaN partial correlations:", 
                     length(features_with_na_pcor)))
      if (length(features_with_na_pcor) <= 10) {
        message(sprintf("    %s", paste(features_with_na_pcor, collapse = ", ")))
      } else {
        message(sprintf("    %s ... (and %d more)", 
                       paste(head(features_with_na_pcor, 10), collapse = ", "),
                       length(features_with_na_pcor) - 10))
      }
    }
    
    # Remove features with NA/NaN from valid_features
    valid_features <- valid_features[na_per_feature == 0]
    
    if (length(valid_features) == 0) {
      stop("No valid features remain after removing features with NA/NaN partial correlations")
    }
  }

  # ============================================================================
  # COMPUTE FEATURE-METADATA CONNECTIONS (WEIGHTS) - ONE MATRIX PER HIDDEN LAYER
  # ============================================================================

  if (verbose) {
    message("Computing connections from visible features to hidden layers...")
  }

  # Extract valid features from expression matrix
  expr_valid <- expr_matrix[valid_features, , drop = FALSE]

  # Create separate weight matrix for each hidden layer
  weights_per_layer <- list()
  features_with_na_weights <- character(0)

  # Setup progress tracking
  if (progressr && requireNamespace("progressr", quietly = TRUE)) {
    progressr::handlers(global = TRUE)
    p <- progressr::progressor(steps = length(valid_features) * length(hiddenFactors))
  } else {
    p <- NULL
  }

  for (factor_name in hiddenFactors) {
    factor_info <- hidden_layers_info[[factor_name]]
    encoded_hidden <- hidden_layers_encoded[[factor_name]]
    n_hidden_units <- ncol(encoded_hidden)
    
    # Initialize weight matrix for this hidden layer
    weights_matrix <- matrix(0, 
                            nrow = length(valid_features), 
                            ncol = n_hidden_units,
                            dimnames = list(valid_features, colnames(encoded_hidden)))
    
    # Compute weights based on type-specific method matching activation function
    for (i in seq_along(valid_features)) {
      feature_name <- valid_features[i]
      feature_expr <- as.numeric(expr_valid[i, ])
      
      for (j in seq_len(n_hidden_units)) {
        hidden_values <- encoded_hidden[, j]
        
        # Type-specific weight computation matching activation function
        if (factor_info$type == "binary") {
          # Binary/Bernoulli: Use logistic regression coefficient approximation
          # Correlation scaled for sigmoid activation
          weight_val <- cor(feature_expr, hidden_values,
                           method = "pearson",
                           use = "pairwise.complete.obs")
          # Scale by ~1.7 to approximate logistic regression (probit approximation)
          weight_val <- weight_val * 1.7
          
        } else if (factor_info$type == "categorical") {
          # Categorical/Softmax: Use correlation for initial weights
          # Each unit in one-hot encoding gets independent weight
          weight_val <- cor(feature_expr, hidden_values,
                           method = "pearson",
                           use = "pairwise.complete.obs")
          
        } else if (factor_info$type == "ordinal") {
          # Ordinal/Gaussian: Use Pearson correlation for linear relationship
          weight_val <- cor(feature_expr, hidden_values,
                           method = "pearson",
                           use = "pairwise.complete.obs")
          
        } else if (factor_info$type == "continuous") {
          # Continuous/Gaussian: Use Pearson correlation for linear relationship
          weight_val <- cor(feature_expr, hidden_values,
                           method = "pearson",
                           use = "pairwise.complete.obs")
          
        } else {
          # Fallback: Spearman correlation
          weight_val <- cor(feature_expr, hidden_values,
                           method = "spearman",
                           use = "pairwise.complete.obs")
        }
        
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
    
    weights_per_layer[[factor_name]] <- weights_matrix
  }
  
  # ============================================================================
  # IDENTIFY FEATURES WITH NA/NaN WEIGHTS
  # ============================================================================
  
  if (length(features_with_na_weights) > 0) {
    if (verbose) {
      message(sprintf("  Identified %d features with NA/NaN weights:", 
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
      stop("No valid features remain after removing features with NA/NaN weights")
    }
    
    valid_features <- features_to_keep
    expr_valid <- expr_valid[valid_features, , drop = FALSE]
    
    # Update all weight matrices
    for (factor_name in hiddenFactors) {
      weights_per_layer[[factor_name]] <- weights_per_layer[[factor_name]][valid_features, , drop = FALSE]
    }
  }
  
  # ============================================================================
  # RECOMPUTE PARTIAL CORRELATIONS ON FINAL FEATURE SET
  # ============================================================================
  
  # Determine if we need to recompute
  need_recompute <- length(features_with_na_pcor) > 0 || length(features_with_na_weights) > 0
  
  if (need_recompute) {
    if (verbose) {
      message(sprintf("  Recomputing partial correlations with final feature set (%d features)...", 
                     length(valid_features)))
    }
    
    # IMPORTANT: Recompute partial correlations with only the final retained features
    # This is necessary because partial correlations condition on ALL features,
    # so removing features changes the definition of the quasilikelihood
    pcor_result_final <- EstimatePartialCorrelations(
      expressionMatrix = expr_matrix[valid_features, , drop = FALSE],
      metadata = NULL,
      family = family,
      minNonZero = minNonZero,
      progressr = progressr,  # Pass through progress settings
      parallel = parallel,     # Pass through parallel settings
      numWorkers = numWorkers, # Pass through worker count
      verbose = verbose        # Pass through verbose setting
    )
    
    pcor_matrix <- pcor_result_final$partial_cor[valid_features, valid_features, drop = FALSE]
  } else {
    # No features were removed, use original partial correlation matrix
    pcor_matrix <- pcor_result$partial_cor[valid_features, valid_features, drop = FALSE]
  }

  # ============================================================================
  # COMPUTE BIAS TERMS - ONE SET PER HIDDEN LAYER
  # ============================================================================

  if (verbose) {
    message("Computing bias terms...")
  }

  # Visible bias: mean expression for each feature
  visible_bias <- rowMeans(expr_valid)
  names(visible_bias) <- valid_features

  # Hidden bias: mean value for each encoded hidden layer
  hidden_bias_per_layer <- list()
  
  for (factor_name in hiddenFactors) {
    encoded_hidden <- hidden_layers_encoded[[factor_name]]
    hidden_bias_per_layer[[factor_name]] <- colMeans(encoded_hidden)
  }

  # ============================================================================
  # TRAIN RBM WITH CONTRASTIVE DIVERGENCE
  # ============================================================================
  
  if (verbose) {
    message(sprintf("\nTraining RBM with Contrastive Divergence (CD-%d)...", cd_k))
    if (persistent) {
      message("  Using Persistent Contrastive Divergence")
    }
    message(sprintf("  Learning rate: %.4f, Epochs: %d, Batch size: %d", 
                   learning_rate, n_epochs, batch_size))
  }
  
  # Train the RBM
  training_result <- .train_rbm_cd(
    visible_data = expr_valid,
    hidden_layers_info = hidden_layers_info,
    hidden_layers_encoded = hidden_layers_encoded,
    weights_per_layer = weights_per_layer,
    visible_bias = visible_bias,
    hidden_bias_per_layer = hidden_bias_per_layer,
    cd_k = cd_k,
    learning_rate = learning_rate,
    n_epochs = n_epochs,
    batch_size = batch_size,
    persistent = persistent,
    momentum = momentum,
    weight_decay = weight_decay,
    verbose = verbose,
    progressr = progressr
  )
  
  # Update weights and biases from training
  weights_per_layer <- training_result$weights_per_layer
  visible_bias <- training_result$visible_bias
  hidden_bias_per_layer <- training_result$hidden_bias_per_layer
  training_error <- training_result$training_error
  
  if (verbose) {
    message(sprintf("  Final reconstruction error: %.6f", training_error[length(training_error)]))
  }

  # ============================================================================
  # PREPARE OUTPUT - MULTI-LAYER RBM STRUCTURE
  # ============================================================================
  
  # Combine all excluded features
  all_excluded_features <- unique(c(
    pcor_result$excluded_features,
    features_with_na_pcor,
    features_with_na_weights
  ))
  
  # Count valid pairs from the pruned pcor_matrix
  n_valid_pairs <- sum(!is.na(pcor_matrix) & upper.tri(pcor_matrix))
  
  # Count total hidden units across all layers
  total_hidden_units <- sum(unlist(hidden_layers_dim))

  fit_info <- list(
    n_features = length(valid_features),
    n_hidden_layers = length(hiddenFactors),
    n_hidden_units_total = total_hidden_units,
    n_cells = ncol(expr_matrix),
    family = family,
    minNonZero = minNonZero,
    n_pairs = n_valid_pairs,
    excluded_features = all_excluded_features,
    n_excluded_low_counts = length(pcor_result$excluded_features),
    n_excluded_na_pcor = length(features_with_na_pcor),
    n_excluded_na_weights = length(features_with_na_weights),
    hidden_layers_info = hidden_layers_info,
    hidden_layers_dim = hidden_layers_dim
  )

  rbm <- structure(
    list(
      weights_per_layer = weights_per_layer,
      visible_bias = visible_bias,
      hidden_bias_per_layer = hidden_bias_per_layer,
      partial_correlations = pcor_matrix,
      visible_features = valid_features,
      hidden_factors = hiddenFactors,
      hidden_layers_info = hidden_layers_info,
      hidden_layers_encoded = hidden_layers_encoded,
      family = family,
      metadata = metadata_df,
      training_error = training_error,
      fit_info = fit_info
    ),
    class = "RBM"
  )

  if (verbose) {
    message("RBM fitting complete!")
    message(sprintf("  Visible layer: %d features", fit_info$n_features))
    message(sprintf("  Hidden layers: %d layers with %d total units", 
                   fit_info$n_hidden_layers, fit_info$n_hidden_units_total))
    for (factor_name in hiddenFactors) {
      factor_info <- hidden_layers_info[[factor_name]]
      n_units <- hidden_layers_dim[[factor_name]]
      message(sprintf("    - %s: %d units (%s)", factor_name, n_units, factor_info$type))
    }
  }

  return(rbm)
}


#' Print method for RBM objects
#' @param x An RBM object
#' @param ... Additional arguments (unused)
#' @export
print.RBM <- function(x, ...) {
  cat("Multi-Layer Restricted Boltzmann Machine\n")
  cat("=========================================\n\n")
  cat(sprintf("Visible layer:  %d features\n", x$fit_info$n_features))
  cat(sprintf("Hidden layers:  %d layers (%s)\n",
              x$fit_info$n_hidden_layers,
              paste(x$hidden_factors, collapse = ", ")))
  cat(sprintf("Total hidden units: %d\n", x$fit_info$n_hidden_units_total))
  
  cat("\nHidden layer details:\n")
  for (factor_name in x$hidden_factors) {
    factor_info <- x$hidden_layers_info[[factor_name]]
    n_units <- x$fit_info$hidden_layers_dim[[factor_name]]
    cat(sprintf("  %s: %d units, type=%s\n", factor_name, n_units, factor_info$type))
  }
  
  cat(sprintf("\nFamily:         %s\n", x$family))
  cat(sprintf("Observations:   %d cells\n", x$fit_info$n_cells))
  cat(sprintf("Valid feature pairs: %d\n", x$fit_info$n_pairs))
  
  # Training information
  if (!is.null(x$training_error)) {
    cat(sprintf("\nTraining epochs: %d\n", length(x$training_error)))
    cat(sprintf("Final reconstruction error: %.6f\n", x$training_error[length(x$training_error)]))
  }

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
  cat("Multi-Layer Restricted Boltzmann Machine Summary\n")
  cat("=================================================\n\n")

  print(object)

  cat("\nWeight statistics per layer:\n")
  for (factor_name in object$hidden_factors) {
    weights <- object$weights_per_layer[[factor_name]]
    cat(sprintf("\n  %s:\n", factor_name))
    cat(sprintf("    Min:    %.4f\n", min(weights, na.rm = TRUE)))
    cat(sprintf("    Q1:     %.4f\n", quantile(weights, 0.25, na.rm = TRUE)))
    cat(sprintf("    Median: %.4f\n", median(weights, na.rm = TRUE)))
    cat(sprintf("    Mean:   %.4f\n", mean(weights, na.rm = TRUE)))
    cat(sprintf("    Q3:     %.4f\n", quantile(weights, 0.75, na.rm = TRUE)))
    cat(sprintf("    Max:    %.4f\n", max(weights, na.rm = TRUE)))
  }

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
