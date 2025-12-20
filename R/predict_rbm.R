#' Predict Hidden Layer Activations from RBM
#'
#' Given a fitted RBM and new expression data, predict the hidden layer activations
#' (metadata factor values) based on the learned weights and biases.
#'
#' @param object An RBM object fitted using FitRBM().
#' @param newdata A matrix or data frame of expression values for visible features.
#'   Rows should be features (matching those in the RBM) and columns are new observations.
#'   If NULL, uses the training data (default: NULL).
#' @param type Character string specifying the type of prediction.
#'   Options: "activation" (hidden unit activations), "probability" (sigmoid-transformed
#'   activations), "label" (discretized labels for factors) (default: "activation").
#' @param ... Additional arguments (unused).
#'
#' @return A matrix of predicted hidden layer values with rows as observations and
#'   columns as hidden factors. The exact form depends on the 'type' argument:
#'   \itemize{
#'     \item activation: Raw activation values (continuous)
#'     \item probability: Sigmoid-transformed activations in [0, 1]
#'     \item label: Discretized labels (for categorical factors)
#'   }
#'
#' @details
#' Predictions are computed using the standard RBM forward pass:
#' \deqn{h = \sigma(W^T v + b_h)}
#' where v is the visible layer (expression), W is the weight matrix,
#' b_h is the hidden bias, and σ is an activation function.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = "CellType", family = "zinb")
#'
#' # Predict on new data
#' predictions <- predict(rbm, newdata = new_expr_matrix)
#'
#' # Get probability-transformed predictions
#' probs <- predict(rbm, newdata = new_expr_matrix, type = "probability")
#' }
predict.RBM <- function(object, newdata = NULL, type = "activation", ...) {
  
  # Validate type
  valid_types <- c("activation", "probability", "label")
  if (!type %in% valid_types) {
    stop(sprintf("type must be one of: %s", paste(valid_types, collapse = ", ")))
  }
  
  # If no new data, return NULL (training data predictions not stored)
  if (is.null(newdata)) {
    stop("newdata must be provided for predictions")
  }
  
  # Validate newdata format
  if (!is.matrix(newdata) && !is.data.frame(newdata)) {
    stop("newdata must be a matrix or data frame")
  }
  
  # Convert to matrix if needed
  if (is.data.frame(newdata)) {
    newdata <- as.matrix(newdata)
  }
  
  # Check that features match
  if (is.null(rownames(newdata))) {
    stop("newdata must have row names matching RBM visible features")
  }
  
  # Subset to visible features in the RBM
  missing_features <- setdiff(object$visible_features, rownames(newdata))
  if (length(missing_features) > 0) {
    warning(sprintf("%d features missing from newdata, will be treated as zero",
                   length(missing_features)))
  }
  
  # Get features present in both
  common_features <- intersect(object$visible_features, rownames(newdata))
  if (length(common_features) == 0) {
    stop("No common features found between newdata and RBM visible layer")
  }
  
  # Extract expression for common features
  expr_data <- newdata[common_features, , drop = FALSE]
  
  # Get weights for common features
  weights <- object$weights[common_features, , drop = FALSE]
  
  # ============================================================================
  # COMPUTE HIDDEN ACTIVATIONS
  # ============================================================================
  
  # Forward pass: h = W^T * v + b_h
  # expr_data: features x observations
  # weights: features x hidden_units
  # Result: observations x hidden_units
  
  activations <- t(expr_data) %*% weights  # observations x hidden_units
  
  # Add hidden bias
  hidden_bias_matrix <- matrix(
    rep(object$hidden_bias, each = ncol(expr_data)),
    nrow = ncol(expr_data),
    ncol = length(object$hidden_bias)
  )
  activations <- activations + hidden_bias_matrix
  
  # Set column names
  colnames(activations) <- object$hidden_factors
  rownames(activations) <- colnames(expr_data)
  
  # ============================================================================
  # TRANSFORM BASED ON TYPE
  # ============================================================================
  
  if (type == "activation") {
    # Return raw activations
    result <- activations
    
  } else if (type == "probability") {
    # Apply sigmoid transformation to get probabilities
    result <- 1 / (1 + exp(-activations))
    
  } else if (type == "label") {
    # Discretize into labels
    # For each factor, assign to the most likely category
    result <- activations
    
    # Normalize each column to probabilities
    for (j in seq_len(ncol(result))) {
      # Shift to positive range
      shifted <- result[, j] - min(result[, j]) + 1
      # Normalize
      result[, j] <- shifted / sum(shifted)
    }
    
    # Convert to discrete labels (1, 2, 3, ...)
    result <- apply(result, 2, function(col) {
      # Simple discretization: round to nearest integer
      round(col * (max(col) / min(col[col > 0])))
    })
    
    if (!is.matrix(result)) {
      result <- as.matrix(result)
      colnames(result) <- object$hidden_factors
      rownames(result) <- colnames(expr_data)
    }
  }
  
  return(result)
}


#' Reconstruct Visible Layer from Hidden Layer (RBM Backward Pass)
#'
#' Given hidden layer activations, reconstruct the visible layer (expression values)
#' using the RBM weights. This is useful for understanding what patterns of
#' expression are associated with specific metadata factors.
#'
#' @param object An RBM object fitted using FitRBM().
#' @param hidden A matrix of hidden layer values with observations as rows and
#'   hidden factors as columns. If NULL, uses the metadata from the training data
#'   (default: NULL).
#'
#' @return A matrix of reconstructed visible layer values (expression) with
#'   observations as rows and features as columns.
#'
#' @details
#' Reconstruction uses the RBM backward pass:
#' \deqn{v_{reconstructed} = W h + b_v}
#' where h is the hidden layer, W is the weight matrix, and b_v is the visible bias.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = "CellType", family = "zinb")
#'
#' # Reconstruct expression from metadata
#' reconstructed <- ReconstructRBM(rbm)
#'
#' # Reconstruct from specific hidden values
#' hidden_vals <- matrix(c(1, 0, 0, 2), nrow = 2, ncol = 2)
#' colnames(hidden_vals) <- c("Factor1", "Factor2")
#' reconstructed <- ReconstructRBM(rbm, hidden = hidden_vals)
#' }
ReconstructRBM <- function(object, hidden = NULL) {
  
  if (!inherits(object, "RBM")) {
    stop("object must be an RBM object")
  }
  
  # If no hidden data provided, use training metadata
  if (is.null(hidden)) {
    # Convert metadata to numeric
    hidden <- as.matrix(as.data.frame(lapply(object$metadata, function(col) {
      if (is.factor(col) || is.character(col)) {
        as.numeric(as.factor(col))
      } else {
        as.numeric(col)
      }
    })))
    colnames(hidden) <- object$hidden_factors
  }
  
  # Validate hidden format
  if (!is.matrix(hidden) && !is.data.frame(hidden)) {
    stop("hidden must be a matrix or data frame")
  }
  
  if (is.data.frame(hidden)) {
    hidden <- as.matrix(hidden)
  }
  
  # Check dimensions
  if (ncol(hidden) != length(object$hidden_factors)) {
    stop(sprintf("hidden must have %d columns (one per hidden factor)",
                length(object$hidden_factors)))
  }
  
  # ============================================================================
  # RECONSTRUCT VISIBLE LAYER
  # ============================================================================
  
  # Backward pass: v = W * h + b_v
  # hidden: observations x hidden_units
  # weights: features x hidden_units
  # Result: observations x features
  
  reconstructed <- hidden %*% t(object$weights)  # observations x features
  
  # Add visible bias
  visible_bias_matrix <- matrix(
    rep(object$visible_bias, each = nrow(hidden)),
    nrow = nrow(hidden),
    ncol = length(object$visible_bias)
  )
  reconstructed <- reconstructed + visible_bias_matrix
  
  # Set column names
  colnames(reconstructed) <- object$visible_features
  if (!is.null(rownames(hidden))) {
    rownames(reconstructed) <- rownames(hidden)
  }
  
  return(reconstructed)
}
