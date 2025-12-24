#' Predict Hidden Layer Activations from Multi-Layer RBM
#'
#' Given a fitted multi-layer RBM and new expression data, predict the hidden layer 
#' activations for each metadata factor based on the learned weights and biases.
#'
#' @param object An RBM object fitted using FitRBM().
#' @param newdata A matrix or data frame of expression values for visible features.
#'   Rows should be features (matching those in the RBM) and columns are new observations.
#'   If NULL, uses the training data (default: NULL).
#' @param type Character string specifying the type of prediction.
#'   Only "activation" is supported. Note that this returns the *post-activation* output:
#'   probabilities for binary (sigmoid) and categorical (softmax) layers, and linear scores
#'   for continuous/ordinal layers (default: "activation").
#' @param return_list Logical indicating whether to return a list of activations per layer
#'   (TRUE) or a combined matrix (FALSE, default).
#' @param ... Additional arguments (unused).
#'
#' @return If return_list=FALSE (default): A combined matrix with all hidden unit activations.
#'   If return_list=TRUE: A list with one element per hidden layer containing activations.
#'
#' @details
#' For each hidden layer, predictions are computed using the forward pass:
#' \deqn{h_layer = activation_function(W_layer^T \%*\% v + b_layer)}
#' where v is the visible layer (expression), W_layer is the weight matrix for that layer,
#' b_layer is the hidden bias for that layer, and activation_function depends on the
#' metadata type (sigmoid for binary, softmax for categorical, linear for continuous/ordinal).
#'
#' For binary layers, the returned value can be interpreted as $P(h=1\mid v)$.
#' For categorical layers, each row is a probability vector over levels (rows sum to 1).
#' For continuous/ordinal layers, the output is a score on the model's internal scale and
#' is not a calibrated likelihood.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = c("CellType", "Treatment"), family = "zinb")
#'
#' # Predict on new data - combined matrix
#' predictions <- predict(rbm, newdata = new_expr_matrix)
#' 
#' # Predict per layer
#' predictions_list <- predict(rbm, newdata = new_expr_matrix, return_list = TRUE)
#' }
predict.RBM <- function(object, newdata = NULL, type = "activation", 
                       return_list = FALSE, ...) {
  
  # Validate type
  if (type != "activation") {
    stop("type must be 'activation' (only activation predictions supported for RBM)")
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
  n_obs <- ncol(expr_data)
  
  # ============================================================================
  # COMPUTE HIDDEN ACTIVATIONS FOR EACH LAYER
  # ============================================================================
  
  activations_per_layer <- list()
  
  for (factor_name in object$hidden_factors) {
    # Get weights and bias for this layer
    weights <- object$weights_per_layer[[factor_name]][common_features, , drop = FALSE]
    hidden_bias <- object$hidden_bias_per_layer[[factor_name]]
    factor_info <- object$hidden_layers_info[[factor_name]]
    
    # Forward pass: h = W^T * v + b_h
    # expr_data: features x observations
    # weights: features x hidden_units
    # Result: observations x hidden_units
    activations <- t(expr_data) %*% weights  # observations x hidden_units
    
    # Add hidden bias
    for (j in seq_len(ncol(activations))) {
      activations[, j] <- activations[, j] + hidden_bias[j]
    }
    
    # Apply activation function based on factor type
    activation_func <- .get_activation_function(factor_info$type)
    activations <- activation_func$forward(activations)
    
    # Set names
    if (is.matrix(activations)) {
      rownames(activations) <- colnames(expr_data)
    }
    
    activations_per_layer[[factor_name]] <- activations
  }
  
  # ============================================================================
  # RETURN RESULTS
  # ============================================================================
  
  if (return_list) {
    return(activations_per_layer)
  } else {
    # Combine all activations into a single matrix
    combined_activations <- do.call(cbind, activations_per_layer)
    rownames(combined_activations) <- colnames(expr_data)
    return(combined_activations)
  }
}


#' Reconstruct Visible Layer from Hidden Layers (Multi-Layer RBM Backward Pass)
#'
#' Given hidden layer activations from multiple layers, reconstruct the visible layer 
#' (expression values) using the RBM weights. This is useful for understanding what 
#' patterns of expression are associated with specific metadata factors.
#'
#' @param object An RBM object fitted using FitRBM().
#' @param hidden A list of matrices (one per hidden layer) or NULL. If NULL, uses the 
#'   encoded metadata from the training data (default: NULL). If provided, must be a 
#'   named list with names matching object$hidden_factors.
#'
#' @return A matrix of reconstructed visible layer values (expression) with
#'   observations as rows and features as columns.
#'
#' @details
#' Reconstruction uses the multi-layer RBM backward pass:
#' \deqn{v_{reconstructed} = \sum_{layers} W_layer \%*\% h_layer + b_v}
#' where h_layer is each hidden layer, W_layer is the corresponding weight matrix, 
#' and b_v is the visible bias.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = c("CellType", "Treatment"), family = "zinb")
#'
#' # Reconstruct expression from training metadata
#' reconstructed <- ReconstructRBM(rbm)
#'
#' # Reconstruct from specific hidden values
#' hidden_vals <- list(
#'   CellType = matrix(c(1, 0, 1), ncol = 3),  # 3 cell type categories
#'   Treatment = matrix(c(0, 1), ncol = 1)     # Binary treatment
#' )
#' reconstructed <- ReconstructRBM(rbm, hidden = hidden_vals)
#' }
ReconstructRBM <- function(object, hidden = NULL) {
  
  if (!inherits(object, "RBM")) {
    stop("object must be an RBM object")
  }
  
  # If no hidden data provided, use training metadata (encoded)
  if (is.null(hidden)) {
    hidden <- object$hidden_layers_encoded
  }
  
  # Validate hidden format
  if (!is.list(hidden)) {
    stop("hidden must be a list with one element per hidden layer")
  }
  
  # Check that all hidden factors are present
  missing_factors <- setdiff(object$hidden_factors, names(hidden))
  if (length(missing_factors) > 0) {
    stop(sprintf("hidden must contain all hidden factors. Missing: %s",
                paste(missing_factors, collapse = ", ")))
  }
  
  # ============================================================================
  # RECONSTRUCT VISIBLE LAYER FROM ALL HIDDEN LAYERS
  # ============================================================================
  
  # Initialize reconstruction with visible bias
  # Get number of observations from first hidden layer
  n_obs <- nrow(hidden[[object$hidden_factors[1]]])
  n_features <- length(object$visible_features)
  
  reconstructed <- matrix(0, nrow = n_obs, ncol = n_features)
  
  # Sum contributions from all hidden layers
  for (factor_name in object$hidden_factors) {
    hidden_vals <- hidden[[factor_name]]
    weights <- object$weights_per_layer[[factor_name]]
    
    # Backward pass for this layer: contribution = h * W^T
    # hidden_vals: observations x hidden_units
    # weights: features x hidden_units
    # Result: observations x features
    contribution <- hidden_vals %*% t(weights)
    
    reconstructed <- reconstructed + contribution
  }
  
  # Add visible bias
  for (j in seq_len(n_features)) {
    reconstructed[, j] <- reconstructed[, j] + object$visible_bias[j]
  }
  
  # Set column names
  colnames(reconstructed) <- object$visible_features
  if (!is.null(rownames(hidden[[object$hidden_factors[1]]]))) {
    rownames(reconstructed) <- rownames(hidden[[object$hidden_factors[1]]])
  }
  
  return(reconstructed)
}
