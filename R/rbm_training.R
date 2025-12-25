#' Gibbs Sample Visible Layer
#'
#' Performs Gibbs sampling to sample the visible layer given hidden activations
#' 
#' @param hidden_activations Matrix of hidden layer activations (observations x hidden_units)
#' @param weights_per_layer List of weight matrices per hidden layer
#' @param visible_bias Bias vector for visible units
#' @param sample Logical, whether to sample or return probabilities (default: TRUE)
#' 
#' @return Matrix of sampled visible layer (features x observations)
#' @keywords internal
#' @noRd
.gibbs_sample_visible <- function(hidden_activations_per_layer, 
                                   weights_per_layer,
                                   visible_bias,
                                   sample = TRUE) {
  
  # Combine contributions from all hidden layers
  # visible_recon = sum over layers of (W_layer %*% h_layer) + b_v
  
  n_obs <- nrow(hidden_activations_per_layer[[1]])
  n_visible <- length(visible_bias)
  
  # Initialize reconstruction
  visible_input <- matrix(0, nrow = n_visible, ncol = n_obs)
  
  # Add contribution from each hidden layer
  for (layer_name in names(weights_per_layer)) {
    weights <- weights_per_layer[[layer_name]]  # features x hidden_units
    hidden_act <- hidden_activations_per_layer[[layer_name]]  # observations x hidden_units
    
    # W %*% h^T gives features x observations
    visible_input <- visible_input + weights %*% t(hidden_act)
  }
  
  # Add visible bias (broadcast across observations)
  for (i in seq_len(n_visible)) {
    visible_input[i, ] <- visible_input[i, ] + visible_bias[i]
  }
  
  # For continuous/count data, we use Gaussian approximation
  # In a proper ZINB RBM, we'd sample from ZINB, but for simplicity we use mean
  if (sample) {
    # Add Gaussian noise
    visible_sample <- visible_input + rnorm(n_visible * n_obs, mean = 0, sd = 0.1)
    # Ensure non-negative for count data
    visible_sample[visible_sample < 0] <- 0
    return(visible_sample)
  } else {
    return(visible_input)
  }
}


#' Gibbs Sample Hidden Layer
#'
#' Performs Gibbs sampling to sample a hidden layer given visible data
#' 
#' @param visible_data Matrix of visible layer data (features x observations)
#' @param weights Weight matrix for this layer (features x hidden_units)
#' @param hidden_bias Bias vector for hidden units
#' @param factor_type Type of hidden layer ("binary", "categorical", "ordinal", "continuous")
#' @param sample Logical, whether to sample or return probabilities/activations
#' 
#' @return Matrix of sampled/activated hidden layer (observations x hidden_units)
#' @keywords internal
#' @noRd
.gibbs_sample_hidden <- function(visible_data, 
                                  weights,
                                  hidden_bias,
                                  factor_type,
                                  sample = TRUE) {
  
  # hidden_input = W^T %*% v + b_h
  # visible_data: features x observations
  # weights: features x hidden_units
  # Result: observations x hidden_units
  
  hidden_input <- t(visible_data) %*% weights  # observations x hidden_units
  
  # Add bias
  for (j in seq_len(ncol(hidden_input))) {
    hidden_input[, j] <- hidden_input[, j] + hidden_bias[j]
  }
  
  # Apply activation function based on type
  activation_func <- .get_activation_function(factor_type)
  hidden_prob <- activation_func$forward(hidden_input)
  
  if (!sample) {
    return(hidden_prob)
  }
  
  # Sample based on type
  if (factor_type == "binary") {
    # Bernoulli sampling
    hidden_sample <- matrix(rbinom(length(hidden_prob), 1, hidden_prob),
                           nrow = nrow(hidden_prob),
                           ncol = ncol(hidden_prob))
    return(hidden_sample)
    
  } else if (factor_type == "categorical") {
    # Multinomial sampling (one-hot)
    hidden_sample <- matrix(0, nrow = nrow(hidden_prob), ncol = ncol(hidden_prob))
    for (i in seq_len(nrow(hidden_prob))) {
      sampled_cat <- sample(seq_len(ncol(hidden_prob)), 1, prob = hidden_prob[i, ])
      hidden_sample[i, sampled_cat] <- 1
    }
    return(hidden_sample)
    
  } else {
    # Ordinal/continuous: use probabilities as-is (or add small Gaussian noise)
    if (sample) {
      hidden_sample <- hidden_prob + rnorm(length(hidden_prob), mean = 0, sd = 0.01)
      return(hidden_sample)
    } else {
      return(hidden_prob)
    }
  }
}


#' Train RBM with Contrastive Divergence
#'
#' Trains RBM weights using Contrastive Divergence (CD-k) or Persistent CD
#' 
#' @param visible_data Matrix of visible layer data (features x observations)
#' @param hidden_layers_info List of hidden layer information (type, encoding, etc.)
#' @param hidden_layers_encoded List of encoded hidden layer matrices
#' @param weights_per_layer Initial weight matrices per layer (will be updated)
#' @param visible_bias Initial visible bias (will be updated)
#' @param hidden_bias_per_layer Initial hidden bias per layer (will be updated)
#' @param cd_k Number of Gibbs sampling steps (default: 1)
#' @param learning_rate Learning rate for weight updates (default: 0.01)
#' @param n_epochs Number of training epochs (default: 10)
#' @param batch_size Batch size for mini-batch training (default: 100)
#' @param persistent Logical, whether to use Persistent CD (default: FALSE)
#' @param momentum Momentum coefficient for weight updates (default: 0.5)
#' @param weight_decay L2 regularization coefficient (default: 0.0001)
#' @param verbose Logical, whether to print progress (default: TRUE)
#' @param progressr Logical, whether to use progressr (default: FALSE)
#' 
#' @return List with updated weights_per_layer, visible_bias, hidden_bias_per_layer, and training_error
#' @keywords internal
#' @noRd
.train_rbm_cd <- function(visible_data,
                          hidden_layers_info,
                          hidden_layers_encoded,
                          weights_per_layer,
                          visible_bias,
                          hidden_bias_per_layer,
                          cd_k = 1,
                          learning_rate = 0.01,
                          n_epochs = 10,
                          batch_size = 100,
                          persistent = FALSE,
                          momentum = 0.5,
                          weight_decay = 0.0001,
                          verbose = TRUE,
                          progressr = FALSE) {
  
  n_features <- nrow(visible_data)
  n_obs <- ncol(visible_data)
  layer_names <- names(weights_per_layer)
  
  # Initialize momentum terms
  weight_velocity_per_layer <- lapply(weights_per_layer, function(w) matrix(0, nrow = nrow(w), ncol = ncol(w)))
  visible_bias_velocity <- rep(0, length(visible_bias))
  hidden_bias_velocity_per_layer <- lapply(hidden_bias_per_layer, function(b) rep(0, length(b)))
  
  # For persistent CD, maintain chains
  if (persistent) {
    persistent_chains <- lapply(hidden_layers_info, function(info) {
      matrix(0, nrow = batch_size, ncol = info$n_units)
    })
    names(persistent_chains) <- layer_names
  }
  
  # Setup progress tracking
  if (progressr) {
    p <- progressr::progressor(steps = n_epochs)
  } else {
    p <- NULL
  }
  
  training_errors <- numeric(n_epochs)
  
  for (epoch in seq_len(n_epochs)) {
    
    # Shuffle data
    shuffle_idx <- sample(seq_len(n_obs))
    n_batches <- ceiling(n_obs / batch_size)
    
    epoch_error <- 0
    
    for (batch_idx in seq_len(n_batches)) {
      
      # Get batch indices
      start_idx <- (batch_idx - 1) * batch_size + 1
      end_idx <- min(batch_idx * batch_size, n_obs)
      batch_indices <- shuffle_idx[start_idx:end_idx]
      actual_batch_size <- length(batch_indices)
      
      # Extract batch
      v0 <- visible_data[, batch_indices, drop = FALSE]  # features x batch_size
      
      # ======================================================================
      # POSITIVE PHASE: Compute hidden activations from data
      # ======================================================================
      
      h0_per_layer <- list()
      for (layer_name in layer_names) {
        weights <- weights_per_layer[[layer_name]]
        hidden_bias <- hidden_bias_per_layer[[layer_name]]
        factor_type <- hidden_layers_info[[layer_name]]$type
        
        # Sample hidden layer given visible data
        h0 <- .gibbs_sample_hidden(v0, weights, hidden_bias, factor_type, sample = TRUE)
        h0_per_layer[[layer_name]] <- h0
      }
      
      # ======================================================================
      # NEGATIVE PHASE: Run k steps of Gibbs sampling
      # ======================================================================
      
      if (persistent && batch_idx == 1) {
        # Initialize persistent chains with positive phase
        hk_per_layer <- h0_per_layer
      } else if (persistent) {
        # Use persistent chains
        hk_per_layer <- lapply(layer_names, function(ln) {
          persistent_chains[[ln]][seq_len(actual_batch_size), , drop = FALSE]
        })
        names(hk_per_layer) <- layer_names
      } else {
        # Start from positive phase
        hk_per_layer <- h0_per_layer
      }
      
      # Run k Gibbs steps
      for (gibbs_step in seq_len(cd_k)) {
        # Sample visible from hidden
        vk <- .gibbs_sample_visible(hk_per_layer, weights_per_layer, visible_bias, sample = TRUE)
        
        # Sample hidden from visible
        for (layer_name in layer_names) {
          weights <- weights_per_layer[[layer_name]]
          hidden_bias <- hidden_bias_per_layer[[layer_name]]
          factor_type <- hidden_layers_info[[layer_name]]$type
          
          # For last Gibbs step, don't sample (use probabilities for better gradient)
          do_sample <- (gibbs_step < cd_k)
          hk_per_layer[[layer_name]] <- .gibbs_sample_hidden(vk, weights, hidden_bias, 
                                                              factor_type, sample = do_sample)
        }
      }
      
      # Update persistent chains
      if (persistent) {
        for (layer_name in layer_names) {
          persistent_chains[[layer_name]][seq_len(actual_batch_size), ] <- hk_per_layer[[layer_name]]
        }
      }
      
      # ======================================================================
      # COMPUTE GRADIENTS AND UPDATE WEIGHTS
      # ======================================================================
      
      # Update weights and biases for each layer
      for (layer_name in layer_names) {
        weights <- weights_per_layer[[layer_name]]
        h0 <- h0_per_layer[[layer_name]]
        hk <- hk_per_layer[[layer_name]]
        
        # Compute gradients: <v0 h0^T> - <vk hk^T>
        positive_grad <- (v0 %*% h0) / actual_batch_size
        negative_grad <- (vk %*% hk) / actual_batch_size
        
        weight_grad <- positive_grad - negative_grad - weight_decay * weights
        
        # Momentum update
        weight_velocity_per_layer[[layer_name]] <- momentum * weight_velocity_per_layer[[layer_name]] + 
                                                     learning_rate * weight_grad
        weights_per_layer[[layer_name]] <- weights + weight_velocity_per_layer[[layer_name]]
        
        # Update hidden bias
        hidden_bias_grad <- colMeans(h0) - colMeans(hk)
        hidden_bias_velocity_per_layer[[layer_name]] <- momentum * hidden_bias_velocity_per_layer[[layer_name]] +
                                                          learning_rate * hidden_bias_grad
        hidden_bias_per_layer[[layer_name]] <- hidden_bias_per_layer[[layer_name]] + 
                                                 hidden_bias_velocity_per_layer[[layer_name]]
      }
      
      # Update visible bias
      visible_bias_grad <- rowMeans(v0) - rowMeans(vk)
      visible_bias_velocity <- momentum * visible_bias_velocity + learning_rate * visible_bias_grad
      visible_bias <- visible_bias + visible_bias_velocity
      
      # Compute reconstruction error
      batch_error <- mean((v0 - vk)^2)
      epoch_error <- epoch_error + batch_error
    }
    
    # Record training error
    training_errors[epoch] <- epoch_error / n_batches
    
    if (verbose) {
      message(sprintf("  Epoch %d/%d: Reconstruction Error = %.6f", 
                     epoch, n_epochs, training_errors[epoch]))
    }
    
    if (!is.null(p)) p()
  }
  
  return(list(
    weights_per_layer = weights_per_layer,
    visible_bias = visible_bias,
    hidden_bias_per_layer = hidden_bias_per_layer,
    training_error = training_errors
  ))
}
