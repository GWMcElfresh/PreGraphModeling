#' Detect Metadata Factor Type
#'
#' Automatically detects the type of a metadata factor (binary, categorical, ordinal, continuous)
#' 
#' @param column A vector representing a metadata column
#' @param col_name Name of the column (for informative messages)
#' @param verbose Logical indicating whether to print detection info
#' 
#' @return A list with type and additional info
#' @keywords internal
#' @noRd
.detect_factor_type <- function(column, col_name = "column", verbose = FALSE) {
  
  # Handle factor columns
  if (is.factor(column)) {
    levels_count <- nlevels(column)
    
    if (levels_count == 2) {
      if (verbose) message(sprintf("  %s: Binary (2 levels)", col_name))
      return(list(
        type = "binary",
        levels = levels(column),
        n_levels = 2,
        is_ordered = is.ordered(column)
      ))
    } else {
      if (is.ordered(column)) {
        if (verbose) message(sprintf("  %s: Ordinal (%d ordered levels)", col_name, levels_count))
        return(list(
          type = "ordinal",
          levels = levels(column),
          n_levels = levels_count,
          is_ordered = TRUE
        ))
      } else {
        if (verbose) message(sprintf("  %s: Categorical (%d levels)", col_name, levels_count))
        return(list(
          type = "categorical",
          levels = levels(column),
          n_levels = levels_count,
          is_ordered = FALSE
        ))
      }
    }
  }
  
  # Handle character columns
  if (is.character(column)) {
    unique_vals <- unique(column[!is.na(column)])
    n_unique <- length(unique_vals)
    
    if (n_unique == 2) {
      if (verbose) message(sprintf("  %s: Binary (2 unique values)", col_name))
      return(list(
        type = "binary",
        levels = sort(unique_vals),
        n_levels = 2,
        is_ordered = FALSE
      ))
    } else {
      if (verbose) message(sprintf("  %s: Categorical (%d unique values)", col_name, n_unique))
      return(list(
        type = "categorical",
        levels = sort(unique_vals),
        n_levels = n_unique,
        is_ordered = FALSE
      ))
    }
  }
  
  # Handle numeric columns
  if (is.numeric(column)) {
    unique_vals <- unique(column[!is.na(column)])
    n_unique <- length(unique_vals)
    
    # Check if it's binary (0/1 or TRUE/FALSE encoded as 0/1)
    if (n_unique == 2 && all(unique_vals %in% c(0, 1))) {
      if (verbose) message(sprintf("  %s: Binary (0/1)", col_name))
      return(list(
        type = "binary",
        levels = c(0, 1),
        n_levels = 2,
        is_ordered = FALSE
      ))
    }
    
    # Check if it looks like discrete ordinal (small number of integer values)
    if (all(column == floor(column), na.rm = TRUE) && n_unique <= 10) {
      if (verbose) message(sprintf("  %s: Ordinal (%d integer levels)", col_name, n_unique))
      return(list(
        type = "ordinal",
        levels = sort(unique_vals),
        n_levels = n_unique,
        is_ordered = TRUE
      ))
    }
    
    # Otherwise, treat as continuous
    if (verbose) message(sprintf("  %s: Continuous (range: %.2f to %.2f)", 
                                col_name, min(column, na.rm = TRUE), max(column, na.rm = TRUE)))
    return(list(
      type = "continuous",
      levels = NULL,
      n_levels = NA,
      is_ordered = FALSE,
      range = range(column, na.rm = TRUE)
    ))
  }
  
  # Default fallback
  warning(sprintf("Column %s has unrecognized type, treating as continuous", col_name))
  return(list(
    type = "continuous",
    levels = NULL,
    n_levels = NA,
    is_ordered = FALSE
  ))
}


#' Encode Metadata Factor
#'
#' Encodes a metadata factor according to its detected type
#' 
#' @param column A vector representing a metadata column
#' @param factor_info List with factor type information from .detect_factor_type
#' @param col_name Name of the column
#' 
#' @return Encoded matrix (n_obs x n_units)
#' @keywords internal
#' @noRd
.encode_factor <- function(column, factor_info, col_name = "column") {
  
  n_obs <- length(column)
  
  if (factor_info$type == "binary") {
    # Binary: encode as 0/1
    if (is.factor(column)) {
      # Convert factor to 0/1 (first level = 0, second level = 1)
      encoded <- as.integer(column) - 1
    } else if (is.character(column)) {
      # Convert character to 0/1 based on sorted unique values
      encoded <- as.integer(factor(column, levels = factor_info$levels)) - 1
    } else {
      # Already numeric, ensure 0/1
      encoded <- as.integer(column)
    }
    return(matrix(encoded, ncol = 1, dimnames = list(NULL, col_name)))
    
  } else if (factor_info$type == "categorical") {
    # Categorical: one-hot encoding
    if (is.factor(column)) {
      levels_to_use <- levels(column)
    } else {
      levels_to_use <- factor_info$levels
    }
    
    # Create one-hot matrix
    one_hot <- matrix(0, nrow = n_obs, ncol = length(levels_to_use))
    colnames(one_hot) <- paste0(col_name, "_", levels_to_use)
    
    for (i in seq_along(levels_to_use)) {
      one_hot[column == levels_to_use[i], i] <- 1
    }
    
    return(one_hot)
    
  } else if (factor_info$type == "ordinal") {
    # Ordinal: encode as continuous numeric (maintaining order)
    if (is.factor(column)) {
      encoded <- as.integer(column)
    } else {
      # Map to integer positions
      encoded <- match(column, sort(factor_info$levels))
    }
    
    # Normalize to [0, 1] range
    encoded <- (encoded - 1) / (factor_info$n_levels - 1)
    
    return(matrix(encoded, ncol = 1, dimnames = list(NULL, col_name)))
    
  } else if (factor_info$type == "continuous") {
    # Continuous: standardize (mean 0, sd 1)
    encoded <- as.numeric(column)
    encoded <- (encoded - mean(encoded, na.rm = TRUE)) / (sd(encoded, na.rm = TRUE) + 1e-8)
    
    return(matrix(encoded, ncol = 1, dimnames = list(NULL, col_name)))
  }
  
  stop(sprintf("Unknown factor type for column %s", col_name))
}


#' Get Activation Function for Factor Type
#'
#' Returns the appropriate activation function for a given factor type
#' 
#' @param factor_type Character string: "binary", "categorical", "ordinal", "continuous"
#' 
#' @return A list with forward and inverse activation functions
#' @keywords internal
#' @noRd
.get_activation_function <- function(factor_type) {
  
  if (factor_type == "binary") {
    # Sigmoid activation for binary
    return(list(
      forward = function(x) 1 / (1 + exp(-x)),
      inverse = function(p) log(p / (1 - p + 1e-8)),
      name = "sigmoid"
    ))
    
  } else if (factor_type == "categorical") {
    # Softmax activation for categorical
    return(list(
      forward = function(x) {
        # x is matrix: rows = observations, cols = categories
        exp_x <- exp(x - apply(x, 1, max))  # Subtract max for numerical stability
        exp_x / rowSums(exp_x)
      },
      inverse = function(p) {
        # Log probabilities (relative to last category)
        log(p[, -ncol(p), drop = FALSE] / p[, ncol(p)] + 1e-8)
      },
      name = "softmax"
    ))
    
  } else if (factor_type == "ordinal" || factor_type == "continuous") {
    # Identity/linear activation for ordinal and continuous
    return(list(
      forward = function(x) x,
      inverse = function(x) x,
      name = "linear"
    ))
  }
  
  stop(sprintf("Unknown factor type: %s", factor_type))
}
