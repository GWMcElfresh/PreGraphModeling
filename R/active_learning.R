#' Active Learning Scoring for RBM Candidate Cells
#'
#' Scores candidate observations (cells) for active learning by estimating the expected
#' impact each candidate would have on the RBM. Two scoring strategies are provided:
#'
#' - **"expected_gradient"**: a fast proxy for expected parameter change computed from
#'   the conditional expectation of hidden activations under the model-induced
#'   conditional activation distributions (positive-phase only). For each hidden layer,
#'   this uses the fact that the positive-phase weight gradient for a
#'   single observation is proportional to \eqn{v \otimes E[h | v]}.
#'   We score candidates by the Frobenius norm of this outer product, which equals
#'   \eqn{||v||_2 \cdot ||E[h | v]||_2}.
#'
#' - **"latent_entropy"**: total entropy of the hidden-layer conditional activation
#'   distributions for the
#'   candidate (Bernoulli entropy for binary layers; categorical entropy for softmax
#'   layers). Higher entropy corresponds to higher latent uncertainty.
#'
#' The implementation is chunked and can be parallelized with `future.apply`.
#'
#' @param rbmObject An RBM object fitted using `FitRBM()`.
#' @param candidateExpression A numeric matrix (or sparse `Matrix`) of expression values
#'   with rows as features and columns as candidate observations. Row names must be
#'   feature names. Missing RBM features are treated as zero.
#' @param method Scoring method: `"expected_gradient"` or `"latent_entropy"`.
#' @param aggregate How to aggregate across hidden layers: `"sum"` (default) or `"max"`.
#' @param transform Optional transform applied to candidate expression before scoring:
#'   `"none"` (default) or `"log1p"`.
#' @param chunkSize Integer, number of candidate columns to score per chunk (default: 1000).
#' @param parallel Logical; if `TRUE`, uses `future.apply::future_lapply()` over chunks.
#' @param numWorkers Optional integer number of parallel workers. If `NULL`, uses
#'   `parallel::detectCores() - 1`.
#' @param progressr Logical indicating whether to use `progressr` progress reporting.
#' @param verbose Logical indicating whether to print progress messages.
#'
#' @return A numeric vector of scores (length = number of candidate observations) with
#'   names matching `colnames(candidateExpression)` when available.
#'
#' @export
ScoreRBMActiveLearningCandidates <- function(rbmObject,
                                            candidateExpression,
                                            method = c("expected_gradient", "latent_entropy"),
                                            aggregate = c("sum", "max"),
                                            transform = c("none", "log1p"),
                                            chunkSize = 1000,
                                            parallel = FALSE,
                                            numWorkers = NULL,
                                            progressr = TRUE,
                                            verbose = TRUE) {
  method <- match.arg(method)
  aggregate <- match.arg(aggregate)
  transform <- match.arg(transform)

  if (!inherits(rbmObject, "RBM")) {
    stop("rbmObject must be an RBM object")
  }

  if (!is.matrix(candidateExpression) && !inherits(candidateExpression, "Matrix")) {
    stop("candidateExpression must be a matrix or Matrix")
  }

  if (is.null(rownames(candidateExpression))) {
    stop("candidateExpression must have row names (feature names)")
  }

  n_candidates <- ncol(candidateExpression)
  if (n_candidates == 0) {
    stop("candidateExpression must have at least one column (candidate observation)")
  }

  if (!is.numeric(chunkSize) || length(chunkSize) != 1 || chunkSize < 1) {
    stop("chunkSize must be a positive integer")
  }
  chunkSize <- as.integer(chunkSize)

  if (verbose) {
    message(sprintf("Scoring %d candidate observations (method=%s, chunkSize=%d)...",
      n_candidates, method, chunkSize
    ))
  }

  # Feature alignment (missing RBM features treated as zero)
  common_features <- intersect(rbmObject$visible_features, rownames(candidateExpression))
  if (length(common_features) == 0) {
    stop("No common features found between candidateExpression and rbmObject$visible_features")
  }

  # Create chunk index ranges over columns
  starts <- seq.int(1, n_candidates, by = chunkSize)
  ranges <- lapply(starts, function(s) c(s, min(s + chunkSize - 1L, n_candidates)))

  score_chunk <- function(col_range) {
    s <- col_range[1]
    e <- col_range[2]

    expr_chunk <- candidateExpression[common_features, s:e, drop = FALSE]

    if (transform == "log1p") {
      expr_chunk <- log1p(expr_chunk)
    }

    # Expected-gradient proxy needs ||v|| per observation
    if (method == "expected_gradient") {
      vnorm <- .rbm_col_l2_norm(expr_chunk)
    } else {
      vnorm <- NULL
    }

    # Compute per-layer scores and aggregate
    per_layer_scores <- list()

    for (factor_name in rbmObject$hidden_factors) {
      factor_info <- rbmObject$hidden_layers_info[[factor_name]]
      weights <- rbmObject$weights_per_layer[[factor_name]][common_features, , drop = FALSE]
      hidden_bias <- rbmObject$hidden_bias_per_layer[[factor_name]]

      # Forward pass pre-activation: observations x hidden_units
      # expr_chunk: features x observations
      if (inherits(expr_chunk, "Matrix")) {
        hidden_input <- Matrix::t(expr_chunk) %*% weights
      } else {
        hidden_input <- t(expr_chunk) %*% weights
      }

      # Add bias
      for (j in seq_len(ncol(hidden_input))) {
        hidden_input[, j] <- hidden_input[, j] + hidden_bias[j]
      }

      activation_func <- .get_activation_function(factor_info$type)
      hidden_act <- activation_func$forward(hidden_input)

      # Ensure 2D shape (some ops may drop dimensions for 1-unit layers)
      if (is.null(dim(hidden_act))) {
        hidden_act <- matrix(hidden_act, nrow = nrow(hidden_input), ncol = ncol(hidden_input))
      }

      if (method == "latent_entropy") {
        layer_score <- .rbm_hidden_entropy(hidden_act, factor_info$type)
      } else {
        # expected_gradient
        hnorm <- .rbm_row_l2_norm(hidden_act)
        layer_score <- as.numeric(vnorm * hnorm)
      }

      per_layer_scores[[factor_name]] <- layer_score
    }

    per_layer_mat <- do.call(cbind, per_layer_scores)

    if (aggregate == "sum") {
      scores <- rowSums(per_layer_mat, na.rm = TRUE)
    } else {
      # max
      scores <- apply(per_layer_mat, 1, max, na.rm = TRUE)
    }

    return(scores)
  }

  if (parallel) {
    if (is.null(numWorkers)) {
      numWorkers <- max(1, parallel::detectCores() - 1)
    }

    old_plan <- future::plan()
    on.exit(future::plan(old_plan), add = TRUE)
    future::plan(future::multisession, workers = numWorkers)

    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = length(ranges))
      scores_list <- progressr::with_progress({
        future.apply::future_lapply(ranges, function(rng) {
          out <- score_chunk(rng)
          p()
          out
        })
      })
    } else {
      scores_list <- future.apply::future_lapply(ranges, score_chunk)
    }
  } else {
    if (progressr) {
      progressr::handlers(global = TRUE)
      p <- progressr::progressor(steps = length(ranges))
      scores_list <- progressr::with_progress({
        lapply(ranges, function(rng) {
          out <- score_chunk(rng)
          p()
          out
        })
      })
    } else {
      scores_list <- lapply(ranges, score_chunk)
    }
  }

  scores <- unlist(scores_list, use.names = FALSE)

  # Attach candidate names if available
  cn <- colnames(candidateExpression)
  if (!is.null(cn)) {
    names(scores) <- cn
  }

  return(scores)
}


#' Select a Batch of RBM Active-Learning Candidates
#'
#' Convenience wrapper around `ScoreRBMActiveLearningCandidates()` that returns the
#' top `batchSize` candidates by score.
#'
#' @param rbmObject An RBM object fitted using `FitRBM()`.
#' @param candidateExpression Candidate expression matrix (features x candidates).
#' @param batchSize Integer, number of candidates to select.
#' @param ... Passed to `ScoreRBMActiveLearningCandidates()`.
#'
#' @return A list with:
#'   - `indices`: integer indices of selected candidates (into columns of candidateExpression)
#'   - `scores`: numeric scores for the selected candidates
#'
#' @export
SelectRBMActiveLearningCandidates <- function(rbmObject,
                                             candidateExpression,
                                             batchSize,
                                             ...) {
  if (!is.numeric(batchSize) || length(batchSize) != 1 || batchSize < 1) {
    stop("batchSize must be a positive integer")
  }
  batchSize <- as.integer(batchSize)

  scores <- ScoreRBMActiveLearningCandidates(rbmObject = rbmObject, candidateExpression = candidateExpression, ...)

  n <- length(scores)
  k <- min(batchSize, n)

  # Order descending, stable for ties
  ord <- order(scores, decreasing = TRUE)
  idx <- ord[seq_len(k)]

  list(indices = idx, scores = scores[idx])
}


.rbm_hidden_entropy <- function(hidden_act, factor_type) {
  if (factor_type == "binary") {
    p <- as.numeric(hidden_act)
    return(.rbm_entropy_bernoulli(p))
  }

  if (factor_type == "categorical") {
    return(.rbm_entropy_categorical(hidden_act))
  }

  # continuous/ordinal: current implementation is deterministic linear output,
  # so we do not treat it as a calibrated likelihood.
  rep(NA_real_, nrow(hidden_act))
}

.rbm_entropy_bernoulli <- function(p) {
  # Clamp for numerical stability
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  -(p * log(p) + (1 - p) * log(1 - p))
}

.rbm_entropy_categorical <- function(prob_mat) {
  # prob_mat: n_obs x k
  prob_mat <- pmin(pmax(prob_mat, 1e-12), 1)
  -rowSums(prob_mat * log(prob_mat))
}

.rbm_row_l2_norm <- function(mat) {
  # mat: n_obs x k
  # If a single-unit layer drops to a vector, treat it as (n_obs x 1)
  if (is.null(dim(mat))) {
    return(abs(as.numeric(mat)))
  }

  if (inherits(mat, "Matrix")) {
    return(sqrt(Matrix::rowSums(mat^2)))
  }

  sqrt(rowSums(mat^2))
}

.rbm_col_l2_norm <- function(mat) {
  # mat: features x n_obs; supports base and sparse Matrix
  if (inherits(mat, "Matrix")) {
    return(sqrt(Matrix::colSums(mat^2)))
  }
  sqrt(colSums(mat^2))
}
