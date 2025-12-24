test_that("ScoreRBMActiveLearningCandidates returns scores for candidates", {
  skip_if_not_installed("Matrix")

  set.seed(1)
  # Mock RBM with one binary and one categorical layer
  rbm <- structure(
    list(
      weights_per_layer = list(
        Bin = matrix(rnorm(6 * 1), nrow = 6, ncol = 1,
          dimnames = list(paste0("Gene", 1:6), "u1")
        ),
        Cat = matrix(rnorm(6 * 3), nrow = 6, ncol = 3,
          dimnames = list(paste0("Gene", 1:6), c("c1", "c2", "c3"))
        )
      ),
      visible_bias = setNames(rnorm(6), paste0("Gene", 1:6)),
      hidden_bias_per_layer = list(
        Bin = setNames(rnorm(1), "u1"),
        Cat = setNames(rnorm(3), c("c1", "c2", "c3"))
      ),
      partial_correlations = diag(1, 6),
      visible_features = paste0("Gene", 1:6),
      hidden_factors = c("Bin", "Cat"),
      family = "zinb",
      hidden_layers_info = list(
        Bin = list(type = "binary"),
        Cat = list(type = "categorical")
      ),
      fit_info = list(n_features = 6)
    ),
    class = "RBM"
  )

  cand <- matrix(abs(rnorm(6 * 10)), nrow = 6, ncol = 10,
    dimnames = list(paste0("Gene", 1:6), paste0("Cell", 1:10))
  )

  scores_e <- ScoreRBMActiveLearningCandidates(
    rbmObject = rbm,
    candidateExpression = cand,
    method = "expected_gradient",
    progressr = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )
  expect_true(is.numeric(scores_e))
  expect_length(scores_e, 10)
  expect_named(scores_e, paste0("Cell", 1:10))

  scores_h <- ScoreRBMActiveLearningCandidates(
    rbmObject = rbm,
    candidateExpression = cand,
    method = "latent_entropy",
    progressr = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )
  expect_true(is.numeric(scores_h))
  expect_length(scores_h, 10)
})


test_that("ScoreRBMActiveLearningCandidates accepts sparse Matrix", {
  skip_if_not_installed("Matrix")

  set.seed(2)
  rbm <- structure(
    list(
      weights_per_layer = list(
        Bin = matrix(rnorm(4 * 1), nrow = 4, ncol = 1,
          dimnames = list(paste0("Gene", 1:4), "u1")
        )
      ),
      visible_bias = setNames(rnorm(4), paste0("Gene", 1:4)),
      hidden_bias_per_layer = list(Bin = setNames(rnorm(1), "u1")),
      partial_correlations = diag(1, 4),
      visible_features = paste0("Gene", 1:4),
      hidden_factors = "Bin",
      family = "zinb",
      hidden_layers_info = list(Bin = list(type = "binary")),
      fit_info = list(n_features = 4)
    ),
    class = "RBM"
  )

  dense <- matrix(rpois(4 * 7, lambda = 2), nrow = 4, ncol = 7,
    dimnames = list(paste0("Gene", 1:4), paste0("Cell", 1:7))
  )
  sparse <- Matrix::Matrix(dense, sparse = TRUE)

  scores <- ScoreRBMActiveLearningCandidates(
    rbmObject = rbm,
    candidateExpression = sparse,
    method = "expected_gradient",
    progressr = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )

  expect_true(is.numeric(scores))
  expect_length(scores, 7)
})


test_that("SelectRBMActiveLearningCandidates returns top indices", {
  set.seed(3)
  rbm <- structure(
    list(
      weights_per_layer = list(
        Bin = matrix(rnorm(5 * 1), nrow = 5, ncol = 1,
          dimnames = list(paste0("Gene", 1:5), "u1")
        )
      ),
      visible_bias = setNames(rnorm(5), paste0("Gene", 1:5)),
      hidden_bias_per_layer = list(Bin = setNames(rnorm(1), "u1")),
      partial_correlations = diag(1, 5),
      visible_features = paste0("Gene", 1:5),
      hidden_factors = "Bin",
      family = "zinb",
      hidden_layers_info = list(Bin = list(type = "binary")),
      fit_info = list(n_features = 5)
    ),
    class = "RBM"
  )

  cand <- matrix(abs(rnorm(5 * 9)), nrow = 5, ncol = 9,
    dimnames = list(paste0("Gene", 1:5), paste0("Cell", 1:9))
  )

  sel <- SelectRBMActiveLearningCandidates(
    rbmObject = rbm,
    candidateExpression = cand,
    batchSize = 3,
    method = "expected_gradient",
    progressr = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )

  expect_type(sel, "list")
  expect_named(sel, c("indices", "scores"))
  expect_length(sel$indices, 3)
  expect_length(sel$scores, 3)
  expect_true(all(order(sel$scores, decreasing = TRUE) == seq_along(sel$scores)))
})
