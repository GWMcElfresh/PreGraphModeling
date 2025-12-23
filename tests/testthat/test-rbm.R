test_that("EstimatePartialCorrelations works with simple data", {
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_samples <- 50
  n_features <- 10
  
  # Create expression matrix with some structure
  expr_matrix <- matrix(
    rpois(n_features * n_samples, lambda = 8),
    nrow = n_features,
    ncol = n_samples,
    dimnames = list(paste0("Gene", 1:n_features), paste0("Cell", 1:n_samples))
  )
  
  # Add some zeros
  zero_mask <- matrix(rbinom(n_features * n_samples, 1, 0.3), nrow = n_features)
  expr_matrix[zero_mask == 1] <- 0
  
  # Estimate partial correlations
  result <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    family = "zinb",
    minNonZero = 5,
    progressr = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )
  
  # Verify structure
  expect_type(result, "list")
  expect_named(result, c("partial_cor", "se", "pvalues", "n_pairs", "family", 
                        "features", "excluded_features"))
  
  # Verify partial correlation matrix
  expect_true(is.matrix(result$partial_cor))
  expect_equal(nrow(result$partial_cor), n_features)
  expect_equal(ncol(result$partial_cor), n_features)
  
  # Verify family
  expect_equal(result$family, "zinb")
  
  # Diagonal should be 1 (or NA for excluded features)
  diag_vals <- diag(result$partial_cor)
  valid_diag <- diag_vals[!is.na(diag_vals)]
  if (length(valid_diag) > 0) {
    expect_true(all(abs(valid_diag - 1) < 0.01))
  }
})


test_that("EstimatePartialCorrelations validates input correctly", {
  # Test with invalid expression matrix
  expect_error(
    EstimatePartialCorrelations("not a matrix", family = "zinb"),
    "must be a matrix or Matrix object"
  )
  
  # Test with matrix without row names
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10)
  expect_error(
    EstimatePartialCorrelations(expr_matrix, family = "zinb"),
    "must have row names"
  )
  
  # Test with invalid family
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10,
                       dimnames = list(paste0("Gene", 1:10), paste0("Cell", 1:10)))
  expect_error(
    EstimatePartialCorrelations(expr_matrix, family = "invalid"),
    "family must be one of"
  )
})


test_that("EstimatePartialCorrelations handles different families", {
  set.seed(123)
  n_samples <- 40
  n_features <- 8
  
  expr_matrix <- matrix(
    rpois(n_features * n_samples, lambda = 10),
    nrow = n_features,
    ncol = n_samples,
    dimnames = list(paste0("Gene", 1:n_features), paste0("Cell", 1:n_samples))
  )
  
  # Test gaussian family
  result_gaussian <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    family = "gaussian",
    minNonZero = 5,
    progressr = FALSE,
    verbose = FALSE
  )
  expect_equal(result_gaussian$family, "gaussian")
  expect_true(is.matrix(result_gaussian$partial_cor))
  
  # Test nb family
  result_nb <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    family = "nb",
    minNonZero = 5,
    progressr = FALSE,
    verbose = FALSE
  )
  expect_equal(result_nb$family, "nb")
  
  # Test poisson family
  result_poisson <- EstimatePartialCorrelations(
    expressionMatrix = expr_matrix,
    family = "poisson",
    minNonZero = 5,
    progressr = FALSE,
    verbose = FALSE
  )
  expect_equal(result_poisson$family, "poisson")
})


test_that("FitRBM validates input correctly", {
  skip_if_not_installed("SeuratObject")
  
  # Create a simple mock Seurat-like object
  # This is a simplified test - in real use, would use actual Seurat object
  
  # Test missing hiddenFactors
  expect_error(
    FitRBM(list(), visibleFeatures = NULL),
    "hiddenFactors must be specified"
  )
})


test_that("predict.RBM validates input correctly", {
  # Create a mock RBM object
  rbm <- structure(
    list(
      weights = matrix(rnorm(20), nrow = 10, ncol = 2,
                      dimnames = list(paste0("Gene", 1:10), c("Factor1", "Factor2"))),
      visible_bias = setNames(rnorm(10), paste0("Gene", 1:10)),
      hidden_bias = setNames(rnorm(2), c("Factor1", "Factor2")),
      partial_correlations = matrix(0, nrow = 10, ncol = 10),
      visible_features = paste0("Gene", 1:10),
      hidden_factors = c("Factor1", "Factor2"),
      family = "zinb",
      metadata = data.frame(Factor1 = 1:5, Factor2 = 1:5),
      fit_info = list(n_features = 10, n_hidden = 2)
    ),
    class = "RBM"
  )
  
  # Test with missing newdata
  expect_error(
    predict(rbm, newdata = NULL),
    "newdata must be provided"
  )
  
  # Test with invalid type
  new_expr <- matrix(rnorm(50), nrow = 10, ncol = 5,
                    dimnames = list(paste0("Gene", 1:10), paste0("Cell", 1:5)))
  expect_error(
    predict(rbm, newdata = new_expr, type = "probability"),
    "type must be 'activation'"
  )
  
  expect_error(
    predict(rbm, newdata = new_expr, type = "label"),
    "type must be 'activation'"
  )
  
  # Test valid prediction
  predictions <- predict(rbm, newdata = new_expr, type = "activation")
  expect_true(is.matrix(predictions))
  expect_equal(ncol(predictions), 2)  # 2 hidden factors
  expect_equal(nrow(predictions), 5)  # 5 observations
})


test_that("ReconstructRBM works correctly", {
  # Create a mock RBM object
  rbm <- structure(
    list(
      weights = matrix(rnorm(20), nrow = 10, ncol = 2,
                      dimnames = list(paste0("Gene", 1:10), c("Factor1", "Factor2"))),
      visible_bias = setNames(rnorm(10), paste0("Gene", 1:10)),
      hidden_bias = setNames(rnorm(2), c("Factor1", "Factor2")),
      partial_correlations = matrix(0, nrow = 10, ncol = 10),
      visible_features = paste0("Gene", 1:10),
      hidden_factors = c("Factor1", "Factor2"),
      family = "zinb",
      metadata = data.frame(Factor1 = c(1, 2, 1), Factor2 = c(2, 1, 2)),
      fit_info = list(n_features = 10, n_hidden = 2)
    ),
    class = "RBM"
  )
  
  # Test reconstruction from training data
  reconstructed <- ReconstructRBM(rbm)
  expect_true(is.matrix(reconstructed))
  expect_equal(ncol(reconstructed), 10)  # 10 features
  expect_equal(nrow(reconstructed), 3)   # 3 observations from metadata
  
  # Test reconstruction from specific hidden values
  hidden_vals <- matrix(c(1, 2), nrow = 1, ncol = 2,
                       dimnames = list(NULL, c("Factor1", "Factor2")))
  reconstructed_specific <- ReconstructRBM(rbm, hidden = hidden_vals)
  expect_true(is.matrix(reconstructed_specific))
  expect_equal(nrow(reconstructed_specific), 1)
  expect_equal(ncol(reconstructed_specific), 10)
})


test_that("PlotRBMHeatmap validates input correctly", {
  skip_if_not_installed("ComplexHeatmap")
  
  # Create a mock RBM object
  rbm <- structure(
    list(
      weights = matrix(rnorm(20), nrow = 10, ncol = 2),
      visible_bias = rnorm(10),
      hidden_bias = rnorm(2),
      partial_correlations = cor(matrix(rnorm(100), nrow = 10)),
      visible_features = paste0("Gene", 1:10),
      hidden_factors = c("Factor1", "Factor2"),
      family = "zinb",
      metadata = data.frame(Factor1 = 1:5, Factor2 = 1:5),
      fit_info = list(n_features = 10, n_hidden = 2)
    ),
    class = "RBM"
  )
  rownames(rbm$partial_correlations) <- paste0("Gene", 1:10)
  colnames(rbm$partial_correlations) <- paste0("Gene", 1:10)
  
  # Test with invalid object
  expect_error(
    PlotRBMHeatmap("not an RBM"),
    "must be an RBM object"
  )
  
  # Test valid heatmap creation (just check it doesn't error)
  # Note: We skip actually drawing the heatmap in tests
  expect_no_error({
    heatmap <- PlotRBMHeatmap(
      rbm,
      cluster_rows = FALSE,
      cluster_columns = FALSE,
      color_palette = "RdBu"
    )
  })
})
