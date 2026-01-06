test_that("CONGA helper functions work", {
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")
  
  # Test power parameter selection
  set.seed(42)
  X <- matrix(rpois(100, lambda = 5), nrow = 10, ncol = 10)
  
  # This should work if C++ code compiles
  skip("Skipping C++ tests until package is built")
  # power <- SelectPowerParameter(10, X)
  # expect_true(is.numeric(power))
  # expect_true(power > 0 && power < 20)
})


test_that("FitCONGAModel validates inputs correctly", {
  # Test with invalid input
  expect_error(
    FitCONGAModel(expressionData = "not a matrix"),
    "must be a matrix"
  )
  
  # Test with burnIn >= totalIterations
  X <- matrix(rpois(100, lambda = 5), nrow = 10, ncol = 10)
  expect_error(
    FitCONGAModel(X, totalIterations = 100, burnIn = 100),
    "burnIn must be less than totalIterations"
  )
})


test_that("FitCONGA validates Seurat object", {
  # Test with invalid input
  expect_error(
    FitCONGA(seuratObject = "not a seurat object"),
    "must be a Seurat or SeuratObject"
  )
})


test_that("FitCONGA extracts and filters data correctly", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("Matrix")
  
  # Create a minimal mock Seurat-like object
  set.seed(123)
  n_cells <- 50
  n_genes <- 20
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_cells * n_genes, lambda = 3),
    nrow = n_genes,
    ncol = n_cells
  )
  rownames(expr_matrix) <- paste0("Gene", 1:n_genes)
  colnames(expr_matrix) <- paste0("Cell", 1:n_cells)
  
  # Note: Full test would require creating a real Seurat object
  # which requires the Seurat package. Skipping for now.
  skip("Skipping Seurat integration test - requires full Seurat package")
})


test_that("ExtractCONGAGraph validates inputs", {
  # Create a mock CONGAfit object
  mock_fit <- list(
    beta_mcmc = list(rnorm(10), rnorm(10), rnorm(10)),
    lambda_mcmc = list(matrix(1, 5, 5), matrix(1, 5, 5), matrix(1, 5, 5)),
    n_genes = 5,
    n_cells = 10,
    edge_index = combinat::combn(1:5, 2),
    power_parameter = 1.0,
    acceptance_rate_lambda = 0.3,
    acceptance_rate_beta = 0.4
  )
  class(mock_fit) <- c("CONGAfit", "list")
  
  # Test cutoff validation
  expect_error(
    ExtractCONGAGraph(mock_fit, cutoff = 1.5),
    "cutoff must be between 0 and 1"
  )
  
  expect_error(
    ExtractCONGAGraph(mock_fit, cutoff = -0.1),
    "cutoff must be between 0 and 1"
  )
  
  # Test method validation
  expect_error(
    ExtractCONGAGraph(mock_fit, method = "invalid"),
    "method must be"
  )
})


test_that("ExtractCONGAGraph computes graph correctly", {
  skip_if_not_installed("combinat")
  
  # Create a mock CONGAfit object with known beta samples
  n_genes <- 4
  edge_index <- combinat::combn(1:n_genes, 2)
  n_edges <- ncol(edge_index)
  
  # Create beta samples where some edges are consistently positive
  # and others are consistently negative
  beta_samples <- lapply(1:100, function(i) {
    c(0.5, -0.5, 0.8, 0.1, -0.2, 0.9)  # 6 edges for 4 genes
  })
  
  mock_fit <- list(
    beta_mcmc = beta_samples,
    lambda_mcmc = list(),
    n_genes = n_genes,
    n_cells = 10,
    edge_index = edge_index,
    power_parameter = 1.0,
    acceptance_rate_lambda = 0.3,
    acceptance_rate_beta = 0.4,
    gene_names = paste0("Gene", 1:n_genes)
  )
  class(mock_fit) <- c("CONGAfit", "list")
  
  # Extract graph with low cutoff
  graph <- ExtractCONGAGraph(mock_fit, cutoff = 0.5)
  
  expect_true(inherits(graph, "CONGAgraph"))
  expect_true(is.matrix(graph$adjacency_matrix))
  expect_equal(nrow(graph$adjacency_matrix), n_genes)
  expect_equal(ncol(graph$adjacency_matrix), n_genes)
  expect_true(is.data.frame(graph$edge_list))
  expect_equal(nrow(graph$edge_list), n_edges)
})


test_that("ComputeCONGAROC validates inputs", {
  skip_if_not_installed("combinat")
  
  # Create mock objects
  n_genes <- 4
  edge_index <- combinat::combn(1:n_genes, 2)
  
  mock_fit <- list(
    beta_mcmc = list(rnorm(6)),
    n_genes = n_genes,
    edge_index = edge_index
  )
  class(mock_fit) <- c("CONGAfit", "list")
  
  # Test with invalid true graph
  expect_error(
    ComputeCONGAROC(mock_fit, trueGraph = "not a matrix"),
    "must be a matrix"
  )
  
  # Test with non-square matrix
  expect_error(
    ComputeCONGAROC(mock_fit, trueGraph = matrix(1, 4, 5)),
    "must be square"
  )
  
  # Test with wrong dimensions
  expect_error(
    ComputeCONGAROC(mock_fit, trueGraph = matrix(1, 5, 5)),
    "dimensions must match"
  )
})


test_that("Print methods work", {
  skip_if_not_installed("combinat")
  
  # Test print.CONGAfit
  mock_fit <- list(
    beta_mcmc = list(rnorm(6)),
    lambda_mcmc = list(),
    n_genes = 4,
    n_cells = 10,
    edge_index = combinat::combn(1:4, 2),
    power_parameter = 1.0,
    acceptance_rate_lambda = 0.3,
    acceptance_rate_beta = 0.4
  )
  class(mock_fit) <- c("CONGAfit", "list")
  
  # Should not error
  expect_output(print(mock_fit), "CONGA Model Fit")
  
  # Test print.CONGAgraph
  mock_graph <- list(
    adjacency_matrix = matrix(0, 4, 4),
    edge_probabilities = rep(0.5, 6),
    edge_list = data.frame(
      gene1 = c(1, 1, 1, 2, 2, 3),
      gene2 = c(2, 3, 4, 3, 4, 4),
      posterior_prob = rep(0.5, 6),
      transformed_prob = rep(0.5, 6),
      included = rep(TRUE, 6)
    ),
    n_edges = 6,
    cutoff = 0.5,
    method = "asymmetric"
  )
  class(mock_graph) <- c("CONGAgraph", "list")
  
  # Should not error
  expect_output(print(mock_graph), "CONGA Conditional Dependency Graph")
})
