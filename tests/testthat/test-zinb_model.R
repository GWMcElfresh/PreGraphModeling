test_that("FitZeroInflatedModels works with simple data", {
  skip_if_not_installed("pscl")
  
  set.seed(789)
  n_samples <- 50
  n_genes <- 10
  
  # Create expression matrix with some zeros
  expr_matrix <- matrix(
    rpois(n_genes * n_samples, lambda = 5),
    nrow = n_genes,
    ncol = n_samples,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Sample", 1:n_samples))
  )
  
  # Add some zeros to simulate zero-inflation
  zero_mask <- matrix(rbinom(n_genes * n_samples, 1, 0.3), nrow = n_genes)
  expr_matrix[zero_mask == 1] <- 0
  
  # Fit models
  result <- FitZeroInflatedModels(expr_matrix, verbose = FALSE)
  
  # Verify structure
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), n_genes)
  expect_named(result, c("gene", "mu", "phi", "pi", "converged", "n_nonzero", "n_datapoints"))
  
  # Verify all genes are present
  expect_equal(result$gene, rownames(expr_matrix))
  
  # Verify parameter types
  expect_type(result$mu, "double")
  expect_type(result$phi, "double")
  expect_type(result$pi, "double")
  expect_type(result$converged, "logical")
  
  # Verify n_datapoints
  expect_equal(unique(result$n_datapoints), n_samples)
})

test_that("FitZeroInflatedModels handles gene subset correctly", {
  skip_if_not_installed("pscl")
  
  set.seed(321)
  n_samples <- 30
  n_genes <- 20
  
  expr_matrix <- matrix(
    rpois(n_genes * n_samples, lambda = 4),
    nrow = n_genes,
    ncol = n_samples,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Sample", 1:n_samples))
  )
  
  # Fit models for subset of genes
  gene_subset <- c("Gene1", "Gene5", "Gene10")
  result <- FitZeroInflatedModels(expr_matrix, geneSubset = gene_subset, verbose = FALSE)
  
  expect_equal(nrow(result), length(gene_subset))
  expect_equal(result$gene, gene_subset)
})

test_that("FitZeroInflatedModels validates input correctly", {
  # Test with invalid input
  expect_error(
    FitZeroInflatedModels("not a matrix"),
    "must be a matrix or Matrix object"
  )
  
  # Test with matrix without row names
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10)
  expect_error(
    FitZeroInflatedModels(expr_matrix),
    "must have row names"
  )
  
  # Test with invalid minNonZero
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10,
                       dimnames = list(paste0("Gene", 1:10), paste0("Sample", 1:10)))
  expect_error(
    FitZeroInflatedModels(expr_matrix, minNonZero = -1),
    "must be a positive integer"
  )
  
  # Test with invalid geneSubset
  expect_error(
    FitZeroInflatedModels(expr_matrix, geneSubset = 123),
    "must be a character vector"
  )
})

test_that("FitZeroInflatedModels handles genes with few non-zeros", {
  skip_if_not_installed("pscl")
  
  set.seed(654)
  n_samples <- 20
  
  # Create matrix where some genes have very few non-zero values
  expr_matrix <- matrix(0, nrow = 5, ncol = n_samples,
                       dimnames = list(paste0("Gene", 1:5), paste0("Sample", 1:n_samples)))
  
  # Gene1: many non-zeros (ensure at least minNonZero)
  expr_matrix[1, ] <- rpois(n_samples, lambda = 5)
  expr_matrix[1, 1:5] <- c(5, 10, 15, 20, 25)  # Force at least 5 non-zeros
  
  # Gene2: only 2 non-zeros (below default minNonZero = 3)
  expr_matrix[2, 1:2] <- c(10, 15)
  
  # Gene3: exactly 3 non-zeros (at threshold)
  expr_matrix[3, 1:3] <- c(5, 10, 15)
  
  # Gene4 and Gene5: all zeros
  
  result <- FitZeroInflatedModels(expr_matrix, minNonZero = 3, verbose = FALSE)
  
  # Gene1 should have valid parameters (or may fail to converge, which is ok)
  # Just check that it attempted to fit (n_nonzero >= 3)
  expect_true(result$n_nonzero[1] >= 3)
  
  # Gene3 should attempt to fit (at threshold), may or may not converge
  # but should have at least minNonZero non-zeros
  expect_true(result$n_nonzero[3] >= 3)
  
  # Gene2 (only 2 non-zeros) should have NA parameters
  expect_true(is.na(result$mu[2]))
  
  # Gene4 and Gene5 (all zeros) should have NA parameters
  expect_true(all(is.na(result$mu[4:5])))
  
  # Check n_nonzero counts
  expect_true(result$n_nonzero[1] >= 5)  # At least 5 due to forced non-zeros
  expect_equal(result$n_nonzero[2], 2)
  expect_equal(result$n_nonzero[3], 3)
  expect_equal(result$n_nonzero[4], 0)
})

test_that("FitZeroInflatedModels parameters are in valid ranges", {
  skip_if_not_installed("pscl")
  
  set.seed(999)
  n_samples <- 40
  n_genes <- 5
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_samples, lambda = 10),
    nrow = n_genes,
    ncol = n_samples,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Sample", 1:n_samples))
  )
  
  # Add some zeros
  zero_mask <- matrix(rbinom(n_genes * n_samples, 1, 0.2), nrow = n_genes)
  expr_matrix[zero_mask == 1] <- 0
  
  result <- FitZeroInflatedModels(expr_matrix, verbose = FALSE)
  
  # For converged models, check parameter ranges
  converged_rows <- which(result$converged)
  
  if (length(converged_rows) > 0) {
    # Mu should be positive
    expect_true(all(result$mu[converged_rows] > 0))
    
    # Phi (dispersion) should be positive
    expect_true(all(result$phi[converged_rows] > 0))
    
    # Pi (probability) should be between 0 and 1
    expect_true(all(result$pi[converged_rows] >= 0))
    expect_true(all(result$pi[converged_rows] <= 1))
  }
})
