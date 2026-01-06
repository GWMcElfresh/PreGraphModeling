#!/usr/bin/env Rscript
# Simple validation script for CONGA implementation
# This script tests the core functionality with synthetic data

library(testthat)

cat("=== CONGA Implementation Validation ===\n\n")

# Test 1: Power parameter selection (C++ function)
cat("Test 1: Power parameter selection...\n")
tryCatch({
  set.seed(42)
  X <- matrix(rpois(100, lambda = 5), nrow = 10, ncol = 10)
  
  # This will only work if C++ code is compiled
  # For now, we'll skip this test
  cat("  SKIPPED: Requires compiled C++ code\n")
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

# Test 2: FitCONGAModel basic functionality
cat("\nTest 2: FitCONGAModel basic functionality...\n")
tryCatch({
  set.seed(123)
  # Create small synthetic dataset
  n_cells <- 20
  n_genes <- 5
  
  X <- matrix(rpois(n_cells * n_genes, lambda = 3), 
              nrow = n_cells, 
              ncol = n_genes)
  colnames(X) <- paste0("Gene", 1:n_genes)
  rownames(X) <- paste0("Cell", 1:n_cells)
  
  cat("  Data dimensions:", nrow(X), "cells x", ncol(X), "genes\n")
  
  # Source the functions directly
  source("R/RcppExports.R")
  source("R/conga_fit.R")
  source("R/conga_wrapper.R")
  
  # Run a very short MCMC chain for testing
  cat("  Running MCMC (short chain for testing)...\n")
  result <- FitCONGAModel(
    expressionData = X,
    totalIterations = 50,
    burnIn = 25,
    lambdaShrinkage = 1,
    verbose = FALSE
  )
  
  # Check result structure
  expect_true(inherits(result, "CONGAfit"))
  expect_true("beta_mcmc" %in% names(result))
  expect_true("lambda_mcmc" %in% names(result))
  expect_equal(length(result$beta_mcmc), 25)  # After burn-in
  
  cat("  ✓ FitCONGAModel completed successfully\n")
  cat("  ✓ Returned CONGAfit object with correct structure\n")
  cat("  ✓ Lambda acceptance rate:", sprintf("%.1f%%", result$acceptance_rate_lambda * 100), "\n")
  cat("  ✓ Beta acceptance rate:", sprintf("%.1f%%", result$acceptance_rate_beta * 100), "\n")
  
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
  traceback()
})

# Test 3: ExtractCONGAGraph functionality
cat("\nTest 3: ExtractCONGAGraph functionality...\n")
tryCatch({
  # Create a mock CONGAfit object
  n_genes <- 4
  edge_index <- combinat::combn(1:n_genes, 2)
  n_edges <- ncol(edge_index)
  
  # Create beta samples where some edges are consistently positive
  beta_samples <- lapply(1:50, function(i) {
    c(0.8, -0.7, 0.9, 0.2, -0.3, 0.85)  # 6 edges for 4 genes
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
  
  # Extract graph
  graph <- ExtractCONGAGraph(mock_fit, cutoff = 0.5)
  
  expect_true(inherits(graph, "CONGAgraph"))
  expect_equal(nrow(graph$adjacency_matrix), n_genes)
  expect_equal(nrow(graph$edge_list), n_edges)
  
  cat("  ✓ ExtractCONGAGraph completed successfully\n")
  cat("  ✓ Found", graph$n_edges, "edges with cutoff 0.5\n")
  cat("  ✓ Graph density:", sprintf("%.2f", graph$n_edges / n_edges), "\n")
  
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
  traceback()
})

# Test 4: Print methods
cat("\nTest 4: Print methods...\n")
tryCatch({
  # Test print.CONGAfit
  n_genes <- 4
  mock_fit <- list(
    beta_mcmc = list(rnorm(6)),
    lambda_mcmc = list(),
    n_genes = n_genes,
    n_cells = 10,
    edge_index = combinat::combn(1:n_genes, 2),
    power_parameter = 1.0,
    acceptance_rate_lambda = 0.3,
    acceptance_rate_beta = 0.4
  )
  class(mock_fit) <- c("CONGAfit", "list")
  
  cat("  Testing print.CONGAfit:\n")
  print(mock_fit)
  
  cat("\n  ✓ Print methods work correctly\n")
  
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
})

cat("\n=== Validation Complete ===\n")
cat("\nNOTE: Full integration tests with Seurat objects require the Seurat package.\n")
cat("NOTE: C++ functions require compilation before use.\n")
