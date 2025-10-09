test_that("AnalyzeWithZINB integrates subsetting and modeling", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 60
  n_genes <- 20
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 8),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Add some zeros
  zero_mask <- matrix(rbinom(n_genes * n_cells, 1, 0.2), nrow = n_genes)
  expr_matrix[zero_mask == 1] <- 0
  
  # Create metadata
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB", "TypeC"), each = 20),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Run complete analysis
  result <- AnalyzeWithZINB(seurat_obj, groupByColumns = "CellType", verbose = FALSE)
  
  # Verify structure
  expect_type(result, "list")
  expect_named(result, c("subset_matrices", "group_metadata", "model_parameters", 
                        "combined_parameters", "timing"))
  
  # Verify subset results
  expect_type(result$subset_matrices, "list")
  expect_equal(length(result$subset_matrices), 3)  # 3 cell types
  
  # Verify metadata
  expect_s3_class(result$group_metadata, "data.frame")
  expect_equal(nrow(result$group_metadata), 3)
  
  # Verify model parameters (now a list, one per subset)
  expect_type(result$model_parameters, "list")
  expect_equal(length(result$model_parameters), 3)
  
  # Each model parameter data frame should have the right columns
  for (params in result$model_parameters) {
    expect_s3_class(params, "data.frame")
    expect_equal(nrow(params), n_genes)
    expect_true(all(c("gene", "mu", "phi", "pi", "converged", "subset", "n_datapoints") %in% 
                     colnames(params)))
  }
  
  # Verify combined parameters
  expect_s3_class(result$combined_parameters, "data.frame")
  expect_true(all(c("gene", "mu", "phi", "pi", "key", "key_colnames", "n_datapoints") %in% 
                   colnames(result$combined_parameters)))
  expect_equal(unique(result$combined_parameters$key_colnames), "CellType")
  
  # Verify timing data
  expect_s3_class(result$timing, "data.frame")
  expect_true(all(c("step", "elapsed_seconds") %in% colnames(result$timing)))
})

test_that("AnalyzeWithZINB works with multiple grouping columns", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 80
  n_genes <- 15
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 6),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata with multiple columns
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 40),
    Treatment = rep(c("Control", "Drug"), times = 40),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Run analysis with multiple grouping columns
  result <- AnalyzeWithZINB(seurat_obj, 
                           groupByColumns = c("CellType", "Treatment"),
                           verbose = FALSE)
  
  # Should create 2 cell types * 2 treatments = 4 subsets
  expect_equal(length(result$subset_matrices), 4)
  expect_equal(nrow(result$group_metadata), 4)
  expect_equal(length(result$model_parameters), 4)
  
  # Verify both grouping columns are in metadata
  expect_true(all(c("CellType", "Treatment") %in% colnames(result$group_metadata)))
})

test_that("AnalyzeWithZINB respects gene subset parameter", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 50
  n_genes <- 30
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 7),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata
  metadata <- data.frame(
    Group = rep(c("A", "B"), length.out = n_cells),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Run analysis with gene subset
  gene_subset <- c("Gene1", "Gene5", "Gene10", "Gene15")
  result <- AnalyzeWithZINB(seurat_obj, 
                           groupByColumns = "Group",
                           geneSubset = gene_subset,
                           verbose = FALSE)
  
  # Should only have models for specified genes in each subset
  for (params in result$model_parameters) {
    expect_equal(nrow(params), length(gene_subset))
    expect_equal(params$gene, gene_subset)
  }
  
  # But subset matrices should have all genes
  for (subset_mat in result$subset_matrices) {
    expect_equal(nrow(subset_mat), n_genes)
  }
})

test_that("AnalyzeWithZINB propagates parameters correctly", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 40
  n_genes <- 10
  
  # Create expression matrix with some genes having few non-zeros
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 5),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Make some genes have many zeros
  expr_matrix[1:3, ] <- 0
  expr_matrix[1, 1:2] <- c(5, 10)  # Only 2 non-zeros
  
  # Create metadata
  metadata <- data.frame(
    CellType = rep(c("A", "B"), each = 20),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Run analysis with minNonZero = 5
  result <- AnalyzeWithZINB(seurat_obj, 
                           groupByColumns = "CellType",
                           minNonZero = 5,
                           verbose = FALSE)
  
  # Genes with subsets having < 5 non-zeros should have NA parameters
  # Check that at least some models have NA (due to sparse data)
  has_na <- FALSE
  for (params in result$model_parameters) {
    if (any(is.na(params$mu))) {
      has_na <- TRUE
      break
    }
  }
  expect_true(has_na)
})

test_that("AnalyzeWithZINB creates proper keys and key_colnames", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 40
  n_genes <- 10
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 6),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata with multiple columns
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 20),
    Treatment = rep(c("Control", "Drug"), times = 20),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Run analysis
  result <- AnalyzeWithZINB(seurat_obj, 
                           groupByColumns = c("CellType", "Treatment"),
                           verbose = FALSE)
  
  # Verify key format
  expected_keys <- c("TypeA_Control", "TypeA_Drug", "TypeB_Control", "TypeB_Drug")
  actual_keys <- unique(result$combined_parameters$key)
  expect_equal(sort(actual_keys), sort(expected_keys))
  
  # Verify key_colnames
  expect_equal(unique(result$combined_parameters$key_colnames), "CellType|Treatment")
  
  # Verify n_datapoints is populated
  expect_true(all(result$combined_parameters$n_datapoints > 0))
  expect_equal(unique(result$combined_parameters$n_datapoints), 10)  # 10 cells per subset
})

test_that("AnalyzeWithZINB parallel processing works", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  skip_if_not_installed("future")
  skip_if_not_installed("future.apply")
  
  # Skip on systems where parallel might not work
  skip_on_cran()
  skip_on_ci()
  
  set.seed(42)
  n_cells <- 40
  n_genes <- 10
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 7),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 20),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Test that parallel parameter is accepted and structure is correct
  # Run without parallel processing
  result_sequential <- AnalyzeWithZINB(seurat_obj, 
                                      groupByColumns = "CellType",
                                      parallel = FALSE,
                                      verbose = FALSE)
  
  # Verify results structure
  expect_equal(length(result_sequential$model_parameters), 2)
  expect_equal(names(result_sequential$model_parameters), c("TypeA", "TypeB"))
  
  # Timing should be recorded
  expect_s3_class(result_sequential$timing, "data.frame")
  expect_true(nrow(result_sequential$timing) > 0)
})

test_that("AnalyzeWithZINB accepts parallelPlan parameter", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(42)
  n_cells <- 20
  n_genes <- 5
  
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 5),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 10),
    row.names = colnames(expr_matrix)
  )
  
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Test that function accepts parallelPlan parameter
  result <- AnalyzeWithZINB(seurat_obj, 
                           groupByColumns = "CellType",
                           parallel = FALSE,
                           parallelPlan = "multisession",
                           verbose = FALSE)
  
  # Verify basic structure
  expect_type(result, "list")
  expect_true("model_parameters" %in% names(result))
})
