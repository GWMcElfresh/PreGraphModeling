test_that("AnalyzeWithZINB integrates pseudobulking and modeling", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(111)
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
  expect_named(result, c("pseudobulk_matrix", "group_metadata", "model_parameters"))
  
  # Verify pseudobulk results
  expect_true(is.matrix(result$pseudobulk_matrix))
  expect_equal(nrow(result$pseudobulk_matrix), n_genes)
  expect_equal(ncol(result$pseudobulk_matrix), 3)  # 3 cell types
  
  # Verify metadata
  expect_s3_class(result$group_metadata, "data.frame")
  expect_equal(nrow(result$group_metadata), 3)
  
  # Verify model parameters
  expect_s3_class(result$model_parameters, "data.frame")
  expect_equal(nrow(result$model_parameters), n_genes)
  expect_true(all(c("gene", "mu", "phi", "pi", "converged") %in% 
                   colnames(result$model_parameters)))
})

test_that("AnalyzeWithZINB works with multiple grouping columns", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(222)
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
  
  # Should create 2 cell types * 2 treatments = 4 pseudobulk samples
  expect_equal(ncol(result$pseudobulk_matrix), 4)
  expect_equal(nrow(result$group_metadata), 4)
  
  # Verify both grouping columns are in metadata
  expect_true(all(c("CellType", "Treatment") %in% colnames(result$group_metadata)))
})

test_that("AnalyzeWithZINB respects gene subset parameter", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(333)
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
  
  # Should only have models for specified genes
  expect_equal(nrow(result$model_parameters), length(gene_subset))
  expect_equal(result$model_parameters$gene, gene_subset)
  
  # But pseudobulk matrix should have all genes
  expect_equal(nrow(result$pseudobulk_matrix), n_genes)
})

test_that("AnalyzeWithZINB propagates parameters correctly", {
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("pscl")
  
  set.seed(444)
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
  
  # Genes with pseudobulk samples having < 5 non-zeros should have NA parameters
  # Since we group into 2 samples, genes 1-3 will have very few non-zeros per sample
  expect_true(any(is.na(result$model_parameters$mu)))
})
