test_that("PseudobulkSeurat works with single metadata column", {
  skip_if_not_installed("SeuratObject")
  
  # Create a minimal test Seurat-like object
  set.seed(123)
  n_cells <- 100
  n_genes <- 50
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 5),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB", "TypeC"), length.out = n_cells),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Test pseudobulking
  result <- PseudobulkSeurat(seurat_obj, groupByColumns = "CellType")
  
  # Verify structure
  expect_type(result, "list")
  expect_named(result, c("pseudobulk_matrix", "group_metadata"))
  
  # Verify pseudobulk matrix
  expect_true(is.matrix(result$pseudobulk_matrix))
  expect_equal(nrow(result$pseudobulk_matrix), n_genes)
  expect_equal(ncol(result$pseudobulk_matrix), 3)  # 3 cell types
  
  # Verify metadata
  expect_s3_class(result$group_metadata, "data.frame")
  expect_equal(nrow(result$group_metadata), 3)
  expect_true("CellType" %in% colnames(result$group_metadata))
  expect_true("n_cells" %in% colnames(result$group_metadata))
  
  # Verify cell counts sum correctly
  expect_equal(sum(result$group_metadata$n_cells), n_cells)
})

test_that("PseudobulkSeurat works with multiple metadata columns", {
  skip_if_not_installed("SeuratObject")
  
  set.seed(456)
  n_cells <- 120
  n_genes <- 30
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 3),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata with multiple columns
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 60),
    Condition = rep(c("Control", "Treatment"), length.out = n_cells),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = expr_matrix,
    meta.data = metadata
  )
  
  # Test pseudobulking with multiple columns
  result <- PseudobulkSeurat(seurat_obj, groupByColumns = c("CellType", "Condition"))
  
  # Should have 2 cell types * 2 conditions = 4 groups
  expect_equal(ncol(result$pseudobulk_matrix), 4)
  expect_equal(nrow(result$group_metadata), 4)
  
  # Verify both columns are in metadata
  expect_true(all(c("CellType", "Condition") %in% colnames(result$group_metadata)))
})

test_that("PseudobulkSeurat validates input correctly", {
  # Test with invalid input object
  expect_error(
    PseudobulkSeurat("not a seurat object", groupByColumns = "CellType"),
    "must be a Seurat or SeuratObject"
  )
  
  # Test with invalid groupByColumns
  skip_if_not_installed("SeuratObject")
  
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10,
                       dimnames = list(paste0("Gene", 1:10), paste0("Cell", 1:10)))
  metadata <- data.frame(CellType = rep("TypeA", 10), row.names = colnames(expr_matrix))
  seurat_obj <- SeuratObject::CreateSeuratObject(counts = expr_matrix, meta.data = metadata)
  
  expect_error(
    PseudobulkSeurat(seurat_obj, groupByColumns = character(0)),
    "must be a non-empty character vector"
  )
  
  expect_error(
    PseudobulkSeurat(seurat_obj, groupByColumns = 123),
    "must be a non-empty character vector"
  )
  
  # Test with non-existent metadata column
  expect_error(
    PseudobulkSeurat(seurat_obj, groupByColumns = "NonExistent"),
    "Metadata columns not found"
  )
})

test_that("PseudobulkSeurat aggregates counts correctly", {
  skip_if_not_installed("SeuratObject")
  
  # Create a simple test case where we can verify the sums
  expr_matrix <- matrix(
    c(1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12),
    nrow = 3,
    ncol = 4,
    byrow = TRUE,
    dimnames = list(paste0("Gene", 1:3), paste0("Cell", 1:4))
  )
  
  metadata <- data.frame(
    Group = c("A", "A", "B", "B"),
    row.names = colnames(expr_matrix)
  )
  
  seurat_obj <- SeuratObject::CreateSeuratObject(counts = expr_matrix, meta.data = metadata)
  result <- PseudobulkSeurat(seurat_obj, groupByColumns = "Group")
  
  # Verify sums for each gene/group combination
  # Gene1, Group A: 1 + 2 = 3
  # Gene1, Group B: 3 + 4 = 7
  expect_equal(as.numeric(result$pseudobulk_matrix[1, ]), c(3, 7))
  
  # Gene2, Group A: 5 + 6 = 11
  # Gene2, Group B: 7 + 8 = 15
  expect_equal(as.numeric(result$pseudobulk_matrix[2, ]), c(11, 15))
})
