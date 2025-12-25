test_that("SubsetSeurat works with single metadata column", {
  skip_if_not_installed("SeuratObject")
  
  # Create a minimal test Seurat-like object
  set.seed(42)
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
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  
  # Test subsetting
  result <- SubsetSeurat(seurat_obj, groupByColumns = "CellType")
  
  # Verify structure
  expect_type(result, "list")
  expect_named(result, c("subset_matrices", "group_metadata"))
  
  # Verify subset matrices
  expect_type(result$subset_matrices, "list")
  expect_equal(length(result$subset_matrices), 3)  # 3 cell types
  
  # Each subset should be a matrix with all genes and subset of cells
  for (subset_mat in result$subset_matrices) {
    expect_true(inherits(subset_mat, "Matrix"))
    expect_equal(nrow(subset_mat), n_genes)
  }
  
  # Verify metadata
  expect_s3_class(result$group_metadata, "data.frame")
  expect_equal(nrow(result$group_metadata), 3)
  expect_true("CellType" %in% colnames(result$group_metadata))
  expect_true("n_cells" %in% colnames(result$group_metadata))
  
  # Verify cell counts sum correctly
  expect_equal(sum(result$group_metadata$n_cells), n_cells)
})

test_that("SubsetSeurat works with multiple metadata columns", {
  skip_if_not_installed("SeuratObject")
  
  set.seed(42)
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
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  
  # Test subsetting with multiple columns
  result <- SubsetSeurat(seurat_obj, groupByColumns = c("CellType", "Condition"))
  
  # Should have 2 cell types * 2 conditions = 4 groups
  expect_equal(length(result$subset_matrices), 4)
  expect_equal(nrow(result$group_metadata), 4)
  
  # Verify both columns are in metadata
  expect_true(all(c("CellType", "Condition") %in% colnames(result$group_metadata)))
})

test_that("SubsetSeurat validates input correctly", {
  # Test with invalid input object
  expect_error(
    SubsetSeurat("not a seurat object", groupByColumns = "CellType"),
    "must be a Seurat or SeuratObject"
  )
  
  # Test with invalid groupByColumns
  skip_if_not_installed("SeuratObject")
  
  expr_matrix <- matrix(1:100, nrow = 10, ncol = 10,
                       dimnames = list(paste0("Gene", 1:10), paste0("Cell", 1:10)))
  metadata <- data.frame(CellType = rep("TypeA", 10), row.names = colnames(expr_matrix))
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  
  expect_error(
    SubsetSeurat(seurat_obj, groupByColumns = character(0)),
    "must be a non-empty character vector"
  )
  
  expect_error(
    SubsetSeurat(seurat_obj, groupByColumns = 123),
    "must be a non-empty character vector"
  )
  
  # Test with non-existent metadata column
  expect_error(
    SubsetSeurat(seurat_obj, groupByColumns = "NonExistent"),
    "Metadata columns not found"
  )
})

test_that("SubsetSeurat subsets cells correctly", {
  skip_if_not_installed("SeuratObject")
  
  # Create a simple test case where we can verify the subsetting
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
  
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  result <- SubsetSeurat(seurat_obj, groupByColumns = "Group")
  
  # Verify subsets contain the right cells
  # Group A should have cells 1 and 2
  expect_equal(ncol(result$subset_matrices$A), 2)
  expect_equal(colnames(result$subset_matrices$A), c("Cell1", "Cell2"))
  
  # Group B should have cells 3 and 4
  expect_equal(ncol(result$subset_matrices$B), 2)
  expect_equal(colnames(result$subset_matrices$B), c("Cell3", "Cell4"))
  
  # Verify values are preserved (not summed)
  expect_equal(as.numeric(result$subset_matrices$A[1, ]), c(1, 2))
  expect_equal(as.numeric(result$subset_matrices$B[1, ]), c(3, 4))
})

test_that("SubsetSeurat extracts saturation values correctly", {
  skip_if_not_installed("SeuratObject")
  
  set.seed(42)
  n_cells <- 60
  n_genes <- 20
  
  # Create expression matrix
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 5),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  # Create metadata with saturation values
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB", "TypeC"), each = 20),
    Saturation.RNA = runif(n_cells, min = 0.5, max = 0.95),
    row.names = colnames(expr_matrix)
  )
  
  # Create SeuratObject
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  
  # Test subsetting with saturation extraction
  result <- SubsetSeurat(seurat_obj, 
                        groupByColumns = "CellType",
                        saturationColumn = "Saturation.RNA")
  
  # Verify structure includes saturation_vectors
  expect_type(result, "list")
  expect_named(result, c("subset_matrices", "group_metadata", "saturation_vectors"))
  
  # Verify saturation vectors are present for each subset
  expect_type(result$saturation_vectors, "list")
  expect_equal(length(result$saturation_vectors), 3)
  expect_equal(names(result$saturation_vectors), names(result$subset_matrices))
  
  # Verify saturation vector lengths match cell counts
  for (subset_name in names(result$subset_matrices)) {
    expect_equal(length(result$saturation_vectors[[subset_name]]), 
                ncol(result$subset_matrices[[subset_name]]))
    
    # Verify saturation values are numeric and in valid range
    expect_true(is.numeric(result$saturation_vectors[[subset_name]]))
    expect_true(all(result$saturation_vectors[[subset_name]] >= 0))
    expect_true(all(result$saturation_vectors[[subset_name]] <= 1))
  }
  
  # Test without saturation extraction
  result_no_sat <- SubsetSeurat(seurat_obj, groupByColumns = "CellType")
  expect_named(result_no_sat, c("subset_matrices", "group_metadata"))
  expect_false("saturation_vectors" %in% names(result_no_sat))
})

test_that("SubsetSeurat validates saturation column", {
  skip_if_not_installed("SeuratObject")
  
  n_cells <- 30
  n_genes <- 10
  
  expr_matrix <- matrix(
    rpois(n_genes * n_cells, lambda = 5),
    nrow = n_genes,
    ncol = n_cells,
    dimnames = list(paste0("Gene", 1:n_genes), paste0("Cell", 1:n_cells))
  )
  
  metadata <- data.frame(
    CellType = rep(c("TypeA", "TypeB"), each = 15),
    Saturation.RNA = runif(n_cells),
    row.names = colnames(expr_matrix)
  )
  
  seurat_obj <- SeuratObject::CreateSeuratObject(
    counts = Matrix::Matrix(expr_matrix, sparse = TRUE),
    meta.data = metadata
  )
  
  # Test with non-existent saturation column
  expect_error(
    SubsetSeurat(seurat_obj, groupByColumns = "CellType", 
                saturationColumn = "NonExistent"),
    "Saturation column not found"
  )
  
  # Test with invalid saturation column type
  expect_error(
    SubsetSeurat(seurat_obj, groupByColumns = "CellType",
                saturationColumn = c("Saturation.RNA", "Other")),
    "must be a single character string"
  )
})
