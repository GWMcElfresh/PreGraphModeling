#' Subset Seurat Object by Metadata Columns
#'
#' This function takes a Seurat object and subsets the expression data
#' according to specified metadata columns. Returns a list of expression matrices,
#' one for each unique combination of metadata values.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data
#' @param groupByColumns Character vector of metadata column names to group by.
#'   Multiple columns can be specified to create fine-grained subsets.
#' @param assay Character string specifying which assay to use (default: "RNA")
#' @param layer Character string specifying which layer to use (default: "counts")
#' @param saturationColumn Character string specifying the metadata column containing
#'   cellular saturation values. If NULL, no saturation data is extracted (default: NULL)
#'
#' @return A list with three elements:
#'   \itemize{
#'     \item subset_matrices: A list of expression matrices (sparse `dgCMatrix`), one per unique metadata combination
#'     \item group_metadata: A data frame containing the metadata for each subset
#'     \item saturation_vectors: A list of numeric vectors containing saturation values for each subset (only if saturationColumn is provided)
#'   }
#'
#' @export
#' @importFrom SeuratObject GetAssayData
#' @importFrom methods is
#' @examples
#' \dontrun{
#' # Subset by cell type
#' result <- SubsetSeurat(seurat_obj, groupByColumns = "CellType")
#'
#' # Subset by multiple columns
#' result <- SubsetSeurat(seurat_obj,
#'                        groupByColumns = c("CellType", "Sample", "Condition"))
#'
#' # Subset and extract saturation values for residualization
#' result <- SubsetSeurat(seurat_obj,
#'                        groupByColumns = "CellType",
#'                        saturationColumn = "Saturation.RNA")
#'
#' # Access saturation vectors
#' head(result$saturation_vectors[[1]])
#' }
SubsetSeurat <- function(seuratObject,
                         groupByColumns,
                         assay = "RNA",
                         layer = "counts",
                         saturationColumn = NULL) {

  # Input validation
  if (!methods::is(seuratObject, "Seurat") && !methods::is(seuratObject, "SeuratObject")) {
    stop("seuratObject must be a Seurat or SeuratObject")
  }

  if (!is.character(groupByColumns) || length(groupByColumns) == 0) {
    stop("groupByColumns must be a non-empty character vector")
  }

  # Check if metadata columns exist
  metadata <- seuratObject[[]]
  missing_cols <- setdiff(groupByColumns, colnames(metadata))
  if (length(missing_cols) > 0) {
    stop(paste("Metadata columns not found:", paste(missing_cols, collapse = ", ")))
  }

  # Check saturation column if provided
  if (!is.null(saturationColumn)) {
    if (!is.character(saturationColumn) || length(saturationColumn) != 1) {
      stop("saturationColumn must be a single character string")
    }
    if (!saturationColumn %in% colnames(metadata)) {
      stop(paste("Saturation column not found in metadata:", saturationColumn))
    }
  }

  # Get expression data
  if (methods::is(seuratObject, "Seurat")) {
    expr_data <- SeuratObject::GetAssayData(seuratObject, assay = assay, layer = layer)
  } else {
    # For SeuratObject
    expr_data <- SeuratObject::GetAssayData(seuratObject, layer = layer)
  }

  # Ensure sparse representation to avoid downstream coercion warnings and
  # accidental densification for scRNA-seq-sized objects.
  if (!inherits(expr_data, "Matrix")) {
    expr_data <- tryCatch(
      Matrix::Matrix(expr_data, sparse = TRUE),
      error = function(e) Matrix::Matrix(as.matrix(expr_data), sparse = TRUE)
    )
  }

  # Create grouping factor by combining all specified columns
  if (length(groupByColumns) == 1) {
    group_factor <- gsub("_", ".", as.character(metadata[[groupByColumns[1]]]), fixed = TRUE)
  } else {
    group_list <- lapply(groupByColumns, function(col) {
      gsub("_", ".", as.character(metadata[[col]]), fixed = TRUE)
    })
    group_factor <- do.call(paste, c(group_list, sep = "_"))
  }

  # Get unique groups
  unique_groups <- unique(group_factor)

  # Create list of subset matrices and saturation vectors
  subset_matrices <- list()
  saturation_vectors <- list()

  for (i in seq_along(unique_groups)) {
    group_name <- unique_groups[i]
    group_cells <- which(group_factor == group_name)

    # Extract subset of cells for this group
    subset_matrices[[group_name]] <- expr_data[, group_cells, drop = FALSE]

    # Extract saturation values if requested
    if (!is.null(saturationColumn)) {
      saturation_vectors[[group_name]] <- as.numeric(metadata[[saturationColumn]][group_cells])
    }
  }

  # Create metadata for subsets
  group_metadata <- data.frame(
    subset_id = unique_groups,
    stringsAsFactors = FALSE
  )

  # Add original metadata columns
  for (col in groupByColumns) {
    group_metadata[[col]] <- sapply(unique_groups, function(g) {
      idx <- which(group_factor == g)[1]
      as.character(metadata[[col]][idx])
    })
  }

  # Add cell counts
  group_metadata$n_cells <- sapply(unique_groups, function(g) {
    sum(group_factor == g)
  })

  # Build return list
  result <- list(
    subset_matrices = subset_matrices,
    group_metadata = group_metadata
  )

  # Add saturation vectors if they were extracted
  if (!is.null(saturationColumn)) {
    result$saturation_vectors <- saturation_vectors
  }

  return(result)
}


