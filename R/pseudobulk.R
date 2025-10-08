#' Pseudobulk Seurat Object by Metadata Columns
#'
#' This function takes a Seurat object and pseudobulks the expression data
#' according to specified metadata columns. Cells sharing the same values
#' across all specified metadata columns are aggregated together.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data
#' @param groupByColumns Character vector of metadata column names to group by.
#'   Multiple columns can be specified to create fine-grained pseudobulk groups.
#' @param assay Character string specifying which assay to use (default: "RNA")
#' @param slot Character string specifying which slot to use (default: "counts")
#'
#' @return A list with two elements:
#'   \itemize{
#'     \item pseudobulk_matrix: A matrix where rows are genes and columns are pseudobulk samples
#'     \item group_metadata: A data frame containing the metadata for each pseudobulk sample
#'   }
#'
#' @export
#' @importFrom SeuratObject GetAssayData
#' @importFrom methods is
#' @examples
#' \dontrun{
#' # Pseudobulk by cell type
#' result <- PseudobulkSeurat(seurat_obj, groupByColumns = "CellType")
#' 
#' # Pseudobulk by multiple columns
#' result <- PseudobulkSeurat(seurat_obj, 
#'                            groupByColumns = c("CellType", "Sample", "Condition"))
#' }
PseudobulkSeurat <- function(seuratObject, 
                             groupByColumns, 
                             assay = "RNA",
                             slot = "counts") {
  
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
  
  # Get expression data
  if (methods::is(seuratObject, "Seurat")) {
    expr_data <- SeuratObject::GetAssayData(seuratObject, assay = assay, slot = slot)
  } else {
    # For SeuratObject
    expr_data <- SeuratObject::GetAssayData(seuratObject, layer = slot)
  }
  
  # Create grouping factor by combining all specified columns
  if (length(groupByColumns) == 1) {
    group_factor <- as.character(metadata[[groupByColumns[1]]])
  } else {
    group_list <- lapply(groupByColumns, function(col) {
      as.character(metadata[[col]])
    })
    group_factor <- do.call(paste, c(group_list, sep = "_"))
  }
  
  # Get unique groups
  unique_groups <- unique(group_factor)
  n_groups <- length(unique_groups)
  
  # Initialize pseudobulk matrix
  pseudobulk_matrix <- matrix(0, 
                               nrow = nrow(expr_data), 
                               ncol = n_groups,
                               dimnames = list(rownames(expr_data), unique_groups))
  
  # Aggregate expression by summing across cells in each group
  for (i in seq_along(unique_groups)) {
    group_name <- unique_groups[i]
    group_cells <- which(group_factor == group_name)
    
    if (length(group_cells) == 1) {
      pseudobulk_matrix[, i] <- expr_data[, group_cells]
    } else {
      pseudobulk_matrix[, i] <- Matrix::rowSums(expr_data[, group_cells])
    }
  }
  
  # Create metadata for pseudobulk samples
  group_metadata <- data.frame(
    pseudobulk_id = unique_groups,
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
  
  return(list(
    pseudobulk_matrix = pseudobulk_matrix,
    group_metadata = group_metadata
  ))
}
