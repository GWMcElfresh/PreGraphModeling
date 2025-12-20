#' Plot RBM Partial Correlations as Heatmap
#'
#' Creates a heatmap visualization of the partial correlation matrix from an RBM
#' using ComplexHeatmap. The heatmap shows the relationships between features
#' (genes) after conditioning on all other features.
#'
#' @param rbmObject An RBM object fitted using FitRBM().
#' @param features Optional character vector of specific features to plot.
#'   If NULL, plots all features (default: NULL).
#' @param cluster_rows Logical indicating whether to cluster rows (default: TRUE).
#' @param cluster_columns Logical indicating whether to cluster columns (default: TRUE).
#' @param show_row_names Logical indicating whether to show row names (default: TRUE).
#' @param show_column_names Logical indicating whether to show column names (default: TRUE).
#' @param color_palette Character string specifying color palette.
#'   Options: "RdBu", "viridis", "plasma", "RdYlBu" (default: "RdBu").
#' @param title Character string for heatmap title (default: "RBM Partial Correlations").
#' @param name Character string for color legend name (default: "Partial\nCorrelation").
#' @param ... Additional arguments passed to ComplexHeatmap::Heatmap().
#'
#' @return A ComplexHeatmap object that can be drawn or further customized.
#'
#' @details
#' The heatmap displays partial correlations between features, which represent
#' direct relationships after accounting for all other features. Strong positive
#' correlations (red) indicate features that co-vary together, while strong negative
#' correlations (blue) indicate features with opposing patterns.
#'
#' Clustering helps identify groups of features with similar correlation patterns.
#'
#' @export
#' @importFrom grDevices colorRampPalette
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = "CellType", family = "zinb")
#'
#' # Plot all partial correlations
#' heatmap_plot <- PlotRBMHeatmap(rbm)
#' print(heatmap_plot)
#'
#' # Plot specific features
#' heatmap_plot <- PlotRBMHeatmap(
#'   rbm,
#'   features = c("CD3D", "CD8A", "CD4", "CD19"),
#'   title = "T and B Cell Marker Correlations"
#' )
#'
#' # Use different color palette
#' heatmap_plot <- PlotRBMHeatmap(rbm, color_palette = "viridis")
#' }
PlotRBMHeatmap <- function(rbmObject,
                          features = NULL,
                          cluster_rows = TRUE,
                          cluster_columns = TRUE,
                          show_row_names = TRUE,
                          show_column_names = TRUE,
                          color_palette = "RdBu",
                          title = "RBM Partial Correlations",
                          name = "Partial\nCorrelation",
                          ...) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  
  if (!inherits(rbmObject, "RBM")) {
    stop("rbmObject must be an RBM object created by FitRBM()")
  }
  
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    stop("Package 'ComplexHeatmap' is required for this function. ",
         "Install it with: BiocManager::install('ComplexHeatmap')")
  }
  
  # Extract partial correlation matrix
  pcor_matrix <- rbmObject$partial_correlations
  
  # Filter to specific features if requested
  if (!is.null(features)) {
    if (!is.character(features)) {
      stop("features must be a character vector of feature names")
    }
    
    # Check which features are available
    available_features <- intersect(features, rownames(pcor_matrix))
    missing_features <- setdiff(features, rownames(pcor_matrix))
    
    if (length(missing_features) > 0) {
      warning(sprintf("Features not found in RBM: %s",
                     paste(missing_features, collapse = ", ")))
    }
    
    if (length(available_features) == 0) {
      stop("None of the specified features are available in the RBM")
    }
    
    # Subset matrix
    pcor_matrix <- pcor_matrix[available_features, available_features, drop = FALSE]
  } else {
    # Use only valid features (those with non-NA values)
    valid_rows <- rowSums(!is.na(pcor_matrix)) > 0
    pcor_matrix <- pcor_matrix[valid_rows, valid_rows, drop = FALSE]
  }
  
  # Check for empty matrix
  if (nrow(pcor_matrix) == 0 || ncol(pcor_matrix) == 0) {
    stop("No valid partial correlations to plot")
  }
  
  # ============================================================================
  # PREPARE COLOR PALETTE
  # ============================================================================
  
  # Determine color range
  # Correlations range from -1 to 1, but actual values may be more constrained
  cor_range <- range(pcor_matrix, na.rm = TRUE)
  max_abs_cor <- max(abs(cor_range))
  
  # Create color function based on palette
  if (color_palette == "RdBu") {
    # Red-Blue diverging (standard for correlations)
    col_fun <- circlize::colorRamp2(
      c(-max_abs_cor, 0, max_abs_cor),
      c("blue", "white", "red")
    )
  } else if (color_palette == "RdYlBu") {
    # Red-Yellow-Blue diverging
    col_fun <- circlize::colorRamp2(
      c(-max_abs_cor, 0, max_abs_cor),
      c("blue", "yellow", "red")
    )
  } else if (color_palette == "viridis") {
    # Viridis sequential
    if (requireNamespace("viridisLite", quietly = TRUE)) {
      colors <- viridisLite::viridis(100)
      col_fun <- circlize::colorRamp2(
        seq(min(cor_range, na.rm = TRUE), max(cor_range, na.rm = TRUE), length.out = 100),
        colors
      )
    } else {
      # Fallback to simple gradient
      col_fun <- circlize::colorRamp2(
        c(min(cor_range, na.rm = TRUE), max(cor_range, na.rm = TRUE)),
        c("white", "darkblue")
      )
    }
  } else if (color_palette == "plasma") {
    # Plasma sequential
    if (requireNamespace("viridisLite", quietly = TRUE)) {
      colors <- viridisLite::plasma(100)
      col_fun <- circlize::colorRamp2(
        seq(min(cor_range, na.rm = TRUE), max(cor_range, na.rm = TRUE), length.out = 100),
        colors
      )
    } else {
      # Fallback
      col_fun <- circlize::colorRamp2(
        c(min(cor_range, na.rm = TRUE), max(cor_range, na.rm = TRUE)),
        c("white", "purple")
      )
    }
  } else {
    warning(sprintf("Unknown color_palette '%s', using 'RdBu'", color_palette))
    col_fun <- circlize::colorRamp2(
      c(-max_abs_cor, 0, max_abs_cor),
      c("blue", "white", "red")
    )
  }
  
  # ============================================================================
  # CREATE HEATMAP
  # ============================================================================
  
  # Handle row/column name display based on matrix size
  if (nrow(pcor_matrix) > 50 && show_row_names) {
    show_row_names <- FALSE
    message("Note: Hiding row names due to large matrix size (>50 features)")
  }
  if (ncol(pcor_matrix) > 50 && show_column_names) {
    show_column_names <- FALSE
    message("Note: Hiding column names due to large matrix size (>50 features)")
  }
  
  # Create heatmap using ComplexHeatmap
  heatmap <- ComplexHeatmap::Heatmap(
    matrix = pcor_matrix,
    name = name,
    col = col_fun,
    cluster_rows = cluster_rows,
    cluster_columns = cluster_columns,
    show_row_names = show_row_names,
    show_column_names = show_column_names,
    column_title = title,
    row_title = "Features",
    column_title_gp = grid::gpar(fontsize = 14, fontface = "bold"),
    heatmap_legend_param = list(
      title = name,
      title_gp = grid::gpar(fontsize = 10, fontface = "bold"),
      labels_gp = grid::gpar(fontsize = 8),
      legend_direction = "vertical",
      legend_width = grid::unit(4, "cm")
    ),
    ...
  )
  
  return(heatmap)
}


#' Plot RBM Weights as Heatmap
#'
#' Creates a heatmap visualization of the weight matrix from an RBM, showing
#' the connections between visible features and hidden factors.
#'
#' @param rbmObject An RBM object fitted using FitRBM().
#' @param features Optional character vector of specific features to plot.
#'   If NULL, plots all features (default: NULL).
#' @param factors Optional character vector of specific hidden factors to plot.
#'   If NULL, plots all factors (default: NULL).
#' @param cluster_rows Logical indicating whether to cluster rows (default: TRUE).
#' @param cluster_columns Logical indicating whether to cluster columns (default: FALSE).
#' @param show_row_names Logical indicating whether to show row names (default: TRUE).
#' @param show_column_names Logical indicating whether to show column names (default: TRUE).
#' @param color_palette Character string specifying color palette (default: "RdBu").
#' @param title Character string for heatmap title (default: "RBM Weights").
#' @param name Character string for color legend name (default: "Weight").
#' @param ... Additional arguments passed to ComplexHeatmap::Heatmap().
#'
#' @return A ComplexHeatmap object that can be drawn or further customized.
#'
#' @details
#' The heatmap displays the learned weights connecting visible features (genes)
#' to hidden factors (metadata). Strong positive weights indicate features that
#' activate when the factor is present, while negative weights indicate suppression.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit RBM
#' rbm <- FitRBM(pbmc, hiddenFactors = c("CellType", "Treatment"), family = "zinb")
#'
#' # Plot weights
#' weight_plot <- PlotRBMWeights(rbm)
#' print(weight_plot)
#'
#' # Plot specific features and factors
#' weight_plot <- PlotRBMWeights(
#'   rbm,
#'   features = c("CD3D", "CD8A", "CD4"),
#'   factors = "CellType"
#' )
#' }
PlotRBMWeights <- function(rbmObject,
                          features = NULL,
                          factors = NULL,
                          cluster_rows = TRUE,
                          cluster_columns = FALSE,
                          show_row_names = TRUE,
                          show_column_names = TRUE,
                          color_palette = "RdBu",
                          title = "RBM Weights: Features to Hidden Factors",
                          name = "Weight",
                          ...) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  
  if (!inherits(rbmObject, "RBM")) {
    stop("rbmObject must be an RBM object created by FitRBM()")
  }
  
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    stop("Package 'ComplexHeatmap' is required. Install with: BiocManager::install('ComplexHeatmap')")
  }
  
  # Extract weight matrix
  weight_matrix <- rbmObject$weights
  
  # Filter features if specified
  if (!is.null(features)) {
    available <- intersect(features, rownames(weight_matrix))
    if (length(available) == 0) {
      stop("None of the specified features are available in the RBM")
    }
    weight_matrix <- weight_matrix[available, , drop = FALSE]
  }
  
  # Filter factors if specified
  if (!is.null(factors)) {
    available <- intersect(factors, colnames(weight_matrix))
    if (length(available) == 0) {
      stop("None of the specified factors are available in the RBM")
    }
    weight_matrix <- weight_matrix[, available, drop = FALSE]
  }
  
  # ============================================================================
  # PREPARE COLOR PALETTE
  # ============================================================================
  
  weight_range <- range(weight_matrix, na.rm = TRUE)
  max_abs_weight <- max(abs(weight_range))
  
  if (color_palette == "RdBu") {
    col_fun <- circlize::colorRamp2(
      c(-max_abs_weight, 0, max_abs_weight),
      c("blue", "white", "red")
    )
  } else if (color_palette == "RdYlBu") {
    col_fun <- circlize::colorRamp2(
      c(-max_abs_weight, 0, max_abs_weight),
      c("blue", "yellow", "red")
    )
  } else {
    col_fun <- circlize::colorRamp2(
      c(-max_abs_weight, 0, max_abs_weight),
      c("blue", "white", "red")
    )
  }
  
  # ============================================================================
  # CREATE HEATMAP
  # ============================================================================
  
  if (nrow(weight_matrix) > 50 && show_row_names) {
    show_row_names <- FALSE
    message("Note: Hiding row names due to large matrix size")
  }
  
  heatmap <- ComplexHeatmap::Heatmap(
    matrix = weight_matrix,
    name = name,
    col = col_fun,
    cluster_rows = cluster_rows,
    cluster_columns = cluster_columns,
    show_row_names = show_row_names,
    show_column_names = show_column_names,
    column_title = title,
    row_title = "Features (Visible Layer)",
    column_title_gp = grid::gpar(fontsize = 14, fontface = "bold"),
    row_title_gp = grid::gpar(fontsize = 12),
    heatmap_legend_param = list(
      title = name,
      title_gp = grid::gpar(fontsize = 10, fontface = "bold"),
      labels_gp = grid::gpar(fontsize = 8)
    ),
    ...
  )
  
  return(heatmap)
}
