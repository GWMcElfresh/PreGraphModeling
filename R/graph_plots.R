#' Plot Partial Correlations as an igraph Graph
#'
#' Visualizes the (subsampled) partial correlation matrix as an undirected graph.
#' By default, only the strongest edges are used to avoid rendering extremely large
#' graphs for typical single-cell feature sets.
#'
#' @param rbmObject An RBM object fitted using FitRBM().
#' @param features Optional character vector of features (genes) to include.
#'   If NULL, features are chosen from the strongest partial correlations.
#' @param nEdges Integer. If features is NULL, use the top nEdges absolute partial
#'   correlations as candidate edges (default: 50).
#' @param nFeatures Optional integer. If provided and features is NULL, cap the
#'   number of vertices to this value by keeping features with the strongest
#'   incident edge weights (default: 50).
#' @param layout Character. Graph layout algorithm: "fr" (Fruchterman-Reingold)
#'   or "kk" (Kamada-Kawai) (default: "fr").
#' @param main Plot title (default: "Partial Correlation Graph").
#' @param ... Additional arguments forwarded to igraph::plot().
#'
#' @return Invisibly returns a list with elements: graph (igraph), layout (numeric matrix).
#' @export
PlotPartialCorrelationGraph <- function(rbmObject,
                                       features = NULL,
                                       nEdges = 50,
                                       nFeatures = 50,
                                       layout = c("fr", "kk"),
                                       main = "Partial Correlation Graph",
                                       ...) {
  if (!inherits(rbmObject, "RBM")) {
    stop("rbmObject must be an RBM object created by FitRBM()")
  }
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Package 'igraph' is required. Install it with: install.packages('igraph')")
  }

  layout <- match.arg(layout)

  pcor <- rbmObject$partial_correlations
  if (is.null(pcor) || !is.matrix(pcor)) {
    stop("rbmObject$partial_correlations must be a matrix")
  }

  # Drop all-NA rows/cols
  valid_rows <- rowSums(!is.na(pcor)) > 0
  pcor <- pcor[valid_rows, valid_rows, drop = FALSE]

  if (is.null(features)) {
    selected <- .select_features_from_partial_cor(pcor_matrix = pcor, nEdges = nEdges, nFeatures = nFeatures)
  } else {
    if (!is.character(features)) stop("features must be a character vector")
    selected <- intersect(features, rownames(pcor))
    if (length(selected) == 0) stop("None of the specified features are available")
  }

  sub <- pcor[selected, selected, drop = FALSE]
  if (nrow(sub) < 2) {
    stop("Not enough features to build a graph")
  }

  ut <- upper.tri(sub, diag = FALSE)
  valid <- ut & !is.na(sub)
  idx <- which(valid)
  if (length(idx) == 0) {
    stop("No valid partial correlations among selected features")
  }

  vals <- abs(sub[idx])
  ord <- order(vals, decreasing = TRUE)
  top_n <- min(as.integer(nEdges), length(ord))
  top_idx <- idx[ord[seq_len(top_n)]]

  rc <- arrayInd(top_idx, dim(sub))
  genes <- rownames(sub)
  from <- genes[rc[, 1]]
  to <- genes[rc[, 2]]
  w <- sub[top_idx]

  edges <- data.frame(from = from, to = to, weight = as.numeric(w), stringsAsFactors = FALSE)
  g <- igraph::graph_from_data_frame(edges, directed = FALSE, vertices = genes)

  # Edge aesthetics
  ew <- abs(igraph::E(g)$weight)
  if (length(ew) > 0 && max(ew, na.rm = TRUE) > 0) {
    igraph::E(g)$width <- 1 + 4 * (ew / max(ew, na.rm = TRUE))
  } else {
    igraph::E(g)$width <- 1
  }
  igraph::E(g)$color <- ifelse(igraph::E(g)$weight >= 0, "red", "blue")

  # Layout: stronger edges should pull closer
  layout_mat <- if (layout == "fr") {
    igraph::layout_with_fr(g, weights = abs(igraph::E(g)$weight))
  } else {
    igraph::layout_with_kk(g, weights = abs(igraph::E(g)$weight))
  }

  igraph::plot.igraph(
    g,
    layout = layout_mat,
    main = main,
    vertex.size = 12,
    vertex.label.cex = 0.7,
    ...
  )

  invisible(list(graph = g, layout = layout_mat))
}


#' Plot RBM Layers and Activations as an igraph Graph
#'
#' Visualizes the RBM architecture as a simple bipartite graph: visible features
#' connected to hidden layers (metadata factors). Edges are aggregated per
#' feature-factor pair using the maximum absolute weight across that factor's
#' hidden units. This keeps the plot interpretable while still highlighting
#' which hidden layers are most connected to which genes.
#'
#' @param rbmObject An RBM object fitted using FitRBM().
#' @param features Optional character vector of visible features to include.
#'   If NULL, selects the top nFeatures by weight magnitude approximately evenly
#'   across the chosen factors (default: NULL).
#' @param factors Optional character vector of hidden factors (layers) to include.
#'   If NULL, uses all rbmObject$hidden_factors (default: NULL).
#' @param nFeatures Integer. If features is NULL, select approximately this many
#'   visible features across layers (default: 50).
#' @param main Plot title (default: "RBM Layer Graph").
#' @param ... Additional arguments forwarded to igraph::plot().
#'
#' @return Invisibly returns a list with elements: graph (igraph), layout (numeric matrix).
#' @export
PlotRBMLayerGraph <- function(rbmObject,
                              features = NULL,
                              factors = NULL,
                              nFeatures = 50,
                              main = "RBM Layer Graph",
                              ...) {
  if (!inherits(rbmObject, "RBM")) {
    stop("rbmObject must be an RBM object created by FitRBM()")
  }
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Package 'igraph' is required. Install it with: install.packages('igraph')")
  }

  factors_to_plot <- if (is.null(factors)) rbmObject$hidden_factors else factors
  missing_factors <- setdiff(factors_to_plot, rbmObject$hidden_factors)
  if (length(missing_factors) > 0) {
    stop(sprintf("Factors not found in RBM: %s", paste(missing_factors, collapse = ", ")))
  }

  if (is.null(features)) {
    features_to_plot <- .select_features_from_weights_per_layer(
      weights_per_layer = rbmObject$weights_per_layer,
      factors_to_plot = factors_to_plot,
      nFeatures = nFeatures
    )
  } else {
    if (!is.character(features)) stop("features must be a character vector")
    features_to_plot <- unique(features)
  }

  # Build aggregated edges (gene -> factor)
  edge_list <- list()
  for (factor_name in factors_to_plot) {
    w <- rbmObject$weights_per_layer[[factor_name]]
    if (is.null(w)) next

    common <- intersect(features_to_plot, rownames(w))
    if (length(common) == 0) next

    subw <- w[common, , drop = FALSE]
    # pick max-abs weight and its sign for each gene
    max_abs <- apply(abs(subw), 1, max, na.rm = TRUE)
    idx <- apply(abs(subw), 1, which.max)
    signed <- vapply(seq_along(common), function(i) subw[i, idx[i]], numeric(1))

    edge_list[[factor_name]] <- data.frame(
      from = common,
      to = factor_name,
      weight = as.numeric(signed),
      abs_weight = as.numeric(max_abs),
      stringsAsFactors = FALSE
    )
  }

  edges <- do.call(rbind, edge_list)
  if (is.null(edges) || nrow(edges) == 0) {
    stop("No edges available for the selected features/factors")
  }

  # Vertex data: label factors with activation name if available
  activation_labels <- factors_to_plot
  if (!is.null(rbmObject$hidden_layers_info) && is.list(rbmObject$hidden_layers_info)) {
    activation_labels <- vapply(factors_to_plot, function(f) {
      info <- rbmObject$hidden_layers_info[[f]]
      if (!is.null(info$type)) {
        act <- .get_activation_function(info$type)$name
        sprintf("%s (%s)", f, act)
      } else {
        f
      }
    }, character(1))
  }

  vertices <- data.frame(
    name = unique(c(edges$from, edges$to)),
    type = c(
      rep("feature", length(unique(edges$from))),
      rep("factor", length(unique(edges$to)))
    ),
    stringsAsFactors = FALSE
  )

  g <- igraph::graph_from_data_frame(edges[, c("from", "to", "weight")], directed = FALSE, vertices = vertices)

  # Set labels
  igraph::V(g)$label <- igraph::V(g)$name
  factor_idx <- igraph::V(g)$name %in% factors_to_plot
  igraph::V(g)$label[factor_idx] <- activation_labels[match(igraph::V(g)$name[factor_idx], factors_to_plot)]

  # Colors and sizes (keep simple)
  igraph::V(g)$color <- ifelse(igraph::V(g)$name %in% factors_to_plot, "grey80", "grey95")
  igraph::V(g)$size <- ifelse(igraph::V(g)$name %in% factors_to_plot, 20, 12)

  ew <- abs(igraph::E(g)$weight)
  igraph::E(g)$color <- ifelse(igraph::E(g)$weight >= 0, "red", "blue")
  if (length(ew) > 0 && max(ew, na.rm = TRUE) > 0) {
    igraph::E(g)$width <- 1 + 4 * (ew / max(ew, na.rm = TRUE))
  } else {
    igraph::E(g)$width <- 1
  }

  # Manual, formulaic layout: features left, factors right
  v_names <- igraph::V(g)$name
  is_factor <- v_names %in% factors_to_plot
  feat_names <- v_names[!is_factor]
  fac_names <- v_names[is_factor]

  y_feat <- seq(from = 1, to = length(feat_names), by = 1)
  y_fac <- seq(from = 1, to = length(fac_names), by = 1)

  coords <- matrix(0, nrow = length(v_names), ncol = 2)
  rownames(coords) <- v_names
  coords[feat_names, 1] <- 0
  coords[feat_names, 2] <- y_feat
  coords[fac_names, 1] <- 1
  coords[fac_names, 2] <- y_fac

  layout_mat <- coords[v_names, , drop = FALSE]

  igraph::plot.igraph(
    g,
    layout = layout_mat,
    main = main,
    vertex.label.cex = 0.7,
    ...
  )

  invisible(list(graph = g, layout = layout_mat))
}
