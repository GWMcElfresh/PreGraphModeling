#' Fit CONGA Model to Seurat Object
#'
#' @description
#' Wrapper function to fit the CONGA (Conditional Nonparametric Graphical Analysis)
#' model directly to a Seurat object. This function extracts expression data from
#' the Seurat object, optionally performs feature selection, and then estimates
#' a conditional dependency graph using Dirichlet Process mixtures and MCMC.
#'
#' @param seuratObject A Seurat object or SeuratObject containing single-cell data
#' @param assay Character string specifying which assay to use (default: "RNA")
#' @param layer Character string specifying which layer to use (default: "counts")
#' @param geneSubset Optional character vector of gene names to include in the model.
#'   If NULL, uses all genes (default: NULL). Recommended to use feature selection
#'   for large datasets (e.g., highly variable genes).
#' @param minCells Minimum number of cells expressing a gene for inclusion (default: 10)
#' @param minGenes Minimum number of genes expressed in a cell for inclusion (default: 200)
#' @param totalIterations Total number of MCMC iterations (default: 5000)
#' @param burnIn Number of burn-in iterations to discard (default: 2500)
#' @param lambdaShrinkage Shrinkage parameter for beta prior (default: 1)
#' @param verbose Logical indicating whether to print progress (default: TRUE)
#'
#' @return A list with components (CONGAfit object):
#'   \itemize{
#'     \item beta_mcmc: List of MCMC samples for beta (precision matrix elements) after burn-in
#'     \item lambda_mcmc: List of MCMC samples for lambda (Poisson intensities) after burn-in
#'     \item power_parameter: Selected power parameter for atan transformation
#'     \item acceptance_rate_lambda: Acceptance rate for lambda updates
#'     \item acceptance_rate_beta: Acceptance rate for beta updates
#'     \item n_cells: Number of cells in the analysis
#'     \item n_genes: Number of genes in the analysis
#'     \item edge_index: Matrix of edge indices (gene pairs)
#'     \item gene_names: Names of genes included in the analysis
#'     \item cell_names: Names of cells included in the analysis
#'   }
#'
#' @details
#' This function provides a convenient interface for applying CONGA to Seurat objects.
#' It handles data extraction, filtering, and preprocessing before calling the core
#' CONGA fitting function.
#'
#' ## Feature Selection Recommendations
#'
#' For large datasets (>5000 genes), it is strongly recommended to perform feature
#' selection to reduce computational burden and focus on biologically relevant genes.
#' Options include:
#' - Highly variable genes (HVGs): Use Seurat::FindVariableFeatures()
#' - Marker genes: Use Seurat::FindAllMarkers()
#' - Custom gene set: Provide specific genes of interest
#'
#' ## Computational Complexity
#'
#' - Time complexity: O(iterations * cells * genes^2)
#' - Space complexity: O(iterations * cells * genes)
#' - Typical runtime: 10-30 minutes for 1000 cells x 100 genes with 5000 iterations
#'
#' @export
#' @importFrom SeuratObject GetAssayData
#' @examples
#' \dontrun{
#' # Load Seurat object
#' library(Seurat)
#' data("pbmc_small")
#'
#' # Option 1: Use highly variable genes
#' pbmc_small <- FindVariableFeatures(pbmc_small, nfeatures = 100)
#' hvg <- VariableFeatures(pbmc_small)
#'
#' result <- FitCONGA(
#'   seuratObject = pbmc_small,
#'   geneSubset = hvg,
#'   totalIterations = 1000,
#'   burnIn = 500
#' )
#'
#' # Option 2: Use specific genes
#' genes_of_interest <- c("CD3D", "CD4", "CD8A", "MS4A1", "CD14")
#' result <- FitCONGA(
#'   seuratObject = pbmc_small,
#'   geneSubset = genes_of_interest
#' )
#'
#' # Extract graph
#' graph <- ExtractCONGAGraph(result, cutoff = 0.7)
#' }
FitCONGA <- function(seuratObject,
                     assay = "RNA",
                     layer = "counts",
                     geneSubset = NULL,
                     minCells = 10,
                     minGenes = 200,
                     totalIterations = 5000,
                     burnIn = 2500,
                     lambdaShrinkage = 1,
                     verbose = TRUE) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  if (!inherits(seuratObject, c("Seurat", "SeuratObject"))) {
    stop("seuratObject must be a Seurat or SeuratObject")
  }
  
  # ============================================================================
  # EXTRACT EXPRESSION DATA
  # ============================================================================
  if (verbose) {
    message("Extracting expression data from Seurat object...")
  }
  
  # Get expression matrix
  expression_matrix <- SeuratObject::GetAssayData(
    object = seuratObject,
    assay = assay,
    layer = layer
  )
  
  # Convert to regular matrix if sparse
  if (inherits(expression_matrix, "Matrix")) {
    expression_matrix <- as.matrix(expression_matrix)
  }
  
  # Transpose to cells x genes format (CONGA expects this)
  expression_matrix <- t(expression_matrix)
  
  if (verbose) {
    message(sprintf("Initial data dimensions: %d cells x %d genes",
                    nrow(expression_matrix), ncol(expression_matrix)))
  }
  
  # ============================================================================
  # FEATURE SELECTION
  # ============================================================================
  if (!is.null(geneSubset)) {
    # Filter to specified genes
    available_genes <- intersect(geneSubset, colnames(expression_matrix))
    
    if (length(available_genes) == 0) {
      stop("None of the genes in geneSubset are present in the expression matrix")
    }
    
    if (length(available_genes) < length(geneSubset)) {
      missing_genes <- setdiff(geneSubset, available_genes)
      warning(sprintf("%d genes not found in expression matrix: %s",
                      length(missing_genes),
                      paste(head(missing_genes, 5), collapse = ", ")))
    }
    
    expression_matrix <- expression_matrix[, available_genes, drop = FALSE]
    
    if (verbose) {
      message(sprintf("After gene subset: %d cells x %d genes",
                      nrow(expression_matrix), ncol(expression_matrix)))
    }
  }
  
  # ============================================================================
  # QUALITY FILTERING
  # ============================================================================
  
  # Filter genes by minimum cells
  genes_per_cell_count <- colSums(expression_matrix > 0)
  genes_to_keep <- genes_per_cell_count >= minCells
  
  if (sum(genes_to_keep) == 0) {
    stop(sprintf("No genes pass the minCells filter (%d)", minCells))
  }
  
  expression_matrix <- expression_matrix[, genes_to_keep, drop = FALSE]
  
  if (verbose) {
    message(sprintf("After filtering genes (minCells=%d): %d cells x %d genes",
                    minCells, nrow(expression_matrix), ncol(expression_matrix)))
  }
  
  # Filter cells by minimum genes
  cells_gene_count <- rowSums(expression_matrix > 0)
  cells_to_keep <- cells_gene_count >= minGenes
  
  if (sum(cells_to_keep) == 0) {
    stop(sprintf("No cells pass the minGenes filter (%d)", minGenes))
  }
  
  expression_matrix <- expression_matrix[cells_to_keep, , drop = FALSE]
  
  if (verbose) {
    message(sprintf("After filtering cells (minGenes=%d): %d cells x %d genes",
                    minGenes, nrow(expression_matrix), ncol(expression_matrix)))
  }
  
  # Warn if too many genes (computational burden)
  if (ncol(expression_matrix) > 500) {
    warning(paste0(
      "You are fitting CONGA to ", ncol(expression_matrix), " genes. ",
      "This may be computationally expensive. Consider using feature selection ",
      "(e.g., highly variable genes) to reduce the gene set."
    ))
  }
  
  # ============================================================================
  # FIT CONGA MODEL
  # ============================================================================
  if (verbose) {
    message("Fitting CONGA model...")
  }
  
  result <- FitCONGAModel(
    expressionData = expression_matrix,
    totalIterations = totalIterations,
    burnIn = burnIn,
    lambdaShrinkage = lambdaShrinkage,
    verbose = verbose
  )
  
  # Add gene and cell names to result
  result$gene_names <- colnames(expression_matrix)
  result$cell_names <- rownames(expression_matrix)
  
  if (verbose) {
    message("CONGA model fitting complete!")
  }
  
  return(result)
}


#' Extract Graph from CONGA Fit
#'
#' @description
#' Post-processes CONGA MCMC samples to extract a conditional dependency graph.
#' Computes posterior probabilities for each edge and applies a threshold to
#' determine which edges to include in the final graph.
#'
#' @param congaFit A CONGAfit object returned by FitCONGA() or FitCONGAModel()
#' @param cutoff Threshold for posterior probability to include an edge (default: 0.7).
#'   Higher values produce sparser graphs. Must be between 0 and 1.
#' @param method Method for computing posterior edge probabilities (default: "asymmetric").
#'   Options:
#'   - "asymmetric": P(beta > 0) or P(beta < 0), whichever is larger
#'   - "symmetric": P(|beta| > 0)
#'
#' @return A list with components:
#'   \itemize{
#'     \item adjacency_matrix: Binary adjacency matrix (1 = edge present, 0 = no edge)
#'     \item edge_probabilities: Posterior probability for each edge
#'     \item edge_list: Data frame of edges with probabilities
#'     \item n_edges: Number of edges in the graph
#'     \item cutoff: Cutoff value used
#'   }
#'
#' @details
#' The posterior edge probability is computed as:
#' - Asymmetric method: max(P(beta > 0), P(beta < 0))
#' - Symmetric method: P(beta != 0)
#'
#' The cutoff is then applied: abs(P - 0.5) / 0.5 > cutoff
#' This means:
#' - cutoff = 0.0: include edges with P > 0.5 or P < 0.5 (all non-zero)
#' - cutoff = 0.5: include edges with P > 0.75 or P < 0.25
#' - cutoff = 1.0: include edges with P ~ 1.0 or P ~ 0.0 (very confident)
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit CONGA model
#' result <- FitCONGA(seurat_obj, geneSubset = hvg)
#'
#' # Extract graph with default cutoff
#' graph <- ExtractCONGAGraph(result)
#'
#' # Extract sparser graph with higher cutoff
#' sparse_graph <- ExtractCONGAGraph(result, cutoff = 0.9)
#'
#' # Visualize adjacency matrix
#' library(ComplexHeatmap)
#' Heatmap(graph$adjacency_matrix)
#'
#' # Convert to igraph for network analysis
#' library(igraph)
#' g <- graph_from_adjacency_matrix(graph$adjacency_matrix, mode = "undirected")
#' plot(g)
#' }
ExtractCONGAGraph <- function(congaFit,
                              cutoff = 0.7,
                              method = "asymmetric") {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  if (!inherits(congaFit, "CONGAfit")) {
    stop("congaFit must be a CONGAfit object returned by FitCONGA() or FitCONGAModel()")
  }
  
  if (cutoff < 0 || cutoff > 1) {
    stop("cutoff must be between 0 and 1")
  }
  
  if (!method %in% c("asymmetric", "symmetric")) {
    stop("method must be 'asymmetric' or 'symmetric'")
  }
  
  # ============================================================================
  # EXTRACT MCMC SAMPLES
  # ============================================================================
  beta_samples <- congaFit$beta_mcmc
  n_samples <- length(beta_samples)
  n_beta_params <- length(beta_samples[[1]])
  
  # ============================================================================
  # COMPUTE POSTERIOR PROBABILITIES
  # ============================================================================
  # Create matrix: rows = parameters, columns = MCMC samples
  beta_matrix <- matrix(unlist(beta_samples), nrow = n_beta_params, ncol = n_samples)
  
  # Compute posterior probability that beta > 0 for each parameter
  if (method == "asymmetric") {
    prob_positive <- apply(beta_matrix, 1, function(x) mean(x >= 0))
  } else {
    # For symmetric, we want P(beta != 0), approximated by counting non-zeros
    prob_positive <- apply(beta_matrix, 1, function(x) mean(abs(x) > 1e-10))
  }
  
  # ============================================================================
  # APPLY CUTOFF
  # ============================================================================
  # Transform probabilities to [-1, 1] scale where 0.5 -> 0
  # Edge is included if |transformed_prob| > cutoff
  transformed_probs <- (abs(prob_positive - 0.5) / 0.5)
  edge_indicator <- transformed_probs > cutoff
  
  # ============================================================================
  # CONSTRUCT ADJACENCY MATRIX
  # ============================================================================
  n_genes <- congaFit$n_genes
  adjacency_matrix <- matrix(0, n_genes, n_genes)
  
  # Fill lower triangle
  adjacency_matrix[lower.tri(adjacency_matrix)] <- edge_indicator
  
  # Make symmetric
  adjacency_matrix <- adjacency_matrix + t(adjacency_matrix)
  
  # Add gene names if available
  if (!is.null(congaFit$gene_names)) {
    rownames(adjacency_matrix) <- congaFit$gene_names
    colnames(adjacency_matrix) <- congaFit$gene_names
  }
  
  # ============================================================================
  # CREATE EDGE LIST
  # ============================================================================
  edge_index <- congaFit$edge_index
  edge_list <- data.frame(
    gene1 = edge_index[1, ],
    gene2 = edge_index[2, ],
    posterior_prob = prob_positive,
    transformed_prob = transformed_probs,
    included = edge_indicator,
    stringsAsFactors = FALSE
  )
  
  # Add gene names if available
  if (!is.null(congaFit$gene_names)) {
    edge_list$gene1_name <- congaFit$gene_names[edge_list$gene1]
    edge_list$gene2_name <- congaFit$gene_names[edge_list$gene2]
  }
  
  # Sort by probability (descending)
  edge_list <- edge_list[order(-edge_list$transformed_prob), ]
  
  # ============================================================================
  # RETURN RESULTS
  # ============================================================================
  result <- list(
    adjacency_matrix = adjacency_matrix,
    edge_probabilities = prob_positive,
    edge_list = edge_list,
    n_edges = sum(edge_indicator),
    cutoff = cutoff,
    method = method
  )
  
  class(result) <- c("CONGAgraph", "list")
  return(result)
}


#' Compute ROC Curve for CONGA Graph
#'
#' @description
#' Computes ROC curve for CONGA edge predictions if the true graph is known.
#' This is primarily used for simulation studies and method evaluation.
#'
#' @param congaFit A CONGAfit object returned by FitCONGA() or FitCONGAModel()
#' @param trueGraph True adjacency matrix or precision matrix
#' @param cutoffs Vector of cutoff values to evaluate (default: seq(0, 1, by = 0.05))
#'
#' @return A list with components:
#'   \itemize{
#'     \item fpr: False positive rates for each cutoff
#'     \item tpr: True positive rates for each cutoff
#'     \item cutoffs: Cutoff values used
#'     \item auc: Area under the ROC curve
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Assuming we have a true graph structure
#' roc <- ComputeCONGAROC(result, trueGraph = true_precision_matrix)
#'
#' # Plot ROC curve
#' plot(roc$fpr, roc$tpr, type = "l",
#'      xlab = "False Positive Rate",
#'      ylab = "True Positive Rate",
#'      main = sprintf("ROC Curve (AUC = %.3f)", roc$auc))
#' abline(0, 1, lty = 2)
#' }
ComputeCONGAROC <- function(congaFit,
                            trueGraph,
                            cutoffs = seq(0, 0.9, by = 0.05)) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  if (!inherits(congaFit, "CONGAfit")) {
    stop("congaFit must be a CONGAfit object")
  }
  
  if (!is.matrix(trueGraph)) {
    stop("trueGraph must be a matrix")
  }
  
  if (nrow(trueGraph) != ncol(trueGraph)) {
    stop("trueGraph must be square")
  }
  
  if (nrow(trueGraph) != congaFit$n_genes) {
    stop("trueGraph dimensions must match number of genes in congaFit")
  }
  
  # ============================================================================
  # BINARIZE TRUE GRAPH
  # ============================================================================
  # Consider non-zero entries as edges
  true_adjacency <- (trueGraph != 0) * 1
  
  # Extract lower triangle (since graph is undirected)
  edge_index <- congaFit$edge_index
  true_edges <- true_adjacency[t(edge_index)]
  
  # ============================================================================
  # COMPUTE ROC CURVE
  # ============================================================================
  fpr <- numeric(length(cutoffs))
  tpr <- numeric(length(cutoffs))
  
  for (i in seq_along(cutoffs)) {
    # Extract graph at this cutoff
    graph <- ExtractCONGAGraph(congaFit, cutoff = cutoffs[i])
    predicted_edges <- graph$edge_list$included
    
    # Compute confusion matrix elements
    true_positives <- sum(predicted_edges == 1 & true_edges == 1)
    false_positives <- sum(predicted_edges == 1 & true_edges == 0)
    true_negatives <- sum(predicted_edges == 0 & true_edges == 0)
    false_negatives <- sum(predicted_edges == 0 & true_edges == 1)
    
    # Compute rates
    tpr[i] <- true_positives / max(true_positives + false_negatives, 1)
    fpr[i] <- false_positives / max(false_positives + true_negatives, 1)
  }
  
  # ============================================================================
  # COMPUTE AUC
  # ============================================================================
  # Use trapezoidal rule
  auc <- 0
  for (i in 2:length(cutoffs)) {
    auc <- auc + (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
  }
  auc <- abs(auc)  # Make sure it's positive
  
  # ============================================================================
  # RETURN RESULTS
  # ============================================================================
  result <- list(
    fpr = fpr,
    tpr = tpr,
    cutoffs = cutoffs,
    auc = auc
  )
  
  class(result) <- c("CONGAROC", "list")
  return(result)
}


#' Print Method for CONGAfit
#'
#' @param x A CONGAfit object
#' @param ... Additional arguments (not used)
#'
#' @export
print.CONGAfit <- function(x, ...) {
  cat("CONGA Model Fit\n")
  cat("===============\n\n")
  cat(sprintf("Number of cells: %d\n", x$n_cells))
  cat(sprintf("Number of genes: %d\n", x$n_genes))
  cat(sprintf("Number of MCMC samples: %d\n", length(x$beta_mcmc)))
  cat(sprintf("Power parameter: %.2f\n", x$power_parameter))
  cat(sprintf("Lambda acceptance rate: %.2f%%\n", x$acceptance_rate_lambda * 100))
  cat(sprintf("Beta acceptance rate: %.2f%%\n", x$acceptance_rate_beta * 100))
  cat("\nUse ExtractCONGAGraph() to extract the conditional dependency graph.\n")
}


#' Print Method for CONGAgraph
#'
#' @param x A CONGAgraph object
#' @param ... Additional arguments (not used)
#'
#' @export
print.CONGAgraph <- function(x, ...) {
  cat("CONGA Conditional Dependency Graph\n")
  cat("===================================\n\n")
  cat(sprintf("Number of genes: %d\n", nrow(x$adjacency_matrix)))
  cat(sprintf("Number of edges: %d\n", x$n_edges))
  cat(sprintf("Graph density: %.3f\n", 
              x$n_edges / (nrow(x$adjacency_matrix) * (nrow(x$adjacency_matrix) - 1) / 2)))
  cat(sprintf("Cutoff used: %.2f\n", x$cutoff))
  cat(sprintf("Method: %s\n", x$method))
  cat("\nTop edges:\n")
  print(head(x$edge_list[x$edge_list$included, ], 10))
}
