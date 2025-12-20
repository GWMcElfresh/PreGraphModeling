\name{PlotRBMHeatmap}
\alias{PlotRBMHeatmap}
\title{Plot RBM Partial Correlations as Heatmap}
\usage{
PlotRBMHeatmap(
  rbmObject,
  features = NULL,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  color_palette = "RdBu",
  title = "RBM Partial Correlations",
  name = "Partial\\nCorrelation",
  ...
)
}
\arguments{
\item{rbmObject}{An RBM object fitted using FitRBM().}

\item{features}{Optional character vector of specific features to plot.}

\item{cluster_rows}{Logical indicating whether to cluster rows.}

\item{cluster_columns}{Logical indicating whether to cluster columns.}

\item{show_row_names}{Logical indicating whether to show row names.}

\item{show_column_names}{Logical indicating whether to show column names.}

\item{color_palette}{Character string specifying color palette. Options: "RdBu", "viridis", "plasma", "RdYlBu".}

\item{title}{Character string for heatmap title.}

\item{name}{Character string for color legend name.}

\item{...}{Additional arguments passed to ComplexHeatmap::Heatmap().}
}
\value{
A ComplexHeatmap object.
}
\description{
Creates a heatmap visualization of the partial correlation matrix from an RBM using ComplexHeatmap.
}
\examples{
\dontrun{
rbm <- FitRBM(pbmc, hiddenFactors = "CellType")
heatmap_plot <- PlotRBMHeatmap(rbm)
print(heatmap_plot)
}
}
