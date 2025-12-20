\name{PlotRBMWeights}
\alias{PlotRBMWeights}
\title{Plot RBM Weights as Heatmap}
\usage{
PlotRBMWeights(
  rbmObject,
  features = NULL,
  factors = NULL,
  cluster_rows = TRUE,
  cluster_columns = FALSE,
  show_row_names = TRUE,
  show_column_names = TRUE,
  color_palette = "RdBu",
  title = "RBM Weights: Features to Hidden Factors",
  name = "Weight",
  ...
)
}
\arguments{
\item{rbmObject}{An RBM object fitted using FitRBM().}

\item{features}{Optional character vector of specific features to plot.}

\item{factors}{Optional character vector of specific hidden factors to plot.}

\item{cluster_rows}{Logical indicating whether to cluster rows.}

\item{cluster_columns}{Logical indicating whether to cluster columns.}

\item{show_row_names}{Logical indicating whether to show row names.}

\item{show_column_names}{Logical indicating whether to show column names.}

\item{color_palette}{Character string specifying color palette.}

\item{title}{Character string for heatmap title.}

\item{name}{Character string for color legend name.}

\item{...}{Additional arguments passed to ComplexHeatmap::Heatmap().}
}
\value{
A ComplexHeatmap object.
}
\description{
Creates a heatmap visualization of the weight matrix from an RBM, showing connections between visible features and hidden factors.
}
\examples{
\dontrun{
rbm <- FitRBM(pbmc, hiddenFactors = c("CellType", "Treatment"))
weight_plot <- PlotRBMWeights(rbm)
print(weight_plot)
}
}
