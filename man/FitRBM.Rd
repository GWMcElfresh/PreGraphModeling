\name{FitRBM}
\alias{FitRBM}
\title{Fit Restricted Boltzmann Machine}
\usage{
FitRBM(
  seuratObject,
  visibleFeatures = NULL,
  hiddenFactors,
  assay = "RNA",
  layer = "counts",
  family = "zinb",
  minNonZero = 10,
  progressr = TRUE,
  parallel = FALSE,
  numWorkers = NULL,
  verbose = TRUE
)
}
\arguments{
\item{seuratObject}{A Seurat object or SeuratObject containing single-cell data.}

\item{visibleFeatures}{Character vector of feature names to use as the visible layer.}

\item{hiddenFactors}{Character vector of metadata column names to use as hidden units.}

\item{assay}{Character string specifying which assay to use.}

\item{layer}{Character string specifying which layer to use.}

\item{family}{Character string specifying the distribution family. Options: "zinb", "nb", "poisson", "gaussian".}

\item{minNonZero}{Minimum number of non-zero observations required.}

\item{progressr}{Logical indicating whether to use progressr.}

\item{parallel}{Logical indicating whether to use parallel processing.}

\item{numWorkers}{Integer specifying number of workers.}

\item{verbose}{Logical indicating whether to print messages.}
}
\value{
An object of class "RBM" containing weights, biases, partial correlations, and fit information.
}
\description{
Fits a Restricted Boltzmann Machine to connect features from a Seurat object to a hidden layer populated by metadata factors using partial correlations via quasilikelihood.
}
\examples{
\dontrun{
rbm <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = "CellType",
  family = "zinb",
  parallel = TRUE
)
}
}
