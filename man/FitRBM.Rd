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
  cd_k = 1,
  learning_rate = 0.01,
  n_epochs = 10,
  batch_size = 100,
  persistent = FALSE,
  momentum = 0.5,
  weight_decay = 0.0001,
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

\item{cd_k}{Number of Gibbs sampling steps for Contrastive Divergence (default: 1).}

\item{learning_rate}{Learning rate for weight updates (default: 0.01).}

\item{n_epochs}{Number of training epochs (default: 10).}

\item{batch_size}{Batch size for mini-batch training (default: 100).}

\item{persistent}{Logical indicating whether to use Persistent CD (default: FALSE).}

\item{momentum}{Momentum coefficient for weight updates (default: 0.5).}

\item{weight_decay}{L2 regularization coefficient (default: 0.0001).}

\item{progressr}{Logical indicating whether to use progressr.}

\item{parallel}{Logical indicating whether to use parallel processing.}

\item{numWorkers}{Integer specifying number of workers.}

\item{verbose}{Logical indicating whether to print messages.}
}
\value{
An object of class "RBM" containing weights, biases, partial correlations, training error, and fit information.
}
\description{
Fits a Restricted Boltzmann Machine to connect features from a Seurat object to a hidden layer populated by metadata factors using partial correlations via quasilikelihood, then trains using Contrastive Divergence.
}
\examples{
\dontrun{
rbm <- FitRBM(
  seuratObject = pbmc,
  hiddenFactors = "CellType",
  family = "zinb",
  cd_k = 1,
  n_epochs = 10,
  parallel = TRUE
)
}
}
