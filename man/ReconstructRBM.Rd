\name{ReconstructRBM}
\alias{ReconstructRBM}
\title{Reconstruct Visible Layer from Hidden Layer}
\usage{
ReconstructRBM(object, hidden = NULL)
}
\arguments{
\item{object}{An RBM object fitted using FitRBM().}

\item{hidden}{A matrix of hidden layer values.}
}
\value{
A matrix of reconstructed visible layer values (expression).
}
\description{
Given hidden layer activations, reconstruct the visible layer (expression values) using the RBM weights.
}
\examples{
\dontrun{
rbm <- FitRBM(pbmc, hiddenFactors = "CellType")
reconstructed <- ReconstructRBM(rbm)
}
}
