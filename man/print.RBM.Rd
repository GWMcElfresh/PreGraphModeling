\name{print.RBM}
\alias{print.RBM}
\alias{summary.RBM}
\title{Print and Summary Methods for RBM Objects}
\usage{
\method{print}{RBM}(x, ...)
\method{summary}{RBM}(object, ...)
}
\arguments{
\item{x}{An RBM object.}

\item{object}{An RBM object.}

\item{...}{Additional arguments (unused).}
}
\value{
Invisibly returns the RBM object.
}
\description{
Print and summary methods for RBM objects fitted using FitRBM().
}
\examples{
\dontrun{
rbm <- FitRBM(pbmc, hiddenFactors = "CellType")
print(rbm)
summary(rbm)
}
}
