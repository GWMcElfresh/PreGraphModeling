\name{EstimatePartialCorrelations}
\alias{EstimatePartialCorrelations}
\title{Estimate Partial Correlations via Quasilikelihood}
\usage{
EstimatePartialCorrelations(
  expressionMatrix,
  metadata = NULL,
  family = "zinb",
  method = "glm",
  minNonZero = 10,
  progressr = TRUE,
  parallel = FALSE,
  numWorkers = NULL,
  verbose = TRUE
)
}
\arguments{
\item{expressionMatrix}{A numeric matrix where rows are features (genes) and columns are observations (cells).}

\item{metadata}{Optional data frame of metadata variables to include as covariates.}

\item{family}{Character string specifying the distribution family for quasilikelihood. Options: "zinb", "nb", "poisson", "gaussian", "quasipoisson".}

\item{method}{Character string specifying the estimation method. Options: "glm", "correlation".}

\item{minNonZero}{Minimum number of non-zero observations required for a feature pair.}

\item{progressr}{Logical indicating whether to use progressr for progress reporting.}

\item{parallel}{Logical indicating whether to use parallel processing.}

\item{numWorkers}{Integer specifying number of workers for parallel processing.}

\item{verbose}{Logical indicating whether to print progress messages.}
}
\value{
A list with components including partial_cor matrix, standard errors, p-values, and metadata.
}
\description{
Computes partial correlations between features using quasilikelihood estimation for various families including zero-inflated negative binomial.
}
\examples{
\dontrun{
result <- EstimatePartialCorrelations(
  expressionMatrix = expr_matrix,
  family = "zinb",
  parallel = TRUE
)
}
}
