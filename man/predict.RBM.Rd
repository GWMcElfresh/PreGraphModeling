\name{predict.RBM}
\alias{predict.RBM}
\title{Predict Hidden Layer Activations from RBM}
\usage{
\method{predict}{RBM}(object, newdata = NULL, type = "activation", ...)
}
\arguments{
\item{object}{An RBM object fitted using FitRBM().}

\item{newdata}{A matrix of expression values for visible features.}

\item{type}{Character string specifying the type of prediction. Options: "activation", "probability", "label".}

\item{...}{Additional arguments (unused).}
}
\value{
A matrix of predicted hidden layer values.
}
\description{
Given a fitted RBM and new expression data, predict the hidden layer activations based on the learned weights and biases.
}
\examples{
\dontrun{
rbm <- FitRBM(pbmc, hiddenFactors = "CellType")
predictions <- predict(rbm, newdata = new_expr_matrix)
}
}
