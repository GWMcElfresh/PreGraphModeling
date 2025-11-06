
#' Residualize a numeric vector using a GAM fit from mgcv
#'
#' Fit a univariate GAM of y on x (a smooth of x) using mgcv and return residuals.
#'
#' @param y Numeric vector to be residualized (e.g. sizeFactor).
#' @param x Numeric predictor (e.g. saturation).
#' @param k Integer, basis dimension for the smooth (passed to mgcv::s). Default 10.
#' @param bs Character, smooth basis type (passed to mgcv::s). Default "tp".
#' @param method Character, smoothing selection method passed to mgcv::gam/bam. Default "REML".
#' @param family A family object for the GAM. Default gaussian().
#' @param use_bam Logical, use mgcv::bam instead of mgcv::gam for large data. Default FALSE.
#' @param ... Additional arguments passed to mgcv::gam or mgcv::bam.
#'
#' @return A list with elements:
#' \describe{
#'   \item{residuals}{Numeric vector of residuals (same length as y).}
#'   \item{fitted}{Fitted values from the GAM (NA where fit not possible).}
#'   \item{model}{The fitted mgcv model object (or NULL if not fit).}
#' }
#'
#' @examples
#' \dontrun{
#' df <- data.frame(sizeFactor = rnorm(200, 10, 2), saturation = runif(200))
#' out <- residualize_gam(df$sizeFactor, df$saturation, k = 15)
#' df$sizeFactor_resid <- out$residuals
#' }
#' @importFrom stats as.formula
#' @export
residualize_gam <- function(y, x,
                            k = 10, bs = "tp",
                            method = "REML",
                            family = stats::gaussian(),
                            use_bam = FALSE,
                            ...) {
  if (!is.numeric(y) || !is.numeric(x)) stop("y and x must be numeric")
  if (length(y) != length(x)) stop("y and x must have the same length")
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("Install the 'mgcv' package: install.packages('mgcv')")
  }

  idx <- which(!is.na(y) & !is.na(x))
  fitted_vals <- rep(NA_real_, length(y))
  model <- NULL

  if (length(idx) > 1) {
    df <- data.frame(y = y[idx], saturation = x[idx])
    fmla <- as.formula(sprintf("y ~ s(saturation, k = %d, bs = '%s')", k, bs))
    if (use_bam) {
      model <- mgcv::bam(fmla, data = df, method = method, family = family, ...)
    } else {
      model <- mgcv::gam(fmla, data = df, method = method, family = family, ...)
    }
    fitted_vals[idx] <- stats::predict(model, newdata = data.frame(saturation = x[idx]))
  }

  residuals <- y - fitted_vals
  list(residuals = residuals, fitted = fitted_vals, model = model)
}
