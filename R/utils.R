
#' Residualize a numeric vector using a GAM fit from mgcv
#'
#' Fit a univariate GAM of y on x (a smooth of x) using mgcv and return residuals.
#' The function automatically adapts the basis dimension (k) based on the number of
#' unique values in x to avoid overfitting. For very small datasets (< 4 unique x values),
#' falls back to linear regression.
#'
#' @param y Numeric vector to be residualized (e.g. sizeFactor).
#' @param x Numeric predictor (e.g. saturation).
#' @param k Integer, maximum basis dimension for the smooth (passed to mgcv::s). Default 10.
#'   Will be automatically reduced if there are fewer unique x values.
#' @param bs Character, smooth basis type (passed to mgcv::s). Default "tp".
#' @param method Character, smoothing selection method passed to mgcv::gam/bam. Default "REML".
#' @param family A family object for the GAM. Default gaussian().
#' @param use_bam Logical, use mgcv::bam instead of mgcv::gam for large data. Default FALSE.
#' @param ... Additional arguments passed to mgcv::gam or mgcv::bam.
#'
#' @return A list with elements:
#' \describe{
#'   \item{residuals}{Numeric vector of residuals (same length as y).}
#'   \item{fitted}{Fitted values from the GAM or linear model (NA where fit not possible).}
#'   \item{model}{The fitted mgcv model object or lm object (or NULL if not fit).}
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
    
    # Adaptive k: ensure k is less than the number of unique x values
    # GAM requires k < number of unique covariate combinations
    n_unique <- length(unique(x[idx]))
    k_adaptive <- min(k, max(3, n_unique - 1))
    
    # Try to fit the model with error handling
    tryCatch({
      # If we have very few unique values, use linear regression instead
      if (n_unique < 4) {
        # Fall back to simple linear model when too few unique values
        model <- stats::lm(y ~ saturation, data = df)
        fitted_vals[idx] <- stats::predict(model, newdata = data.frame(saturation = x[idx]))
      } else {
        # Use GAM with adaptive k
        fmla <- as.formula(sprintf("y ~ s(saturation, k = %d, bs = '%s')", k_adaptive, bs))
        if (use_bam) {
          model <- mgcv::bam(fmla, data = df, method = method, family = family, ...)
        } else {
          model <- mgcv::gam(fmla, data = df, method = method, family = family, ...)
        }
        fitted_vals[idx] <- stats::predict(model, newdata = data.frame(saturation = x[idx]))
      }
    }, error = function(e) {
      # If GAM fails, fall back to linear model
      warning(sprintf("GAM fitting failed (%s), falling back to linear model", e$message))
      model <<- stats::lm(y ~ saturation, data = df)
      fitted_vals[idx] <<- stats::predict(model, newdata = data.frame(saturation = x[idx]))
    })
  }

  residuals <- y - fitted_vals
  list(residuals = residuals, fitted = fitted_vals, model = model)
}
