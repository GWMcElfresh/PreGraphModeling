test_that("residualize_gam works with simple data", {
  skip_if_not_installed("mgcv")
  
  set.seed(42)
  n <- 100
  
  # Create correlated data (saturation affects size factors)
  saturation <- runif(n, min = 0.3, max = 0.95)
  size_factors <- 1.5 + 0.8 * saturation + rnorm(n, sd = 0.2)
  
  # Residualize
  result <- residualize_gam(y = size_factors, x = saturation)
  
  # Verify structure
  expect_type(result, "list")
  expect_named(result, c("residuals", "fitted", "model"))
  
  # Verify lengths
  expect_equal(length(result$residuals), n)
  expect_equal(length(result$fitted), n)
  
  # Verify model was fit
  expect_false(is.null(result$model))
  
  # Residuals should have reduced correlation with saturation
  cor_original <- cor(size_factors, saturation)
  cor_residual <- cor(result$residuals, saturation, use = "complete.obs")
  
  # Residual correlation should be smaller (in absolute value)
  expect_true(abs(cor_residual) < abs(cor_original))
})

test_that("residualize_gam handles NAs correctly", {
  skip_if_not_installed("mgcv")
  
  set.seed(123)
  n <- 50
  
  saturation <- runif(n)
  size_factors <- 1 + saturation + rnorm(n, sd = 0.1)
  
  # Introduce some NAs
  saturation[1:3] <- NA
  size_factors[4:5] <- NA
  
  result <- residualize_gam(y = size_factors, x = saturation)
  
  # Should still return vectors of correct length
  expect_equal(length(result$residuals), n)
  expect_equal(length(result$fitted), n)
  
  # NAs should produce NA residuals
  expect_true(is.na(result$residuals[1]))
  expect_true(is.na(result$residuals[4]))
})

test_that("residualize_gam validates input", {
  expect_error(
    residualize_gam(y = "not numeric", x = 1:10),
    "must be numeric"
  )
  
  expect_error(
    residualize_gam(y = 1:10, x = 1:5),
    "must have the same length"
  )
})

test_that("residualize_gam works with different parameters", {
  skip_if_not_installed("mgcv")
  
  set.seed(42)
  n <- 100
  
  x <- runif(n)
  y <- 1 + 2 * x + rnorm(n, sd = 0.1)
  
  # Different k values
  result_k5 <- residualize_gam(y = y, x = x, k = 5)
  result_k15 <- residualize_gam(y = y, x = x, k = 15)
  
  expect_type(result_k5, "list")
  expect_type(result_k15, "list")
  
  # Both should fit successfully
  expect_false(is.null(result_k5$model))
  expect_false(is.null(result_k15$model))
})
