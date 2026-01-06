#' Fit CONGA Graphical Model
#'
#' @description
#' Implements the CONGA (Conditional Nonparametric Graphical Analysis) algorithm
#' for estimating conditional dependency graphs using Dirichlet Process mixtures
#' and MCMC sampling. This is a translation of the algorithm from:
#' Roy & Dunson - "Nonparametric graphical model for counts"
#' (https://github.com/royarkaprava/CONGA)
#'
#' @param expressionData A numeric matrix where rows are cells/observations and
#'   columns are genes/features. Can be a regular matrix or sparse Matrix.
#' @param totalIterations Total number of MCMC iterations (default: 5000)
#' @param burnIn Number of burn-in iterations to discard (default: 2500)
#' @param lambdaShrinkage Shrinkage parameter for beta prior, similar to Wang (2012).
#'   Controls regularization strength (default: 1)
#' @param verbose Logical indicating whether to print progress (default: TRUE)
#'
#' @return A list with components:
#'   \itemize{
#'     \item beta_mcmc: List of MCMC samples for beta (precision matrix elements) after burn-in
#'     \item lambda_mcmc: List of MCMC samples for lambda (Poisson intensities) after burn-in
#'     \item power_parameter: Selected power parameter for atan transformation
#'     \item acceptance_rate_lambda: Acceptance rate for lambda updates
#'     \item acceptance_rate_beta: Acceptance rate for beta updates
#'     \item n_cells: Number of cells in the input data
#'     \item n_genes: Number of genes in the input data
#'   }
#'
#' @details
#' ## CONGA Model Structure
#'
#' The CONGA model assumes:
#' 1. **Data Model**: X[i,j] | lambda[i,j], beta ~ Poisson(lambda[i,j]) * exp(beta' * atan(X[i,-j])^power)
#'    - X[i,j]: Count for cell i, gene j
#'    - lambda[i,j]: Cell-gene specific Poisson intensity
#'    - beta: Precision matrix elements capturing conditional dependencies
#'    - power: Transformation parameter to handle non-Gaussian data
#'
#' 2. **Dirichlet Process Prior on lambda**: lambda[,j] ~ DP(M, G0)
#'    - Allows clustering of cells with similar intensity patterns
#'    - M: Concentration parameter (controls number of clusters)
#'    - G0: Base distribution (Gamma distribution)
#'
#' 3. **Horseshoe-like Prior on beta**: Encourages sparsity in the graph
#'    - beta_k ~ N(0, tau^2 * phi_k * psi_k)
#'    - Spike-and-slab structure with latent indicators Z
#'
#' ## MCMC Sampling Strategy
#'
#' The algorithm alternates between:
#' 1. **Update lambda** via modified Gibbs sampler with auxiliary variable method
#'    - Uses Dirichlet Process clustering structure
#'    - Metropolis-Hastings acceptance step
#' 2. **Update beta** via blocked Gibbs sampler
#'    - One column at a time to exploit conditional independence
#'    - Metropolis-Hastings acceptance step
#' 3. **Update hyperparameters** (M, tau, phi, psi, Z)
#'
#' ## UNCERTAINTY NOTES
#'
#' - **Power parameter selection**: Uses heuristic method that may not be optimal for all datasets
#' - **Normalizing constant**: Approximated by truncating infinite sum at max_val=100
#' - **Convergence**: Standard MCMC diagnostics should be applied to check convergence
#' - **Identifiability**: The model is not fully identified; focus on posterior edge probabilities
#'
#' @export
#' @examples
#' \dontrun{
#' # Fit CONGA model to expression matrix
#' result <- FitCONGAModel(
#'   expressionData = expression_matrix,
#'   totalIterations = 5000,
#'   burnIn = 2500
#' )
#'
#' # Extract posterior samples
#' beta_samples <- result$beta_mcmc
#'
#' # Compute edge probabilities
#' edges <- ExtractCONGAGraph(result, cutoff = 0.7)
#' }
FitCONGAModel <- function(expressionData,
                          totalIterations = 5000,
                          burnIn = 2500,
                          lambdaShrinkage = 1,
                          verbose = TRUE) {
  
  # ============================================================================
  # INPUT VALIDATION
  # ============================================================================
  if (!is.matrix(expressionData) && !inherits(expressionData, "Matrix")) {
    stop("expressionData must be a matrix or Matrix object")
  }
  
  # Convert to regular matrix if sparse
  if (inherits(expressionData, "Matrix")) {
    expressionData <- as.matrix(expressionData)
  }
  
  if (burnIn >= totalIterations) {
    stop("burnIn must be less than totalIterations")
  }
  
  # ============================================================================
  # INITIALIZE DIMENSIONS AND DATA
  # ============================================================================
  n_cells <- nrow(expressionData)  # Number of observations (cells)
  n_genes <- ncol(expressionData)  # Number of features (genes)
  
  if (verbose) {
    message(sprintf("Fitting CONGA model to %d cells and %d genes", n_cells, n_genes))
  }
  
  # Maximum count value in the data
  max_count <- max(expressionData)
  
  # ============================================================================
  # SELECT POWER PARAMETER
  # ============================================================================
  # This parameter controls the atan transformation: atan(x)^power
  # It balances between capturing dependencies and computational tractability
  if (verbose) {
    message("Selecting optimal power parameter...")
  }
  
  power_parameter <- SelectPowerParameter(po = as.integer(max_count), X = expressionData)
  
  if (verbose) {
    message(sprintf("Selected power parameter: %.2f", power_parameter))
  }
  
  # ============================================================================
  # INITIALIZE MCMC VARIABLES
  # ============================================================================
  
  # --- Lambda (Poisson intensities) ---
  # Initialize at observed counts + small constant to avoid zeros
  lambda <- expressionData + 1e-10
  
  # Store MCMC samples
  lambda_mcmc_samples <- vector("list", totalIterations)
  
  # --- Beta (Precision matrix elements) ---
  # Number of unique beta parameters: choose(n_genes, 2)
  n_beta_params <- n_genes * (n_genes - 1) / 2
  
  # Create index matrix for all possible edges (i,j) where i < j
  edge_index <- combinat::combn(1:n_genes, 2)
  
  # Initialize beta randomly from standard normal
  beta <- rnorm(n_beta_params)
  
  # Store MCMC samples
  beta_mcmc_samples <- vector("list", totalIterations)
  
  # --- Dirichlet Process Parameters ---
  # M: Concentration parameter for each gene (controls clustering strength)
  concentration_M <- rep(2, n_genes)
  
  # --- Hyperparameters for Gamma base distribution G0 ---
  gamma_shape_alpha <- 1
  gamma_rate_beta <- 1
  
  # --- Compute centered atan-transformed data ---
  # This is used frequently in beta updates
  atan_data <- atan(expressionData)^power_parameter
  atan_means <- colMeans(atan_data)
  atan_centered <- sweep(atan_data, 2, atan_means, "-")
  
  # Cross-product matrix for beta proposal
  crossprod_atan <- crossprod(atan_centered)
  
  # --- Spike-and-Slab Prior Components ---
  # Hyperparameters for the Dirichlet prior on mixture weights
  dirichlet_alpha <- 1 / n_beta_params
  
  # Spike and slab variances
  spike_variance_s0 <- 100  # Variance for slab (large = non-sparse)
  slab_variance_s1 <- 100   # Variance for spike (should be smaller, but code uses same)
  
  # Latent indicators for spike vs slab
  latent_indicators_Z <- rep(1, n_beta_params)
  
  # Hierarchical variance components
  psi <- rexp(n_beta_params, rate = 0.5)
  phi <- MCMCpack::rdirichlet(1, rep(dirichlet_alpha, n_beta_params))[1,]
  tau <- rgamma(1, shape = n_beta_params * dirichlet_alpha, rate = 0.5)
  
  # --- Construct full Beta matrix from vector ---
  # This is a symmetric matrix with zeros on diagonal
  Beta_matrix <- matrix(0, n_genes, n_genes)
  Beta_matrix[lower.tri(Beta_matrix)] <- beta
  Beta_matrix <- Beta_matrix + t(Beta_matrix)
  
  # --- Acceptance counters ---
  accept_count_lambda <- 0
  accept_count_beta <- 0
  
  # --- Pre-compute inverse of diagonal precision matrix ---
  # This is used in the proposal distribution for beta
  # UNCERTAINTY NOTE: This inversion can be unstable if covariance is near-singular
  const <- 1e-20
  repeat {
    atan_cov <- cov(atan_data)
    atan_cov_plus_const <- atan_cov + const * diag(n_genes)
    
    # Try to invert
    inv_result <- tryCatch({
      diag(solve(atan_cov_plus_const))
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(inv_result)) {
      crossprod_atan_inv_diag <- inv_result
      break
    }
    
    # Increase regularization if inversion failed
    const <- 10 * const
    
    if (const > 1e-10) {
      warning("Covariance matrix is nearly singular. Using strong regularization.")
      crossprod_atan_inv_diag <- rep(1 / var(atan_data), n_genes)
      break
    }
  }
  
  # --- Pre-compute gamma densities for lambda proposals ---
  # This is reused in the lambda update step
  gamma_densities <- matrix(0, n_cells, n_genes)
  for (j in 1:n_genes) {
    gamma_densities[, j] <- dgamma(lambda[, j], 
                                    shape = gamma_shape_alpha + expressionData[, j],
                                    rate = gamma_rate_beta + 1)
  }
  
  # ============================================================================
  # MCMC LOOP
  # ============================================================================
  if (verbose) {
    message("Starting MCMC iterations...")
    pb <- txtProgressBar(min = 0, max = totalIterations, style = 3)
  }
  
  for (iteration in 1:totalIterations) {
    
    # ==========================================================================
    # UPDATE LAMBDA (Poisson intensities) - One gene at a time
    # ==========================================================================
    # This uses a modified Gibbs sampler with auxiliary variables
    # based on the Dirichlet Process prior structure
    
    for (gene_j in 1:n_genes) {
      
      lambda_candidate <- lambda
      
      # Compute unnormalized probabilities for auxiliary variable sampling
      # Q[i] = probability of selecting cell i's lambda value for proposal
      proposal_probabilities_Q <- rep(0, n_cells)
      
      for (cell_i in 1:n_cells) {
        # Poisson likelihood contribution for all cells
        # exp(-lambda + X*log(lambda) - log(X!))
        log_likelihood <- -lambda[, gene_j] + 
          log(lambda[, gene_j]) * expressionData[cell_i, gene_j] - 
          lgamma(expressionData[cell_i, gene_j] + 1)
        
        proposal_probabilities_Q <- exp(log_likelihood) + 1e-100
        
        # For the current cell, weight by the Gamma proposal density
        proposal_probabilities_Q[cell_i] <- concentration_M[gene_j] * gamma_densities[cell_i, gene_j]
        
        # Normalize
        proposal_probabilities_Q <- proposal_probabilities_Q / sum(proposal_probabilities_Q)
        
        # Sample an auxiliary cell index
        selected_cell_index <- sample(1:n_cells, size = 1, prob = proposal_probabilities_Q)
        
        if (selected_cell_index == cell_i) {
          # Propose new lambda from the base distribution
          lambda_proposal <- rgamma(1, 
                                     shape = gamma_shape_alpha + expressionData[cell_i, gene_j],
                                     rate = gamma_rate_beta + 1)
          lambda_candidate[cell_i, gene_j] <- lambda_proposal
          
          # --- Compute Metropolis-Hastings ratio ---
          # log likelihood for lambda proposal
          log_likelihood_new <- ComputeLogLikelihoodLambda(
            cell_i = cell_i,
            gene_j = gene_j,
            X = expressionData,
            lambda = lambda_candidate,
            beta = beta,
            edge_index = edge_index,
            Beta_matrix = Beta_matrix,
            power = power_parameter
          )
          
          # log likelihood for current lambda
          log_likelihood_old <- ComputeLogLikelihoodLambda(
            cell_i = cell_i,
            gene_j = gene_j,
            X = expressionData,
            lambda = lambda,
            beta = beta,
            edge_index = edge_index,
            Beta_matrix = Beta_matrix,
            power = power_parameter
          )
          
          # Prior ratio
          log_prior_new <- dgamma(lambda_proposal, 
                                   shape = gamma_shape_alpha, 
                                   rate = gamma_rate_beta, 
                                   log = TRUE)
          log_prior_old <- dgamma(lambda[cell_i, gene_j], 
                                   shape = gamma_shape_alpha, 
                                   rate = gamma_rate_beta, 
                                   log = TRUE)
          
          # Proposal ratio (symmetric, cancels out)
          log_proposal_new <- dgamma(lambda_proposal,
                                      shape = gamma_shape_alpha + expressionData[cell_i, gene_j],
                                      rate = gamma_rate_beta + 1,
                                      log = TRUE)
          log_proposal_old <- dgamma(lambda[cell_i, gene_j],
                                      shape = gamma_shape_alpha + expressionData[cell_i, gene_j],
                                      rate = gamma_rate_beta + 1,
                                      log = TRUE)
          
          # Metropolis-Hastings ratio
          log_MH_ratio <- (log_likelihood_new - log_likelihood_old) +
            (log_prior_new - log_prior_old) -
            (log_proposal_new - log_proposal_old)
          
          # Handle NaN or NA (can happen with numerical issues)
          if (is.na(log_MH_ratio) || is.nan(log_MH_ratio)) {
            log_MH_ratio <- 0  # Neutral: accept with 50% probability
          }
          
          # Accept or reject
          if (log(runif(1)) < log_MH_ratio) {
            lambda[, gene_j] <- lambda_candidate[, gene_j]
            accept_count_lambda <- accept_count_lambda + 1
          }
          
        } else {
          # Use lambda from another cell (Dirichlet Process clustering)
          lambda[cell_i, gene_j] <- lambda[selected_cell_index, gene_j]
          lambda_candidate[cell_i, gene_j] <- lambda[selected_cell_index, gene_j]
        }
        
        # Update pre-computed gamma density
        gamma_densities[cell_i, gene_j] <- dgamma(lambda[cell_i, gene_j],
                                                    shape = gamma_shape_alpha + expressionData[cell_i, gene_j],
                                                    rate = gamma_rate_beta + 1)
      }
      
      # --- Update concentration parameter M ---
      # Number of unique lambda values (number of clusters)
      n_unique_lambda <- length(unique(lambda[, gene_j]))
      
      # Sample auxiliary variable delta
      delta <- rbeta(1, concentration_M[gene_j], n_cells)
      
      # Update M from Gamma distribution
      concentration_M[gene_j] <- rgamma(1, 
                                         shape = 10 + n_unique_lambda, 
                                         rate = 10 - log(delta))
    }
    
    # Store lambda sample
    lambda_mcmc_samples[[iteration]] <- lambda
    
    # ==========================================================================
    # UPDATE BETA (Precision matrix elements) - One gene at a time
    # ==========================================================================
    # Uses blocked Gibbs sampler with multivariate normal proposals
    
    # Construct variance matrix for beta prior
    beta_prior_variance_matrix <- matrix(0, n_genes, n_genes)
    beta_prior_variance_matrix[lower.tri(beta_prior_variance_matrix)] <- 
      latent_indicators_Z * slab_variance_s1 + (1 - latent_indicators_Z) * spike_variance_s0
    beta_prior_variance_matrix <- beta_prior_variance_matrix + t(beta_prior_variance_matrix)
    
    Beta_matrix_candidate <- Beta_matrix
    
    for (gene_i in 1:n_genes) {
      
      # --- Construct proposal distribution for beta[i, -i] ---
      # This is a multivariate normal centered at the conditional mode
      
      # Mean of proposal: -crossprod_atan[i, -i]
      proposal_mean <- -crossprod_atan[gene_i, -gene_i]
      
      # Variance of proposal (from prior)
      proposal_variance_diag <- beta_prior_variance_matrix[gene_i, -gene_i]
      
      # Construct precision matrix for proposal
      # UNCERTAINTY NOTE: This is a complex calculation that can be numerically unstable
      proposal_precision_submatrix <- Beta_matrix[-gene_i, -gene_i]
      diag(proposal_precision_submatrix) <- crossprod_atan_inv_diag[-gene_i]
      
      # Eigendecomposition for inversion
      eigen_result <- eigen(proposal_precision_submatrix, symmetric = TRUE)
      eigen_values <- eigen_result$values
      eigen_vectors <- eigen_result$vectors
      
      # Invert using eigendecomposition (more stable)
      # Avoid division by very small eigenvalues
      eigen_values_inv <- 1 / pmax(abs(eigen_values), 1e-10)
      eigen_U_scaled <- t(eigen_vectors) / sqrt(abs(eigen_values_inv))
      precision_inverse <- crossprod(eigen_U_scaled)
      
      # Combine with prior variance
      proposal_precision <- (var(atan_data[, gene_i]) + lambdaShrinkage) * n_cells * 
        precision_inverse + 
        diag(1 / proposal_variance_diag)
      
      # Invert again to get proposal variance
      eigen_result2 <- eigen(proposal_precision, symmetric = TRUE)
      eigen_values2 <- eigen_result2$values
      eigen_vectors2 <- eigen_result2$vectors
      eigen_values2_inv <- 1 / pmax(abs(eigen_values2), 1e-10)
      eigen_U2_scaled <- t(eigen_vectors2) / sqrt(abs(eigen_values2_inv))
      proposal_variance <- crossprod(eigen_U2_scaled)
      
      # Sample beta proposal
      beta_proposal <- mvtnorm::rmvnorm(1, 
                                         mean = proposal_variance %*% proposal_mean,
                                         sigma = proposal_variance)
      beta_proposal <- as.vector(beta_proposal)
      
      # Handle NAs in proposal (can happen with numerical issues)
      if (any(is.na(beta_proposal))) {
        beta_proposal[is.na(beta_proposal)] <- 0
      }
      
      # Update candidate Beta matrix
      Beta_matrix_candidate[gene_i, -gene_i] <- beta_proposal
      Beta_matrix_candidate[-gene_i, gene_i] <- beta_proposal
      
      # Extract beta vector for candidate
      beta_candidate <- Beta_matrix_candidate[lower.tri(Beta_matrix_candidate)]
      
      # --- Compute Metropolis-Hastings ratio ---
      # Log likelihood for beta proposal
      log_likelihood_new <- ComputeLogLikelihoodBeta(
        gene_j = gene_i,
        X = expressionData,
        lambda = lambda,
        Beta_matrix = Beta_matrix_candidate,
        edge_index = edge_index,
        power = power_parameter
      )
      
      # Log likelihood for current beta
      log_likelihood_old <- ComputeLogLikelihoodBeta(
        gene_j = gene_i,
        X = expressionData,
        lambda = lambda,
        Beta_matrix = Beta_matrix,
        edge_index = edge_index,
        power = power_parameter
      )
      
      # Prior ratio
      log_prior_new <- sum(dnorm(Beta_matrix_candidate[gene_i, -gene_i],
                                  mean = 0,
                                  sd = sqrt(beta_prior_variance_matrix[gene_i, -gene_i]),
                                  log = TRUE))
      log_prior_old <- sum(dnorm(Beta_matrix[gene_i, -gene_i],
                                  mean = 0,
                                  sd = sqrt(beta_prior_variance_matrix[gene_i, -gene_i]),
                                  log = TRUE))
      
      # Proposal ratio
      log_proposal_new <- mvtnorm::dmvnorm(Beta_matrix[-gene_i, gene_i],
                                            mean = proposal_variance %*% proposal_mean,
                                            sigma = proposal_variance,
                                            log = TRUE)
      log_proposal_old <- mvtnorm::dmvnorm(beta_proposal,
                                            mean = proposal_variance %*% proposal_mean,
                                            sigma = proposal_variance,
                                            log = TRUE)
      
      # Metropolis-Hastings ratio
      log_MH_ratio <- (log_likelihood_new - log_likelihood_old) +
        (log_prior_new - log_prior_old) +
        (log_proposal_new - log_proposal_old)
      
      # Handle NaN or NA
      if (is.na(log_MH_ratio) || is.nan(log_MH_ratio)) {
        log_MH_ratio <- 0
      }
      
      # Accept or reject
      if (log(runif(1)) < log_MH_ratio) {
        beta <- beta_candidate
        Beta_matrix <- Beta_matrix_candidate
        accept_count_beta <- accept_count_beta + 1
      }
    }
    
    # Store beta sample
    beta_mcmc_samples[[iteration]] <- beta
    
    # Update progress bar
    if (verbose) {
      setTxtProgressBar(pb, iteration)
    }
  }
  
  if (verbose) {
    close(pb)
    message("MCMC sampling complete!")
  }
  
  # ============================================================================
  # COMPUTE ACCEPTANCE RATES
  # ============================================================================
  acceptance_rate_lambda <- accept_count_lambda / (totalIterations * n_genes * n_cells)
  acceptance_rate_beta <- accept_count_beta / (totalIterations * n_genes)
  
  if (verbose) {
    message(sprintf("Lambda acceptance rate: %.2f%%", acceptance_rate_lambda * 100))
    message(sprintf("Beta acceptance rate: %.2f%%", acceptance_rate_beta * 100))
  }
  
  # ============================================================================
  # RETURN RESULTS
  # ============================================================================
  # Return only post-burn-in samples
  post_burnin_indices <- (burnIn + 1):totalIterations
  
  result <- list(
    beta_mcmc = beta_mcmc_samples[post_burnin_indices],
    lambda_mcmc = lambda_mcmc_samples[post_burnin_indices],
    power_parameter = power_parameter,
    acceptance_rate_lambda = acceptance_rate_lambda,
    acceptance_rate_beta = acceptance_rate_beta,
    n_cells = n_cells,
    n_genes = n_genes,
    edge_index = edge_index
  )
  
  class(result) <- c("CONGAfit", "list")
  return(result)
}


#' Compute Log Likelihood for Lambda Parameter
#'
#' Helper function to compute the log likelihood for a single lambda parameter
#' in the CONGA model. This includes both the Poisson component and the
#' conditional dependency structure through beta.
#'
#' @param cell_i Cell index
#' @param gene_j Gene index  
#' @param X Data matrix
#' @param lambda Current lambda matrix
#' @param beta Current beta vector
#' @param edge_index Matrix of edge indices
#' @param Beta_matrix Full beta matrix
#' @param power Power parameter
#'
#' @return Log likelihood value
#'
#' @keywords internal
ComputeLogLikelihoodLambda <- function(cell_i, gene_j, X, lambda, beta, 
                                        edge_index, Beta_matrix, power) {
  
  # Poisson component: -lambda + X*log(lambda)
  log_poisson <- -lambda[cell_i, gene_j] + 
    X[cell_i, gene_j] * log(lambda[cell_i, gene_j]) +
    lambda[cell_i, gene_j]  # Add lambda back (part of exponential family)
  
  # Compute normalizing constant
  # This is: log(sum_k dpois(k, lambda) * exp(lambda + beta_sum * atan(k)^power))
  beta_sum <- sum(-Beta_matrix[gene_j, -gene_j] * (atan(X[cell_i, -gene_j])^power))
  
  log_normalizing_constant <- ComputeLogNormalizingConstant(
    lambda_val = lambda[cell_i, gene_j],
    beta_sum = beta_sum,
    power = power,
    max_val = 100
  )
  
  return(log_poisson - log_normalizing_constant)
}


#' Compute Log Likelihood for Beta Parameter
#'
#' Helper function to compute the log likelihood for beta parameters
#' for a specific gene in the CONGA model.
#'
#' @param gene_j Gene index
#' @param X Data matrix
#' @param lambda Current lambda matrix
#' @param Beta_matrix Full beta matrix
#' @param edge_index Matrix of edge indices
#' @param power Power parameter
#'
#' @return Log likelihood value
#'
#' @keywords internal
ComputeLogLikelihoodBeta <- function(gene_j, X, lambda, Beta_matrix, 
                                      edge_index, power) {
  
  n_cells <- nrow(X)
  log_likelihood_sum <- 0
  
  # Sum over all cells
  for (cell_i in 1:n_cells) {
    # Compute beta interaction term
    beta_sum <- sum(-Beta_matrix[gene_j, -gene_j] * (atan(X[cell_i, -gene_j])^power) * 
                      (atan(X[cell_i, gene_j])^power))
    
    # Compute normalizing constant
    beta_sum_for_norm <- sum(-Beta_matrix[gene_j, -gene_j] * (atan(X[cell_i, -gene_j])^power))
    
    log_normalizing_constant <- ComputeLogNormalizingConstant(
      lambda_val = lambda[cell_i, gene_j],
      beta_sum = beta_sum_for_norm,
      power = power,
      max_val = 100
    )
    
    log_likelihood_sum <- log_likelihood_sum + beta_sum - log_normalizing_constant
  }
  
  return(log_likelihood_sum)
}
