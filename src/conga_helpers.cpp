// CONGA Helper Functions in C++ for Performance
// These functions are used to optimize the power parameter selection in CONGA algorithm
// Based on pocal.cpp from https://github.com/royarkaprava/CONGA

#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace arma;

//' Select Optimal Power Parameter for CONGA
//'
//' This function selects the optimal power parameter (theta) by finding the value
//' that minimizes the distance between the covariance of atan-transformed data
//' and the original covariance. This transformation is used in CONGA to handle
//' non-Gaussian data.
//'
//' @param po Initial power parameter (typically the maximum value in the data matrix)
//' @param X Data matrix (cells x genes)
//' @return Optimal power parameter value
//'
//' @details
//' The algorithm:
//' 1. Tests power parameters from 0.1 to 2*po in increments of 0.1
//' 2. For each power, computes: sum((cov(atan(X)^power) - cov(X))^2)
//' 3. Stops early if the distance increases for 10 consecutive iterations
//' 4. Returns the power that minimizes the distance
//'
//' UNCERTAINTY NOTE: This is a heuristic method for selecting the transformation
//' parameter. The paper uses this to balance between capturing dependencies and
//' maintaining computational tractability.
//'
//' @export
// [[Rcpp::export]]
double SelectPowerParameter(int po, const arma::mat& X) {
  // Allocate vector to store test results for different power parameters
  arma::vec test_results = arma::zeros(2 * po);
  
  // Counter for consecutive increases in distance
  // Used to stop early if we've passed the minimum
  int consecutive_increases = 0;
  
  // Test different power parameters from 0.1 to 2*po
  for(int i = 0; i < (2 * po); i++){
    // Current power parameter being tested
    double current_power = (i + 1) / 10.0;
    
    // Compute covariance of atan-transformed data raised to current power
    // atan() is used to bound the data and handle extreme values
    arma::mat atan_X_powered = arma::pow(arma::atan(X), current_power);
    arma::mat cov_atan = arma::cov(atan_X_powered);
    
    // Compute covariance of original data
    arma::mat cov_X = arma::cov(X);
    
    // Calculate sum of squared differences between covariances
    // This measures how well the transformation preserves the covariance structure
    arma::mat diff = cov_atan - cov_X;
    test_results(i) = arma::accu(arma::pow(diff, 2));
    
    // Check if distance is increasing (we may have passed the minimum)
    if(i > 0){
      if(test_results(i) > test_results(i - 1)){
        consecutive_increases = consecutive_increases + 1;
      } else {
        consecutive_increases = 0;
      }
    }
    
    // Early stopping: if distance has been increasing for 10 iterations, stop
    // This avoids unnecessary computation once we've found the region of the minimum
    if(consecutive_increases > 10){
      break;
    }
  }
  
  // Find the index of the minimum test result (only considering positive values)
  arma::uvec positive_indices = arma::find(test_results > 0);
  double min_power = 1.0;  // Default to 1.0 if something goes wrong
  
  if(positive_indices.n_elem > 0){
    arma::vec positive_results = test_results.elem(positive_indices);
    arma::uword min_idx = arma::index_min(positive_results);
    min_power = (positive_indices(min_idx) + 1) / 10.0;
  }
  
  return min_power;
}


//' Compute Arctangent Mean for Poisson Distribution
//'
//' Computes the expected value of atan(x)^power under a Poisson distribution
//' with parameter theta. This is used in the CONGA algorithm for computing
//' normalizing constants.
//'
//' @param theta Poisson parameter (intensity)
//' @param power Power parameter for the transformation
//' @param max_val Maximum value to sum over (default 100)
//' @return Expected value of atan(x)^power under Poisson(theta)
//'
//' @details
//' Computes: E[atan(X)^power] where X ~ Poisson(theta)
//' Approximated by summing over 0 to max_val
//'
//' UNCERTAINTY NOTE: The truncation at max_val=100 is a practical approximation.
//' For large theta, this may introduce small errors.
//'
//' @export
// [[Rcpp::export]]
double ComputeAtanMean(double theta, double power, int max_val = 100) {
  double sum = 0.0;
  
  // Sum over possible values 0 to max_val
  for(int x = 0; x <= max_val; x++){
    // Compute atan(x)^power
    double atan_x_powered = std::pow(std::atan(static_cast<double>(x)), power);
    
    // Compute Poisson probability: P(X=x) = exp(-theta) * theta^x / x!
    // Using R::dpois for numerical stability
    double poisson_prob = R::dpois(static_cast<double>(x), theta, 0);
    
    // Add weighted term to sum
    sum += atan_x_powered * poisson_prob;
  }
  
  return sum;
}


//' Compute Log Normalizing Constant for CONGA Likelihood
//'
//' Computes the log normalizing constant for a single observation in the
//' CONGA model. This is used in the MCMC sampler.
//'
//' @param lambda_val Current lambda (Poisson intensity) value
//' @param beta_sum Sum of beta[j,-j] * atan(X[i,-j])^power terms
//' @param power Power parameter for transformation
//' @param max_val Maximum value to sum over in approximation
//' @return Log of the normalizing constant
//'
//' @details
//' Computes: log(sum_{k=0}^{max_val} dpois(k, lambda) * exp(lambda + beta_sum * atan(k)^power))
//'
//' UNCERTAINTY NOTE: This is a key computational bottleneck. The approximation
//' quality depends on max_val being large enough relative to lambda.
//'
//' @export
// [[Rcpp::export]]
double ComputeLogNormalizingConstant(double lambda_val, 
                                      double beta_sum, 
                                      double power,
                                      int max_val = 100) {
  double sum = 0.0;
  
  for(int k = 0; k <= max_val; k++){
    // Poisson probability mass
    double poisson_density = R::dpois(static_cast<double>(k), lambda_val, 0);
    
    // Exponential term with beta interaction
    double atan_k_powered = std::pow(std::atan(static_cast<double>(k)), power);
    double exp_term = std::exp(lambda_val + beta_sum * atan_k_powered);
    
    sum += poisson_density * exp_term;
  }
  
  // Return log of sum for numerical stability
  return std::log(sum);
}
