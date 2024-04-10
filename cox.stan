data {
  int<lower=0> N;          // Number of observations
  int<lower=0> p;          // Number of predictors
  matrix[N, p] X;          // Matrix of predictor variables
  vector<lower=0>[N] time; // Time to event or censoring
  int<lower=0, upper=1> event[N]; // Event indicator (1=event, 0=censored)
}

parameters {
  real intercept;          // Intercept
  vector[p] betas;         // Coefficients for predictors
}

model {
  // Priors for coefficients
  intercept ~ normal(-5, 2);
  betas ~ normal(0, 2);

  // Likelihood function for Cox proportional hazards model
  for (i in 1:N) {
    if (event[i] == 1) {   // Event occurred
      target += exponential_lpdf(time[i] | exp(intercept + dot_product(X[i], betas)));
    } else {               // Censored
      target += exponential_lccdf(time[i] | exp(intercept + dot_product(X[i], betas)));
    }
  }
}
