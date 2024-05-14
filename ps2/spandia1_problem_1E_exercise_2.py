from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.random import default_rng
import os
import sys

def neg_log_likelihood_gauss(g, X, r, sigma, sigma_prior):
    residuals = r - X @ g
    prior_term = (1 / (2 * sigma_prior**2)) * np.sum(g**2)
    return (1 / (2 * sigma**2)) * np.sum(residuals**2) + prior_term

def neg_log_likelihood_poiss(g, X, r, sigma, sigma_prior):
    residuals = r - X @ g
    prior_term = (1 / (2 * sigma_prior**2)) * np.sum(g**2)
    return (1 / (2 * sigma**2)) * np.sum(np.gradient(residuals)**2) + prior_term

def estimate_g_gauss(X, r, sigma, sigma_prior):
    g_initial = np.zeros(X.shape[1])
    result = minimize(neg_log_likelihood_gauss, g_initial, args=(X, r, sigma, sigma_prior), method='BFGS')
    return result.x

def estimate_g_poiss(X, r, sigma, sigma_prior):
    g_initial = np.zeros(X.shape[1])
    result = minimize(neg_log_likelihood_poiss, g_initial, args=(X, r, sigma, sigma_prior), method='BFGS')
    return result.x

def gausswin(N, alpha=5):
    n = np.arange(0, N) - (N - 1) / 2
    return np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)

def param_regime(N, M, sigma, sigma_prior, A):
  # Generate the true g vector for this N
  # Generate stimulus and responses
  X = 2 * np.random.rand(M, N) * A
  g = gausswin(N) * np.cos(2 * np.pi * np.arange(0, N) / 10)
  r = X @ g + np.random.normal(0, sigma, M)

  # Estimations
  estimated_g_gauss = estimate_g_gauss(X, r, sigma, sigma_prior)
  estimated_g_poiss = estimate_g_poiss(X, r, sigma, sigma_prior)


  # Plotting
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))

  title = f'Estimation Results with N={N}, M={M}, A={A:.1f}, sigma={sigma:.1f}, sigma_prior={sigma_prior:.1f}'
  fig.suptitle(title)

  # Gaussian prior estimation plot
  axes[0].plot(g, label='True g', linestyle='--', color='black')
  axes[0].plot(estimated_g_gauss, label='Gaussian Prior')
  axes[0].set_title('Gaussian Prior Estimation')
  axes[0].legend()

  # Poisson prior estimation plot
  axes[1].plot(g, label='True g', linestyle='--', color='black')
  axes[1].plot(estimated_g_poiss, label='Poisson Prior')
  axes[1].set_title('Poisson Prior Estimation')
  axes[1].legend()

  ## Histogram of spike rates
  #axes[2].hist(r_poiss, bins=30, color='gray')
  axes[2].set_title('Histogram of Spike Rates')
  axes[2].set_xlabel('Spike Count')
  axes[2].set_ylabel('Frequency')

  for i in range(M):
    curr_X = X[i]
    g = gausswin(N) * np.cos(2 * np.pi * np.arange(0, N) / 10)

    lmbda = np.exp(np.dot(g, A * curr_X))

    rng = default_rng()
    r_samples = rng.poisson(lmbda, 1000)
    axes[2].hist(r_samples, bins=30, alpha=0.5)
    axes[2].set_title('Histogram of Spike Rates')
    axes[2].set_xlabel('Spike Count')
    axes[2].set_ylabel('Frequency')

  plt.tight_layout()
  plt.show()

