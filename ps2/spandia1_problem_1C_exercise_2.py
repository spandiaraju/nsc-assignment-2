import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.random import default_rng

def neg_log_likelihood(g, X, r, sigma):
    residuals = r - X @ g
    return (1 / (2 * sigma**2)) * np.sum(residuals**2)

def estimate_g(X, r, sigma):
    g_initial = np.zeros(X.shape[1])
    result = minimize(neg_log_likelihood, g_initial, args=(X, r, sigma), method='BFGS')
    return result.x

def gausswin(N, alpha=5):
    n = np.arange(0, N) - (N - 1) / 2
    return np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)


def main():
    # Standard deviation of the Gaussian noise
    sigma = 0.2
    
    # Set of values for N and corresponding M
    N_values = [50, 100, 200]
    M_values_factors = [1, 2, 0.5]
    
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle('True vs Estimated Tuning Curve g for Different N and M')
    
    for i, N in enumerate(N_values):
        # Generate the true g vector for this N
        g = gausswin(N) * np.cos(2 * np.pi * np.arange(0, N) / 10)
    
        for j, factor in enumerate(M_values_factors):
            M = int(N * factor)
    
            # Generate data matrix X and observations
            X = 2 * np.random.rand(M, N)
            r = X @ g + np.random.normal(0, sigma, M)
    
            # Estimate g using the optimization approach
            estimated_g = estimate_g(X, r, sigma)
    
            # Plotting
            ax = axes[i, j]
            ax.plot(g, label='True g', linestyle='--', color='black')
            ax.plot(estimated_g, label=f'Estimated g for M={M}')
            ax.set_title(f'N = {N}, M = {M}')
            ax.set_xlabel('Component')
            ax.set_ylabel('Value')
            ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
