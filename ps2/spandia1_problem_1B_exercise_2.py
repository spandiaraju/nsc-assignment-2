import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from numpy.random import default_rng

def gausswin(N, alpha=5):
    n = np.arange(0, N) - (N - 1) / 2
    return np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)


def main():
    # Standard deviation of the Gaussian noise
    sigma = 0.2
    
    # Set of values for N and corresponding M
    N_values = [50, 100, 200]
    M_values_factors = [1, 2, 0.5]
    
    # Number of components in vector g
    g_size = max(N_values)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    fig.suptitle('True vs Estimated Tuning Curve g for Different N and M')
    
    for i, N in enumerate(N_values):
        for j, factor in enumerate(M_values_factors):
            g = gausswin(N) * np.cos(2 * np.pi * np.arange(0, N) / 10)
    
            M = int(N * factor)
    
            # Generate data matrix X and observations
            X = 2 * np.random.rand(M, N)
            true_r = X @ g[:N]  # Make sure g has the right dimensions
            observed_r = true_r + np.random.normal(0, sigma, M)
    
            # Estimate g using least squares
            estimated_g = np.linalg.pinv(X.T @ X) @ X.T @ observed_r
    
            # Plotting
            ax = axes[i, j]
            ax.plot(g[:N], label='True g')
            ax.plot(estimated_g, label='Estimated g', linestyle='--')
            ax.set_title(f'N = {N}, M = {M}')
            ax.set_xlabel('Component')
            ax.set_ylabel('Value')
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
