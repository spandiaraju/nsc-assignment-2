import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from numpy.random import default_rng

def gausswin(N, alpha=5):
    n = np.arange(0, N) - (N - 1) / 2
    return np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)

def main():
    N = 100  # Stim dimensionality

    g = gausswin(N) * np.cos(2 * np.pi * np.arange(0, N) / 10)
    
    for i in range(10):
        X = 2 * np.random.rand(N, 1)
        
        dot_product = np.dot(g, X)
        print(f"Dot product (g . X): {dot_product}")
        
        lmbda = np.exp(dot_product)
        print(f"Lambda: {lmbda}")
        
        # Generate samples from the Poisson distribution
        rng = default_rng()
        r_samples = rng.poisson(lmbda, 1000)
        
        print(f"Poisson samples: {r_samples[:10]}")  # Print first 10 samples for reference
        
        # Plot the distribution of r for a single draw of X as a histogram
        plt.hist(r_samples, bins=30)  # Adjust bins for better visualization
        plt.title('Histogram of Poisson samples')
        plt.xlabel('Number of spikes')
        plt.ylabel('Number of windows')
        plt.xlim(0, np.max(r_samples))
        plt.ylim(0, 1000)
        plt.show()

if __name__ == '__main__':
    main()
