import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.spatial.distance
import sys
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def gp_smoothing(A, l, x):
    '''
    Smooths the input data using a Gaussian Process (GP) with a specified kernel.

    :param A: float, scalar hyperparameter of the GP kernel, controls the variance
    :param l: float, length scale of the GP kernel, controls the smoothness
    :param x: 1-d numpy array, the data to be smoothed
    :return: 1-d numpy array, the GP smoothed data
    '''
    # Time points, assuming uniform spacing and appropriately scaled
    t = np.expand_dims(np.linspace(0, len(x), x.shape[0]), 1)/1000

    # Compute the squared Euclidean distance matrix for time points
    d = scipy.spatial.distance.cdist(t, t, 'sqeuclidean')

    # Compute the kernel matrix
    K = A * np.exp(-d/l)  # Adjusted kernel computation

    # Compute the inverse of the kernel matrix
    invK = np.linalg.pinv(K)

    # Fit the GP for predictive distribution such that
    mu = K.T.dot(invK).dot(x)  # mean of the predictive distribution
    cov = K - K.T.dot(invK).dot(K)  # covariance of the predictive distribution

    # Sample from the multivariate normal distribution defined by the predictive distribution
    gp_smooth = np.random.multivariate_normal(mean=mu, cov=cov, size=1)

    # Reshape the output to match the size of the input array
    gp_smooth = gp_smooth.reshape(x.size)

    return gp_smooth

def normalize(data):
    """ Normalize data to have mean 0 and standard deviation 1 along each row (component). """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def pca_3comp(input_data):

  # Assuming smoothed_psth_direct is your smoothed data with shape (53, 400)
  data_transposed = input_data.T  # Transpose to shape (400, 53)

  # Initialize PCA with 3 components
  pca = PCA(n_components=3)

  # Fit PCA on the transposed data
  pca.fit(data_transposed)

  # Transform the data using PCA
  reduced_data = pca.transform(data_transposed)  # This will have shape (400, 3)

  # Optional: If you want it back in the shape (3, 400)
  reduced_data_transposed = reduced_data.T

  return reduced_data_transposed
    
def main():

    A = 5  # Amplitude
    l = 0.02  # Length scale


    current_working_dir = os.getcwd()

    # Construct the path to the data folder
    data_folder_path = os.path.join(current_working_dir, 'data')
    
    # Append the data folder to the sys.path if it's not already there
    if data_folder_path not in sys.path:
        sys.path.append(data_folder_path)
    
    # Specify the path to your .mat file within the data folder
    sample_data_path = os.path.join(data_folder_path, 'sample_dat.mat')
    

    data = scipy.io.loadmat(sample_data_path)
    
    all_trial_dat = []
    for i in range(data['dat'][0].shape[0]):
      trial_dat = data['dat'][0][i][1]
      all_trial_dat.append(trial_dat)
    
    spike_data = np.array(all_trial_dat) # Trials x # Neurons x # Timepoints
    
    # Assuming spike_data is already defined and has the shape (Trials, 53, Timepoints)
    psth = spike_data.sum(axis=0)  # Average over trials
    
    # Assuming the timepoints are uniformly spaced and start from 0
    timepoints = np.arange(psth.shape[1])
    
    A = 5.0  # Amplitude
    l = 0.02  # Length scale
    
    smoothed_psth = []
    for x in psth:
      # Smooth the data
      smoothed_data = gp_smoothing(A, l, x)
      smoothed_psth.append(smoothed_data)
    
    smoothed_psth = np.array(smoothed_psth)
    
    n_components = 3

    normalized_raw = normalize(psth)  # Transpose if necessary to shape (3, 400)
    normalized_smooth = normalize(smoothed_psth)
    
    normalized_pca_smooth = pca_3comp(normalized_smooth)
    normalized_pca_raw = pca_3comp(normalized_raw)
    
    # Create subplots for each of the first three components
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Titles for plots
    titles = ['Component 0', 'Component 1', 'Component 2']
    
    for i in range(3):
        axes[i].bar(timepoints, normalized_pca_raw[i], label='Raw')
        axes[i].bar(timepoints, normalized_pca_smooth[i], label='Smoothed')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Timepoints')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    normalized_raw = normalize(psth)  # Transpose if necessary to shape (3, 400)
    normalized_smooth = normalize(smoothed_psth)
    
    normalized_pca_smooth = pca_3comp(normalized_smooth)
    normalized_pca_raw = pca_3comp(normalized_raw)
    
    # Create a figure with two subplots (side by side) for 3D plotting
    fig = plt.figure(figsize=(14, 6))
    
    # Add a 3D subplot for raw data
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(normalized_pca_raw[0], normalized_pca_raw[1], normalized_pca_raw[2], label='Raw')
    ax1.set_title('3D Line Plot for Raw Data')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')
    ax1.legend()
    
    # Add a 3D subplot for smoothed data
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(normalized_pca_smooth[0], normalized_pca_smooth[1], normalized_pca_smooth[2], label='Smoothed')
    ax2.set_title('3D Line Plot for Smoothed Data')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()