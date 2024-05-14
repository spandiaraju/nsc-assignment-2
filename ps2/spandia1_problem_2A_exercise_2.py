import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.spatial.distance
import sys
import os

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
    
    fig, axes = plt.subplots(8, 7, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle('PSTH of Each Neuron')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Assuming the timepoints are uniformly spaced and start from 0
    timepoints = np.arange(psth.shape[1])
    
    # Plot PSTH for each neuron
    for i in range(53):
        axes[i].bar(timepoints, psth[i], width=0.8)  # Adjust the bar width as needed
        axes[i].set_title(f'Neuron {i+1}')
        axes[i].set_xlabel('Timepoints')
        axes[i].set_ylabel('Total Spike Count')
    
    # Hide the empty subplots if any (in this case, 3 subplots will be empty)
    for i in range(53, 56):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    A = 5.0  # Amplitude
    l = 0.02  # Length scale
    
    smoothed_psth = []
    for x in psth:
      # Smooth the data
      smoothed_data = gp_smoothing(A, l, x)
      smoothed_psth.append(smoothed_data)
    
    smoothed_psth = np.array(smoothed_psth)
    
    fig, axes = plt.subplots(8, 7, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle('Smoothed PSTH of Each Neuron')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot PSTH for each neuron
    for i in range(53):
        axes[i].bar(timepoints, smoothed_psth[i], width=0.8)  # Adjust the bar width as needed
        axes[i].set_title(f'Neuron {i+1}')
        axes[i].set_xlabel('Timepoints')
        axes[i].set_ylabel('Total Spike Count')
    
    # Hide the empty subplots if any (in this case, 3 subplots will be empty)
    for i in range(53, 56):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()