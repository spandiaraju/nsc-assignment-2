import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.spatial.distance
import sys
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import quantities as pq
from neo.core import SpikeTrain
from elephant.gpfa import GPFA


def convert_to_spiketrains(spike_data):
    num_trials, num_neurons, num_bins = spike_data.shape
    trials_spiketrains = []

    for trial in range(num_trials):
        trial_spiketrains = []
        for neuron in range(num_neurons):
            # Extract spike times based on counts
            spike_times = np.where(spike_data[trial, neuron] > 0)[0]  # Get indices where spikes occur
            spike_counts = spike_data[trial, neuron, spike_times]  # Get the spike counts at these times

            # Convert indices to times with units
            spike_times = spike_times * pq.ms

            # For each count greater than 1, we need to repeat the spike time
            repeated_spike_times = np.repeat(spike_times, spike_counts)

            # Create a SpikeTrain for each neuron
            spiketrain = SpikeTrain(repeated_spike_times, t_stop=num_bins * pq.ms)
            trial_spiketrains.append(spiketrain)

        trials_spiketrains.append(trial_spiketrains)

    return trials_spiketrains

def main():
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

    # Load data
    data = scipy.io.loadmat(sample_data_path)
    
    all_trial_dat = []
    for i in range(data['dat'][0].shape[0]):
      trial_dat = data['dat'][0][i][1]
      all_trial_dat.append(trial_dat)
    
    spike_data = np.array(all_trial_dat)

    formatted_data = convert_to_spiketrains(spike_data)

    # Initialize GPFA with specified parameters
    gpfa = GPFA(bin_size=1 * pq.ms, x_dim=3)
    
    # Run GPFA on the data
    fitted_model = gpfa.fit(formatted_data)
    
    # Transform data to extract latent variables
    extracted_dimensions = gpfa.transform(formatted_data)
    
    ex_dim = []
    for i in range(len(extracted_dimensions)):
      ex_dim.append(extracted_dimensions[i])
    
    ex_dim = np.array(ex_dim)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Assuming ex_dim is the array with shape (56, 3, 400)
    # 56 trials, 3 dimensions, 400 time points per trial
    
    # Set up the figure and the grid of subplots
    fig = plt.figure(figsize=(20, 20))
    
    # Iterate through each trial and plot its 3D trajectory
    for trial in range(56):
        ax = fig.add_subplot(8, 7, trial + 1, projection='3d')
    
        # Data for this trial
        data = ex_dim[trial]  # This will be 3 x 400
    
        # Plot the trajectory in 3D
        ax.plot(data[0], data[1], data[2])
    
        # Setting the labels for axes
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    
        # Setting the title to indicate the trial number
        ax.set_title(f'Trial {trial + 1}')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()