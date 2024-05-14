import os
import sys
import scipy
import numpy as np
import jPCA
import matplotlib.pyplot as plt
from jPCA.util import load_churchland_data, plot_projections
from sklearn.decomposition import PCA
import scipy.io

def calcA_multi(datasets, eps, dt=0.01):
    sum_x1T_x1 = 0
    sum_y_x1T = 0

    for x in datasets:
        x2 = x[:, 1:]
        x1 = x[:, :-1]
        y = (x2 - x1) / dt + np.random.normal(0, eps, x2.shape)

        x1T_x1 = x1 @ x1.T
        y_x1T = y @ x1.T

        sum_x1T_x1 += x1T_x1
        sum_y_x1T += y_x1T

    Xpinv = np.linalg.pinv(sum_x1T_x1)
    A = sum_y_x1T @ Xpinv
    return A

def predict_next_activity(A, x, dt=0.01):
    return A @ x[:, 0:-1] * dt + x[:, 0:-1]

def main():
    current_working_dir = os.getcwd()
    
    # Construct the path to the data folder
    data_folder_path = os.path.join(current_working_dir, 'data')
    
    # Append the data folder to the sys.path if it's not already there
    if data_folder_path not in sys.path:
        sys.path.append(data_folder_path)
    
    # Specify the path to your .mat file within the data folder
    example_data_path = os.path.join(data_folder_path, 'exampleData.mat')
    
    data = scipy.io.loadmat(example_data_path)
    time = data['Data'][0][0][1].T[0]
    # second index of Data (26) represents condition number
    
    all_data = []
    for i in range(len(data['Data'][0])):
      matrix = data['Data'][0][i][0].T
      all_data.append(matrix)
    
    all_data = np.array(all_data)
    
    # Example usage with multiple datasets
    datasets = [all_data[i] for i in range(all_data.shape[0])]  # Assuming 'all_data' is a list of all datasets
    datasets = np.array(datasets)
    
    datasets = datasets[:, :, time>=0]
    time = time[time>=0]
    
    datas, times = load_churchland_data(example_data_path)
    times = np.array(times)
    datas = np.stack(datas, axis=0)
    datas = datas[:, times >= 0, :]
    num_conditions, num_time_bins, num_units = datas.shape
    X_full = np.concatenate(datas, axis=0)
    num_pcs = 6
    pca = PCA(num_pcs)
    pca_data = pca.fit_transform(X_full)
    pca_variance_captured = pca.explained_variance_
    pca_data_reconstructed = pca.inverse_transform(pca_data)
    pca_data = pca_data.reshape(num_conditions, num_time_bins, num_pcs)
    pca_data = np.transpose(pca_data, (0, 2, 1))  # Reshape to (conditions, pcs, time_bins)
    
    # Assume calcA_multi is defined and pca_data is available
    A_multi_pca = calcA_multi(pca_data, eps=0.44)
    
    datas = pca_data
    
    all_errors = []
    
    all_predicted_activity = []
    all_actual_activity = []
    
    for trial_num in range(datas.shape[0]):
      predicted_activity = predict_next_activity(A_multi_pca, datas[trial_num])
      actual_activity = datas[trial_num][:, 1:]
    
      all_predicted_activity.append(predicted_activity)
      all_actual_activity.append(actual_activity)
        
      error = np.linalg.norm(predicted_activity - actual_activity, axis=0)
      all_errors.append(error)
    
    all_errors = np.array(all_errors)
    all_predicted_activity = np.array(all_predicted_activity)
    all_actual_activity = np.array(all_actual_activity)
    
    plt.hist(all_errors[26], alpha=0.7)
    plt.title('Histogram of Prediction Errors for Only Condition 27 via A derived from All Datasets together')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.hist(all_errors, alpha=0.7)
    plt.title('Histogram of Prediction Errors Across all Conditions via A derived from All Datasets together')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.show()
    
    predicted_activity_27 = all_predicted_activity[26, :, :]
    actual_activity_27 = all_actual_activity[26, :, :]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))  # Adjust figsize to fit your screen or preferences
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for curr_neuron in range(6):
        # Extract the data for the current neuron
        pred = predicted_activity_27[curr_neuron, :]
        acc = actual_activity_27[curr_neuron, :]
     
        # Plot on the corresponding subplot
        ax = axes[curr_neuron]
        ax.plot(pred, label='Predicted')
        ax.plot(acc, label='Actual')
        ax.set_title(f'Neuron {curr_neuron+1}')
        ax.legend()
    
    # Set an overall title for the figure
    fig.suptitle('Predicted vs. Actual Activity for 6 Neurons in Condition 27', fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()