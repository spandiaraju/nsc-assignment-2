import os
import sys
import scipy
import numpy as np
import jPCA
import matplotlib.pyplot as plt
from jPCA.util import load_churchland_data, plot_projections
from sklearn.decomposition import PCA
import scipy.io
import seaborn as sns


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
    
# Define the prediction function
def predict_next_activity(A, x, dt=0.01):
    return A @ x * dt + x
    
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
    cc_mean = [np.mean(sublist) for sublist in datas]
    datas = [[value - mean for value in sublist] for sublist, mean in zip(datas, cc_mean)]

    times = np.array(times)
    datas = np.stack(datas, axis=0)
    datas = datas[:, times >= 0, :]
    num_conditions, num_time_bins, num_units = datas.shape
    
    
    # PCA transformation
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
    
    # Initial conditions
    initial_conditions = pca_data[:, :, 0]
    datas_predicted = np.zeros((108, 6, 56))
    datas_predicted[:, :, 0] = initial_conditions
    
    # Predict the activity over time
    for t in range(1, pca_data.shape[2]):
        for trial in range(pca_data.shape[0]):
            datas_predicted[trial, :, t] = predict_next_activity(A_multi_pca, datas_predicted[trial, :, t-1])
    
    # Plot the results
    trial_index = 26  # First trial
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for neuron_index in range(6):
        actual_data = pca_data[trial_index, neuron_index, :]
        predicted_data = datas_predicted[trial_index, neuron_index, :]
    
        ax = axes[neuron_index]
        ax.plot(actual_data, label='Actual')
        ax.plot(predicted_data, label='Predicted')
        ax.set_title(f'Component {neuron_index+1}')
        ax.legend()
    
    fig.suptitle('Predicted vs. Actual Component Activity for Condition 27', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(12, 9, figsize=(18, 24))
    axes = axes.flatten()
    
    # Create dummy plots for the legend
    lines = []
    labels = ["Actual", "Forwarded/Predicted"]
    line1, = plt.plot([], [], color='blue', label='Actual')
    line2, = plt.plot([], [], color='red', label='Forwarded/Predicted')
    lines.append(line1)
    lines.append(line2)
    
    for i in range(pca_data.shape[0]):
        # Extract the first two principal components
        actual_pc1 = pca_data[i, 0, :]
        actual_pc2 = pca_data[i, 1, :]
        predicted_pc1 = datas_predicted[i, 0, :]
        predicted_pc2 = datas_predicted[i, 1, :]
    
        # Plot actual data
        axes[i].plot(actual_pc1, actual_pc2, color='blue')
    
        # Plot predicted data
        axes[i].plot(predicted_pc1, predicted_pc2, color='red')
    
        axes[i].set_title(f'Trial {i+1}')
        axes[i].set_xlabel('PC 1')
        axes[i].set_ylabel('PC 2')
    
    fig.legend(lines, labels, loc='upper center', fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('First Two Principal Dimensions of the Projection for Each Trial', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
    # Plot the first two principal components for all forwarded/predicted data
    fig_predicted, ax_predicted = plt.subplots(figsize=(10, 8))
    
    for i in range(datas_predicted.shape[0]):
        # Extract the first two principal components
        predicted_pc1 = datas_predicted[i, 0, :]
        predicted_pc2 = datas_predicted[i, 1, :]
    
        # Plot all predicted data on the same axis
        ax_predicted.plot(predicted_pc1, predicted_pc2, color='red', alpha=0.3, label='Forwarded/Predicted' if i == 0 else "")
    
    # Customize the plot
    ax_predicted.set_title('First Two Principal Dimensions of the Projection (Forwarded/Predicted Data)')
    ax_predicted.set_xlabel('PC 1')
    ax_predicted.set_ylabel('PC 2')
    ax_predicted.legend()
    plt.show()
    
    # Plot the first two principal components for all actual data
    fig_actual, ax_actual = plt.subplots(figsize=(10, 8))
    
    for i in range(pca_data.shape[0]):
        # Extract the first two principal components
        actual_pc1 = pca_data[i, 0, :]
        actual_pc2 = pca_data[i, 1, :]
    
        # Plot all actual data on the same axis
        ax_actual.plot(actual_pc1, actual_pc2, color='blue', alpha=0.3, label='Actual' if i == 0 else "")
    
    # Customize the plot
    ax_actual.set_title('First Two Principal Dimensions of the Projection (Actual Data)')
    ax_actual.set_xlabel('PC 1')
    ax_actual.set_ylabel('PC 2')
    ax_actual.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(A_multi_pca, cmap='viridis', annot=False)
    plt.title('Heatmap of PCA-derived A, Dynamical Systems Matrix')
    plt.show()
