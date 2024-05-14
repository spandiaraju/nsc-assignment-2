import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io
from jPCA.jPCA import JPCA
from jPCA.util import load_churchland_data, plot_projections
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

def predict_next_activity_M_skew(A, x, dt=0.01):
    return A @ x + x

def main():
    # Set up directories and load data
    current_working_dir = os.getcwd()
    data_folder_path = os.path.join(current_working_dir, 'data')
    
    # Append the data folder to the sys.path if it's not already there
    if data_folder_path not in sys.path:
        sys.path.append(data_folder_path)
    
    # Specify the path to your .mat file within the data folder
    example_data_path = os.path.join(data_folder_path, 'exampleData.mat')
    datas, times = load_churchland_data(example_data_path)
    cc_mean = [np.mean(sublist) for sublist in datas]
    datas = [[value - mean for value in sublist] for sublist, mean in zip(datas, cc_mean)]
    #cc_mean = np.mean(datas, axis=0, keepdims=True)
    #datas -= cc_mean
    
    ### PCA PART
    
    datas = np.stack(datas, axis=0)
    num_conditions, num_time_bins, num_units = datas.shape
    num_pcs = 6
    
    tstart = 0
    tend = 550
    idx_start = times.index(tstart)
    idx_end = times.index(tend) + 1 # Add one so idx is inclusive
    num_time_bins = idx_end - idx_start
    datas = datas[:, idx_start:idx_end, :]
    
    X_full = np.concatenate(datas)
    full_data_var = np.sum(np.var(X_full, axis=0))
    pca_variance_captured = None
    
    pca = PCA(num_pcs)
    datas = pca.fit_transform(X_full)
    datas = datas.reshape(num_conditions, num_time_bins, num_pcs)
    pca_variance_captured = pca.explained_variance_
    
    ###########
    
    datas, times = load_churchland_data(example_data_path)

    jpca = JPCA(num_jpcs=6)
    
    # Fit the jPCA object to data
    (projected, full_data_var, pca_var_capt, jpca_var_capt) = jpca.fit(datas, times=times, tstart=0, tend=550)
    
    
    projected_np = np.array(projected)
    full_data_var_np = np.array(full_data_var)
    pca_var_capt_np = np.array(pca_var_capt)
    jpca_var_capt_np = np.array(jpca_var_capt)
    
    pc1_projected = projected_np[:, :, 0]
    pc2_projected = projected_np[:, :, 1]
    
    projected_np_reshaped = projected_np.transpose(0, 2, 1)
    pc1_projected_reshaped = projected_np_reshaped[:, 0, :]
    pc2_projected_reshaped = projected_np_reshaped[:, 1, :]
    
    times = np.array(times)
    datas = np.stack(datas, axis=0)
    datas = datas[:, times >= 0, :]
    times = times[times >= 0]
    
    jpca_data = projected_np
    pca_data = jpca_data.transpose(0, 2, 1)
    
    A_multi_pca = jpca.M_skew.T
    # Initial conditions
    initial_conditions = pca_data[:, :, 0]
    datas_predicted = np.zeros((pca_data.shape[0], pca_data.shape[1], pca_data.shape[2]))
    datas_predicted[:, :, 0] = initial_conditions
    
    # Predict the activity over time
    for t in range(1, pca_data.shape[2]):
        for trial in range(pca_data.shape[0]):
            datas_predicted[trial, :, t] = predict_next_activity_M_skew(A_multi_pca, datas_predicted[trial, :, t-1])
    
    pc1_datas_predicted = datas_predicted[:, 0, :]
    pc2_datas_predicted = datas_predicted[:, 1, :]
    
    times = np.array(times)
    datas = np.stack(datas, axis=0)
    datas = datas[:, times >= 0, :]
    times = times[times >= 0]
    
    jpca_data = projected_np
    pca_data = jpca_data.transpose(0, 2, 1)
    
    A_multi_pca = jpca.M_skew.T
    # Initial conditions
    initial_conditions = pca_data[:, :, 0]
    datas_predicted = np.zeros((pca_data.shape[0], pca_data.shape[1], pca_data.shape[2]))
    datas_predicted[:, :, 0] = initial_conditions
    
    # Predict the activity over time
    for t in range(1, pca_data.shape[2]):
        for trial in range(pca_data.shape[0]):
            datas_predicted[trial, :, t] = predict_next_activity_M_skew(A_multi_pca, datas_predicted[trial, :, t-1])
    
    pc1_datas_predicted = datas_predicted[:, 0, :]
    pc2_datas_predicted = datas_predicted[:, 1, :]
    
    pca_data = projected_np_reshaped
    
    
    # Plot the results
    trial_index = 0  # First trial
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
    
    fig.suptitle('Predicted vs. Actual Component Activity for the First Trial in Condition 27', fontsize=16, y=1.02)
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
    
    M = jpca.M_skew
    jpca_eigvals, jpca_eigvecs = np.linalg.eig(M)
    
    pca_variance = pca.explained_variance_
    jpca_variance = jpca_var_capt
    
    pca_components = pca.components_
    jpca_components = jpca.jpcs
    
    # Plotting the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca_variance) + 1), pca_variance, color='blue', alpha=0.7)
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by PCA Component')
    plt.xticks(range(1, len(pca_variance) + 1))
    plt.show()
    
    # Plotting the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(jpca_variance) + 1), jpca_variance, color='blue', alpha=0.7)
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by jPCA Component')
    plt.xticks(range(1, len(jpca_variance) + 1))
    plt.show()
    
    # Plotting the heatmap with components on the x-axis, features on the y-axis, and no labels on the y-axis
    plt.figure(figsize=(10, 6))
    sns.heatmap(pca_components.T, cmap='viridis', cbar=True, xticklabels=[f'PC {i+1}' for i in range(pca_components.shape[0])], yticklabels=False)
    plt.xlabel('PCA Components')
    plt.ylabel('Features')
    plt.title('Heatmap of PCA Components')
    plt.show()
    
    # Plotting the heatmap with components on the x-axis, features on the y-axis, and no labels on the y-axis
    plt.figure(figsize=(10, 6))
    sns.heatmap(jpca_components.T, cmap='viridis', cbar=True, xticklabels=[f'PC {i+1}' for i in range(jpca_components.shape[0])], yticklabels=False)
    plt.xlabel('jPCA Components')
    plt.ylabel('Features')
    plt.title('Heatmap of jPCA Components')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(jpca.M_skew.T, cmap='viridis', annot=False)
    plt.title('Heatmap of jPCA-derived A (M Skew), Dynamical Systems Matrix')
    plt.show()

if __name__ == '__main__':
    main()