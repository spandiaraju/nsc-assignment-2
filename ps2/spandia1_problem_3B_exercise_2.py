import numpy as np
import scipy.io
import os
import sys
import matplotlib.pyplot as plt

def calcA(x, eps, dt=0.01):
    x2 = x[:, 1:];
    x1 = x[:, 0:-1]
    y = (x2 - x1) / dt + np.random.normal(0, eps, x2.shape)  # dt is bin width, or 10ms
    Xpinv = x1.T @ np.linalg.pinv(x1 @ x1.T)
    A = y @ Xpinv
    return A

def predict_next_activity(A, x, dt=0.01):
    return A @ x[:, 0:-1] * dt + x[:, 0:-1]

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


    time = time = data['Data'][0][0][1].T[0]
    # second index of Data (26) represents condition number
    
    all_data = []
    for i in range(len(data['Data'][0])):
      matrix = data['Data'][0][i][0].T
      all_data.append(matrix)
    
    all_data = np.array(all_data)
    
    cond_27 = all_data[26]

    baseline_data = all_data[:, :, time<=0]
    baseline_std = np.std(baseline_data, axis=2)
    eps = np.mean(baseline_std)
    print("Average standard deviation in signals is " + str(eps))
    
    all_errors = []
    all_A = []

    print("Now removing baseline data from dataset")

    all_data = all_data[:, :, time>=0]
    time = time[time>=0]

    print(all_data.shape)
    
    for trial_num in range(all_data.shape[0]):
      A = calcA(all_data[trial_num], eps=eps)
      all_A.append(A)
      predicted_activity = predict_next_activity(A, all_data[trial_num])
      actual_activity = all_data[trial_num][:, 1:]
      error = np.linalg.norm(predicted_activity - actual_activity, axis=0)
      all_errors.append(error)
    
    all_errors = np.array(all_errors)
    all_A = np.array(all_A)
    
    plt.hist(all_errors[26], alpha=0.7)
    plt.title('Histogram of Prediction Errors for Condition 27 Using A constructed via only Condition 27')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.show()
    
    curr_cond = 26
    
    # Setup for subplots - 4 rows and 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # Adjust figsize to fit your screen or preferences
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for curr_neuron in range(20):
        pred_dat = predict_next_activity(all_A[curr_cond], all_data[curr_cond])[curr_neuron, :]
        acc_dat = all_data[curr_cond][curr_neuron, 1:]
    
        # Plot on the corresponding subplot
        ax = axes[curr_neuron]
        ax.plot(pred_dat, label='Predicted')
        ax.plot(acc_dat, label='Actual')
        ax.set_title(f'Neuron {curr_neuron+1}')
        ax.legend()
    
    # Set an overall title for the figure
    fig.suptitle('Predicted vs. Actual Activity for the First 20 Neurons in Condition 27', fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    plt.hist(all_errors, alpha=0.7)
    plt.title('Histogram of Prediction Errors for Across All A constructed via only that Condition Data')
    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.show()
    
    # Example usage with multiple datasets
    datasets = [all_data[i] for i in range(all_data.shape[0])]  # Assuming 'all_data' is a list of all datasets
    A_multi = calcA_multi(datasets, eps=0.1)
    print("Shape of A:", A_multi.shape)
    
    all_errors = []
    for trial_num in range(all_data.shape[0]):
      predicted_activity = predict_next_activity(A_multi, all_data[trial_num])
      actual_activity = all_data[trial_num][:, 1:]
      error = np.linalg.norm(predicted_activity - actual_activity, axis=0)
      all_errors.append(error)
    
    all_errors = np.array(all_errors)
    
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
    
    # Setup for subplots - 4 rows and 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # Adjust figsize to fit your screen or preferences
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for curr_neuron in range(20):
        pred_dat = predict_next_activity(A_multi, all_data[curr_cond])[curr_neuron, :]
        acc_dat = all_data[curr_cond][curr_neuron, 1:]
    
        # Plot on the corresponding subplot
        ax = axes[curr_neuron]
        ax.plot(pred_dat, label='Predicted')
        ax.plot(acc_dat, label='Actual')
        ax.set_title(f'Neuron {curr_neuron+1}')
        ax.legend()
    
    # Set an overall title for the figure
    fig.suptitle('Predicted vs. Actual Activity for the First 20 Neurons in Condition 27 using A derived from whole dataset', fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


