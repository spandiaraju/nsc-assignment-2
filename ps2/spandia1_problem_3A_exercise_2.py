import numpy as np
import scipy.io
import os
import sys
import matplotlib.pyplot as plt

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

    plt.figure(figsize=(10, 8))
    plt.imshow(cond_27, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation level')
    plt.title('Neural Activity Heatmap')
    plt.xlabel('Timepoints')
    plt.ylabel('Neurons')
    plt.show()
