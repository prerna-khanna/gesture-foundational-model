import numpy as np
import os
from utils import set_seeds

def load_combined_datasets(dataset_version='20_120', seed=42):
    """
    Load and combine UCI, HHAR, and MotionSense datasets.
    Handles different label dimensions by focusing on activity labels.
    
    Args:
        dataset_version (str): Version of datasets to load (e.g., '20_120')
        seed (int): Random seed for reproducibility
        
    Returns:
        combined_data: Combined sensor data from all datasets
        combined_labels: Combined labels from all datasets, aligned on activity labels
    """
    set_seeds(seed)
    datasets = ['uci', 'hhar', 'motion']
    all_data = []
    all_labels = []
    
    for dataset in datasets:
        try:
            # Load data and labels from each dataset
            data_path = os.path.join('dataset', dataset, f'data_{dataset_version}.npy')
            label_path = os.path.join('dataset', dataset, f'label_{dataset_version}.npy')
            
            if os.path.exists(data_path) and os.path.exists(label_path):
                data = np.load(data_path).astype(np.float32)
                labels = np.load(label_path).astype(np.float32)
                
                # Ensure data has consistent dimensions (use only accelerometer and gyroscope)
                if data.shape[2] > 6:
                    data = data[:, :, :6]  # Keep only acc and gyro data
                
                # Extract only the activity labels (first label dimension)
                # Reshape to match expected dimensions (N, W, 1)
                activity_labels = labels[:, :, 0:1]  # Take only first label (activity) and keep dimension
                
                print(f"Loaded {dataset} dataset:")
                print(f"  Data shape: {data.shape}")
                print(f"  Original labels shape: {labels.shape}")
                print(f"  Processed activity labels shape: {activity_labels.shape}")
                
                all_data.append(data)
                all_labels.append(activity_labels)
                
            else:
                print(f"Warning: Could not find {dataset} dataset files")
                
        except Exception as e:
            print(f"Error loading {dataset} dataset: {str(e)}")
            
    # Combine all datasets
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Shuffle the combined dataset
    indices = np.arange(combined_data.shape[0])
    np.random.shuffle(indices)
    combined_data = combined_data[indices]
    combined_labels = combined_labels[indices]
    
    print("\nCombined dataset:")
    print(f"  Data shape: {combined_data.shape}")
    print(f"  Labels shape: {combined_labels.shape}")
    
    return combined_data, combined_labels