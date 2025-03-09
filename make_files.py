import os
import numpy as np
import glob

def process_accel_data(base_dir):
    """
    Process acceleration data from a hierarchical folder structure.
    
    Args:
        base_dir: Base directory containing user folders
        
    Returns:
        data_array: NumPy array of shape (n, 120, 6)
        labels_array: NumPy array of shape (n, 120, 2) - 3D structure
    """
    all_data = []
    all_labels = []
    
    # Find all user directories (starting with 'U')
    user_dirs = sorted(glob.glob(os.path.join(base_dir, 'U*')))
    
    for user_dir in user_dirs:
        # Extract user index from folder name (e.g., 'U01' -> 1)
        user_idx = int(os.path.basename(user_dir)[1:])
        
        # Find all gesture directories
        gesture_dirs = sorted(glob.glob(os.path.join(user_dir, '*')))
        
        for gesture_dir in gesture_dirs:
            # Extract gesture index from folder name (e.g., '01' -> 1)
            gesture_idx = int(os.path.basename(gesture_dir))
            
            # Find all text files in this gesture directory
            txt_files = glob.glob(os.path.join(gesture_dir, '*.txt'))
            
            for txt_file in txt_files:
                # Process the text file
                sample_data = process_txt_file(txt_file)
                
                # Create a 3D label array: repeat [gesture_idx, user_idx] 120 times
                # This creates a (120, 2) array for each sample, matching the time dimension
                sample_label = np.array([[gesture_idx, user_idx]] * 120)
                
                # Append to our lists
                all_data.append(sample_data)
                all_labels.append(sample_label)
    
    # Convert lists to NumPy arrays
    data_array = np.array(all_data)
    labels_array = np.array(all_labels)
    
    return data_array, labels_array

def process_txt_file(file_path):
    """
    Process a single text file containing acceleration data.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        padded_data: NumPy array of shape (120, 6) where the 3-axis accel data is repeated
    """
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract acceleration data (last 3 columns)
    accel_data = []
    for line in lines:
        columns = line.strip().split()
        
        # Last three columns are acceleration data
        x = float(columns[-3])
        y = float(columns[-2])
        z = float(columns[-1])
        
        accel_data.append([x, y, z])
    
    # Convert to NumPy array
    accel_array = np.array(accel_data)
    
    # Pad or truncate to 120 samples such that gesture is in the middle
    if len(accel_array) < 120:
        # Pad with zeros such that accel_array is in the middle
        padded_data = np.zeros((120, 3))
        start_idx = (120 - len(accel_array)) // 2
        padded_data[start_idx:start_idx + len(accel_array)] = accel_array
    else:
        # Truncate to 120 samples
        padded_data = accel_array[:120]
    
    # Repeat the 3-axis accel data to make it 6 columns
    # Create an array of shape (120, 6) by repeating each row
    repeated_data = np.zeros((120, 6))
    repeated_data[:, 0:3] = padded_data  # First 3 columns
    repeated_data[:, 3:6] = padded_data  # Repeat for next 3 columns
    
    return repeated_data

def main():
    # Specify the base directory containing your data
    base_dir = '/Users/tanmay-s/Downloads/gestures-dataset'  # Replace with your actual directory path
    
    # Process all data
    data_array, labels_array = process_accel_data(base_dir)
    
    # Print shapes
    print(f"Data shape: {data_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    # Save as NumPy files
    np.save('/Users/tanmay-s/Downloads/gestures-dataset/data.npy', data_array)
    np.save('/Users/tanmay-s/Downloads/gestures-dataset/labels.npy', labels_array)
    
    print(f"Saved data.npy ({data_array.shape}) and labels.npy ({labels_array.shape})")

if __name__ == '__main__':
    main()