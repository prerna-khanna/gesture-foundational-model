import numpy as np
import os


def diagnose_dataset():
    """
    Diagnoses dataset issues and creates a fixed version if necessary
    """
   # Define paths
    data_path = 'dataset/blind_user/data_20_120.npy'
    label_path = 'dataset/blind_user/label_20_120.npy'
    fixed_data_path = 'dataset/blind_user_filtered/data_20_120.npy'
    fixed_label_path = 'dataset/blind_user_filtered/label_20_120.npy'
        
    print(f"Loading data from {data_path}")
    try:
        data = np.load(data_path)
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loading labels from {label_path}")
    try:
        labels = np.load(label_path)
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return
    
    # Check label data types and ranges
    print("\n=== Label Analysis ===")
    print(f"Label data type: {labels.dtype}")
    
    # Analyze the activity labels (first column, index 0)
    activity_labels = labels[:, 0, 0]  # First time step's activity label for each sample
    unique_activities = np.unique(activity_labels)
    print(f"Unique activity labels: {unique_activities}")
    print(f"Min activity label: {np.min(activity_labels)}")
    print(f"Max activity label: {np.max(activity_labels)}")
    
    # Count instances of each activity
    activity_counts = {}
    for act in unique_activities:
        activity_counts[act] = np.sum(activity_labels == act)
    
    print("\nActivity label counts:")
    for act, count in activity_counts.items():
        print(f"  Activity {act}: {count} samples")
    
    # Check for consistency within sequences (all time steps should have same label)
    print("\nChecking label consistency across time steps...")
    inconsistent_samples = 0
    for i in range(labels.shape[0]):
        if not np.all(labels[i, :, 0] == labels[i, 0, 0]):
            inconsistent_samples += 1
            print(f"  Sample {i} has inconsistent activity labels")
            if inconsistent_samples <= 5:  # Limit the output
                print(f"    Labels: {labels[i, :, 0]}")
    
    if inconsistent_samples == 0:
        print("  All samples have consistent activity labels across time steps.")
    else:
        print(f"  {inconsistent_samples} samples have inconsistent activity labels.")
    
    # Check for NaN or inf values in data
    print("\nChecking for NaN or inf values in data...")
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    # Fix the dataset if needed
    print("\n=== Dataset Fixes ===")
    
    # Problem 1: Check for label indices that are out of range (>=13)
    expected_max_label = 12  # 0-based indexing for 13 classes
    fix_needed = False
    
    if np.max(activity_labels) > expected_max_label:
        fix_needed = True
        print(f"Found labels greater than expected max ({expected_max_label})")
        
        # Create a mapping for fixing out-of-range labels
        # Option 1: Remap them to valid labels (recommended if these are just off-by-one errors)
        # Option 2: Filter out samples with invalid labels
        
        # We'll implement Option 2 here - filtering out invalid samples
        valid_indices = []
        for i in range(labels.shape[0]):
            if labels[i, 0, 0] <= expected_max_label:
                valid_indices.append(i)
        
        print(f"Keeping {len(valid_indices)} samples with valid labels")
        
        # Create fixed datasets
        fixed_data = data[valid_indices]
        fixed_labels = labels[valid_indices]
        
        # Save fixed datasets
        print(f"Saving fixed data to {fixed_data_path}")
        np.save(fixed_data_path, fixed_data)
        print(f"Saving fixed labels to {fixed_label_path}")
        np.save(fixed_label_path, fixed_labels)
        
        # Verify fixed labels
        fixed_activity_labels = fixed_labels[:, 0, 0]
        print(f"Fixed dataset - Min activity label: {np.min(fixed_activity_labels)}")
        print(f"Fixed dataset - Max activity label: {np.max(fixed_activity_labels)}")
    else:
        print("All labels are within expected range.")
    
    # Problem 2: Check for remapping needed to ensure labels are consecutive
    # Sometimes labels might be like [0,1,2,4,7] instead of [0,1,2,3,4]
    sorted_unique = np.sort(unique_activities)
    if not np.array_equal(sorted_unique, np.arange(len(sorted_unique))):
        fix_needed = True
        print("\nLabels are not consecutive.")
        
        # Create a mapping of old labels to new consecutive labels
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_unique)}
        print("Label remapping:")
        for old, new in label_map.items():
            print(f"  {old} -> {new}")
        
        # Apply the mapping to create fixed labels
        # If we already created fixed labels above, update those. Otherwise, start from original
        if 'fixed_labels' not in locals():
            fixed_labels = labels.copy()
            fixed_data = data
        
        for i in range(fixed_labels.shape[0]):
            old_label = fixed_labels[i, 0, 0]
            new_label = label_map[old_label]
            fixed_labels[i, :, 0] = new_label
        
        # Save fixed datasets if not already saved
        if not os.path.exists(fixed_data_path) or not os.path.exists(fixed_label_path):
            print(f"Saving fixed data to {fixed_data_path}")
            np.save(fixed_data_path, fixed_data)
            print(f"Saving fixed labels to {fixed_label_path}")
            np.save(fixed_label_path, fixed_labels)
        
        # Verify fixed labels
        fixed_activity_labels = fixed_labels[:, 0, 0]
        unique_fixed = np.unique(fixed_activity_labels)
        print(f"Fixed dataset - Unique activity labels: {unique_fixed}")
    else:
        print("Labels are already consecutive, no remapping needed.")
    
    if not fix_needed:
        print("\nNo fixes needed for this dataset.")

# Run the diagnostic function
if __name__ == "__main__":
    diagnose_dataset()