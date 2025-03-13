import numpy as np
import os

def filter_and_remap_gestures():
    # Define paths
    data_path = 'dataset/earbud/data_20_120.npy'
    label_path = 'dataset/earbud/label_20_120.npy'
    filtered_data_path = 'dataset/earbud_filtered/data_20_120.npy'
    filtered_label_path = 'dataset/earbud_filtered/label_20_120.npy'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filtered_data_path), exist_ok=True)
    
    # Load data and labels
    data = np.load(data_path)
    labels = np.load(label_path)
    
    # Check data types before processing
    print(f"Original data type: {data.dtype}")
    print(f"Original labels type: {labels.dtype}")
    
    # Analyze original dataset
    print("\n=== Original Dataset Analysis ===")
    activity_labels = labels[:, 0, 0]  # Get first time step's activity label for each sample
    unique_activities = np.unique(activity_labels)
    print(f"Unique activity labels: {unique_activities}")
    
    # Step 1: Filter out gesture IDs 9 and 11
    gesture_ids_to_remove = [2, 5, 10, 11]
    print(f"\nRemoving gesture IDs: {gesture_ids_to_remove}")
    
    indices_to_keep = []
    for i in range(len(labels)):
        gesture_id = labels[i, 0, 0]
        if gesture_id not in gesture_ids_to_remove:
            indices_to_keep.append(i)
    
    filtered_data = data[indices_to_keep]
    filtered_labels = labels[indices_to_keep].copy()  # Ensure we have a copy
    
    # Verify data integrity after filtering
    print(f"\nFiltered data shape: {filtered_data.shape}")
    print(f"Filtered labels shape: {filtered_labels.shape}")
    print(f"Filtered data type: {filtered_data.dtype}")
    print(f"Filtered labels type: {filtered_labels.dtype}")
    
    # Step 2: Create a mapping to make labels consecutive from 1 to 13
    # Sort the unique filtered labels
    filtered_activity_labels = filtered_labels[:, 0, 0]
    unique_filtered = np.unique(filtered_activity_labels)
    sorted_unique = np.sort(unique_filtered)
    
    # Create the mapping - ensuring we keep 1-indexed labels
    label_map = {}
    new_label = 1  # Start at 1 (keep 1-indexed)
    for old_label in sorted_unique:
        label_map[old_label] = new_label
        new_label += 1
    
    print("\nLabel remapping:")
    for old, new in sorted(label_map.items()):
        print(f"  {old} -> {new}")
    
    # Apply the mapping to create remapped labels
    remapped_labels = filtered_labels.copy()
    
    # Verify all time steps have same label for each sample before remapping
    for i in range(filtered_labels.shape[0]):
        unique_labels_in_sample = np.unique(filtered_labels[i, :, 0])
        if len(unique_labels_in_sample) > 1:
            print(f"Warning: Sample {i} has multiple activity labels: {unique_labels_in_sample}")
    
    # Apply remapping
    print("\nApplying remapping...")
    for i in range(remapped_labels.shape[0]):
        old_label = filtered_labels[i, 0, 0]
        new_label = label_map[old_label]
        
        # Apply the new label to all time steps for this sample
        remapped_labels[i, :, 0] = new_label
    
    # Verify remapping consistency across time steps
    inconsistent_samples = 0
    for i in range(remapped_labels.shape[0]):
        unique_labels_in_sample = np.unique(remapped_labels[i, :, 0])
        if len(unique_labels_in_sample) > 1:
            inconsistent_samples += 1
            if inconsistent_samples <= 5:  # Limit printing to first 5 inconsistent samples
                print(f"Error: Sample {i} has inconsistent labels after remapping: {unique_labels_in_sample}")
    
    if inconsistent_samples > 0:
        print(f"Found {inconsistent_samples} samples with inconsistent labels across time steps!")
    else:
        print("✓ All samples have consistent labels across time steps")
    
    # Verify remapped labels
    remapped_activity_labels = remapped_labels[:, 0, 0]
    unique_remapped = np.unique(remapped_activity_labels)
    print(f"\nUnique activity labels after remapping: {unique_remapped}")
    
    # Check data types after remapping
    print(f"Remapped labels type: {remapped_labels.dtype}")
    
    # Verify we have 13 consecutive classes from 1-13
    expected_labels = np.arange(1, 14)  # 1 to 13
    if np.array_equal(np.sort(unique_remapped), expected_labels):
        print("✓ Successfully remapped to consecutive labels 1-13")
    else:
        print("⚠ Warning: Labels are not consecutively 1-13 as expected")
        print(f"  Expected: {expected_labels}")
        print(f"  Actual: {np.sort(unique_remapped)}")
    
    # Save filtered and remapped datasets
    print(f"\nSaving filtered and remapped data to {filtered_data_path}")
    np.save(filtered_data_path, filtered_data)
    print(f"Saving filtered and remapped labels to {filtered_label_path}")
    np.save(filtered_label_path, remapped_labels)
    
    print("\nFiltering and remapping complete!")
    
    # Additional verification - load saved files and check
    print("\n=== Verification of Saved Files ===")
    loaded_data = np.load(filtered_data_path)
    loaded_labels = np.load(filtered_label_path)
    
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Loaded labels shape: {loaded_labels.shape}")
    print(f"Loaded data type: {loaded_data.dtype}")
    print(f"Loaded labels type: {loaded_labels.dtype}")
    
    # Verify loaded labels have expected values
    loaded_unique = np.unique(loaded_labels[:, 0, 0])
    print(f"Unique labels in loaded file: {loaded_unique}")
    
    if np.array_equal(np.sort(loaded_unique), expected_labels):
        print("✓ Saved labels have correct consecutive values 1-13")
    else:
        print("⚠ Warning: Saved labels do not have expected values")

# Run the function
if __name__ == "__main__":
    filter_and_remap_gestures()