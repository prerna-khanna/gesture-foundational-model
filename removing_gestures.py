import numpy as np
import os

def filter_and_remap_gestures():
    """
    Specifically removes gesture IDs 9 and 11, then remaps the remaining gestures
    to be consecutive 1-indexed labels (1 to 13)
    """
    # Define paths
    data_path = 'dataset/blind_user/data_20_120.npy'
    label_path = 'dataset/blind_user/label_20_120.npy'
    filtered_data_path = 'dataset/blind_user_filtered/data_20_120.npy'
    filtered_label_path = 'dataset/blind_user_filtered/label_20_120.npy'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filtered_data_path), exist_ok=True)
    
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
    
    # Analyze original dataset
    print("\n=== Original Dataset Analysis ===")
    activity_labels = labels[:, 0, 0]  # Get first time step's activity label for each sample
    unique_activities = np.unique(activity_labels)
    print(f"Unique activity labels: {unique_activities}")
    
    # Count instances of each activity
    activity_counts = {}
    for act in unique_activities:
        activity_counts[act] = np.sum(activity_labels == act)
    
    print("\nActivity label counts in original dataset:")
    for act, count in sorted(activity_counts.items()):
        print(f"  Gesture ID {act}: {count} samples")
    
    # Step 1: Filter out gesture IDs 9 and 11
    gesture_ids_to_remove = [9, 11]
    print(f"\nRemoving gesture IDs: {gesture_ids_to_remove}")
    
    indices_to_keep = []
    for i in range(len(labels)):
        gesture_id = labels[i, 0, 0]
        if gesture_id not in gesture_ids_to_remove:
            indices_to_keep.append(i)
    
    filtered_data = data[indices_to_keep]
    filtered_labels = labels[indices_to_keep]
    
    # Analyze filtered dataset
    print(f"\nFiltered data shape: {filtered_data.shape}")
    filtered_activity_labels = filtered_labels[:, 0, 0]
    unique_filtered = np.unique(filtered_activity_labels)
    print(f"Unique activity labels after filtering: {unique_filtered}")
    
    # Step 2: Create a mapping to make labels consecutive from 1 to 13
    # Sort the unique filtered labels
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
    
    for i in range(remapped_labels.shape[0]):
        old_label = remapped_labels[i, 0, 0]
        new_label = label_map[old_label]
        
        # Apply the new label to all time steps for this sample
        remapped_labels[i, :, 0] = new_label
    
    # Verify remapped labels
    remapped_activity_labels = remapped_labels[:, 0, 0]
    unique_remapped = np.unique(remapped_activity_labels)
    print(f"\nUnique activity labels after remapping: {unique_remapped}")
    
    # Verify we have 13 consecutive classes from 1-13
    expected_labels = np.arange(1, 14)  # 1 to 13
    if np.array_equal(np.sort(unique_remapped), expected_labels):
        print("✓ Successfully remapped to consecutive labels 1-13")
    else:
        print("⚠ Warning: Labels are not consecutively 1-13 as expected")
        print(f"  Expected: {expected_labels}")
        print(f"  Actual: {np.sort(unique_remapped)}")
    
    # Count instances of each activity in remapped dataset
    remapped_counts = {}
    for act in unique_remapped:
        remapped_counts[act] = np.sum(remapped_activity_labels == act)
    
    print("\nActivity label counts in remapped dataset:")
    for act, count in sorted(remapped_counts.items()):
        print(f"  Gesture ID {act}: {count} samples")
    
    # Save filtered and remapped datasets
    print(f"\nSaving filtered and remapped data to {filtered_data_path}")
    np.save(filtered_data_path, filtered_data)
    print(f"Saving filtered and remapped labels to {filtered_label_path}")
    np.save(filtered_label_path, remapped_labels)
    
    print("\nFiltering and remapping complete!")

# Run the function
if __name__ == "__main__":
    filter_and_remap_gestures()