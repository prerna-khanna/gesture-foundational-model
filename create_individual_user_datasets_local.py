#!/usr/bin/env python3
"""
Create individual user datasets from raw CSV files.
Each user gets their own directory with data_20_120.npy and label_20_120.npy files.

Run from your LIMU-BERT-blind-users directory:
    python create_individual_user_datasets_local.py

This script uses hardcoded paths from your local machine.
"""

import os
import numpy as np
import pandas as pd


# Configuration - ADJUST THESE PATHS FOR YOUR LOCAL MACHINE
RAW_DATA_PATH = '/Users/prerna/Documents/gesture modeling/user_study/'
OUTPUT_BASE_DIR = 'dataset'  # Relative to current working directory
TARGET_SR = 20
CURR_SR = 50
SEQ_LEN = 120


def down_sample(data, target_sr, curr_sr, seq_len=120):
    """
    Downsample data and pad/truncate to fixed sequence length.
    """
    # Ensure data is a numpy array
    data = np.asarray(data, dtype=np.float32)
    
    # Handle case where data might be 1D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    factor = int(curr_sr / target_sr)
    if factor <= 0:
        factor = 1
    
    # Downsample
    data = data[::factor]
    
    # Pad or truncate to seq_len
    current_len = len(data)
    if current_len >= seq_len:
        # Truncate if too long
        data = data[:seq_len, :]
    else:
        # Pad if too short
        total_pad = seq_len - current_len
        if total_pad % 2 == 0:
            pad_start = pad_end = total_pad // 2
        else:
            pad_start = total_pad // 2
            pad_end = total_pad - pad_start
        
        data = np.pad(data, ((pad_start, pad_end), (0, 0)), 'constant', constant_values=0)
    
    return data[:seq_len, :]


def process_user_data(user_path, user_name, target_sr=20, curr_sr=50, seq_len=120):
    """
    Process all CSV files for a single user.
    
    Returns:
        data: list of gesture samples
        labels: list of labels (gesture_samples, seq_len, 2) - [gesture_id, user_id]
        gesture_map: dict mapping gesture names to gesture IDs
    """
    data = []
    labels = []
    gesture_map = {}
    gesture_idx_counter = 0
    
    # Track processing stats
    processed_count = 0
    skipped_count = 0
    skipped_by_gesture = {}
    
    # Scan for CSV files
    csv_files = [f for f in os.listdir(user_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"  ⚠ No CSV files found in {user_path}")
        return [], [], {}
    
    print(f"  Found {len(csv_files)} CSV files")
    
    # Count files by gesture ID
    gesture_file_counts = {}
    for file in csv_files:
        parts = file.replace('.csv', '').split('_')
        if len(parts) >= 2:
            try:
                gesture_id = int(parts[0])
                gesture_file_counts[gesture_id] = gesture_file_counts.get(gesture_id, 0) + 1
            except:
                pass
    print(f"  CSV files per gesture: {dict(sorted(gesture_file_counts.items()))}")
    
    for file in sorted(csv_files):
        file_path = os.path.join(user_path, file)
        
        # Extract gesture ID from filename
        # Expected format: gesture_id_trial.csv or gesture_id_trial_*.csv
        parts = file.replace('.csv', '').split('_')
        
        if len(parts) < 2:
            print(f"    ⊘ Skipping {file}: unexpected filename format")
            continue
        
        try:
            gesture_id = int(parts[0])
        except ValueError:
            print(f"    ⊘ Skipping {file}: gesture ID is not a number")
            continue
        
        # Store gesture ID directly (don't reassign)
        gesture_map[gesture_id] = parts[0]  # Map ID to original filename prefix
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) < 20:
                print(f"    ⊘ Skipping {file}: too short ({len(df)} samples)")
                skipped_by_gesture[gesture_id] = skipped_by_gesture.get(gesture_id, 0) + 1
                skipped_count += 1
                continue
            
            # Downsample and pad
            processed_data = down_sample(df.values, target_sr, curr_sr, seq_len)
            
            data.append(processed_data)
            
            # Create label: [gesture_id, user_id (always 0 for single user)]
            label = np.array([[gesture_id, 0]])
            label = np.tile(label, (seq_len, 1))
            labels.append(label)
            
            print(f"    ✓ {file}: gesture_id={gesture_id}, shape={processed_data.shape}")
            processed_count += 1
            
        except Exception as e:
            print(f"    ✗ Error processing {file}: {str(e)}")
            skipped_by_gesture[gesture_id] = skipped_by_gesture.get(gesture_id, 0) + 1
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"\n  Summary: Processed {processed_count}, Skipped {skipped_count}")
        if skipped_by_gesture:
            print(f"  Skipped by gesture: {dict(sorted(skipped_by_gesture.items()))}")
    
    return data, labels, gesture_map


def create_user_dataset(user_name, user_raw_path, output_base_dir, 
                        target_sr=20, curr_sr=50, seq_len=120):
    """
    Create dataset for a single user and save to output directory.
    """
    print(f"\nProcessing user: {user_name}")
    print(f"  Input path: {user_raw_path}")
    
    # Check if input path exists
    if not os.path.exists(user_raw_path):
        print(f"  ✗ Path not found")
        return False
    
    # Process user data
    data, labels, gesture_map = process_user_data(
        user_raw_path, user_name, target_sr, curr_sr, seq_len
    )
    
    if not data:
        print(f"  ✗ No valid data processed for user {user_name}")
        return False
    
    # Stack into arrays
    data_array = np.stack(data, axis=0)
    labels_array = np.stack(labels, axis=0)
    
    print(f"  Data shape: {data_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Gestures found: {sorted(gesture_map.keys())}")
    
    # Check if 0-indexed or 1-indexed
    min_gesture_id = min(gesture_map.keys())
    max_gesture_id = max(gesture_map.keys())
    if min_gesture_id == 0:
        print(f"  ⚠ Gesture IDs are 0-indexed (0-{max_gesture_id})")
    else:
        print(f"  ✓ Gesture IDs are 1-indexed ({min_gesture_id}-{max_gesture_id})")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, user_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data and labels
    data_path = os.path.join(output_dir, f"data_{target_sr}_{seq_len}.npy")
    label_path = os.path.join(output_dir, f"label_{target_sr}_{seq_len}.npy")
    gesture_map_path = os.path.join(output_dir, "gesture_map.txt")
    
    np.save(data_path, data_array.astype(np.float32))
    np.save(label_path, labels_array.astype(np.float32))
    
    # Save gesture map for reference
    with open(gesture_map_path, 'w') as f:
        for gesture_id in sorted(gesture_map.keys()):
            f.write(f"{gesture_id}: {gesture_map[gesture_id]}\n")
    
    print(f"  ✓ Saved to {output_dir}/")
    print(f"    - data_{target_sr}_{seq_len}.npy ({data_array.shape})")
    print(f"    - label_{target_sr}_{seq_len}.npy ({labels_array.shape})")
    print(f"    - gesture_map.txt")
    
    return True


def main():
    print("=" * 80)
    print("Individual User Dataset Creator")
    print("=" * 80)
    print(f"Raw data path: {RAW_DATA_PATH}")
    print(f"Output dir: {OUTPUT_BASE_DIR}")
    print(f"Target SR: {TARGET_SR}, Current SR: {CURR_SR}, Seq len: {SEQ_LEN}")
    print("=" * 80)
    
    # Validate raw data path
    if not os.path.exists(RAW_DATA_PATH):
        print(f"✗ Error: Raw data path not found: {RAW_DATA_PATH}")
        print(f"  Please update RAW_DATA_PATH in the script")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Get list of users from raw data path
    users_to_process = [
        d for d in os.listdir(RAW_DATA_PATH)
        if os.path.isdir(os.path.join(RAW_DATA_PATH, d))
    ]
    users_to_process = sorted(users_to_process)
    
    if not users_to_process:
        print("✗ Error: No users found in raw data path")
        return
    
    print(f"Found {len(users_to_process)} users: {', '.join(users_to_process)}\n")
    
    # Process each user
    successful = 0
    failed = 0
    
    for user_name in users_to_process:
        # Common paths to check
        possible_paths = [
            os.path.join(RAW_DATA_PATH, user_name, 'Watch', 'cropped'),
            os.path.join(RAW_DATA_PATH, user_name, 'cropped'),
            os.path.join(RAW_DATA_PATH, user_name),
        ]
        
        user_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                if csv_files:
                    user_path = path
                    break
        
        if not user_path:
            print(f"✗ {user_name}: Could not find CSV data")
            failed += 1
            continue
        
        success = create_user_dataset(
            user_name,
            user_path,
            OUTPUT_BASE_DIR,
            TARGET_SR,
            CURR_SR,
            SEQ_LEN
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n" + "=" * 80)
    print(f"✓ Complete! {successful} successful, {failed} failed")
    print(f"Output saved to: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
