"""
Script to create individual user datasets from raw CSV files.
Each user gets their own directory with data_20_120.npy and label_20_120.npy files.

Usage:
    python create_individual_user_datasets.py --base_path /path/to/raw/data --output_dir dataset
"""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def down_sample(data, target_sr, curr_sr, seq_len=120):
    """
    Downsample data and pad/truncate to fixed sequence length.
    
    Args:
        data: numpy array of shape (original_len, features)
        target_sr: target sampling rate
        curr_sr: current sampling rate
        seq_len: target sequence length
    
    Returns:
        Downsampled and padded data of shape (seq_len, features)
    """
    factor = int(curr_sr / target_sr)
    data = data[::factor]
    
    total_pad = seq_len - len(data)
    if total_pad % 2 == 0:
        pad_start = pad_end = total_pad // 2
    else:
        pad_start = total_pad // 2
        pad_end = total_pad - pad_start
    
    data = np.pad(data, ((pad_start, pad_end), (0, 0)), 'constant')
    return data[:seq_len, :]


def process_user_data(user_path, user_name, target_sr=20, curr_sr=50, seq_len=120):
    """
    Process all CSV files for a single user.
    
    Args:
        user_path: path to user's data directory
        user_name: name of the user
        target_sr: target sampling rate
        curr_sr: current sampling rate
        seq_len: target sequence length
    
    Returns:
        data: list of gesture samples (gesture_samples, seq_len, features)
        labels: list of labels (gesture_samples, seq_len, 2) - [gesture_id, user_id]
        gesture_map: dict mapping gesture names to gesture IDs
    """
    data = []
    labels = []
    gesture_map = {}
    gesture_idx_counter = 0
    
    # Scan for CSV files
    csv_files = [f for f in os.listdir(user_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"  Warning: No CSV files found in {user_path}")
        return [], [], {}
    
    print(f"  Found {len(csv_files)} CSV files")
    
    for file in sorted(csv_files):
        file_path = os.path.join(user_path, file)
        
        # Extract gesture name and trial from filename
        # Expected format: gesture_name_trial.csv or gesture_name_trial_*.csv
        parts = file.replace('.csv', '').split('_')
        
        if len(parts) < 2:
            print(f"    Skipping {file}: unexpected filename format")
            continue
        
        gesture_name = parts[0]
        
        # Assign gesture ID (consistent across files)
        if gesture_name not in gesture_map:
            gesture_map[gesture_name] = gesture_idx_counter
            gesture_idx_counter += 1
        
        gesture_id = gesture_map[gesture_name]
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) < 20:
                print(f"    Skipping {file}: too short ({len(df)} samples)")
                continue
            
            # Downsample and pad
            processed_data = down_sample(df.values, target_sr, curr_sr, seq_len)
            
            data.append(processed_data)
            
            # Create label: [gesture_id, user_id (dummy, always 0 for single user)]
            label = np.array([[gesture_id, 0]])
            label = np.tile(label, (seq_len, 1))
            labels.append(label)
            
            print(f"    ✓ {file}: gesture={gesture_name}({gesture_id}), shape={processed_data.shape}")
            
        except Exception as e:
            print(f"    Error processing {file}: {str(e)}")
            continue
    
    return data, labels, gesture_map


def create_user_dataset(user_name, user_raw_path, output_base_dir, 
                        target_sr=20, curr_sr=50, seq_len=120):
    """
    Create dataset for a single user and save to output directory.
    
    Args:
        user_name: name of the user
        user_raw_path: path to user's raw data directory
        output_base_dir: base output directory (dataset/)
        target_sr: target sampling rate
        curr_sr: current sampling rate
        seq_len: target sequence length
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing user: {user_name}")
    print(f"  Input path: {user_raw_path}")
    
    # Check if input path exists
    if not os.path.exists(user_raw_path):
        print(f"  Error: Path not found")
        return False
    
    # Process user data
    data, labels, gesture_map = process_user_data(
        user_raw_path, user_name, target_sr, curr_sr, seq_len
    )
    
    if not data:
        print(f"  Error: No valid data processed for user {user_name}")
        return False
    
    # Stack into arrays
    data_array = np.stack(data, axis=0)
    labels_array = np.stack(labels, axis=0)
    
    print(f"  Data shape: {data_array.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Gestures: {gesture_map}")
    
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
        for gesture_name, gesture_id in sorted(gesture_map.items(), key=lambda x: x[1]):
            f.write(f"{gesture_id}: {gesture_name}\n")
    
    print(f"  ✓ Saved to {output_dir}/")
    print(f"    - data_{target_sr}_{seq_len}.npy ({data_array.shape})")
    print(f"    - label_{target_sr}_{seq_len}.npy ({labels_array.shape})")
    print(f"    - gesture_map.txt")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create individual user datasets from raw CSV files'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        required=True,
        help='Base path containing raw user data directories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset',
        help='Output base directory (default: dataset)'
    )
    parser.add_argument(
        '--users',
        type=str,
        nargs='+',
        help='Specific users to process (if not specified, process all)'
    )
    parser.add_argument(
        '--target_sr',
        type=int,
        default=20,
        help='Target sampling rate (default: 20)'
    )
    parser.add_argument(
        '--curr_sr',
        type=int,
        default=50,
        help='Current sampling rate of raw data (default: 50)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=120,
        help='Target sequence length (default: 120)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_path):
        print(f"Error: Base path not found: {args.base_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=" * 80)
    print(f"Individual User Dataset Creator")
    print(f"=" * 80)
    print(f"Base path: {args.base_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Target SR: {args.target_sr}, Current SR: {args.curr_sr}, Seq len: {args.seq_len}")
    print(f"=" * 80)
    
    # Get list of users to process
    if args.users:
        users_to_process = args.users
    else:
        # Auto-detect users from base_path
        users_to_process = [
            d for d in os.listdir(args.base_path)
            if os.path.isdir(os.path.join(args.base_path, d))
        ]
        users_to_process = sorted(users_to_process)
    
    if not users_to_process:
        print("Error: No users found to process")
        return
    
    print(f"Users to process: {', '.join(users_to_process)}\n")
    
    # Process each user
    successful = 0
    failed = 0
    
    for user_name in users_to_process:
        # Common paths to check
        possible_paths = [
            os.path.join(args.base_path, user_name),
            os.path.join(args.base_path, user_name, 'Watch', 'cropped'),
            os.path.join(args.base_path, user_name, 'cropped'),
        ]
        
        user_path = None
        for path in possible_paths:
            if os.path.exists(path) and any(f.endswith('.csv') for f in os.listdir(path)):
                user_path = path
                break
        
        if not user_path:
            print(f"  Error: Could not find data for user {user_name}")
            failed += 1
            continue
        
        success = create_user_dataset(
            user_name,
            user_path,
            args.output_dir,
            args.target_sr,
            args.curr_sr,
            args.seq_len
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n" + "=" * 80)
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
