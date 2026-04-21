import os
import pandas as pd
import numpy as np
import re
from glob import glob

def process_file(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'user_(\d+)_activity_(\d+)_trial_(\d+)', filename)
    if not match:
        return None
    
    user_id = int(match.group(1))
    activity_id = int(match.group(2))
    # Adjust activity label to be zero-based (1-29 -> 0-28)
    activity_label = activity_id - 1
    
    try:
        # Use low_memory=False to avoid warnings, though small files.
        df = pd.read_csv(filepath)
    except Exception as e:
        return None
    
    if df.empty or len(df.columns) < 7:
        return None
    
    # Columns: timestamp(ms), Ax, Ay, Az, Gx, Gy, Gz
    ts_col = df.columns[0]
    data_cols = df.columns[1:7]
    
    # Target: 0 to 5950ms (120 points)
    target_times = np.arange(0, 6000, 50)
    
    # Center or Relative timestamps? 
    # Usually "resample" implies taking actual duration.
    # If duration > 6s, we take first 6s. If < 6s, we pad.
    
    # Normalize start to 0
    df_ts = df[ts_col].values - df[ts_col].iloc[0]
    
    # Find sequence duration in 50ms steps
    duration_ms = df_ts[-1]
    num_points = int(min(120, np.floor(duration_ms / 50) + 1))
    
    # Points to interpolate (actually within the data)
    interp_times = np.arange(0, num_points * 50, 50)
    
    sample_data = []
    for col in data_cols:
        sample_data.append(np.interp(interp_times, df_ts, df[col]))
    
    valid_data = np.stack(sample_data, axis=-1) # (num_points, 6)
    
    # Symmetric zero padding
    pad_total = 120 - num_points
    pad_front = pad_total // 2
    pad_back = pad_total - pad_front
    
    padded_data = np.pad(valid_data, ((pad_front, pad_back), (0, 0)), mode='constant', constant_values=0)
    
    label_array = np.full((120, 2), [user_id, activity_label])
    
    return padded_data, label_array

def main():
    files = sorted(glob('TRACES/**/*.csv', recursive=True))
    all_data = []
    all_labels = []
    
    for f in files:
        res = process_file(f)
        if res:
            all_data.append(res[0])
            all_labels.append(res[1])
            
    if not all_data:
        print("No data processed.")
        return
        
    X = np.array(all_data).astype(np.float32)
    Y = np.array(all_labels).astype(np.int32)
    
    np.save('data_20_120.npy', X)
    np.save('label_20_120.npy', Y)
    
    print(f"Total N: {len(X)}")
    print(f"Data shape: {X.shape}")
    print(f"Label shape: {Y.shape}")
    
    users = Y[:, :, 0]
    activities = Y[:, :, 1]
    print(f"User ID range: {users.min()} to {users.max()}")
    print(f"Activity Label range: {activities.min()} to {activities.max()}")

if __name__ == "__main__":
    main()
