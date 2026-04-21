import os
import re
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

def process_utd_mhad():
    data_dir = 'UTD_MHAD/inertial_raw/Inertial'
    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    all_data = []
    all_labels = []
    
    target_freq = 20
    target_duration = 6
    target_points = target_freq * target_duration # 120
    original_freq = 50
    
    target_t = np.linspace(0, (target_points - 1) / target_freq, target_points)
    
    for f in sorted(files):
        # Format: a{action}_s{subject}_t{trial}_inertial.mat
        match = re.match(r'a(\d+)_s(\d+)_t(\d+)_inertial\.mat', f)
        if not match:
            continue
        
        action = int(match.group(1))
        subject = int(match.group(2))
        # trial = int(match.group(3))
        
        path = os.path.join(data_dir, f)
        mat = scipy.io.loadmat(path)
        # Load d_iner (N, 6)
        d_iner = mat['d_iner']
        
        n_samples = d_iner.shape[0]
        duration = (n_samples - 1) / original_freq
        source_t = np.linspace(0, duration, n_samples)
        
        # Interpolate each channel
        # We only interpolate for the duration we have. 
        # If duration > 5.95s, we sample up to 5.95s.
        # If duration < 5.95s, we interpolate available then pad.
        
        valid_target_t = target_t[target_t <= duration]
        n_valid = len(valid_target_t)
        
        interpolated = np.zeros((n_valid, 6))
        for i in range(6):
            f_interp = interp1d(source_t, d_iner[:, i], kind='linear', fill_value="extrapolate")
            interpolated[:, i] = f_interp(valid_target_t)
            
        # Padding
        pad_total = target_points - n_valid
        if pad_total > 0:
            pad_front = pad_total // 2
            pad_back = pad_total - pad_front
            padded = np.pad(interpolated, ((pad_front, pad_back), (0, 0)), mode='constant', constant_values=0)
        else:
            # Should not really happen given source freq and typical lengths, but safe slice
            padded = interpolated[:target_points, :]
            
        all_data.append(padded)
        
        # label array shape (N,120,2), repeating [user_id, activity_label]
        # user_id is subject, activity_label is action-1
        label_entry = np.full((target_points, 2), [subject, action - 1])
        all_labels.append(label_entry)
        
    all_data = np.array(all_data) # (N, 120, 6)
    all_labels = np.array(all_labels) # (N, 120, 2)
    
    np.save('UTD_MHAD/data_20_120.npy', all_data.astype(np.float32))
    np.save('UTD_MHAD/label_20_120.npy', all_labels.astype(np.int32))
    
    # Summary
    n = all_data.shape[0]
    users = all_labels[:, 0, 0]
    classes = all_labels[:, 0, 1]
    unique_users = np.unique(users)
    unique_classes = np.unique(classes)
    
    print(f"Total samples (N): {n}")
    print(f"Data shape: {all_data.shape}")
    print(f"Label shape: {all_labels.shape}")
    print(f"User range: {users.min()} to {users.max()}")
    print(f"Class range: {classes.min()} to {classes.max()}")
    print(f"Unique users: {len(unique_users)}")
    print(f"Unique classes: {len(unique_classes)}")
    
    print("Sample count per class:")
    for c in sorted(unique_classes):
        print(f"  Class {c}: {np.sum(classes == c)}")

if __name__ == '__main__':
    process_utd_mhad()
