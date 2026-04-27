import os
import glob
import scipy.io
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Constants
TARGET_FREQ = 20
DURATION = 6
MAX_POINTS = TARGET_FREQ * DURATION # 120
SOURCE_DIR_PATTERN = 'HGAG_DATA/raw/**/HGAG-DATA1'

def resample_and_pad(data, target_len=120):
    n_pts = len(data)
    if n_pts < 2:
        return np.zeros(target_len)
    
    # Simple linear interpolation
    x = np.linspace(0, 1, n_pts)
    f = interp1d(x, data, kind='linear', fill_value="extrapolate")
    
    # 200 Hz to 20 Hz script
    duration_s = n_pts / 200.0
    n_resampled = int(round(duration_s * 20))
    if n_resampled > target_len:
        n_resampled = target_len
    
    if n_resampled < 2:
        resampled_base = np.array([data[0]]) if n_pts > 0 else np.array([0])
    else:
        x_new = np.linspace(0, 1, n_resampled)
        resampled_base = f(x_new)
        
    pad_total = target_len - len(resampled_base)
    pad_front = pad_total // 2
    pad_back = pad_total - pad_front
    return np.pad(resampled_base, (pad_front, pad_back), mode='constant')

# Discovery phase
base_paths = glob.glob(SOURCE_DIR_PATTERN, recursive=True)
all_gestures_set = set()
for bp in base_paths:
    for item in os.listdir(bp):
        if os.path.isdir(os.path.join(bp, item)):
            all_gestures_set.add(item)

GESTURES = sorted(list(all_gestures_set))
GESTURE_MAP = {name: i for i, name in enumerate(GESTURES)}
print(f"Discovered gestures: {GESTURES}")

all_data = []
all_labels = []

for bp in base_paths:
    for gesture_name in GESTURES:
        gesture_path = os.path.join(bp, gesture_name)
        if not os.path.exists(gesture_path):
            continue
        
        label_idx = GESTURE_MAP[gesture_name]
        for subject_name in sorted(os.listdir(gesture_path)):
            if not subject_name.startswith('Subject_'):
                continue
            try:
                subject_id = int(subject_name.split('_')[1])
            except (IndexError, ValueError):
                continue
                
            subject_path = os.path.join(gesture_path, subject_name, '.mat')
            if not os.path.exists(subject_path):
                continue
                
            file_names = ['accel_x_data.mat', 'accel_y_data.mat', 'accel_z_data.mat',
                          'gyro_x_data.mat', 'gyro_y_data.mat', 'gyro_z_data.mat']
            
            mats = []
            valid_subject = True
            for f in file_names:
                f_path = os.path.join(subject_path, f)
                if not os.path.exists(f_path):
                    valid_subject = False
                    break
                # Only read combined_data
                m = scipy.io.loadmat(f_path)['combined_data']
                mats.append(m)
            
            if not valid_subject:
                continue
                
            n_trials = mats[0].shape[0]
            for t in range(n_trials):
                # Process 6 channels
                trial_resampled = np.zeros((MAX_POINTS, 6))
                for m_idx in range(6):
                    trial_resampled[:, m_idx] = resample_and_pad(mats[m_idx][t])
                
                all_data.append(trial_resampled)
                lbl = np.tile([subject_id, label_idx], (MAX_POINTS, 1))
                all_labels.append(lbl)

all_data = np.array(all_data, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.float32)

np.save('dataset/HGAG_DATA/data_20_120.npy', all_data)
np.save('dataset/HGAG_DATA/label_20_120.npy', all_labels)

df_map = pd.DataFrame({'activity_label': range(len(GESTURES)), 'gesture_name': GESTURES})
df_map.to_csv('dataset/HGAG_DATA/activity_mapping.csv', index=False)

print(f"\nFinal Summary:")
print(f"N samples: {len(all_data)}")
print(f"Data shape: {all_data.shape}")
print(f"Labels shape: {all_labels.shape}")
u_ids = all_labels[:, 0, 0]
print(f"Users: count={len(np.unique(u_ids))}, range={int(u_ids.min())}-{int(u_ids.max())}")
a_ids = all_labels[:, 0, 1]
print(f"Classes: count={len(np.unique(a_ids))}, range={int(a_ids.min())}-{int(a_ids.max())}")
counts = pd.Series(a_ids).value_counts().sort_index()
print("\nClass Distribution:")
for idx, count in counts.items():
    print(f"  {GESTURES[int(idx)]} (ID {int(idx)}): {int(count)}")
