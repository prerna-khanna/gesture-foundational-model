import os
import csv
import numpy as np

def resample_segment(segment, target_len=120):
    T = segment.shape[0]
    if T == 0:
        return np.zeros((target_len, segment.shape[1]))
    
    # Linear interpolation to target_len
    # Create indices for the original and target
    original_indices = np.linspace(0, T - 1, num=T)
    target_indices = np.linspace(0, T - 1, num=target_len)
    
    resampled = np.zeros((target_len, segment.shape[1]))
    for i in range(segment.shape[1]):
        resampled[:, i] = np.interp(target_indices, original_indices, segment[:, i])
    
    return resampled

def main():
    base_dir = "INF2018/raw/INF2018/Data/Data_Individual_Readings"
    files = sorted([f for f in os.listdir(base_dir) if f.endswith(".csv")])
    
    # 1. Determine unique classes and map users
    all_original_classes = set()
    user_mapping = []
    
    all_segments = [] # List of (user_id, class_id, segment_data)
    
    for user_idx, fname in enumerate(files, 1):
        user_mapping.append((user_idx, f"User_{user_idx}", fname))
        file_path = os.path.join(base_dir, fname)
        
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 11: continue
                try:
                    # Clean and parse
                    numeric_row = [float(x.strip()) for x in row]
                    data.append(numeric_row)
                    all_original_classes.add(int(numeric_row[10]))
                except ValueError:
                    continue
        
        if not data: continue
        data = np.array(data)
        
        # 2. Split into contiguous segments where class label is constant
        if len(data) == 0: continue
        
        labels = data[:, 10].astype(int)
        diff = np.where(labels[:-1] != labels[1:])[0] + 1
        splits = np.split(data, diff)
        
        for seg in splits:
            if len(seg) == 0: continue
            cls = int(seg[0, 10])
            # First 6 columns as channels
            all_segments.append((user_idx, cls, seg[:, :6]))

    # 3. Create Class Mapping
    sorted_classes = sorted(list(all_original_classes))
    class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
    
    processed_data = []
    processed_labels = []
    class_counts = {cls: 0 for cls in sorted_classes}
    
    for user_idx, cls_orig, seg_data in all_segments:
        # Resample
        resampled = resample_segment(seg_data, 120)
        processed_data.append(resampled)
        
        cls_idx = class_to_idx[cls_orig]
        # Label shape (120, 2): [user_id, activity_label_zero_based]
        label_seq = np.zeros((120, 2))
        label_seq[:, 0] = user_idx
        label_seq[:, 1] = cls_idx
        processed_labels.append(label_seq)
        
        class_counts[cls_orig] += 1

    # 4. Save outputs
    data_np = np.array(processed_data, dtype=np.float32)
    label_np = np.array(processed_labels, dtype=np.float32)
    
    np.save("INF2018/data_20_120.npy", data_np)
    np.save("INF2018/label_20_120.npy", label_np)
    
    with open("INF2018/activity_mapping.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["activity_label", "activity_id_original", "activity_name"])
        for i, cls in enumerate(sorted_classes):
            writer.writerow([i, cls, f"Class_{cls}"])
            
    with open("INF2018/user_mapping.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "user_name", "file_name"])
        for row in user_mapping:
            writer.writerow(row)
            
    # 5. Summary
    print(f"Total Segments (N): {len(data_np)}")
    print(f"Data shape: {data_np.shape}")
    print(f"Labels shape: {label_np.shape}")
    print(f"Unique Users: {len(user_mapping)}")
    print(f"Unique Classes: {len(sorted_classes)}")
    print(f"Original Labels: {sorted_classes}")
    print("Per-class segment counts:")
    for cls in sorted_classes:
        print(f"  Class {cls}: {class_counts[cls]}")

if __name__ == '__main__':
    main()
