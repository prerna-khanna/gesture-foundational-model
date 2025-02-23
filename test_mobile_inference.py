## python test_inference.py v1 blind_user 20_120 -l 0

import torch
import numpy as np
from config import load_dataset_stats, load_dataset_label_names
from features import compute_energy, detect_nucleus
from utils import handle_argv

def load_data(dataset, dataset_version):
    """
    Load data and labels from NPY files
    """
    data_path = f'dataset/{dataset}/data_{dataset_version}.npy'
    label_path = f'dataset/{dataset}/label_{dataset_version}.npy'
    
    print(f"Loading data from: {data_path}")
    print(f"Loading labels from: {label_path}")
    
    data = np.load(data_path)
    labels = np.load(label_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return data, labels

def compute_masks(imu_data):
    """
    Compute nucleus and significant axis masks for inference
    """
    if not isinstance(imu_data, torch.Tensor):
        imu_data = torch.tensor(imu_data, dtype=torch.float32)
    
    if len(imu_data.shape) == 2:
        imu_data = imu_data.unsqueeze(0)
        
    # Compute energy
    energy = compute_energy(imu_data)
    batch_nucleus_points = detect_nucleus(energy, window=20, nucleus_thres=8)
    
    # Generate nucleus mask
    seq_len = imu_data.size(1)
    nucleus_mask = torch.zeros((1, seq_len), dtype=torch.long)
    for i, points in enumerate(batch_nucleus_points):
        if len(points) == 2:
            start, end = points
            nucleus_mask[i, start:end] = 1
            
    # Calculate significant axis mask
    abs_rotations = torch.abs(imu_data[:, :, 3:6])
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)
    sig_axis_mask = (abs_rotations.argmax(dim=-1) == sig_axis[:, None]).long()
    
    return nucleus_mask, sig_axis_mask

def run_inference(model_path, data_batch):
    """
    Run inference on a batch of data
    """
    # Load model
    model = torch.jit.load(model_path)
    model.eval()
    
    # Convert data to tensor
    data_tensor = torch.tensor(data_batch, dtype=torch.float32)
    
    # Compute masks
    nucleus_mask, sig_axis_mask = compute_masks(data_tensor)
    
    # Run inference
    with torch.no_grad():
        predictions = model(data_tensor, nucleus_mask, sig_axis_mask)
    
    return predictions

if __name__ == "__main__":
    # Parse command line arguments
    args = handle_argv('pretrain_base', 'pretrain.json', 'base')
    
    try:
        # Load dataset configuration
        dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
        label_names, label_num, _ = load_dataset_label_names(dataset_cfg, args.label_index)
        
        print(f"\nFound {label_num} classes: {label_names}")
        
        # Load data and labels
        data, labels = load_data(args.dataset, args.dataset_version)
        
        # Select some samples for testing
        num_samples = 5  # Change this to test more/fewer samples
        test_indices = np.random.choice(data.shape[0], num_samples, replace=False)
        
        # Load mobile model
        mobile_model_path = "mobile_imu_model.pt"
        print(f"\nLoading mobile model from: {mobile_model_path}")
        
        # Test each sample
        print("\nRunning inference on test samples:")
        for i, idx in enumerate(test_indices):
            test_data = data[idx:idx+1]  # Keep batch dimension
            true_label = labels[idx, 0, args.label_index]  # Assuming this is the correct label indexing
            
            predictions = run_inference(mobile_model_path, test_data)
            predicted_class = torch.argmax(predictions).item()
            confidence = torch.max(predictions).item() * 100
            
            print(f"\nSample {i+1}:")
            print(f"True label: {label_names[int(true_label)]}")
            print(f"Predicted: {label_names[predicted_class]} (Confidence: {confidence:.2f}%)")
            
            # Print top 3 predictions
            top3_values, top3_indices = torch.topk(predictions, 3)
            print("Top 3 predictions:")
            for j, (conf, idx) in enumerate(zip(top3_values[0], top3_indices[0])):
                print(f"  {j+1}. {label_names[idx]}: {conf*100:.2f}%")
        
    except Exception as e:
        print(f"Error during inference testing: {str(e)}")
        import traceback
        traceback.print_exc()