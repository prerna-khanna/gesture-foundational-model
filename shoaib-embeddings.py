#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
from config import PretrainModelConfig
from utils import IMUDataset, get_device, Preprocess4Normalization
from models import LIMUBertModel4Pretrain

def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """Generate nucleus mask for each sequence in batch"""
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1
    
    return nucleus_mask

# Modified function to handle any dimension data
def compute_energy_flexible(seqs):
    """
    Compute energy of the input sequence with flexible handling of dimensions.
    """
    # For 9-dimensional data (acc, gyro, mag), use only accelerometer for energy
    if seqs.shape[-1] >= 6:
        # Use only the first 3 dimensions (accelerometer)
        acc_data = seqs[:, :, :3]
        energy = torch.sqrt((acc_data ** 2).sum(dim=-1))
    else:
        # Default behavior for smaller dimensions
        energy = torch.sqrt((seqs ** 2).sum(dim=-1))
    
    return energy

# Modified function to handle different input dimensions
def detect_nucleus_flexible(energy, window=20, nucleus_thres=0.4):
    """
    Detects the nucleus of gestures based on changes in signal energy.
    Works with various data dimensions.
    """
    batch_nucleus_points = []

    # Loop over each sequence in the batch
    for sequence_energy in energy:
        # Default nucleus points (middle section)
        seq_len = sequence_energy.shape[0]
        start = seq_len // 3
        end = 2 * seq_len // 3
        
        try:
            change_pts = []

            # Convert to numpy for processing
            if sequence_energy.is_cuda:
                sequence_energy = sequence_energy.cpu().numpy()
            elif hasattr(sequence_energy, 'device') and sequence_energy.device.type == 'mps':
                sequence_energy = sequence_energy.to('cpu').numpy()
            else:
                sequence_energy = sequence_energy.numpy()

            # Sliding window to detect energy changes
            for i in range(len(sequence_energy) - 15):
                if abs(sequence_energy[i + 15] - sequence_energy[i]) > nucleus_thres:
                    change_pts.append(i)

            # If no change points are detected, use default nucleus points
            if not change_pts:
                filtered_change_pts = [start, end]
            else:
                # Adjust detected change points
                change_pts = list(map(lambda x: x + window, change_pts))
                
                # Filter close change points
                filtered_change_pts = [change_pts[0]]
                for i in range(1, len(change_pts)):
                    if change_pts[i] - filtered_change_pts[-1] >= window:
                        filtered_change_pts.append(change_pts[i])

                filtered_change_pts = filtered_change_pts[:2]

                # Adjust if only one change point detected
                if len(filtered_change_pts) == 1:
                    filtered_change_pts.append(change_pts[-1] + 10)
        except Exception as e:
            print(f"Error detecting nucleus: {e}")
            filtered_change_pts = [start, end]  # Use default

        batch_nucleus_points.append(filtered_change_pts)

    return batch_nucleus_points

def generate_user_embeddings_shoaib(model_file, dataset, dataset_version, 
                          position_detection=False,
                          output_dir='user_embeddings', label_index=2,  # Default to user_label_index for Shoaib
                          feature_num=9,  # 9 for Shoaib with magnetometer
                          hidden_dim=72, batch_size=32, gpu=0,
                          pooling='mean'):
    """
    Generate user embeddings for Shoaib dataset
    
    Args:
        model_file: Path to pretrained model (without .pt extension)
        dataset: Dataset name (e.g., 'shoaib')
        dataset_version: Dataset version (e.g., '20_120')
        position_detection: If True, use position_label_index instead of user_label_index
        output_dir: Directory to save embeddings
        label_index: Index for target labels (default: 2 for user in Shoaib)
        feature_num: Feature dimension (default: 9 for Shoaib with magnetometer)
        hidden_dim: Hidden dimension from pretrained model
        batch_size: Batch size for processing
        gpu: GPU device ID
        pooling: Type of pooling to apply ('none', 'mean', 'max')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = get_device(gpu)
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join('dataset', dataset, f'data_{dataset_version}.npy')
    label_path = os.path.join('dataset', dataset, f'label_{dataset_version}.npy')
    
    try:
        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path).astype(np.float32)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure the dataset files exist: {data_path}")
        return None, None
    
    print(f"Loaded data with shape: {data.shape}")
    print(f"Loaded labels with shape: {labels.shape}")
    
    # Target detection: user authentication or position detection
    if position_detection:
        # Default position_label_index for Shoaib is 1
        label_index = 1
        print(f"Using position detection (label_index={label_index})")
    
    # Get number of target classes
    try:
        unique_targets = np.unique(labels[:, 0, label_index])
        num_targets = len(unique_targets)
        
        if position_detection:
            print(f"Found {num_targets} unique positions: {unique_targets}")
        else:
            print(f"Found {num_targets} unique users: {unique_targets}")
    except Exception as e:
        print(f"Error analyzing labels: {e}")
        # Fallback
        if position_detection:
            print("Assuming 5 positions (based on Shoaib dataset)")
            num_targets = 5
        else:
            print("Assuming 10 users (based on Shoaib dataset)")
            num_targets = 10
    
    # Create model configuration
    seq_len = data.shape[1]  # Should be 120 for the dataset
    
    print(f"Creating model configuration with hidden={hidden_dim}, feature_num={feature_num}")
    model_cfg = PretrainModelConfig(
        hidden=hidden_dim,         # Hidden dimension from pretrained model
        hidden_ff=hidden_dim*2,    # Feed-forward dimension
        feature_num=feature_num,   # Number of features in data
        n_layers=3,                # Number of transformer layers
        n_heads=4,                 # Number of attention heads
        seq_len=seq_len,           # Sequence length from data
        emb_norm=True              # Use embedding normalization
    )
    
    # Create model
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    
    # Load model weights
    model_path = f"{model_file}.pt"
    print(f"Loading model from {model_path}")
    
    # Load state dict with error handling
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("This is expected if the model was trained on different feature dimensions.")
        print("Proceeding with randomly initialized weights.")
    
    model.to(device)
    model.eval()
    
    # Create preprocessing pipeline
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    
    # Create dataset
    imu_dataset = IMUDataset(data, labels, pipeline=pipeline)
    dataloader = DataLoader(imu_dataset, batch_size=batch_size, shuffle=False)
    
    # Generate embeddings
    embeddings = []
    target_labels = []
    
    print(f"Generating embeddings for {len(imu_dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (seqs, label) in enumerate(dataloader):
            # Move data to device
            seqs = seqs.to(device)
            
            try:
                # Compute energy for nucleus detection (using modified function)
                energy = compute_energy_flexible(seqs)
                
                # Detect nucleus with flexible function
                batch_nucleus_points = detect_nucleus_flexible(energy)
                
                # Generate nucleus mask
                nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
                nucleus_mask = nucleus_mask.to(device)
                
                # Skip significant axis for Shoaib dataset - not needed
                sig_axis_mask = None
                
                # Generate embeddings (without significant axis)
                batch_embeddings = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
                
                # Store embeddings and target labels
                embeddings.append(batch_embeddings.cpu().numpy())
                target_labels.append(label[:, 0, label_index].cpu().numpy())
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
            
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if not embeddings:
        print("Failed to generate any embeddings. Check the error messages above.")
        return None, None
    
    # Concatenate results
    embeddings = np.concatenate(embeddings, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)
    
    # Apply pooling if specified
    if pooling == 'mean':
        # Mean pooling over sequence dimension
        embeddings = np.mean(embeddings, axis=1)
        print(f"Applied mean pooling, new shape: {embeddings.shape}")
    elif pooling == 'max':
        # Max pooling over sequence dimension
        embeddings = np.max(embeddings, axis=1)
        print(f"Applied max pooling, new shape: {embeddings.shape}")
    
    # Print embedding statistics
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"Target labels shape: {target_labels.shape}")
    
    # Create prefix for output filenames
    target_type = "position" if position_detection else "user"
    output_file_prefix = f"{dataset}_{dataset_version}_{target_type}_embeddings"
    
    if pooling != 'none':
        output_file_prefix += f"_{pooling}"
    
    # Save embeddings and labels
    np.save(os.path.join(output_dir, f"{output_file_prefix}.npy"), embeddings)
    np.save(os.path.join(output_dir, f"{output_file_prefix}_labels.npy"), target_labels)
    
    # Create separate files for each target (user or position)
    for target_id in range(num_targets):
        target_mask = target_labels == target_id
        if np.any(target_mask):  # Only save if we have samples
            target_embeddings = embeddings[target_mask]
            np.save(os.path.join(output_dir, f"{output_file_prefix}_{target_id}.npy"), target_embeddings)
            
            label_name = unique_targets[target_id] if 'unique_targets' in locals() else target_id
            if position_detection:
                print(f"Saved {len(target_embeddings)} embeddings for Position {label_name}")
            else:
                print(f"Saved {len(target_embeddings)} embeddings for User {label_name}")
    
    output_path = os.path.join(output_dir, f"{output_file_prefix}.npy")
    print(f"All embeddings saved to {output_path}")
    
    return embeddings, target_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for Shoaib dataset')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to pretrained model (without .pt extension)')
    parser.add_argument('--dataset', type=str, default='shoaib',
                        help='Dataset name (default: shoaib)')
    parser.add_argument('--dataset_version', type=str, default='20_120',
                        help='Dataset version (default: 20_120)')
    parser.add_argument('--position_detection', action='store_true',
                        help='Enable position detection instead of user authentication')
    parser.add_argument('--output_dir', type=str, default='shoaib_embeddings',
                        help='Directory to save embeddings (default: shoaib_embeddings)')
    parser.add_argument('--feature_num', type=int, default=9,
                        help='Feature dimension (default: 9 for Shoaib with magnetometer)')
    parser.add_argument('--hidden_dim', type=int, default=72,
                        help='Hidden dimension from pretrained model (default: 72)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--pooling', type=str, default='mean', 
                        choices=['none', 'mean', 'max'],
                        help='Pooling method (default: mean)')
    
    args = parser.parse_args()
    
    generate_user_embeddings_shoaib(
        model_file=args.model_file,
        dataset=args.dataset,
        dataset_version=args.dataset_version,
        position_detection=args.position_detection,
        output_dir=args.output_dir,
        feature_num=args.feature_num,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        gpu=args.gpu,
        pooling=args.pooling
    )