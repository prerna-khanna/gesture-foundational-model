#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
from config import PretrainModelConfig
from features import detect_nucleus, compute_energy, calculate_significant_axis
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

def generate_user_embeddings_fixed(model_file, dataset, dataset_version, 
                            output_dir='user_embeddings', label_index=1, 
                            hidden_dim=72, feature_num=6, batch_size=32, gpu=0,
                            pooling='none'):
    """
    Generate user embeddings with correctly named output files
    
    Args:
        model_file: Path to pretrained model (without .pt extension)
        dataset: Dataset name (e.g., 'sony_watch')
        dataset_version: Dataset version (e.g., '20_120')
        output_dir: Directory to save embeddings
        label_index: Index for user labels (default: 1)
        hidden_dim: Hidden dimension from pretrained model (default: 72)
        feature_num: Feature dimension (default: 6)
        batch_size: Batch size for processing (default: 32)
        gpu: GPU device ID (default: 0)
        pooling: Type of pooling to apply to embeddings ('none', 'mean', 'nucleus', 'max')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = get_device(gpu)
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join('dataset', dataset, f'data_{dataset_version}.npy')
    label_path = os.path.join('dataset', dataset, f'label_{dataset_version}.npy')
    
    data = np.load(data_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)
    
    print(f"Loaded data with shape: {data.shape}")
    print(f"Loaded labels with shape: {labels.shape}")
    
    # Get number of users
    unique_users = np.unique(labels[:, 0, label_index])
    num_users = len(unique_users)
    print(f"Found {num_users} unique users")
    
    # Create model configuration manually
    seq_len = data.shape[1]  # Should be 120 for your dataset
    
    # Create configuration with CORRECT dimensions (hidden_dim=72, feature_num=6)
    model_cfg = PretrainModelConfig(
        hidden=hidden_dim,         # Hidden dimension from pretrained model (72)
        hidden_ff=hidden_dim*2,    # Feed-forward dimension (usually 2x hidden)
        feature_num=feature_num,   # Number of features in input data (6)
        n_layers=3,                # Number of transformer layers
        n_heads=4,                 # Number of attention heads
        seq_len=seq_len,           # Sequence length from data
        emb_norm=True              # Use embedding normalization
    )
    
    print(f"Created model configuration with hidden={model_cfg.hidden}, feature_num={model_cfg.feature_num}")
    
    # Create model with correct configuration
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    
    # Load model weights
    model_path = f"{model_file}.pt"
    print(f"Loading model from {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Create preprocessing pipeline
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    
    # Create dataset
    imu_dataset = IMUDataset(data, labels, pipeline=pipeline)
    dataloader = DataLoader(imu_dataset, batch_size=batch_size, shuffle=False)
    
    # Generate embeddings
    embeddings = []
    user_labels = []
    nucleus_points_all = []
    
    print(f"Generating embeddings for {len(imu_dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (seqs, label) in enumerate(dataloader):
            # Move data to device
            seqs = seqs.to(device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            
            # Store nucleus points for potential pooling later
            nucleus_points_all.extend(batch_nucleus_points)
            
            # Generate nucleus mask
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(device)
            
            # Generate embeddings
            batch_embeddings = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            
            # Store embeddings and user labels
            embeddings.append(batch_embeddings.cpu().numpy())
            user_labels.append(label[:, 0, label_index].cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate results
    embeddings = np.concatenate(embeddings, axis=0)
    user_labels = np.concatenate(user_labels, axis=0)
    
    # Apply pooling if specified
    if pooling == 'mean':
        # Mean pooling over sequence dimension
        embeddings = np.mean(embeddings, axis=1)
        print(f"Applied mean pooling, new shape: {embeddings.shape}")
    elif pooling == 'max':
        # Max pooling over sequence dimension
        embeddings = np.max(embeddings, axis=1)
        print(f"Applied max pooling, new shape: {embeddings.shape}")
    elif pooling == 'nucleus':
        # Extract embeddings only from nucleus regions
        pooled_embeddings = []
        
        for i, (emb, nucleus) in enumerate(zip(embeddings, nucleus_points_all)):
            if len(nucleus) == 2:
                start, end = nucleus
                # Average the embeddings in the nucleus region
                nucleus_emb = np.mean(emb[start:end], axis=0)
                pooled_embeddings.append(nucleus_emb)
            else:
                # If no nucleus found, use mean of entire sequence
                pooled_embeddings.append(np.mean(emb, axis=0))
        
        embeddings = np.array(pooled_embeddings)
        print(f"Applied nucleus pooling, new shape: {embeddings.shape}")
    
    # Print embedding statistics
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"User labels shape: {user_labels.shape}")
    
    # Create proper output file names with actual dataset name string
    output_file_prefix = f"{dataset}_{dataset_version}_user_embeddings"
    if pooling != 'none':
        output_file_prefix += f"_{pooling}"
    
    # Save embeddings and labels
    np.save(os.path.join(output_dir, f"{output_file_prefix}.npy"), embeddings)
    np.save(os.path.join(output_dir, f"{output_file_prefix}_labels.npy"), user_labels)
    
    # Create separate files for each user
    for user_id in range(num_users):
        user_mask = user_labels == user_id
        user_embeddings = embeddings[user_mask]
        
        if len(user_embeddings) > 0:
            np.save(os.path.join(output_dir, f"{output_file_prefix}_user{user_id}.npy"), user_embeddings)
            print(f"Saved {len(user_embeddings)} embeddings for User {user_id}")
    
    print(f"All embeddings saved to {output_dir}/{output_file_prefix}.npy")
    return embeddings, user_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate user embeddings from pretrained model with fixed naming')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to pretrained model (without .pt extension)')
    parser.add_argument('--dataset', type=str, default='sony_watch',
                        help='Dataset name (default: sony_watch)')
    parser.add_argument('--dataset_version', type=str, default='20_120',
                        help='Dataset version (default: 20_120)')
    parser.add_argument('--output_dir', type=str, default='user_embeddings',
                        help='Directory to save embeddings (default: user_embeddings)')
    parser.add_argument('--label_index', type=int, default=1,
                        help='Label index for user identification (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=72,
                        help='Hidden dimension from pretrained model (default: 72)')
    parser.add_argument('--feature_num', type=int, default=6,
                        help='Feature dimension (default: 6)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--pooling', type=str, default='none', choices=['none', 'mean', 'max', 'nucleus'],
                        help='Pooling method to apply to embeddings (default: none)')
    
    args = parser.parse_args()
    
    generate_user_embeddings_fixed(
        model_file=args.model_file,
        dataset=args.dataset,
        dataset_version=args.dataset_version,
        output_dir=args.output_dir,
        label_index=args.label_index,
        hidden_dim=args.hidden_dim,
        feature_num=args.feature_num,
        batch_size=args.batch_size,
        gpu=args.gpu,
        pooling=args.pooling
    )