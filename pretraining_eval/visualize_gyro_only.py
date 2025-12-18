#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize only GYRO data reconstructions for clearer comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import json

from models import LIMUBertModel4Pretrain
from config import PretrainModelConfig, MaskConfig
from utils import Preprocess4Mask, Preprocess4Normalization
from features import detect_nucleus, compute_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learnable_embed', type=str, required=True)
    parser.add_argument('--original_embed', type=str, required=True)
    parser.add_argument('--learnable_model', type=str, required=True)
    parser.add_argument('--original_model', type=str, required=True)
    parser.add_argument('--data', type=str, default='hhar')
    parser.add_argument('--version', type=str, default='20_120')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load embeddings
    print("\nLoading embeddings...")
    learnable_embeddings = np.load(args.learnable_embed)
    original_embeddings = np.load(args.original_embed)
    print(f"  Learnable: {learnable_embeddings.shape}")
    print(f"  Original: {original_embeddings.shape}")
    
    # Load data
    data_path = f"dataset/{args.data}/data_{args.version}.npy"
    data = np.load(data_path).astype(np.float32)
    print(f"  Data: {data.shape}")
    
    # Load configs
    with open('config/limu_bert.json', 'r') as f:
        model_cfg = PretrainModelConfig.from_json(json.load(f)['base_v1'])
    mask_cfg = MaskConfig.from_json('config/mask.json')
    
    # Enable nucleus-aware masking with 80% probability
    preprocess_mask = Preprocess4Mask(mask_cfg, nucleus_aware=True, nucleus_prob=0.8)
    preprocess_norm = Preprocess4Normalization(model_cfg.feature_num)
    
    # Load models
    print("\nLoading models...")
    learnable_model_path = args.learnable_model if args.learnable_model.endswith('.pt') else f"{args.learnable_model}.pt"
    original_model_path = args.original_model if args.original_model.endswith('.pt') else f"{args.original_model}.pt"
    
    learnable_model = LIMUBertModel4Pretrain(model_cfg)
    learnable_model.load_state_dict(torch.load(learnable_model_path, map_location=device))
    learnable_model.to(device)
    learnable_model.eval()
    
    original_model = LIMUBertModel4Pretrain(model_cfg)
    original_model.load_state_dict(torch.load(original_model_path, map_location=device))
    original_model.to(device)
    original_model.eval()
    
    # Select random samples
    np.random.seed(args.seed)
    n_total = min(len(data), len(learnable_embeddings))
    random_indices = np.random.choice(n_total, args.n_samples, replace=False)
    print(f"\nSelected {args.n_samples} random samples: {random_indices}")
    
    # Generate masked positions using NUCLEUS-AWARE masking
    print("\nGenerating nucleus-aware masked positions...")
    masked_positions_list = []
    for sample_idx in random_indices:
        instance = data[sample_idx].copy()
        instance = preprocess_norm(instance)
        
        # Detect nucleus for this sample
        energy = compute_energy(torch.from_numpy(instance).unsqueeze(0))
        nucleus_points = detect_nucleus(energy)[0]  # Get first (and only) sample's nucleus
        nucleus_start, nucleus_end = nucleus_points if len(nucleus_points) == 2 else (0, len(instance))
        
        # Use nucleus-aware masking
        masked_seq, masked_pos, original_seq = preprocess_mask(instance, nucleus_start=nucleus_start, nucleus_end=nucleus_end)
        masked_positions_list.append(masked_pos)
        print(f"  Sample {sample_idx}: Nucleus=[{nucleus_start}:{nucleus_end}], Masked={len(masked_pos)} positions")
    
    # Create figure - 2 columns per sample (Original Error, Learnable Error)
    fig, axes = plt.subplots(args.n_samples, 2, figsize=(16, 3*args.n_samples))
    
    # Gyro channels only (indices 3, 4, 5)
    gyro_names = ['Gyro X', 'Gyro Y', 'Gyro Z']
    gyro_colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    print("\nGenerating gyro-only reconstructions...")
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(random_indices):
            ground_truth = data[sample_idx]
            learnable_embed = torch.from_numpy(learnable_embeddings[sample_idx:sample_idx+1]).float().to(device)
            original_embed = torch.from_numpy(original_embeddings[sample_idx:sample_idx+1]).float().to(device)
            masked_pos = masked_positions_list[plot_idx]
            
            # Reconstruct
            original_recon = original_model.decoder(original_embed).cpu().numpy()[0]
            learnable_recon = learnable_model.decoder(learnable_embed).cpu().numpy()[0]
            
            # Calculate errors for gyro only
            gyro_gt = ground_truth[:, 3:6]  # Gyro channels
            gyro_orig = original_recon[:, 3:6]
            gyro_learn = learnable_recon[:, 3:6]
            
            original_mse = np.mean((gyro_orig - gyro_gt) ** 2)
            learnable_mse = np.mean((gyro_learn - gyro_gt) ** 2)
            improvement = ((original_mse - learnable_mse) / original_mse * 100)
            
            time_steps = np.arange(120)
            
            # Scale reconstructions to match GT magnitude (for shape comparison)
            def scale_to_match(recon, gt):
                """Scale reconstruction to have same magnitude as GT"""
                scaled_recon = np.zeros_like(recon)
                for i in range(recon.shape[1]):  # For each channel
                    gt_std = np.std(gt[:, i])
                    recon_std = np.std(recon[:, i])
                    if recon_std > 1e-6:  # Avoid division by zero
                        scaled_recon[:, i] = (recon[:, i] - np.mean(recon[:, i])) * (gt_std / recon_std) + np.mean(gt[:, i])
                    else:
                        scaled_recon[:, i] = recon[:, i]
                return scaled_recon
            
            gyro_orig_scaled = scale_to_match(gyro_orig, gyro_gt)
            gyro_learn_scaled = scale_to_match(gyro_learn, gyro_gt)
            
            # Get y-axis limits from ground truth
            y_min = gyro_gt.min() * 1.1
            y_max = gyro_gt.max() * 1.1
            
            # Plot 1: Original Model - GT (dotted) vs Scaled Reconstruction (solid)
            ax_left = axes[plot_idx, 0] if args.n_samples > 1 else axes[0]
            
            # Highlight masked regions
            for pos in masked_pos:
                ax_left.axvspan(pos-0.5, pos+0.5, alpha=0.15, color='gray', zorder=0)
            
            # Plot each channel: GT as dotted, Reconstruction as solid
            for gyro_idx in range(3):
                ax_left.plot(time_steps, gyro_gt[:, gyro_idx], 
                           linestyle=':', linewidth=2.5, alpha=0.8,
                           color=gyro_colors[gyro_idx],
                           label=f'{gyro_names[gyro_idx]} GT')
                ax_left.plot(time_steps, gyro_orig_scaled[:, gyro_idx], 
                           linestyle='-', linewidth=2, alpha=0.9,
                           color=gyro_colors[gyro_idx])
            
            ax_left.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            ax_left.set_ylabel('Gyro Value (Magnitude Normalized)', fontsize=11, fontweight='bold')
            ax_left.set_title(f'Sample {sample_idx}: ORIGINAL Model\nMSE = {original_mse:.4f} | Dotted=GT, Solid=Recon', 
                            fontsize=11, fontweight='bold', color='#e74c3c')
            ax_left.set_ylim(y_min, y_max)
            ax_left.grid(alpha=0.3, linewidth=0.5)
            if plot_idx == 0:
                ax_left.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # Plot 2: Learnable Model - GT (dotted) vs Scaled Reconstruction (solid)
            ax_right = axes[plot_idx, 1] if args.n_samples > 1 else axes[1]
            
            # Highlight masked regions
            for pos in masked_pos:
                ax_right.axvspan(pos-0.5, pos+0.5, alpha=0.15, color='gray', zorder=0)
            
            # Plot each channel: GT as dotted, Reconstruction as solid
            for gyro_idx in range(3):
                ax_right.plot(time_steps, gyro_gt[:, gyro_idx], 
                            linestyle=':', linewidth=2.5, alpha=0.8,
                            color=gyro_colors[gyro_idx],
                            label=f'{gyro_names[gyro_idx]} GT')
                ax_right.plot(time_steps, gyro_learn_scaled[:, gyro_idx], 
                            linestyle='-', linewidth=2, alpha=0.9,
                            color=gyro_colors[gyro_idx])
            
            ax_right.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            ax_right.set_ylabel('Gyro Value (Magnitude Normalized)', fontsize=11, fontweight='bold')
            title_color = '#27ae60' if improvement > 0 else '#e74c3c'
            ax_right.set_title(f'Sample {sample_idx}: LEARNABLE Model | Improvement: {improvement:+.2f}%\nMSE = {learnable_mse:.4f} | Dotted=GT, Solid=Recon', 
                             fontsize=11, fontweight='bold', color=title_color)
            ax_right.set_ylim(y_min, y_max)
            ax_right.grid(alpha=0.3, linewidth=0.5)
            if plot_idx == 0:
                ax_right.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            print(f"  Sample {sample_idx}: Orig MSE={original_mse:.4f}, Learn MSE={learnable_mse:.4f}, Δ={improvement:+.2f}%")
    
    plt.suptitle('GYRO SHAPE COMPARISON: Dotted = Ground Truth, Solid = Reconstruction (Scaled)\nMagnitude normalized to match GT for shape comparison | Gray = Masked Regions', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = f'gyro_only_reconstructions_{args.data}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
