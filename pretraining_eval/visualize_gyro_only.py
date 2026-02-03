#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize significant gyro axis reconstructions for 3 models: Original, Learnable, Learnable+Nucleus.

python pretraining_eval/visualize_gyro_only.py \
  --original_model saved/pretrain_base_hhar_20_120/limu_v1.pt \
  --learnable_model saved/pretrain_base_hhar_20_120/limu_v1_learnable.pt \
  --learnable_nucleus_model saved/pretrain_base_hhar_20_120/limu_v1_learnable_nucleus.pt \
  --data hhar \
  --n_samples 5

  python pretraining_eval/visualize_gyro_only.py \
  --original_model saved/pretrain_base_blind_user_20_120/limu_v1.pt \
  --learnable_model saved/pretrain_base_blind_user_20_120/limu_v1_learnable.pt \
  --learnable_nucleus_model saved/pretrain_base_blind_user_20_120/limu_v1_learnable_nucleus.pt \
  --data blind_user \
  --n_samples 5
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import json

from models import LIMUBertModel4Pretrain
from config import PretrainModelConfig, MaskConfig
from utils import Preprocess4Mask, Preprocess4Normalization
from features import detect_nucleus, compute_energy, calculate_significant_axis


def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """Generate binary mask for nucleus regions"""
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1
    return nucleus_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', type=str, required=True)
    parser.add_argument('--learnable_model', type=str, required=True)
    parser.add_argument('--learnable_nucleus_model', type=str, required=True)
    parser.add_argument('--data', type=str, default='hhar')
    parser.add_argument('--version', type=str, default='20_120')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--seed', type=int, default=800, help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
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
    original_model_path = args.original_model if args.original_model.endswith('.pt') else f"{args.original_model}.pt"
    learnable_model_path = args.learnable_model if args.learnable_model.endswith('.pt') else f"{args.learnable_model}.pt"
    learnable_nucleus_model_path = args.learnable_nucleus_model if args.learnable_nucleus_model.endswith('.pt') else f"{args.learnable_nucleus_model}.pt"
    
    original_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    original_model.load_state_dict(torch.load(original_model_path, map_location=device))
    original_model.to(device)
    original_model.eval()
    
    learnable_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    learnable_model.load_state_dict(torch.load(learnable_model_path, map_location=device))
    learnable_model.to(device)
    learnable_model.eval()
    
    learnable_nucleus_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    learnable_nucleus_model.load_state_dict(torch.load(learnable_nucleus_model_path, map_location=device))
    learnable_nucleus_model.to(device)
    learnable_nucleus_model.eval()
    
    # Select random samples
    np.random.seed(args.seed)
    n_total = len(data)
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
    
    # Create figure - 3 columns per sample (Original, Learnable, Learnable+Nucleus)
    fig, axes = plt.subplots(args.n_samples, 3, figsize=(20, 3*args.n_samples))
    
    # Gyro channels only (indices 3, 4, 5)
    gyro_names = ['Gyro X', 'Gyro Y', 'Gyro Z']
    gyro_colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    print("\nGenerating significant gyro axis reconstructions...")
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(random_indices):
            ground_truth = data[sample_idx]
            ground_truth_norm = preprocess_norm(ground_truth.copy())
            
            # Prepare input tensor
            seq_tensor = torch.from_numpy(ground_truth_norm).unsqueeze(0).float().to(device)
            
            # Compute energy and detect nucleus for this sample
            energy = compute_energy(seq_tensor)
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = generate_nucleus_mask(seq_tensor.size(1), batch_nucleus_points).to(device)
            
            # Calculate significant axis mask
            sig_axis = calculate_significant_axis(seq_tensor)
            sig_axis_mask = (seq_tensor.argmax(dim=-1) == sig_axis[:, None]).long()
            
            # Generate embeddings on-the-fly
            original_embed = original_model(seq_tensor, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            learnable_embed = learnable_model(seq_tensor, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            learnable_nucleus_embed = learnable_nucleus_model(seq_tensor, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            
            masked_pos = masked_positions_list[plot_idx]
            
            # Reconstruct FULL sequence (all timesteps) using proper reconstruction path
            # Apply activ -> linear -> norm -> decoder (matches pretraining)
            h_orig = original_model.activ(original_model.linear(original_embed))
            h_orig = original_model.norm(h_orig)
            original_recon_full = original_model.decoder(h_orig).cpu().numpy()[0]
            
            h_learn = learnable_model.activ(learnable_model.linear(learnable_embed))
            h_learn = learnable_model.norm(h_learn)
            learnable_recon_full = learnable_model.decoder(h_learn).cpu().numpy()[0]
            
            h_nucleus = learnable_nucleus_model.activ(learnable_nucleus_model.linear(learnable_nucleus_embed))
            h_nucleus = learnable_nucleus_model.norm(h_nucleus)
            learnable_nucleus_recon_full = learnable_nucleus_model.decoder(h_nucleus).cpu().numpy()[0]
            
            # Create hybrid sequences: GT for unmasked, reconstruction for masked
            # This shows what the actual training sees
            original_recon = ground_truth.copy()
            learnable_recon = ground_truth.copy()
            learnable_nucleus_recon = ground_truth.copy()
            
            # Replace only masked positions with reconstructions
            for pos in masked_pos:
                original_recon[pos] = original_recon_full[pos]
                learnable_recon[pos] = learnable_recon_full[pos]
                learnable_nucleus_recon[pos] = learnable_nucleus_recon_full[pos]
            
            # Detect significant axis for this sample (gyro channels only)
            gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)
            sig_axis = calculate_significant_axis(gt_tensor).item()  # Returns 3, 4, or 5 for gyro
            sig_axis_local = sig_axis - 3  # Convert to 0, 1, or 2 for gyro indexing
            
            # Extract only the significant gyro axis
            gyro_gt_sig_raw = ground_truth[:, sig_axis]
            gyro_orig_sig_raw = original_recon[:, sig_axis]
            gyro_learn_sig_raw = learnable_recon[:, sig_axis]
            gyro_nucleus_sig_raw = learnable_nucleus_recon[:, sig_axis]
            
            # Calculate MSE for MASKED positions only BEFORE normalization (for fair comparison)
            masked_pos_array = np.array(list(masked_pos))
            original_mse = np.mean((gyro_orig_sig_raw[masked_pos_array] - gyro_gt_sig_raw[masked_pos_array]) ** 2)
            learnable_mse = np.mean((gyro_learn_sig_raw[masked_pos_array] - gyro_gt_sig_raw[masked_pos_array]) ** 2)
            learnable_nucleus_mse = np.mean((gyro_nucleus_sig_raw[masked_pos_array] - gyro_gt_sig_raw[masked_pos_array]) ** 2)
            
            # Normalize each signal to [-1, 1] for visualization
            def normalize_to_range(signal):
                sig_min = signal.min()
                sig_max = signal.max()
                if sig_max - sig_min > 0:
                    return 2 * (signal - sig_min) / (sig_max - sig_min) - 1
                else:
                    return signal * 0  # If constant, return zeros
            
            gyro_gt_sig = normalize_to_range(gyro_gt_sig_raw)
            gyro_orig_sig = normalize_to_range(gyro_orig_sig_raw)
            gyro_learn_sig = normalize_to_range(gyro_learn_sig_raw)
            gyro_nucleus_sig = normalize_to_range(gyro_nucleus_sig_raw)
            
            sig_axis_name = gyro_names[sig_axis_local]
            sig_axis_color = gyro_colors[sig_axis_local]
            
            # Plot full signal (all 120 timestamps)
            view_range = np.arange(120)
            
            # Set fixed y-axis limits for normalized data
            y_min = -1.1
            y_max = 1.1
            
            # Plot 1: Original Model
            ax1 = axes[plot_idx, 0] if args.n_samples > 1 else axes[0]
            
            # Highlight all masked regions
            for pos in masked_pos:
                ax1.axvspan(pos-0.5, pos+0.5, alpha=0.15, color='yellow', zorder=0)
            
            ax1.plot(view_range, gyro_gt_sig, 
                    linestyle='-', linewidth=2, alpha=0.8,
                    color='black', label='GT')
            ax1.plot(view_range, gyro_orig_sig, 
                    linestyle='--', linewidth=1.5, alpha=0.9,
                    color=sig_axis_color, label='Recon')
            
            ax1.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax1.set_ylabel(f'{sig_axis_name}', fontsize=10, fontweight='bold')
            ax1.set_title(f'Sample {sample_idx}: ORIGINAL\nMSE = {original_mse:.4f}', 
                        fontsize=11, fontweight='bold', color='#e74c3c')
            ax1.set_ylim(y_min, y_max)
            ax1.set_xlim(0, 119)
            ax1.grid(alpha=0.3, linewidth=0.5)
            ax1.legend(loc='best', fontsize=8, framealpha=0.9)
            
            # Plot 2: Learnable Model
            ax2 = axes[plot_idx, 1] if args.n_samples > 1 else axes[1]
            
            for pos in masked_pos:
                ax2.axvspan(pos-0.5, pos+0.5, alpha=0.15, color='yellow', zorder=0)
            
            ax2.plot(view_range, gyro_gt_sig, 
                    linestyle='-', linewidth=2, alpha=0.8,
                    color='black', label='GT')
            ax2.plot(view_range, gyro_learn_sig, 
                    linestyle='--', linewidth=1.5, alpha=0.9,
                    color=sig_axis_color, label='Recon')
            
            learnable_improvement = ((original_mse - learnable_mse) / original_mse * 100)
            title_color = '#27ae60' if learnable_improvement > 0 else '#e74c3c'
            ax2.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax2.set_ylabel(f'{sig_axis_name}', fontsize=10, fontweight='bold')
            ax2.set_title(f'Sample {sample_idx}: LEARNABLE\nMSE = {learnable_mse:.4f} ({learnable_improvement:+.1f}%)', 
                        fontsize=11, fontweight='bold', color=title_color)
            ax2.set_ylim(y_min, y_max)
            ax2.set_xlim(0, 119)
            ax2.grid(alpha=0.3, linewidth=0.5)
            ax2.legend(loc='best', fontsize=8, framealpha=0.9)
            
            # Plot 3: Learnable+Nucleus Model
            ax3 = axes[plot_idx, 2] if args.n_samples > 1 else axes[2]
            
            for pos in masked_pos:
                ax3.axvspan(pos-0.5, pos+0.5, alpha=0.15, color='yellow', zorder=0)
            
            ax3.plot(view_range, gyro_gt_sig, 
                    linestyle='-', linewidth=2, alpha=0.8,
                    color='black', label='GT')
            ax3.plot(view_range, gyro_nucleus_sig, 
                    linestyle='--', linewidth=1.5, alpha=0.9,
                    color=sig_axis_color, label='Recon')
            
            nucleus_improvement = ((original_mse - learnable_nucleus_mse) / original_mse * 100)
            title_color = '#27ae60' if nucleus_improvement > 0 else '#e74c3c'
            ax3.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax3.set_ylabel(f'{sig_axis_name}', fontsize=10, fontweight='bold')
            ax3.set_title(f'Sample {sample_idx}: LEARNABLE+NUCLEUS\nMSE = {learnable_nucleus_mse:.4f} ({nucleus_improvement:+.1f}%)', 
                        fontsize=11, fontweight='bold', color=title_color)
            ax3.set_ylim(y_min, y_max)
            ax3.set_xlim(0, 119)
            ax3.grid(alpha=0.3, linewidth=0.5)
            ax3.legend(loc='best', fontsize=8, framealpha=0.9)
            
            print(f"  Sample {sample_idx} ({sig_axis_name}): Orig={original_mse:.4f}, Learn={learnable_mse:.4f} ({learnable_improvement:+.1f}%), Nucleus={learnable_nucleus_mse:.4f} ({nucleus_improvement:+.1f}%)")
    
    plt.suptitle('Full Signal Reconstruction (All 120 Timestamps - Significant Gyro Axis)\nSolid = Ground Truth, Dashed = Model Reconstruction | Yellow Shading = Masked Positions', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = f'pretraining_eval/gyro_sig_axis_{args.data}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
