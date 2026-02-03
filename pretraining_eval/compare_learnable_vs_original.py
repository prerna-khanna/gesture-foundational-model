#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare embeddings from learnable vs original model using decoder performance.
Uses the same decoder as in pretraining to reconstruct sequences.

python pretraining_eval/compare_learnable_vs_original.py \
  --original_model saved/pretrain_base_hhar_20_120/limu_v1.pt \
  --learnable_model saved/pretrain_base_hhar_20_120/limu_v1_learnable.pt \
  --learnable_nucleus_model saved/pretrain_base_hhar_20_120/limu_v1_learnable_nucleus.pt \
  --data hhar \
  --version 20_120

python pretraining_eval/compare_learnable_vs_original.py \
  --original_model saved/pretrain_base_blind_user_20_120/limu_v1.pt \
  --learnable_model saved/pretrain_base_blind_user_20_120/limu_v1_learnable.pt \
  --learnable_nucleus_model saved/pretrain_base_blind_user_20_120/limu_v1_learnable_nucleus.pt \
  --data blind_user \
  --version 20_120
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

from models import LIMUBertModel4Pretrain
from config import PretrainModelConfig
from features import detect_nucleus, compute_energy, calculate_significant_axis
from utils import Preprocess4Normalization


def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """Generate binary mask for nucleus regions"""
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1
    return nucleus_mask


def generate_embeddings(model, data, model_cfg, device, batch_size=32):
    """Generate embeddings from raw data using the model's encoder"""
    model.eval()
    preprocess = Preprocess4Normalization(model_cfg.feature_num)
    
    # Preprocess data
    data_processed = np.array([preprocess(seq) for seq in data])
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(data_processed), batch_size):
            batch = data_processed[i:i+batch_size]
            seqs = torch.from_numpy(batch).float().to(device)
            
            # Compute energy and detect nucleus
            energy = compute_energy(seqs)
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points).to(device)
            
            # Calculate significant axis mask
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).long()
            
            # Get embeddings from encoder (set output_embed=True mode)
            embeddings = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)




def load_embeddings_and_data(embed_file, data_file=None):
    """DEPRECATED - keeping for compatibility but not used"""
    print(f"Loading embeddings from {embed_file}")
    embeddings = np.load(embed_file)
    print(f"  Shape: {embeddings.shape}")
    return embeddings


def evaluate_reconstruction(model, embeddings, original_seqs, device, batch_size=32):
    """
    Evaluate reconstruction performance using pretrained decoder.
    
    Args:
        model: LIMUBertModel4Pretrain with trained decoder
        embeddings: numpy array of embeddings (N, seq_len, hidden)
        original_seqs: numpy array of original sequences (N, seq_len, features)
        device: torch device
        batch_size: batch size for evaluation
    
    Returns:
        avg_mse: average MSE reconstruction loss
        avg_mae: average MAE reconstruction loss
    """
    model.eval()
    criterion_mse = nn.MSELoss(reduction='mean')
    criterion_mae = nn.L1Loss(reduction='mean')
    
    # Convert to tensors
    embed_tensor = torch.from_numpy(embeddings).float()
    seq_tensor = torch.from_numpy(original_seqs).float()
    
    dataset = TensorDataset(embed_tensor, seq_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for embed_batch, seq_batch in dataloader:
            embed_batch = embed_batch.to(device)
            seq_batch = seq_batch.to(device)
            
            # Apply the full reconstruction path: activ -> linear -> norm -> decoder
            # This matches what happens during pretraining
            h = model.activ(model.linear(embed_batch))
            h = model.norm(h)
            seq_recon = model.decoder(h)
            
            # Calculate losses
            mse = criterion_mse(seq_recon, seq_batch)
            mae = criterion_mae(seq_recon, seq_batch)
            
            batch_size_actual = embed_batch.size(0)
            total_mse += mse.item() * batch_size_actual
            total_mae += mae.item() * batch_size_actual
            total_samples += batch_size_actual
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_model', type=str, required=True,
                        help='Path to original pretrained model')
    parser.add_argument('--learnable_model', type=str, required=True,
                        help='Path to learnable pretrained model')
    parser.add_argument('--learnable_nucleus_model', type=str, required=True,
                        help='Path to learnable nucleus pretrained model')
    parser.add_argument('--data', type=str, default='hhar',
                        help='Dataset name (default: hhar)')
    parser.add_argument('--version', type=str, default='20_120',
                        help='Data version (default: 20_120)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    data_path = f"dataset/{args.data}/data_{args.version}.npy"
    labels_path = f"dataset/{args.data}/label_{args.version}.npy"
    data = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    print(f"Loaded {len(data)} sequences with shape {data[0].shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Load model config
    with open('config/limu_bert.json', 'r') as f:
        model_cfg = PretrainModelConfig.from_json(json.load(f)['base_v1'])
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    # Original model
    original_model_path = args.original_model if args.original_model.endswith('.pt') else f"{args.original_model}.pt"
    print(f"\nLoading original model from {original_model_path}")
    original_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    original_model.load_state_dict(torch.load(original_model_path, map_location=device))
    original_model.to(device)
    original_model.eval()
    
    # Learnable model
    learnable_model_path = args.learnable_model if args.learnable_model.endswith('.pt') else f"{args.learnable_model}.pt"
    print(f"Loading learnable model from {learnable_model_path}")
    learnable_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    learnable_model.load_state_dict(torch.load(learnable_model_path, map_location=device))
    learnable_model.to(device)
    learnable_model.eval()
    
    # Learnable Nucleus model
    learnable_nucleus_model_path = args.learnable_nucleus_model if args.learnable_nucleus_model.endswith('.pt') else f"{args.learnable_nucleus_model}.pt"
    print(f"Loading learnable nucleus model from {learnable_nucleus_model_path}")
    learnable_nucleus_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    learnable_nucleus_model.load_state_dict(torch.load(learnable_nucleus_model_path, map_location=device))
    learnable_nucleus_model.to(device)
    learnable_nucleus_model.eval()
    
    # Generate embeddings on-the-fly
    print("\n" + "="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)
    print("Generating embeddings from ORIGINAL model...")
    original_embeddings = generate_embeddings(original_model, data, model_cfg, device, args.batch_size)
    print(f"  Shape: {original_embeddings.shape}")
    
    print("Generating embeddings from LEARNABLE model...")
    learnable_embeddings = generate_embeddings(learnable_model, data, model_cfg, device, args.batch_size)
    print(f"  Shape: {learnable_embeddings.shape}")
    
    print("Generating embeddings from LEARNABLE+NUCLEUS model...")
    learnable_nucleus_embeddings = generate_embeddings(learnable_nucleus_model, data, model_cfg, device, args.batch_size)
    print(f"  Shape: {learnable_nucleus_embeddings.shape}")
    
    # Evaluate reconstruction performance
    print("\n" + "="*80)
    print("EVALUATING RECONSTRUCTION PERFORMANCE")
    print("="*80)
    
    # Use the same data for all (first N samples matching embedding size)
    n_samples = original_embeddings.shape[0]
    original_seqs = np.array(data[:n_samples])
    
    print(f"\nEvaluating on {n_samples} samples")
    print(f"Original sequence shape: {original_seqs.shape}")
    
    # Evaluate original model
    print("\n--- ORIGINAL MODEL ---")
    original_mse, original_mae = evaluate_reconstruction(
        original_model, original_embeddings, original_seqs, device
    )
    print(f"MSE: {original_mse:.6f}")
    print(f"MAE: {original_mae:.6f}")
    
    # Evaluate learnable model
    print("\n--- LEARNABLE MODEL ---")
    learnable_mse, learnable_mae = evaluate_reconstruction(
        learnable_model, learnable_embeddings, original_seqs, device
    )
    print(f"MSE: {learnable_mse:.6f}")
    print(f"MAE: {learnable_mae:.6f}")
    
    # Evaluate learnable nucleus model
    print("\n--- LEARNABLE NUCLEUS MODEL ---")
    learnable_nucleus_mse, learnable_nucleus_mae = evaluate_reconstruction(
        learnable_nucleus_model, learnable_nucleus_embeddings, original_seqs, device
    )
    print(f"MSE: {learnable_nucleus_mse:.6f}")
    print(f"MAE: {learnable_nucleus_mae:.6f}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nReconstruction MSE:")
    print(f"  Original:           {original_mse:.6f}")
    print(f"  Learnable:          {learnable_mse:.6f} ({((original_mse - learnable_mse) / original_mse * 100):+.2f}%)")
    print(f"  Learnable Nucleus:  {learnable_nucleus_mse:.6f} ({((original_mse - learnable_nucleus_mse) / original_mse * 100):+.2f}%)")
    
    print(f"\nReconstruction MAE:")
    print(f"  Original:           {original_mae:.6f}")
    print(f"  Learnable:          {learnable_mae:.6f} ({((original_mae - learnable_mae) / original_mae * 100):+.2f}%)")
    print(f"  Learnable Nucleus:  {learnable_nucleus_mae:.6f} ({((original_mae - learnable_nucleus_mae) / original_mae * 100):+.2f}%)")
    
    # Embedding statistics
    print("\n" + "="*80)
    print("EMBEDDING STATISTICS")
    print("="*80)
    
    original_mean = np.mean(original_embeddings)
    original_std = np.std(original_embeddings)
    learnable_mean = np.mean(learnable_embeddings)
    learnable_std = np.std(learnable_embeddings)
    learnable_nucleus_mean = np.mean(learnable_nucleus_embeddings)
    learnable_nucleus_std = np.std(learnable_nucleus_embeddings)
    
    print(f"\nOriginal embeddings:")
    print(f"  Mean: {original_mean:.6f}")
    print(f"  Std:  {original_std:.6f}")
    
    print(f"\nLearnable embeddings:")
    print(f"  Mean: {learnable_mean:.6f}")
    print(f"  Std:  {learnable_std:.6f}")
    
    print(f"\nLearnable Nucleus embeddings:")
    print(f"  Mean: {learnable_nucleus_mean:.6f}")
    print(f"  Std:  {learnable_nucleus_std:.6f}")
    
    # Cosine similarity between embeddings
    learnable_flat = learnable_embeddings.reshape(n_samples, -1)
    original_flat = original_embeddings.reshape(n_samples, -1)
    
    cosine_sims = []
    for i in range(n_samples):
        l_norm = np.linalg.norm(learnable_flat[i])
        o_norm = np.linalg.norm(original_flat[i])
        if l_norm > 0 and o_norm > 0:
            cos_sim = np.dot(learnable_flat[i], original_flat[i]) / (l_norm * o_norm)
            cosine_sims.append(cos_sim)
    
    avg_cosine_sim = np.mean(cosine_sims)
    print(f"\nAverage cosine similarity between learnable and original embeddings:")
    print(f"  {avg_cosine_sim:.6f}")
    print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_mse = min(original_mse, learnable_mse, learnable_nucleus_mse)
    if best_mse == learnable_nucleus_mse:
        print(f"\n✓ Learnable Nucleus model shows BEST reconstruction performance!")
        print(f"  MSE improvement over Original: {((original_mse - learnable_nucleus_mse) / original_mse * 100):.2f}%")
        print(f"  MSE improvement over Learnable: {((learnable_mse - learnable_nucleus_mse) / learnable_mse * 100):.2f}%")
    elif best_mse == learnable_mse:
        print(f"\n✓ Learnable model shows BEST reconstruction performance!")
        print(f"  MSE improvement over Original: {((original_mse - learnable_mse) / original_mse * 100):.2f}%")
    else:
        print(f"\n✗ Original model shows best reconstruction performance.")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Reconstruction Loss Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['MSE', 'MAE']
    original_vals = [original_mse, original_mae]
    learnable_vals = [learnable_mse, learnable_mae]
    learnable_nucleus_vals = [learnable_nucleus_mse, learnable_nucleus_mae]
    x = np.arange(len(metrics))
    width = 0.25
    ax1.bar(x - width, original_vals, width, label='Original', color='#e74c3c', alpha=0.8)
    ax1.bar(x, learnable_vals, width, label='Learnable', color='#3498db', alpha=0.8)
    ax1.bar(x + width, learnable_nucleus_vals, width, label='Learnable+Nucleus', color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Reconstruction Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Embedding Distribution (Histogram)
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(original_embeddings.flatten(), bins=50, alpha=0.5, label='Original', color='#e74c3c', density=True)
    ax2.hist(learnable_embeddings.flatten(), bins=50, alpha=0.5, label='Learnable', color='#3498db', density=True)
    ax2.hist(learnable_nucleus_embeddings.flatten(), bins=50, alpha=0.5, label='Learnable+Nucleus', color='#2ecc71', density=True)
    ax2.set_xlabel('Embedding Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Embedding Value Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Embedding Statistics (Box Plot)
    ax3 = plt.subplot(2, 3, 3)
    data_to_plot = [learnable_embeddings.flatten(), original_embeddings.flatten()]
    bp = ax3.boxplot(data_to_plot, labels=['Learnable', 'Original'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax3.set_ylabel('Embedding Value', fontsize=12)
    ax3.set_title('Embedding Value Statistics', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. PCA Visualization (2D)
    print("  Computing PCA...")
    ax4 = plt.subplot(2, 3, 4)
    # Flatten embeddings for PCA
    n_samples_viz = min(1000, n_samples)  # Use subset for faster visualization
    learnable_flat_viz = learnable_embeddings[:n_samples_viz].reshape(n_samples_viz, -1)
    original_flat_viz = original_embeddings[:n_samples_viz].reshape(n_samples_viz, -1)
    
    pca = PCA(n_components=2)
    combined = np.vstack([learnable_flat_viz, original_flat_viz])
    pca_result = pca.fit_transform(combined)
    
    learnable_pca = pca_result[:n_samples_viz]
    original_pca = pca_result[n_samples_viz:]
    
    ax4.scatter(learnable_pca[:, 0], learnable_pca[:, 1], alpha=0.5, s=10, label='Learnable', color='#2ecc71')
    ax4.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.5, s=10, label='Original', color='#e74c3c')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax4.set_title('PCA: Embedding Space', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Sample Reconstruction Error per Sequence
    print("  Computing reconstruction errors per sample...")
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate per-sample reconstruction errors
    learnable_model.eval()
    original_model.eval()
    
    learnable_errors = []
    original_errors = []
    
    with torch.no_grad():
        for i in range(min(100, n_samples)):  # First 100 samples
            embed_l = torch.from_numpy(learnable_embeddings[i:i+1]).float().to(device)
            embed_o = torch.from_numpy(original_embeddings[i:i+1]).float().to(device)
            seq = torch.from_numpy(original_seqs[i:i+1]).float().to(device)
            
            # Apply full reconstruction path: activ -> linear -> norm -> decoder
            h_l = learnable_model.activ(learnable_model.linear(embed_l))
            h_l = learnable_model.norm(h_l)
            recon_l = learnable_model.decoder(h_l)
            
            h_o = original_model.activ(original_model.linear(embed_o))
            h_o = original_model.norm(h_o)
            recon_o = original_model.decoder(h_o)
            
            error_l = torch.mean((recon_l - seq) ** 2).cpu().item()
            error_o = torch.mean((recon_o - seq) ** 2).cpu().item()
            
            learnable_errors.append(error_l)
            original_errors.append(error_o)
    
    x_samples = np.arange(len(learnable_errors))
    ax5.plot(x_samples, learnable_errors, label='Learnable', color='#2ecc71', alpha=0.7, linewidth=2)
    ax5.plot(x_samples, original_errors, label='Original', color='#e74c3c', alpha=0.7, linewidth=2)
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('MSE', fontsize=12)
    ax5.set_title('Per-Sample Reconstruction Error', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Improvement Percentage (compared to Original)
    ax6 = plt.subplot(2, 3, 6)
    metrics_names = ['MSE', 'MAE']
    learnable_improvements = [
        ((original_mse - learnable_mse) / original_mse * 100),
        ((original_mae - learnable_mae) / original_mae * 100)
    ]
    learnable_nucleus_improvements = [
        ((original_mse - learnable_nucleus_mse) / original_mse * 100),
        ((original_mae - learnable_nucleus_mae) / original_mae * 100)
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax6.bar(x - width/2, learnable_improvements, width, label='Learnable', color='#3498db', alpha=0.8)
    bars2 = ax6.bar(x + width/2, learnable_nucleus_improvements, width, label='Learnable+Nucleus', color='#2ecc71', alpha=0.8)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Improvement over Original (%)', fontsize=12)
    ax6.set_title('Improvement Comparison', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_names)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'learnable_vs_original_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Additional: t-SNE visualization (separate figure, takes longer)
    print("\n  Computing t-SNE (this may take a minute)...")
    fig2, (ax_tsne1, ax_tsne2) = plt.subplots(1, 2, figsize=(16, 6))
    
    n_samples_tsne = min(500, n_samples)
    learnable_flat_tsne = learnable_embeddings[:n_samples_tsne].reshape(n_samples_tsne, -1)
    original_flat_tsne = original_embeddings[:n_samples_tsne].reshape(n_samples_tsne, -1)
    
    # Separate t-SNE for each
    tsne_l = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_o = TSNE(n_components=2, random_state=42, perplexity=30)
    
    learnable_tsne = tsne_l.fit_transform(learnable_flat_tsne)
    original_tsne = tsne_o.fit_transform(original_flat_tsne)
    
    # Plot learnable
    scatter1 = ax_tsne1.scatter(learnable_tsne[:, 0], learnable_tsne[:, 1], 
                                c=np.arange(n_samples_tsne), cmap='viridis', 
                                alpha=0.6, s=20)
    ax_tsne1.set_title('t-SNE: Learnable Embeddings', fontsize=14, fontweight='bold')
    ax_tsne1.set_xlabel('t-SNE 1', fontsize=12)
    ax_tsne1.set_ylabel('t-SNE 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax_tsne1, label='Sample Index')
    
    # Plot original
    scatter2 = ax_tsne2.scatter(original_tsne[:, 0], original_tsne[:, 1], 
                                c=np.arange(n_samples_tsne), cmap='viridis', 
                                alpha=0.6, s=20)
    ax_tsne2.set_title('t-SNE: Original Embeddings', fontsize=14, fontweight='bold')
    ax_tsne2.set_xlabel('t-SNE 1', fontsize=12)
    ax_tsne2.set_ylabel('t-SNE 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax_tsne2, label='Sample Index')
    
    plt.tight_layout()
    output_path2 = 'tsne_comparison.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE visualization saved to: {output_path2}")
    
    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
