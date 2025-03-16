#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: Multi-Dataset Attention Comparison for 4-Column Paper

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns
import argparse
import matplotlib

# Import project modules
from models import LIMUBertModel4Pretrain
from features import detect_nucleus, compute_energy, calculate_significant_axis
from utils import IMUDataset, Preprocess4Normalization, get_device, set_seeds
from config import load_dataset_stats, load_model_config, load_dataset_label_names

# Set font properties for paper-quality figures
font = {'family': 'sans-serif',
        'size': 10}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class MultiDatasetAttentionVisualizer:
    def __init__(self):
        self.output_dir = f"multi_dataset_attention_vis"
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Dictionary to store dataset-specific data
        self.datasets = {}
        
    def load_dataset(self, dataset_name, version, label_index=0, model_file=None, model_version='v1', gpu=-1):
        """Load a specific dataset and its model"""
        print(f"Loading dataset: {dataset_name}, version: {version}")
        
        # Create dataset-specific entry
        self.datasets[dataset_name] = {}
        dataset_info = self.datasets[dataset_name]
        
        # Load model config
        dataset_info['model_cfg'] = load_model_config('pretrain', 'base', model_version)
        
        # Load dataset stats
        dataset_info['dataset_cfg'] = load_dataset_stats(dataset_name, version)
        
        # Load data
        data_path = os.path.join('dataset', dataset_name, f'data_{version}.npy')
        label_path = os.path.join('dataset', dataset_name, f'label_{version}.npy')
        
        dataset_info['data'] = np.load(data_path).astype(np.float32)
        dataset_info['labels'] = np.load(label_path).astype(np.float32)
        
        # Load label names
        dataset_info['label_names'], dataset_info['label_num'], _ = load_dataset_label_names(
            dataset_info['dataset_cfg'], label_index)
        print(f"Loaded {dataset_info['label_num']} classes: {dataset_info['label_names']}")
        
        # Prepare dataset
        pipeline = [Preprocess4Normalization(dataset_info['model_cfg'].feature_num)]
        dataset_info['dataset'] = IMUDataset(dataset_info['data'], dataset_info['labels'], pipeline=pipeline)
        dataset_info['label_index'] = label_index
        
        # Initialize model
        dataset_info['model'] = LIMUBertModel4Pretrain(dataset_info['model_cfg'], output_embed=True)
        
        # Load model weights if provided
        if model_file:
            model_path = model_file + '.pt'
            print(f"Loading model from {model_path}")
            try:
                dataset_info['model'].load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            except:
                print("Warning: Couldn't load with weights_only=True, trying without this parameter")
                dataset_info['model'].load_state_dict(torch.load(model_path, map_location=self.device))
        
        dataset_info['model'].to(self.device)
        dataset_info['model'].eval()
        
        # Create a hook for capturing attention
        def hook_attention(module, input, output):
            # Store attention scores
            if hasattr(module, 'scores'):
                dataset_info['attention_scores'] = module.scores.detach().clone()
        
        # Register the hook on the attention module
        dataset_info['model'].transformer.attn.register_forward_hook(hook_attention)
        
        # Initialize attention scores
        dataset_info['attention_scores'] = None
        
        return dataset_name
    
    def generate_nucleus_mask(self, seq_len, batch_nucleus_points):
        """Generate a binary mask for nucleus regions"""
        batch_size = len(batch_nucleus_points)
        nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for i, nucleus_points in enumerate(batch_nucleus_points):
            if len(nucleus_points) == 2:
                start, end = nucleus_points
                nucleus_mask[i, start:end] = 1  # Mark nucleus region with 1
                
        return nucleus_mask
    
    def detect_nucleus_directly(self, dataset_name, sample_idx):
        """Uses EXACTLY the same approach as standalone script to detect the nucleus"""
        dataset_info = self.datasets[dataset_name]
        
        # Load data directly
        data_tensor = torch.tensor(dataset_info['data'])
        
        # Compute energy
        energy = compute_energy(data_tensor)
        
        # Detect nucleus
        nucleus_points = detect_nucleus(energy)
        
        # Get the nucleus points for this sample
        if sample_idx < len(nucleus_points):
            nucleus_start, nucleus_end = nucleus_points[sample_idx]
            print(f"Dataset {dataset_name}, Sample {sample_idx} nucleus: {nucleus_start} to {nucleus_end}")
            return [nucleus_start, nucleus_end], energy[sample_idx]
        else:
            print(f"Sample index {sample_idx} out of range for dataset {dataset_name}")
            return [0, 0], None
    
    def process_single_sample(self, dataset_name, idx):
        """Process a single sample and return all necessary data for visualization"""
        dataset_info = self.datasets[dataset_name]
        
        # Get nucleus using direct approach
        nucleus_points, _ = self.detect_nucleus_directly(dataset_name, idx)
        
        # Get sample from dataset
        seqs, label = dataset_info['dataset'][idx]
        original_seqs = seqs.clone()
        seqs = seqs.unsqueeze(0).to(self.device)
        
        # Create masked version for reconstruction
        masked_seqs = original_seqs.clone()
        seq_len, feature_dim = original_seqs.shape
        
        # Create mask indices (mask 15% of sequence)
        mask_percentage = 0.15
        num_masks = int(seq_len * mask_percentage)
        mask_indices = np.random.choice(seq_len, num_masks, replace=False)
        
        # Apply masking
        masked_values = []
        for pos in mask_indices:
            masked_values.append(masked_seqs[pos].clone())
            masked_seqs[pos] = torch.zeros(feature_dim)
        
        masked_positions_tensor = torch.tensor(mask_indices, dtype=torch.long)
        masked_seqs_batch = masked_seqs.unsqueeze(0).to(self.device)
        masked_positions_batch = masked_positions_tensor.unsqueeze(0).to(self.device)
        
        # Create nucleus mask using the direct detection results
        batch_nucleus_points = [nucleus_points]
        nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
        nucleus_mask = nucleus_mask.to(self.device)
        
        # Calculate significant axis
        sig_axis = calculate_significant_axis(seqs)
        sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
        sig_axis_mask = sig_axis_mask.to(self.device)
        
        # Create configurations
        configs = [
            ("Without Structure-Aware Embedding", None, None),
            ("With Structure-Aware Embedding", nucleus_mask, sig_axis_mask)
        ]
        
        # Store results
        attention_matrices = []
        mse_values = {}
        
        # Process both configurations
        for name, nuc_mask, sig_mask in configs:
            # Forward pass to get attention weights and reconstructions
            with torch.no_grad():
                # For attention visualization
                _ = dataset_info['model'](seqs, nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
                
                # For MSE calculation with masked reconstruction
                reconstructed = dataset_info['model'](masked_seqs_batch, masked_positions_batch, 
                                          nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
            
            # Process attention matrix
            if dataset_info['attention_scores'] is not None:
                attn_matrix = dataset_info['attention_scores'].cpu().numpy()
                
                # Handle different shapes
                if len(attn_matrix.shape) == 4:
                    attn_matrix = attn_matrix[0].mean(axis=0)
                elif len(attn_matrix.shape) == 3:
                    attn_matrix = attn_matrix[0]
                
                # Normalize attention matrix from 0-1
                if np.max(attn_matrix) != np.min(attn_matrix):  # Avoid division by zero
                    attn_matrix = (attn_matrix - np.min(attn_matrix)) / (np.max(attn_matrix) - np.min(attn_matrix))
                
                attention_matrices.append((name, attn_matrix))
            
            # Calculate MSE for reconstruction
            reconstructed_trimmed = reconstructed[0, :, :feature_dim].cpu()
            mse_sum = 0
            
            for i, pos in enumerate(mask_indices):
                orig_values = original_seqs[pos]
                recon_values = reconstructed_trimmed[pos]
                pos_mse = torch.mean((orig_values - recon_values)**2).item()
                mse_sum += pos_mse
            
            mse = mse_sum / len(mask_indices)
            mse_values[name] = mse
        
        # Get activity name
        activity_label = int(label[0, dataset_info['label_index']])
        label_names = dataset_info['label_names']
        activity_name = label_names[activity_label] if label_names and activity_label < len(label_names) else f"Class {activity_label}"
        
        return {
            'original_seqs': original_seqs.numpy(),
            'attention_matrices': attention_matrices,
            'mse_values': mse_values,
            'nucleus_points': nucleus_points,
            'mask_indices': mask_indices,
            'sample_idx': idx,
            'activity_name': activity_name,
            'dataset_name': dataset_name
        }
    
    def visualize_multi_dataset_comparison(self, dataset_sample_map):
        """Create multi-dataset comparison with 4 columns"""
        # Process samples from each dataset
        sample_data = []
        
        for dataset_name, sample_idx in dataset_sample_map:
            print(f"Processing dataset {dataset_name}, sample {sample_idx}...")
            if dataset_name in self.datasets:
                sample_data.append(self.process_single_sample(dataset_name, sample_idx))
            else:
                print(f"Dataset {dataset_name} not loaded. Skipping.")
        
        # Create the visualization
        self._plot_multi_dataset_comparison(sample_data)
    
    def _plot_multi_dataset_comparison(self, sample_data):
        """Plot samples from multiple datasets side by side with normalized heatmaps"""
        num_columns = len(sample_data)
        
        # Create figure for paper layout (multi-column)
        fig_width = min(12, 2.5 * num_columns)  # Cap the width at 12 inches
        fig = plt.figure(figsize=(fig_width, 3.5))
        
        # Create a GridSpec layout
        # n columns (for the datasets) and 3 rows (gyro signal, heatmap 1, heatmap 2)
        gs = gridspec.GridSpec(3, num_columns + 1, height_ratios=[1, 2, 2], 
                             width_ratios=[1] * num_columns + [0.05])
        
        # Assuming gyro data is in indices 3-5
        gyro_indices = range(3, 6)
        
        # Create a shared colormap
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=1)  # Normalized from 0-1
        
        # Process each sample (side by side)
        for col, sample in enumerate(sample_data):
            # Get data for this sample
            original_seqs = sample['original_seqs']
            attention_matrices = sample['attention_matrices']
            mse_values = sample['mse_values']
            nucleus_points = sample['nucleus_points']
            mask_indices = sample['mask_indices']
            activity_name = sample['activity_name']
            dataset_name = sample['dataset_name']
            
            # Plot gyro signal
            ax_gyro = fig.add_subplot(gs[0, col])
            
            for j, gyro_idx in enumerate(gyro_indices):
                if gyro_idx < original_seqs.shape[1]:
                    ax_gyro.plot(original_seqs[:, gyro_idx], linewidth=1.0, alpha=0.8)
            
            # Highlight nucleus region
            if len(nucleus_points) == 2:
                start, end = nucleus_points
                ax_gyro.axvline(x=start, color='red', linestyle='--', linewidth=0.8)
                ax_gyro.axvline(x=end, color='red', linestyle='--', linewidth=0.8)
                ax_gyro.axvspan(start, end, alpha=0.3, color='yellow')
            
            # Set title to include dataset name and activity
            ax_gyro.set_title(f"{activity_name}", fontsize=9)
            ax_gyro.set_xlim(0, len(original_seqs))
            ax_gyro.set_xticks([])  # Hide x ticks for gyro plot
            
            if col == 0:
                ax_gyro.set_ylabel("Motion\nSignal")
            ax_gyro.set_yticks([])
            
            # Plot attention matrices
            for row, (name, matrix) in enumerate(attention_matrices):
                ax_attn = fig.add_subplot(gs[row+1, col])
                
                im = ax_attn.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, 
                                  origin='upper', extent=[0, matrix.shape[1], matrix.shape[0], 0])
                
                # Highlight nucleus region with red rectangle
                if len(nucleus_points) == 2:
                    start, end = nucleus_points
                    rect = plt.Rectangle((start, start), end-start, end-start, 
                                  fill=False, edgecolor='red', linewidth=1)
                    ax_attn.add_patch(rect)
                
                if col == 0:
                    ax_attn.set_ylabel(f"{name}")
                
                # Only add title to the first row
                if row == 0 and col == 1:
                    ax_attn.set_title("                                    " + "Without Structure-Aware Embedding", fontsize=9, fontweight="bold")
                elif row == 1 and col == 1:
                    ax_attn.set_title("                                    " + "With Structure-Aware Embedding", fontsize=9, fontweight="bold")
                
                # Only show x label for bottom row
                if row == 1 and col == 0:
                    ax_attn.set_xlabel("Sequence Position\n(a) Smartwatch gesture")
                if row == 1 and col == 1:
                    ax_attn.set_xlabel("Sequence Position\n(b) Earbud gesture")
                if row == 1 and col == 2:
                    ax_attn.set_xlabel("Sequence Position\n(c) Blind user gesture")
                if row == 1 and col == 3:
                    ax_attn.set_xlabel("Sequence Position\n(d) Activity sequence")
                
                # if row == 0, any col, set x ticks to empty
                if row == 0:
                    ax_attn.set_xticks([])
                
                # Remove y ticks for cleaner look after the first column
                if col > 0:
                    ax_attn.set_yticks([])
                else:
                    ax_attn.set_ylabel("Sequence\nPosition")
        
        # Add shared colorbar on the right
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Attention Weight (Normalized)')
        cbar_ax.tick_params(labelsize=7)
        
        # Tight layout and minimize whitespace
        plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
        
        # Generate filename from the datasets and sample indices
        dataset_names = '_'.join([sample['dataset_name'] for sample in sample_data])
        sample_indices = '_'.join([str(sample['sample_idx']) for sample in sample_data])
        filename = f"multi_dataset_attention_{dataset_names}_{sample_indices}"
        
        # Save as high-quality PDF and PNG
        plt.savefig(os.path.join(self.output_dir, f"{filename}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize multi-dataset attention comparison')
    
    parser.add_argument('--datasets', '--dataset', type=str, required=True,
                        help='Comma-separated list of datasets (e.g., "motion,hhar,uci,shoaib")')
    parser.add_argument('--versions', '--version', type=str, required=True,
                        help='Comma-separated list of dataset versions corresponding to each dataset')
    parser.add_argument('--sample_indices', '--sample_index', type=str, required=True,
                        help='Comma-separated list of sample indices for each dataset')
    parser.add_argument('--model_files', '--model_file', type=str, required=True,
                        help='Comma-separated list of model files for each dataset (without .pt extension)')
    parser.add_argument('--model_versions', '--model_version', type=str, default='v1',
                        help='Comma-separated list of model versions or a single version for all')
    parser.add_argument('--label_indices', '--label_index', type=str, default='0',
                        help='Comma-separated list of label indices or a single index for all')
    
    args = parser.parse_args()
    
    # Parse arguments
    datasets = [ds.strip() for ds in args.datasets.split(',')]
    versions = [v.strip() for v in args.versions.split(',')]
    sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
    model_files = [mf.strip() for mf in args.model_files.split(',')]
    
    # Handle model versions (either one for all or one per dataset)
    if ',' in args.model_versions:
        model_versions = [mv.strip() for mv in args.model_versions.split(',')]
    else:
        model_versions = [args.model_versions.strip()] * len(datasets)
    
    # Handle label indices (either one for all or one per dataset)
    if ',' in args.label_indices:
        label_indices = [int(li) for li in args.label_indices.split(',')]
    else:
        label_indices = [int(args.label_indices.strip())] * len(datasets)
    
    # Ensure all lists have the same length
    if not all(len(lst) == len(datasets) for lst in [versions, sample_indices, model_files, model_versions, label_indices]):
        print("Error: All parameter lists must have the same length as the number of datasets")
        exit(1)
    
    # Create visualizer and load datasets
    visualizer = MultiDatasetAttentionVisualizer()
    
    # Load each dataset
    for i, dataset in enumerate(datasets):
        visualizer.load_dataset(
            dataset_name=dataset,
            version=versions[i],
            label_index=label_indices[i],
            model_file=model_files[i],
            model_version=model_versions[i]
        )
    
    # Create dataset-sample mapping
    dataset_sample_map = [(datasets[i], sample_indices[i]) for i in range(len(datasets))]
    
    # Generate visualization
    visualizer.visualize_multi_dataset_comparison(dataset_sample_map)