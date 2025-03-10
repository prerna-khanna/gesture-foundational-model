#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description: Visualizer for nucleus-based embedding and significant axis impact in LIMU-BERT

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import argparse

# Import project modules
from models import LIMUBertModel4Pretrain, Transformer
from features import detect_nucleus, compute_energy, calculate_significant_axis
from utils import IMUDataset, Preprocess4Normalization, get_device, set_seeds, handle_argv
from config import load_dataset_stats, load_model_config, load_dataset_label_names
from plot import plot_matrix
from matplotlib.colors import Normalize

class NucleusImpactVisualizer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.gpu)
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = f"nucleus_impact_analysis_{args.dataset}_{args.version}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model config
        self.model_cfg = load_model_config('pretrain', 'base', args.model_version)
        
        # Load dataset stats
        self.dataset_cfg = load_dataset_stats(args.dataset, args.version)
        
        # Load data
        data_path = os.path.join('dataset', args.dataset, f'data_{args.version}.npy')
        label_path = os.path.join('dataset', args.dataset, f'label_{args.version}.npy')
        
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32)
        
        # Load label names
        self.label_names, self.label_num, _ = load_dataset_label_names(self.dataset_cfg, args.label_index)
        print(f"Loaded {self.label_num} classes: {self.label_names}")
        
        # Prepare dataset and loader
        pipeline = [Preprocess4Normalization(self.model_cfg.feature_num)]
        self.dataset = IMUDataset(self.data, self.labels, pipeline=pipeline)
        self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=16)
        
        # Initialize model
        self.model = LIMUBertModel4Pretrain(self.model_cfg, output_embed=True)
        
        # Load model weights
        model_path = args.model_file + '.pt'
        print(f"Loading model from {model_path}")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except:
            print("Warning: Couldn't load with weights_only=True, trying without this parameter")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Store attention scores
        self.attention_scores = None
        
        # Register hooks to capture attention
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def hook_attention(module, input, output):
            # Store attention scores
            if hasattr(module, 'scores'):
                self.attention_scores = module.scores.detach().clone()
        
        # Register the hook on the attention module
        self.model.transformer.attn.register_forward_hook(hook_attention)
    
    def generate_nucleus_mask(self, seq_len, batch_nucleus_points):
        """Generate a binary mask for nucleus regions"""
        batch_size = len(batch_nucleus_points)
        nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for i, nucleus_points in enumerate(batch_nucleus_points):
            if len(nucleus_points) == 2:
                start, end = nucleus_points
                nucleus_mask[i, start:end] = 1  # Mark nucleus region with 1
                
        return nucleus_mask
    
    def visualize_attention_with_nucleus(self, sample_indices=None, num_samples=3):
        """Visualize attention patterns with nucleus regions highlighted"""
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
            
        
        for idx in sample_indices:
            print(f"Processing sample {idx}...")
            # Get sample
            seqs, label = self.dataset[idx]
            seqs = seqs.unsqueeze(0).to(self.device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(self.device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            # Forward pass to get attention weights
            with torch.no_grad():
                _ = self.model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            
            # Check if attention scores were captured
            if self.attention_scores is not None:
                # Process the attention scores
                attn_matrix = self.attention_scores.cpu().numpy()
                
                # Handle different shapes - we want a 2D matrix
                if len(attn_matrix.shape) == 4:
                    # If shape is [batch_size, num_heads, seq_len, seq_len]
                    # Take the first batch and average across heads
                    attn_matrix = attn_matrix[0].mean(axis=0)
                elif len(attn_matrix.shape) == 3:
                    # If shape is [batch_size, seq_len, seq_len]
                    attn_matrix = attn_matrix[0]
                
                # Create visualization
                self._plot_attention_with_nucleus(
                    attention_matrix=attn_matrix,
                    nucleus_points=batch_nucleus_points[0],
                    sample_idx=idx,
                    label=label
                )
            else:
                print("Warning: Attention scores not captured. Check hook implementation.")
    
    def _plot_attention_with_nucleus(self, attention_matrix, nucleus_points, sample_idx, label):
        """Plot attention heatmap with nucleus region highlighted"""
        plt.figure(figsize=(10, 8))
        
        # Plot attention heatmap
        ax = sns.heatmap(attention_matrix, cmap='viridis')
        
        # Highlight nucleus region if available
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            # Add rectangle patch to highlight nucleus region
            rect = plt.Rectangle((start, start), end-start, end-start, 
                              fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        activity_label = int(label[0, self.args.label_index])
        activity_name = self.label_names[activity_label] if self.label_names else f"Class {activity_label}"
        
        plt.title(f"Attention Map - {activity_name} (Sample {sample_idx})")
        plt.xlabel("Sequence Position")
        plt.ylabel("Attention")
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"attention_map_sample_{sample_idx}.png"))
        plt.close()
    
    def visualize_raw_signal_with_nucleus(self, sample_indices=None, num_samples=3):
        """Visualize raw IMU signals with nucleus regions highlighted"""
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in sample_indices:
            print(f"Visualizing raw signal for sample {idx}...")
            # Get sample
            seqs, label = self.dataset[idx]
            
            # Compute energy
            energy = compute_energy(seqs.unsqueeze(0))
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Convert to numpy for plotting
            seqs_np = seqs.numpy()
            
            # Plot IMU data (first 3 channels - typically accelerometer X, Y, Z)
            seq_len = seqs_np.shape[0]
            time = np.arange(seq_len)
            
            # Plot each axis with a different color
            colors = ['r', 'g', 'b']
            labels = ['X-axis', 'Y-axis', 'Z-axis']
            
            for j in range(min(3, seqs_np.shape[1])):  # Plot first 3 dimensions
                plt.plot(time, seqs_np[:, j], color=colors[j], label=labels[j])
            
            # Highlight nucleus region
            if len(batch_nucleus_points[0]) == 2:
                start, end = batch_nucleus_points[0]
                plt.axvspan(start, end, alpha=0.3, color='yellow')
                plt.axvline(start, color='orange', linestyle='--')
                plt.axvline(end, color='orange', linestyle='--')
            
            activity_label = int(label[0, self.args.label_index])
            activity_name = self.label_names[activity_label] if self.label_names else f"Class {activity_label}"
            
            # Add labels and legend
            plt.title(f"IMU Signal with Nucleus Region - {activity_name} (Sample {idx})")
            plt.xlabel("Time Steps")
            plt.ylabel("Sensor Reading")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f"imu_signal_nucleus_sample_{idx}.png"))
            plt.close()
    
    def compare_attention_nucleus_vs_non_nucleus(self, num_samples=10):
        """Compare attention in nucleus vs non-nucleus regions"""
        print("Comparing attention in nucleus vs non-nucleus regions...")
        # Use fixed seed for reproducibility
        set_seeds(42)
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        nucleus_attention_avg = []
        non_nucleus_attention_avg = []
        
        for idx in sample_indices:
            # Get sample
            seqs, label = self.dataset[idx]
            seqs = seqs.unsqueeze(0).to(self.device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(self.device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            # Forward pass to get attention weights
            with torch.no_grad():
                _ = self.model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            
            # Check if attention scores were captured
            if self.attention_scores is not None:
                # Process the attention scores
                attn_matrix = self.attention_scores.cpu().numpy()
                
                # Handle different shapes
                if len(attn_matrix.shape) == 4:
                    # If shape is [batch_size, num_heads, seq_len, seq_len]
                    # Take the first batch and average across heads
                    attn_matrix = attn_matrix[0].mean(axis=0)
                elif len(attn_matrix.shape) == 3:
                    # If shape is [batch_size, seq_len, seq_len]
                    attn_matrix = attn_matrix[0]
                
                # Calculate average attention in nucleus vs non-nucleus regions
                if len(batch_nucleus_points[0]) == 2:
                    start, end = batch_nucleus_points[0]
                    
                    # Get nucleus region attention
                    nucleus_region = attn_matrix[start:end, start:end]
                    nucleus_attention_avg.append(nucleus_region.mean())
                    
                    # Get non-nucleus region attention
                    # Create mask for non-nucleus regions
                    seq_len = attn_matrix.shape[0]
                    non_nucleus_mask = np.ones((seq_len, seq_len), dtype=bool)
                    non_nucleus_mask[start:end, start:end] = False
                    
                    non_nucleus_region = attn_matrix[non_nucleus_mask].reshape(-1)
                    non_nucleus_attention_avg.append(non_nucleus_region.mean())
        
        # Create visualization if data was collected
        if nucleus_attention_avg and non_nucleus_attention_avg:
            self._plot_nucleus_vs_non_nucleus_comparison(nucleus_attention_avg, non_nucleus_attention_avg)
        else:
            print("Warning: No attention data collected for comparison.")
    
    def _plot_nucleus_vs_non_nucleus_comparison(self, nucleus_attention, non_nucleus_attention):
        """Plot comparison of attention in nucleus vs non-nucleus regions"""
        plt.figure(figsize=(10, 6))
        
        indices = range(len(nucleus_attention))
        width = 0.35
        
        plt.bar([i - width/2 for i in indices], nucleus_attention, width, label='Nucleus Region', color='blue', alpha=0.7)
        plt.bar([i + width/2 for i in indices], non_nucleus_attention, width, label='Non-Nucleus Region', color='orange', alpha=0.7)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Average Attention Weight')
        plt.title('Attention Comparison: Nucleus vs Non-Nucleus Regions')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "nucleus_vs_non_nucleus_attention.png"))
        plt.close()
        
        # Also calculate and print statistics
        nucleus_mean = np.mean(nucleus_attention)
        non_nucleus_mean = np.mean(non_nucleus_attention)
        print(f"Nucleus region average attention: {nucleus_mean:.4f}")
        print(f"Non-nucleus region average attention: {non_nucleus_mean:.4f}")
        print(f"Ratio (nucleus/non-nucleus): {nucleus_mean/non_nucleus_mean:.4f}")
        
        with open(os.path.join(self.output_dir, "attention_stats.txt"), "w") as f:
            f.write(f"Nucleus region average attention: {nucleus_mean:.4f}\n")
            f.write(f"Non-nucleus region average attention: {non_nucleus_mean:.4f}\n")
            f.write(f"Ratio (nucleus/non-nucleus): {nucleus_mean/non_nucleus_mean:.4f}\n")
    
    def generate_embeddings(self, use_nucleus=True, use_sig_axis=True, num_samples=100):
        """Generate embeddings with specified features enabled/disabled"""
        # Use a fixed set of samples for fair comparison
        set_seeds(42)
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, indices)
        subset_loader = DataLoader(subset, batch_size=16, shuffle=False)
        
        all_embeddings = []
        all_labels = []
        
        for batch_idx, (seqs, label) in enumerate(subset_loader):
            seqs = seqs.to(self.device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus if enabled
            if use_nucleus:
                batch_nucleus_points = detect_nucleus(energy)
                nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
                nucleus_mask = nucleus_mask.to(self.device)
            else:
                nucleus_mask = None
            
            # Calculate significant axis if enabled
            if use_sig_axis:
                sig_axis = calculate_significant_axis(seqs)
                sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
                sig_axis_mask = sig_axis_mask.to(self.device)
            else:
                sig_axis_mask = None
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(label.cpu().numpy())
        
        # Concatenate results
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels
    
    def visualize_embeddings_tsne(self):
        """Visualize embeddings with t-SNE for different configurations"""
        label_index = self.args.label_index
        
        # Generate embeddings with different configurations
        print("Generating embeddings with different feature configurations...")
        embeddings_full, labels = self.generate_embeddings(use_nucleus=True, use_sig_axis=True)
        embeddings_no_nucleus, _ = self.generate_embeddings(use_nucleus=False, use_sig_axis=True)
        embeddings_no_sig_axis, _ = self.generate_embeddings(use_nucleus=True, use_sig_axis=False)
        embeddings_baseline, _ = self.generate_embeddings(use_nucleus=False, use_sig_axis=False)
        
        # Extract specific label type
        y = labels[:, 0, label_index].astype(int)
        
        # Apply t-SNE to reduce dimensionality
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(y)-1))
        
        embeddings_list = [
            ('Full (Nucleus + Sig Axis)', embeddings_full),
            ('No Nucleus', embeddings_no_nucleus),
            ('No Sig Axis', embeddings_no_sig_axis),
            ('Baseline (No Nucleus, No Sig Axis)', embeddings_baseline)
        ]
        
        plt.figure(figsize=(20, 15))
        
        for idx, (name, embeddings) in enumerate(embeddings_list):
            # Reshape embeddings and apply t-SNE
            flat_embeddings = embeddings.reshape(embeddings.shape[0], -1)
            reduced_embeddings = tsne.fit_transform(flat_embeddings)
            
            # Plot t-SNE
            plt.subplot(2, 2, idx+1)
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                      c=y, cmap='viridis', alpha=0.8, s=50)
            plt.title(f't-SNE Visualization: {name}')
            plt.colorbar(scatter, label='Class')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_comparison.png'), dpi=300)
        plt.close()
        
        print("t-SNE visualization saved.")
    
    def compare_classification_performance(self):
        """Compare classification performance with different feature configurations"""
        label_index = self.args.label_index
        
        print("Generating embeddings for classification comparison...")
        # Generate embeddings with different configurations
        embeddings_full, labels = self.generate_embeddings(use_nucleus=True, use_sig_axis=True, num_samples=300)
        embeddings_no_nucleus, _ = self.generate_embeddings(use_nucleus=False, use_sig_axis=True, num_samples=300)
        embeddings_no_sig_axis, _ = self.generate_embeddings(use_nucleus=True, use_sig_axis=False, num_samples=300)
        embeddings_baseline, _ = self.generate_embeddings(use_nucleus=False, use_sig_axis=False, num_samples=300)
        
        # Extract specific label type
        y = labels[:, 0, label_index].astype(int)
        
        # Prepare embeddings for classification
        embeddings_list = [
            ('Full (Nucleus + Sig Axis)', embeddings_full.reshape(embeddings_full.shape[0], -1)),
            ('No Nucleus', embeddings_no_nucleus.reshape(embeddings_no_nucleus.shape[0], -1)),
            ('No Sig Axis', embeddings_no_sig_axis.reshape(embeddings_no_sig_axis.shape[0], -1)),
            ('Baseline', embeddings_baseline.reshape(embeddings_baseline.shape[0], -1))
        ]
        
        # Prepare results storage
        results = []
        
        print("Training classifiers and evaluating performance...")
        # Train and evaluate classifier for each embedding type
        for name, embeddings in embeddings_list:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'name': name,
                'accuracy': accuracy,
                'f1': f1
            })
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            plot_matrix(cm, self.label_names)
            plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{name.replace(' ', '_')}.png"))
            plt.close()
        
        # Visualize classification performance
        self._plot_classification_performance(results)
        
        print("Classification performance comparison saved.")
    
    def _plot_classification_performance(self, results):
        """Plot classification performance metrics"""
        plt.figure(figsize=(12, 6))
        
        # Extract data
        names = [r['name'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1'] for r in results]
        
        # Bar width
        width = 0.35
        
        # Positions
        x = np.arange(len(names))
        
        # Create bars
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
        plt.bar(x + width/2, f1_scores, width, label='F1 Score', color='green', alpha=0.7)
        
        # Add labels and legend
        plt.xlabel('Embedding Configuration')
        plt.ylabel('Score')
        plt.title('Classification Performance Comparison')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'classification_performance.png'), dpi=300)
        plt.close()
        
        # Also save as text
        with open(os.path.join(self.output_dir, "classification_results.txt"), "w") as f:
            f.write("Classification Performance Results:\n\n")
            for result in results:
                f.write(f"{result['name']}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {result['f1']:.4f}\n\n")
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import DataLoader
    from matplotlib.colors import Normalize

    # Add this method to your NucleusImpactVisualizer class
    def compare_attention_heatmaps(self, sample_indices=None, num_samples=3):
        """
        Generate and compare attention heatmaps with different embedding configurations:
        1. Full model (nucleus + significant axis)
        2. No nucleus embedding
        3. No significant axis embedding
        4. No embeddings (baseline)
        """
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in sample_indices:
            print(f"Generating heatmap comparison for sample {idx}...")
            
            # Get sample
            seqs, label = self.dataset[idx]
            seqs = seqs.unsqueeze(0).to(self.device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(self.device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            # Create configurations
            configs = [
                ("Full (Nucleus + Sig Axis)", nucleus_mask, sig_axis_mask),
                ("No Nucleus", None, sig_axis_mask),
                ("No Sig Axis", nucleus_mask, None),
                ("No Embeddings (Baseline)", None, None)
            ]
            
            # Store attention matrices for each configuration
            attention_matrices = []
            
            # Process each configuration
            for name, nuc_mask, sig_mask in configs:
                # Forward pass to get attention weights
                with torch.no_grad():
                    _ = self.model(seqs, nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
                
                # Check if attention scores were captured
                if self.attention_scores is not None:
                    # Process the attention scores
                    attn_matrix = self.attention_scores.cpu().numpy()
                    
                    # Handle different shapes - we want a 2D matrix
                    if len(attn_matrix.shape) == 4:
                        # If shape is [batch_size, num_heads, seq_len, seq_len]
                        # Take the first batch and average across heads
                        attn_matrix = attn_matrix[0].mean(axis=0)
                    elif len(attn_matrix.shape) == 3:
                        # If shape is [batch_size, seq_len, seq_len]
                        attn_matrix = attn_matrix[0]
                    
                    attention_matrices.append((name, attn_matrix))
                else:
                    print("Warning: Attention scores not captured for configuration:", name)
            
            # Create visualization with all configurations
            if attention_matrices:
                self._plot_attention_comparison(
                    attention_matrices=attention_matrices,
                    nucleus_points=batch_nucleus_points[0],
                    sample_idx=idx,
                    label=label
                )
            else:
                print("Warning: No attention data collected for comparison.")

    def _plot_attention_comparison(self, attention_matrices, nucleus_points, sample_idx, label):
        """Plot comparison of attention heatmaps for different configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        # Ensure consistent colormap scaling across all heatmaps
        all_values = np.concatenate([matrix.flatten() for _, matrix in attention_matrices])
        vmin, vmax = np.min(all_values), np.max(all_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Get activity name
        activity_label = int(label[0, self.args.label_index])
        activity_name = self.label_names[activity_label] if self.label_names and activity_label < len(self.label_names) else f"Class {activity_label}"
        
        for i, (name, matrix) in enumerate(attention_matrices):
            ax = axes[i]
            im = sns.heatmap(matrix, ax=ax, cmap='viridis', norm=norm)
            
            # Highlight nucleus region if available and not in "No Nucleus" configuration
            if len(nucleus_points) == 2 and "No Nucleus" not in name:
                start, end = nucleus_points
                # Add rectangle patch to highlight nucleus region
                rect = plt.Rectangle((start, start), end-start, end-start, 
                                fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            
            ax.set_title(f"Attention Map - {name}")
            ax.set_xlabel("Sequence Position")
            ax.set_ylabel("Attention")
        
        plt.suptitle(f"Attention Heatmap Comparison - {activity_name} (Sample {sample_idx})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"attention_comparison_{sample_idx}.png"), dpi=300)
        plt.close()

    def compare_diagonal_attention(self, sample_indices=None, num_samples=3):
        """Compare diagonal attention patterns across different configurations"""
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in sample_indices:
            print(f"Comparing diagonal attention for sample {idx}...")
            
            # Get sample
            seqs, label = self.dataset[idx]
            seqs = seqs.unsqueeze(0).to(self.device)
            
            # Compute energy
            energy = compute_energy(seqs)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = self.generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(self.device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            # Create configurations
            configs = [
                ("Full (Nucleus + Sig Axis)", nucleus_mask, sig_axis_mask),
                ("No Nucleus", None, sig_axis_mask),
                ("No Sig Axis", nucleus_mask, None),
                ("No Embeddings (Baseline)", None, None)
            ]
            
            # Store diagonal attention values
            diagonal_attentions = []
            
            # Process each configuration
            for name, nuc_mask, sig_mask in configs:
                # Forward pass to get attention weights
                with torch.no_grad():
                    _ = self.model(seqs, nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
                
                # Check if attention scores were captured
                if self.attention_scores is not None:
                    # Process the attention scores
                    attn_matrix = self.attention_scores.cpu().numpy()
                    
                    # Handle different shapes
                    if len(attn_matrix.shape) == 4:
                        # If shape is [batch_size, num_heads, seq_len, seq_len]
                        attn_matrix = attn_matrix[0].mean(axis=0)
                    elif len(attn_matrix.shape) == 3:
                        # If shape is [batch_size, seq_len, seq_len]
                        attn_matrix = attn_matrix[0]
                    
                    # Extract diagonal attention
                    seq_len = attn_matrix.shape[0]
                    diagonal = np.diagonal(attn_matrix)
                    
                    # Get nucleus region indices if available
                    if len(batch_nucleus_points[0]) == 2:
                        start, end = batch_nucleus_points[0]
                        nucleus_indices = np.arange(start, end)
                    else:
                        nucleus_indices = np.array([])
                    
                    diagonal_attentions.append((name, diagonal, nucleus_indices))
                else:
                    print("Warning: Attention scores not captured for configuration:", name)
            
            # Create visualization of diagonal attention patterns
            if diagonal_attentions:
                self._plot_diagonal_attention_comparison(
                    diagonal_attentions=diagonal_attentions,
                    sample_idx=idx,
                    label=label
                )
            else:
                print("Warning: No diagonal attention data collected.")

    def _plot_diagonal_attention_comparison(self, diagonal_attentions, sample_idx, label):
        """Plot comparison of diagonal attention patterns"""
        plt.figure(figsize=(14, 8))
        
        # Get activity name
        activity_label = int(label[0, self.args.label_index])
        activity_name = self.label_names[activity_label] if self.label_names and activity_label < len(self.label_names) else f"Class {activity_label}"
        
        # Plot diagonal attention for each configuration
        for name, diagonal, nucleus_indices in diagonal_attentions:
            x = np.arange(len(diagonal))
            plt.plot(x, diagonal, label=name, alpha=0.8, linewidth=2)
            
            # Highlight nucleus region if available
            if len(nucleus_indices) > 0:
                plt.axvspan(nucleus_indices[0], nucleus_indices[-1], 
                            alpha=0.2, color='yellow', label='Nucleus Region' if name == diagonal_attentions[0][0] else None)
        
        plt.xlabel("Sequence Position")
        plt.ylabel("Self-Attention Weight")
        plt.title(f"Diagonal Attention Comparison - {activity_name} (Sample {sample_idx})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"diagonal_attention_comparison_{sample_idx}.png"), dpi=300)
        plt.close()

    

    # Add this to the run_all_visualizations method
    def run_all_visualizations(self):
        """Run all visualizations"""
        print("\n=== Starting nucleus and significant axis impact analysis ===")
        
        try:
            print("\nVisualizing attention with nucleus regions...")
            self.visualize_attention_with_nucleus(num_samples=3)
        except Exception as e:
            import traceback
            print(f"Error in attention visualization: {e}")
            traceback.print_exc()
        
        try:
            print("\nComparing attention heatmaps across configurations...")
            self.compare_attention_heatmaps(num_samples=3)
        except Exception as e:
            import traceback
            print(f"Error in attention heatmap comparison: {e}")
            traceback.print_exc()
        
        try:
            print("\nComparing diagonal attention patterns...")
            self.compare_diagonal_attention(num_samples=3)
        except Exception as e:
            import traceback
            print(f"Error in diagonal attention comparison: {e}")
            traceback.print_exc()
        
        try:
            print("\nVisualizing raw signals with nucleus regions...")
            self.visualize_raw_signal_with_nucleus(num_samples=3)
        except Exception as e:
            import traceback
            print(f"Error in raw signal visualization: {e}")
            traceback.print_exc()
        
        try:
            print("\nComparing attention in nucleus vs non-nucleus regions...")
            self.compare_attention_nucleus_vs_non_nucleus(num_samples=10)
        except Exception as e:
            import traceback
            print(f"Error in nucleus vs non-nucleus comparison: {e}")
            traceback.print_exc()
        
        try:
            print("\nVisualizing embeddings with t-SNE...")
            self.visualize_embeddings_tsne()
        except Exception as e:
            import traceback
            print(f"Error in t-SNE visualization: {e}")
            traceback.print_exc()
        
        try:
            print("\nComparing classification performance...")
            self.compare_classification_performance()
        except Exception as e:
            import traceback
            print(f"Error in classification performance comparison: {e}")
            traceback.print_exc()
        
        print(f"\n=== Analysis complete. Results saved to {self.output_dir} ===")

    def compare_mask_reconstruction(self, sample_indices=None, num_samples=3):
        """
        Compares how the model fills masked sequences with and without nucleus/sig axis embeddings.
        Visualizes both the reconstruction quality and attention patterns.
        
        Args:
            sample_indices: Specific indices to use, if None will select random samples
            num_samples: Number of samples to visualize if sample_indices is None
        """
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in sample_indices:
            print(f"Comparing mask reconstruction for sample {idx}...")
            
            # Get sample
            seqs, label = self.dataset[idx]
            original_seqs = seqs.clone()  # Save original sequence
            
            # Create a masked version of the sequence (mask ~15% of values)
            masked_seqs = seqs.clone()
            seq_len, feature_dim = seqs.shape
            
            # Create mask indices (mask 15% of sequence)
            mask_percentage = 0.15
            num_masks = int(seq_len * mask_percentage)
            mask_indices = np.random.choice(seq_len, num_masks, replace=False)
            
            # Keep track of masked positions and values
            masked_positions = []
            masked_values = []
            
            # Apply masking (replace with zeros or another mask value)
            for pos in mask_indices:
                masked_values.append(masked_seqs[pos].clone())
                masked_seqs[pos] = torch.zeros(feature_dim)
                masked_positions.append(pos)
            
            # Convert to tensors
            masked_positions_tensor = torch.tensor(masked_positions, dtype=torch.long)
            masked_values_tensor = torch.stack(masked_values)
            
            # Prepare batch dimension
            masked_seqs_batch = masked_seqs.unsqueeze(0).to(self.device)
            masked_positions_batch = masked_positions_tensor.unsqueeze(0).to(self.device)
            
            # Compute energy for nucleus detection
            energy = compute_energy(masked_seqs_batch)
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = self.generate_nucleus_mask(masked_seqs_batch.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(self.device)
            
            # Calculate significant axis
            sig_axis = calculate_significant_axis(masked_seqs_batch)
            sig_axis_mask = (masked_seqs_batch.argmax(dim=-1) == sig_axis[:, None]).float()
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            # Create configurations for comparison
            configs = [
                ("Full (Nucleus + Sig Axis)", nucleus_mask, sig_axis_mask),
                ("No Nucleus", None, sig_axis_mask),
                ("No Sig Axis", nucleus_mask, None),
                ("No Embeddings (Baseline)", None, None)
            ]
            
            # Store reconstruction results and attention matrices
            reconstruction_results = []
            attention_matrices = []
            
            # Process each configuration
            for name, nuc_mask, sig_mask in configs:
                # Forward pass to get reconstructed values and attention weights
                # For this to work, you need to modify the model to return reconstructed values
                with torch.no_grad():
                    # Get reconstructions from the model
                    reconstructed = self.model(masked_seqs_batch, masked_positions_batch, nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
                    reconstructed = reconstructed.cpu()
                    
                    # Store reconstruction results
                    reconstruction_results.append((name, reconstructed))
                    
                    # Get attention matrices if available
                    if self.attention_scores is not None:
                        attn_matrix = self.attention_scores.cpu().numpy()
                        
                        # Handle different shapes
                        if len(attn_matrix.shape) == 4:
                            # If shape is [batch_size, num_heads, seq_len, seq_len]
                            attn_matrix = attn_matrix[0].mean(axis=0)
                        elif len(attn_matrix.shape) == 3:
                            # If shape is [batch_size, seq_len, seq_len]
                            attn_matrix = attn_matrix[0]
                        
                        attention_matrices.append((name, attn_matrix))
            
            # Create visualization with all configurations
            if reconstruction_results:
                # Create visualization of masked sequence, reconstructions, and attention
                self._plot_mask_reconstruction_comparison(
                    original_seqs=original_seqs.numpy(),
                    masked_seqs=masked_seqs.numpy(),
                    masked_positions=masked_positions,
                    reconstruction_results=reconstruction_results,
                    attention_matrices=attention_matrices,
                    nucleus_points=batch_nucleus_points[0],
                    sample_idx=idx,
                    label=label
                )
            else:
                print("Warning: No reconstruction data collected.")

    def _plot_mask_reconstruction_comparison(self, original_seqs, masked_seqs, masked_positions, 
                                         reconstruction_results, attention_matrices, 
                                         nucleus_points, sample_idx, label):
        """
        Plot comparison of mask reconstruction and attention for different configurations.
        
        Args:
            original_seqs: Original sequence data
            masked_seqs: Masked sequence data
            masked_positions: Positions that were masked
            reconstruction_results: List of (name, reconstructed) tuples
            attention_matrices: List of (name, matrix) tuples for attention visualizations
            nucleus_points: Start and end of nucleus region
            sample_idx: Sample index
            label: Label data
        """
        # Get activity name
        activity_label = int(label[0, self.args.label_index])
        activity_name = self.label_names[activity_label] if self.label_names and activity_label < len(self.label_names) else f"Class {activity_label}"
        
        # Calculate reconstruction error metrics for each configuration
        mse_values = {}
        for name, reconstructed in reconstruction_results:
            # Debug prints for shape information
            print(f"Shape of original_seqs: {original_seqs.shape}")
            print(f"Shape of reconstructed: {reconstructed.shape}")
            
            # Extract only the FIRST 6 FEATURES from reconstructed to match original
            # or use dimensionality reduction if appropriate
            if reconstructed.shape[2] > original_seqs.shape[1]:
                print(f"Reconstructed has more features ({reconstructed.shape[2]}) than original ({original_seqs.shape[1]})")
                print("Using only the first matching features for comparison")
                
                # Option 1: Use only the first N features that match original
                reconstructed_trimmed = reconstructed[0, :, :original_seqs.shape[1]].numpy()
                
                # Calculate MSE only on masked positions
                mse_sum = 0
                for pos in masked_positions:
                    pos_mse = np.mean((original_seqs[pos] - reconstructed_trimmed[pos])**2)
                    mse_sum += pos_mse
                
                mse = mse_sum / len(masked_positions)
                mse_values[name] = mse
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Define grid layout: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1])
        
        # Plot 1: Original vs Masked Signal
        ax1 = fig.add_subplot(gs[0, :])
        seq_len, feature_dim = original_seqs.shape
        
        # Plot first 3 dimensions of original signal
        time = np.arange(seq_len)
        for j in range(min(3, feature_dim)):
            ax1.plot(time, original_seqs[:, j], label=f'Original Dim {j}', alpha=0.7)
        
        # Highlight masked positions
        for pos in masked_positions:
            ax1.axvline(pos, color='red', alpha=0.3, linestyle='--')
        
        # Highlight nucleus region
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            ax1.axvspan(start, end, alpha=0.2, color='yellow', label='Nucleus Region')
        
        ax1.set_title("Original Signal with Masked Positions", fontsize=14)
        ax1.set_xlabel("Sequence Position")
        ax1.set_ylabel("Signal Value")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2-5: Attention Matrices for each configuration
        if attention_matrices:
            # Create subplots for attention matrices
            axes_attn = [fig.add_subplot(gs[1, i]) for i in range(2)]
            
            # Ensure consistent colormap scaling across all heatmaps
            all_values = np.concatenate([matrix.flatten() for _, matrix in attention_matrices[:2]])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            # Plot first two attention matrices
            for i, (name, matrix) in enumerate(attention_matrices[:2]):
                ax = axes_attn[i]
                im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                
                # Highlight nucleus region if available
                if len(nucleus_points) == 2 and i == 0:  # Only for Full model
                    start, end = nucleus_points
                    rect = plt.Rectangle((start, start), end-start, end-start, 
                                    fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                
                # Highlight masked positions
                for pos in masked_positions:
                    ax.axhline(pos, color='white', alpha=0.3, linestyle='--')
                    ax.axvline(pos, color='white', alpha=0.3, linestyle='--')
                
                ax.set_title(f"Attention Map - {name}", fontsize=14)
                ax.set_xlabel("Sequence Position")
                ax.set_ylabel("Attention")
                fig.colorbar(im, ax=ax)
        
        # Plot 6: Reconstruction Error Comparison
        ax_error = fig.add_subplot(gs[2, :])
        
        # Bar chart of MSE values
        names = list(mse_values.keys())
        values = [mse_values[name] for name in names]
        
        bars = ax_error.bar(names, values, color=['blue', 'orange', 'green', 'red'])
        
        # Add error values on top of each bar
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax_error.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.5f}', ha='center', va='bottom', rotation=0, fontsize=12)
        
        ax_error.set_title("Reconstruction Error (MSE) Comparison", fontsize=14)
        ax_error.set_ylabel("Mean Squared Error")
        ax_error.grid(True, alpha=0.3, axis='y')
        
        # Add a title for the entire figure
        plt.suptitle(f"Mask Reconstruction Comparison - {activity_name} (Sample {sample_idx})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"mask_reconstruction_comparison_{sample_idx}.png"), dpi=300)
        plt.close()

    def compare_mask_fill_pattern(self, sample_indices=None, num_samples=3):
        """
        Visualizes how the model fills masked values in different regions of the sequence
        with and without nucleus/sig axis embeddings, using correlation as a metric.
        """
        if sample_indices is None:
            # Use fixed seed for reproducibility
            set_seeds(42)
            sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in sample_indices:
            print(f"Comparing mask fill pattern for sample {idx}...")
            
            # Get sample
            seqs, label = self.dataset[idx]
            original_seqs = seqs.clone()  # Save original sequence
            seq_len, feature_dim = seqs.shape
            
            # Compute energy for nucleus detection
            energy = compute_energy(seqs.unsqueeze(0))
            
            # Detect nucleus
            batch_nucleus_points = detect_nucleus(energy)
            start, end = batch_nucleus_points[0] if len(batch_nucleus_points[0]) == 2 else (seq_len//3, 2*seq_len//3)
            
            # Define three regions: pre-nucleus, nucleus, post-nucleus
            pre_nucleus = list(range(0, start))
            nucleus = list(range(start, end))
            post_nucleus = list(range(end, seq_len))
            
            # Create masks for each region (mask ~30% of each region)
            mask_percentage = 0.3
            regions = [
                ("Pre-Nucleus", pre_nucleus),
                ("Nucleus", nucleus),
                ("Post-Nucleus", post_nucleus)
            ]
            
            # Store results by region
            results_by_region = {}
            
            for region_name, region_indices in regions:
                if not region_indices:  # Skip empty regions
                    continue
                    
                # Create mask indices for this region
                num_masks = max(1, int(len(region_indices) * mask_percentage))
                mask_indices = np.random.choice(region_indices, num_masks, replace=False)
                
                # Apply masking
                masked_seqs = original_seqs.clone()
                masked_values = []
                
                for pos in mask_indices:
                    masked_values.append(masked_seqs[pos].clone())
                    masked_seqs[pos] = torch.zeros(feature_dim)
                
                # Convert to batch format
                masked_seqs_batch = masked_seqs.unsqueeze(0).to(self.device)
                masked_positions_tensor = torch.tensor(mask_indices, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Create nucleus and sig axis masks
                nucleus_mask = self.generate_nucleus_mask(seq_len, batch_nucleus_points)
                nucleus_mask = nucleus_mask.to(self.device)
                
                sig_axis = calculate_significant_axis(masked_seqs_batch)
                sig_axis_mask = (masked_seqs_batch.argmax(dim=-1) == sig_axis[:, None]).float()
                sig_axis_mask = sig_axis_mask.to(self.device)
                
                # Create configurations
                configs = [
                    ("Full (Nucleus + Sig Axis)", nucleus_mask, sig_axis_mask),
                    ("No Nucleus", None, sig_axis_mask),
                    ("No Sig Axis", nucleus_mask, None),
                    ("No Embeddings (Baseline)", None, None)
                ]
                
                # Store reconstruction results for this region
                region_results = []
                
                # Process each configuration
                for name, nuc_mask, sig_mask in configs:
                    # Forward pass to get reconstructed values
                    with torch.no_grad():
                        reconstructed = self.model(masked_seqs_batch, masked_positions_tensor, 
                                                nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
                    
                    # Calculate correlation instead of MSE
                    correlation_sum = 0
                    count = 0
                    
                    # Use only the first feature_dim features from reconstructed
                    reconstructed_trimmed = reconstructed[0, :, :feature_dim].cpu()
                    
                    for pos in mask_indices:
                        # Calculate correlation for matched features
                        orig_values = original_seqs[pos]
                        recon_values = reconstructed_trimmed[pos]
                        
                        # Calculate average absolute difference
                        diff = torch.mean(torch.abs(orig_values - recon_values)).item()
                        correlation_sum += diff
                        count += 1
                    
                    # Calculate average difference (lower is better)
                    avg_diff = correlation_sum / count if count > 0 else 0
                    region_results.append((name, avg_diff))
                
                results_by_region[region_name] = region_results
            
            # Create visualization of reconstruction performance by region
            self._plot_mask_fill_by_region(
                results_by_region=results_by_region,
                sample_idx=idx,
                label=label,
                nucleus_points=(start, end)
            )
    
    def _plot_mask_fill_by_region(self, results_by_region, sample_idx, label, nucleus_points):
        """
        Plot mask fill performance by region for different configurations using correlation metric.
        """
        # Get activity name
        activity_label = int(label[0, self.args.label_index])
        activity_name = self.label_names[activity_label] if self.label_names and activity_label < len(self.label_names) else f"Class {activity_label}"
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Extract configuration names (assuming consistent across regions)
        config_names = [result[0] for result in next(iter(results_by_region.values()))]
        
        # Number of regions and configurations
        num_regions = len(results_by_region)
        num_configs = len(config_names)
        
        # Set up bar positions
        bar_width = 0.2
        r = np.arange(num_regions)
        
        # Plot bars for each configuration
        for i, config_name in enumerate(config_names):
            # Extract correlation values for this configuration across all regions
            diff_values = []
            region_names = []
            
            for region_name, region_results in results_by_region.items():
                diff = next(res[1] for res in region_results if res[0] == config_name)
                diff_values.append(diff)
                region_names.append(region_name)
            
            # Plot bars
            bars = plt.bar(r + i*bar_width, diff_values, width=bar_width, label=config_name)
            
            # Add values on top of each bar
            for bar, value in zip(bars, diff_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        # Add labels and legend
        plt.xlabel('Sequence Region')
        plt.ylabel('Average Absolute Difference (lower is better)')
        plt.title(f'Mask Reconstruction Quality by Region - {activity_name} (Sample {sample_idx})')
        plt.xticks(r + bar_width * (num_configs-1) / 2, region_names)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add annotation for nucleus region
        start, end = nucleus_points
        plt.annotate(f'Nucleus Region: {start}-{end}', xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"mask_fill_by_region_{sample_idx}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize nucleus and significant axis impact')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the pretrained model (without .pt extension)')
    parser.add_argument('--model_version', type=str, default='v1',
                        help='Model version (default: v1)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., motion, hhar, uci, shoaib)')
    parser.add_argument('--version', type=str, default='20_120',
                        help='Dataset version (default: 20_120)')
    parser.add_argument('--label_index', type=int, default=0,
                        help='Label index for classification (default: 0, typically activity)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID (default: -1 for CPU)')
    
    args = parser.parse_args()
    
    visualizer = NucleusImpactVisualizer(args)
    #visualizer.run_all_visualizations()
    visualizer.compare_mask_reconstruction(num_samples=3)
    visualizer.compare_mask_fill_pattern(num_samples=3) 