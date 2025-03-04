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
    
    def run_all_visualizations(self):
        """Run all visualizations"""
        print("\n=== Starting nucleus and significant axis impact analysis ===")
        
        print("\nVisualizing attention with nucleus regions...")
        self.visualize_attention_with_nucleus(num_samples=3)
        
        print("\nVisualizing raw signals with nucleus regions...")
        self.visualize_raw_signal_with_nucleus(num_samples=3)
        
        print("\nComparing attention in nucleus vs non-nucleus regions...")
        self.compare_attention_nucleus_vs_non_nucleus(num_samples=10)
        
        print("\nVisualizing embeddings with t-SNE...")
        self.visualize_embeddings_tsne()
        
        print("\nComparing classification performance...")
        self.compare_classification_performance()
        
        print(f"\n=== Analysis complete. Results saved to {self.output_dir} ===")


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
    visualizer.run_all_visualizations()