#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Visualize embeddings for 5 selected gesture classes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import os

# Try to import UMAP correctly
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap.umap_ as umap
        UMAP = umap.UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        from sklearn.manifold import TSNE
        UMAP_AVAILABLE = False
        print("UMAP not available, falling back to t-SNE")

def visualize_selected_classes(baseline_path, contrastive_path, labels_path, output_path, 
                               selected_classes=[0, 1, 2, 3, 4], class_names=None):
    """
    Create and save a visualization of baseline vs contrastive embeddings for selected classes
    
    Args:
        baseline_path: Path to baseline embeddings .npy file
        contrastive_path: Path to contrastive embeddings .npy file
        labels_path: Path to labels .npy file
        output_path: Path to save the visualization
        selected_classes: List of class indices to include (0-indexed)
        class_names: Optional list of class names
    """
    # Load embeddings and labels
    print(f"Loading embeddings and labels...")
    baseline_embeddings = np.load(baseline_path)
    contrastive_embeddings = np.load(contrastive_path)
    labels = np.load(labels_path)
    
    print(f"Baseline embeddings shape: {baseline_embeddings.shape}")
    print(f"Contrastive embeddings shape: {contrastive_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Process labels if needed
    if len(labels.shape) > 2:
        # Extract first label dimension (usually activity labels)
        labels = labels[:, 0, 0]
    
    # Adjust labels if they're 1-indexed
    if np.min(labels) == 1:
        print("Labels appear to be 1-indexed, adjusting to 0-indexed")
        labels = labels - 1
    
    # Filter for selected classes
    selected_mask = np.isin(labels, selected_classes)
    baseline_embeddings = baseline_embeddings[selected_mask]
    contrastive_embeddings = contrastive_embeddings[selected_mask]
    labels = labels[selected_mask]
    
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    print(f"Selected {n_classes} classes with {len(labels)} samples")
    
    # Use provided class names or generate defaults
    if class_names is None or len(class_names) < max(selected_classes) + 1:
        class_names = [f"Class {i}" for i in range(max(selected_classes) + 1)]
    
    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Use a consistent colormap
    cmap = plt.cm.Set1
    colors = [cmap(i % cmap.N) for i in range(n_classes)]
    
    # Standardize data
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline_embeddings)
    contrastive_scaled = scaler.fit_transform(contrastive_embeddings)
    
    # Apply dimensionality reduction
    if UMAP_AVAILABLE:
        print("Applying UMAP dimensionality reduction...")
        
        # Apply UMAP to baseline embeddings
        reducer_baseline = UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        baseline_reduced = reducer_baseline.fit_transform(baseline_scaled)
        
        # Apply UMAP to contrastive embeddings
        reducer_contrastive = UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        contrastive_reduced = reducer_contrastive.fit_transform(contrastive_scaled)
        
        method_name = "UMAP"
    else:
        print("Applying t-SNE dimensionality reduction...")
        
        # Apply t-SNE to baseline embeddings
        tsne_baseline = TSNE(n_components=2, random_state=42, perplexity=min(30, len(baseline_scaled)-1))
        baseline_reduced = tsne_baseline.fit_transform(baseline_scaled)
        
        # Apply t-SNE to contrastive embeddings
        tsne_contrastive = TSNE(n_components=2, random_state=42, perplexity=min(30, len(contrastive_scaled)-1))
        contrastive_reduced = tsne_contrastive.fit_transform(contrastive_scaled)
        
        method_name = "t-SNE"
    
    # Plot baseline embeddings
    for i, cls in enumerate(unique_classes):
        mask = (labels == cls)
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        
        axes[0].scatter(
            baseline_reduced[mask, 0],
            baseline_reduced[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=0.7,
            s=60
        )
    
    axes[0].set_title(f"Baseline Embeddings ({method_name})", fontsize=14)
    axes[0].set_xlabel(f"Component 1", fontsize=12)
    axes[0].set_ylabel(f"Component 2", fontsize=12)
    axes[0].legend(loc='best', fontsize=10)
    
    # Plot contrastive embeddings
    for i, cls in enumerate(unique_classes):
        mask = (labels == cls)
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        
        axes[1].scatter(
            contrastive_reduced[mask, 0],
            contrastive_reduced[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=0.7,
            s=60
        )
    
    axes[1].set_title(f"Contrastive + Semantic Embeddings ({method_name})", fontsize=14)
    axes[1].set_xlabel(f"Component 1", fontsize=12)
    axes[1].set_ylabel(f"Component 2", fontsize=12)
    axes[1].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle("Embedding Visualization: 5 Selected Gesture Classes", fontsize=16, y=1.05)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize selected classes from embeddings')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline embeddings .npy file')
    parser.add_argument('--contrastive', type=str, required=True, help='Path to contrastive embeddings .npy file')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels .npy file')
    parser.add_argument('--output', type=str, default='five_classes_visualization.png', help='Output path')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 1, 2, 3, 4], 
                        help='Class indices to visualize (0-indexed)')
    args = parser.parse_args()
    
    # Define class names based on your dataset
    class_names = ["up", "down", "left", "forearm right", "rotate wrist and right", "rotate wrist and left", "flick wrist and up", "flick wrist and down", "flick wrist and left", "flick wrist and right", "square", "circle"],

    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Visualize selected classes
    visualize_selected_classes(
        args.baseline,
        args.contrastive,
        args.labels,
        args.output,
        selected_classes=args.classes,
        class_names=class_names
    )

if __name__ == "__main__":
    main()