#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Generate embeddings using the project's contrastive and semantic loss implementation

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import argparse
import random

# Import project-specific modules
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.semantic_loss import SemanticLoss
from contrastive.models import ContrastiveGRUClassifier, ContrastiveTransformerClassifier
from contrastive.augmenter import GestureAugmenter
from utils import get_device, IMUDataset, set_seeds
from embedding import load_embedding_label
from config import load_dataset_label_names, load_dataset_stats

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Fix the label indexing in your training code
def train_baseline(model, train_loader, val_loader, device, learning_rate=0.001, epochs=50, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize best validation accuracy
    best_val_acc = 0.0
    best_model = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            
            # Extract label component for classification
            if len(labels.shape) > 2:
                class_labels = labels[:, 0, 0].long()
            else:
                class_labels = labels.long()
            
            # Adjust label indices to be 0-indexed
            class_labels = class_labels - 1  # Subtract 1 to convert from 1-indexed to 0-indexed
            
            # Double-check that labels are within valid range
            if torch.min(class_labels) < 0 or torch.max(class_labels) >= model.classifier.out_features:
                print(f"Warning: After adjustment, found labels outside valid range: min={torch.min(class_labels)}, max={torch.max(class_labels)}")
                print(f"Expected range: 0 to {model.classifier.out_features-1}")
                # Clip labels to valid range as a safeguard
                class_labels = torch.clamp(class_labels, 0, model.classifier.out_features-1)
            
            class_labels = class_labels.to(device)
            
            # Forward pass (only using classification loss)
            logits = model(inputs)
            
            loss = criterion(logits, class_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                
                # Extract label component for classification
                if len(labels.shape) > 2:
                    class_labels = labels[:, 0, 0].long()
                else:
                    class_labels = labels.long()
                
                # Adjust label indices to be 0-indexed
                class_labels = class_labels - 1  # Subtract 1 to convert from 1-indexed to 0-indexed
                class_labels = torch.clamp(class_labels, 0, model.classifier.out_features-1)
                class_labels = class_labels.to(device)
                
                # Forward pass
                logits = model(inputs)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += class_labels.size(0)
                val_correct += (predicted == class_labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            
            # Save model checkpoint
            if save_path:
                torch.save(best_model, save_path)
                print(f"Model saved to {save_path}")
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

# Similarly, update the train_with_contrastive_semantic function to adjust labels
def train_with_contrastive_semantic(model, criterion, train_loader, val_loader, device, 
                                   learning_rate=0.001, epochs=50, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize best validation accuracy
    best_val_acc = 0.0
    best_model = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        losses_dict = {
            'classification_loss': 0.0,
            'semantic_loss': 0.0,
            'contrastive_loss': 0.0,
            'total_loss': 0.0
        }
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            
            # Extract label component for classification
            if len(labels.shape) > 2:
                class_labels = labels[:, 0, 0].long()
            else:
                class_labels = labels.long()
            
            # Adjust label indices to be 0-indexed
            class_labels = class_labels - 1  # Subtract 1 to convert from 1-indexed to 0-indexed
            
            # Ensure labels are within valid range
            class_labels = torch.clamp(class_labels, 0, model.classifier.out_features-1)
            class_labels = class_labels.to(device)
            
            # Forward pass with features and projected outputs
            logits, features, projected = model(inputs, return_features=True)
            
            # Compute combined loss with all components
            total_loss, batch_losses = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=class_labels,
                epoch=epoch
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += total_loss.item()
            for key, value in batch_losses.items():
                losses_dict[key] += value
        
        # Calculate average losses
        for key in losses_dict:
            losses_dict[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                
                # Extract label component for classification
                if len(labels.shape) > 2:
                    class_labels = labels[:, 0, 0].long()
                else:
                    class_labels = labels.long()
                
                # Adjust label indices to be 0-indexed
                class_labels = class_labels - 1
                class_labels = torch.clamp(class_labels, 0, model.classifier.out_features-1)
                class_labels = class_labels.to(device)
                
                # Forward pass
                logits = model(inputs)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += class_labels.size(0)
                val_correct += (predicted == class_labels).sum().item()
        
        val_acc = val_correct / val_total
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {losses_dict['total_loss']:.4f} | "
              f"Class Loss: {losses_dict['classification_loss']:.4f} | "
              f"Semantic Loss: {losses_dict['semantic_loss']:.4f} | "
              f"Contrastive Loss: {losses_dict['contrastive_loss']:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            
            # Save model checkpoint
            if save_path:
                torch.save(best_model, save_path)
                print(f"Model saved to {save_path}")
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

# Also update the label adjustment in the generate_embeddings function
def generate_embeddings(model, data_loader, device):
    model.eval()
    all_logits = []
    all_features = []
    all_projected = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Store original labels (without adjustment) for visualization
            all_labels.append(labels.cpu().numpy())
            
            # Pass through model to get all outputs
            logits, features, projected = model(inputs, return_features=True)
            
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_projected.append(projected.cpu().numpy())
    
    # Stack all outputs
    logits = np.vstack(all_logits)
    features = np.vstack(all_features)
    projected = np.vstack(all_projected)
    labels = np.vstack(all_labels)
    
    return logits, features, projected, labels
        

# Visualize embeddings using dimensionality reduction
def visualize_embeddings(baseline_embeddings, contrastive_embeddings, labels, class_names=None):
    """
    Visualize embeddings with dimensionality reduction
    
    Args:
        baseline_embeddings: Embeddings without contrastive/semantic loss
        contrastive_embeddings: Embeddings with contrastive/semantic loss
        labels: Label data
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    # Process labels
    if len(labels.shape) > 2:
        labels = labels[:, 0, 0]
    
    # Get unique classes
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Use a consistent colormap
    cmap = plt.cm.tab10
    colors = [cmap(i % cmap.N) for i in range(n_classes)]
    
    # Standardize data
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline_embeddings)
    contrastive_scaled = scaler.fit_transform(contrastive_embeddings)
    
    # Try to use UMAP first, fall back to t-SNE
    try:
        import umap
        
        # Apply UMAP to baseline embeddings
        reducer_baseline = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        baseline_reduced = reducer_baseline.fit_transform(baseline_scaled)
        
        # Apply UMAP to contrastive embeddings
        reducer_contrastive = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        contrastive_reduced = reducer_contrastive.fit_transform(contrastive_scaled)
        
        method_name = "UMAP"
    except ImportError:
        # Fall back to t-SNE
        print("UMAP not available, using t-SNE instead")
        
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
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {cls}"
        
        axes[0].scatter(
            baseline_reduced[mask, 0],
            baseline_reduced[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=0.7,
            s=40
        )
    
    axes[0].set_title(f"Baseline Embeddings ({method_name})", fontsize=14)
    axes[0].set_xlabel(f"Component 1", fontsize=12)
    axes[0].set_ylabel(f"Component 2", fontsize=12)
    axes[0].legend(loc='best')
    
    # Plot contrastive embeddings
    for i, cls in enumerate(unique_classes):
        mask = (labels == cls)
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class {cls}"
        
        axes[1].scatter(
            contrastive_reduced[mask, 0],
            contrastive_reduced[mask, 1],
            color=colors[i],
            label=class_name,
            alpha=0.7,
            s=40
        )
    
    axes[1].set_title(f"Contrastive + Semantic Embeddings ({method_name})", fontsize=14)
    axes[1].set_xlabel(f"Component 1", fontsize=12)
    axes[1].set_ylabel(f"Component 2", fontsize=12)
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.suptitle("Embedding Visualization: Baseline vs Contrastive+Semantic Learning", fontsize=16, y=1.05)
    
    return fig

# Calculate cluster metrics
def calculate_cluster_metrics(embeddings, labels):
    """
    Calculate metrics for cluster separation
    
    Args:
        embeddings: Embedding matrix
        labels: Label array
        
    Returns:
        Dictionary with metrics
    """
    # Process labels
    if len(labels.shape) > 2:
        labels = labels[:, 0, 0]
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Calculate centroids for each class
    centroids = []
    for cls in unique_classes:
        mask = (labels == cls)
        centroids.append(np.mean(embeddings[mask], axis=0))
    centroids = np.array(centroids)
    
    # Calculate inter-cluster distances (between centroids)
    inter_distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_distances.append(dist)
    
    # Calculate intra-cluster distances (within clusters)
    intra_distances = []
    for i, cls in enumerate(unique_classes):
        mask = (labels == cls)
        class_samples = embeddings[mask]
        centroid = centroids[i]
        
        # Calculate distances to centroid
        dists = np.linalg.norm(class_samples - centroid, axis=1)
        intra_distances.append(np.mean(dists))
    
    # Calculate metrics
    avg_inter_distance = np.mean(inter_distances)
    avg_intra_distance = np.mean(intra_distances)
    separation_ratio = avg_inter_distance / avg_intra_distance
    
    return {
        "avg_inter_cluster_distance": avg_inter_distance,
        "avg_intra_cluster_distance": avg_intra_distance,
        "separation_ratio": separation_ratio
    }

# Visualize cluster metrics comparison
def visualize_metrics(baseline_metrics, contrastive_metrics, output_file='cluster_metrics.png'):
    """
    Visualize cluster metrics comparison
    
    Args:
        baseline_metrics: Metrics for baseline embeddings
        contrastive_metrics: Metrics for contrastive embeddings
        output_file: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    metrics = ['Avg Inter-Cluster\nDistance', 'Avg Intra-Cluster\nDistance', 'Separation Ratio']
    baseline_values = [
        baseline_metrics['avg_inter_cluster_distance'],
        baseline_metrics['avg_intra_cluster_distance'],
        baseline_metrics['separation_ratio']
    ]
    contrastive_values = [
        contrastive_metrics['avg_inter_cluster_distance'],
        contrastive_metrics['avg_intra_cluster_distance'],
        contrastive_metrics['separation_ratio']
    ]
    
    # Set positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#1f77b4', alpha=0.8)
    contrastive_bars = ax.bar(x + width/2, contrastive_values, width, label='Contrastive + Semantic', color='#ff7f0e', alpha=0.8)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(baseline_bars)
    add_labels(contrastive_bars)
    
    # Add improvement percentage annotations
    improvements = [
        (contrastive_values[0] / baseline_values[0] - 1) * 100,  # Inter-cluster improvement
        (1 - contrastive_values[1] / baseline_values[1]) * 100,  # Intra-cluster improvement (lower is better)
        (contrastive_values[2] / baseline_values[2] - 1) * 100   # Separation ratio improvement
    ]
    
    for i, (improvement, contrastive_value) in enumerate(zip(improvements, contrastive_values)):
        improvement_text = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        color = 'green' if improvement > 0 else 'red'
        
        # For intra-cluster distance, lower is better so color coding is flipped
        if i == 1:
            color = 'green' if improvement > 0 else 'red'
        
        ax.annotate(improvement_text,
                    xy=(x[i] + width/2, contrastive_value),
                    xytext=(0, 12),  # 12 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color=color, fontweight='bold')
    
    # Add labels and legend
    ax.set_ylabel('Value')
    ax.set_title('Cluster Separation Metrics: Baseline vs Contrastive + Semantic Learning', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics visualization saved to {output_file}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Train and visualize contrastive embeddings with LIMU-BERT')
    parser.add_argument('--dataset', type=str, default='gesture', help='Dataset name')
    parser.add_argument('--version', type=str, default='20_120', help='Dataset version (default: 20_120)')
    parser.add_argument('--model_file', type=str, default=None, help='Pretrained model file name')
    parser.add_argument('--label_index', type=int, default=0, help='Label index to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--output_dir', type=str, default='contrastive_output', help='Output directory')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    try:
        # Try to load dataset stats for label names
        dataset_cfg = load_dataset_stats(args.dataset, args.version)
        label_names, label_num, descriptions = load_dataset_label_names(dataset_cfg, args.label_index)
        
        if descriptions is None:
            print("No descriptions found in dataset config, using class names as descriptions")
            descriptions = [f"{name} gesture" for name in label_names]
        
        print(f"Loaded {label_num} classes: {label_names}")
        print(f"Class descriptions available: {descriptions is not None}")
    except Exception as e:
        print(f"Error loading dataset stats: {e}")
        print("Using default class names and descriptions")
        label_names = [f"Gesture {i+1}" for i in range(5)]
        descriptions = [f"Gesture {i+1}" for i in range(5)]
        label_num = 5
    
    try:
        # Load embeddings from LIMU-BERT
        print(f"Loading embeddings for dataset: {args.dataset} version: {args.version}")
        embeddings, labels = load_embedding_label(args.model_file, args.dataset, args.version)
        print(f"Loaded embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")
        
        if len(labels.shape) > 2:
            # Get number of classes from the labels
            unique_labels = np.unique(labels[:, 0, args.label_index])
            label_num = len(unique_labels)
            print(f"Found {label_num} unique classes in the labels")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Please make sure you have generated embeddings with the LIMU-BERT model first")
        return
    
    # Create dataset with augmentation
    augmenter = GestureAugmenter()
    dataset = IMUDataset(embeddings, labels, pipeline=[augmenter.augment])
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    full_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    # Model parameters
    #input_dim = embeddings.shape[1]
    input_dim = 72
    hidden_dim = 128
    proj_dim = 128
    
    # Initialize baseline model
    baseline_model = ContrastiveTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=label_num,
        proj_dim=proj_dim
    ).to(device)
    
    # Initialize contrastive model (same architecture, different training)
    contrastive_model = ContrastiveTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=label_num,
        proj_dim=proj_dim
    ).to(device)
    
    # Initialize combined loss
    contrastive_criterion = ContrastiveCombinedLoss(
        label_names=label_names,
        descriptions=descriptions,
        pooling="cls",  # Using CLS token pooling for BERT
        device=device,
        temperature=0.07,
        hidden_dim=hidden_dim
    )
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_model = train_baseline(
        baseline_model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        epochs=args.epochs,
        save_path=os.path.join(args.output_dir, "baseline_model.pt")
    )
    
    # Train contrastive model
    print("\nTraining contrastive model with semantic loss...")
    contrastive_model = train_with_contrastive_semantic(
        contrastive_model,
        contrastive_criterion,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        epochs=args.epochs,
        save_path=os.path.join(args.output_dir, "contrastive_model.pt")
    )
    
    # Generate embeddings
    print("\nGenerating embeddings from both models...")
    
    # Baseline embeddings
    _, baseline_features, _, labels_test = generate_embeddings(
        baseline_model,
        full_loader,
        device
    )
    
    # Contrastive embeddings
    _, _, contrastive_projected, _ = generate_embeddings(
        contrastive_model,
        full_loader,
        device
    )
    
    # Save embeddings
    np.save(os.path.join(args.output_dir, "baseline_embeddings.npy"), baseline_features)
    np.save(os.path.join(args.output_dir, "contrastive_embeddings.npy"), contrastive_projected)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels_test)
    
    # Calculate cluster metrics
    baseline_metrics = calculate_cluster_metrics(baseline_features, labels_test)
    contrastive_metrics = calculate_cluster_metrics(contrastive_projected, labels_test)
    
    # Print metrics
    print("\nBaseline Embedding Metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nContrastive Embedding Metrics:")
    for k, v in contrastive_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calculate improvement
    improvement = contrastive_metrics['separation_ratio'] / baseline_metrics['separation_ratio']
    print(f"\nSeparation Ratio Improvement: {improvement:.2f}x better ({(improvement-1)*100:.1f}%)")
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    vis_fig = visualize_embeddings(
        baseline_features,
        contrastive_projected,
        labels_test,
        class_names=label_names
    )
    
    # Save visualization
    vis_fig.savefig(os.path.join(args.output_dir, "embedding_visualization.png"), dpi=300, bbox_inches='tight')
    print(f"Embedding visualization saved to {os.path.join(args.output_dir, 'embedding_visualization.png')}")
    
    # Visualize metrics
    print("\nVisualizing cluster metrics...")
    metrics_fig = visualize_metrics(
        baseline_metrics,
        contrastive_metrics,
        output_file=os.path.join(args.output_dir, "cluster_metrics.png")
    )
    
    print("\nDone! All outputs saved to", args.output_dir)

if __name__ == "__main__":
    main()