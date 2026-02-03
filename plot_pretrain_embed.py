import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os

def load_model_and_data(embed_path, input_data_path, label_data_path):
    """
    Load the saved embeddings and data files
    """
    input_data = np.load(input_data_path)
    label_data = np.load(label_data_path)
    embed_data = np.load(embed_path)    
    
    print(f"Loaded input data with shape: {input_data.shape}")
    print(f"Loaded label data with shape: {label_data.shape}")
    print(f"Loaded embed data with shape: {embed_data.shape}")
    
    return embed_data, input_data, label_data

def prepare_embeddings_for_pca(embeddings):
    """
    Prepare 3D embeddings for PCA by averaging over sequence dimension
    """
    # Mean pooling over sequence length
    mean_embeddings = np.mean(embeddings, axis=1)
    print(f"Prepared embeddings with shape: {mean_embeddings.shape}")
    return mean_embeddings

def select_representative_samples(embeddings, labels, n_samples_per_class=10):
    """
    Select representative samples from each class that maximize inter-class distance
    and minimize intra-class distance.
    
    Args:
        embeddings: 2D array of shape (n_samples, n_features)
        labels: 1D array of shape (n_samples,)
        n_samples_per_class: Number of samples to select per class
        
    Returns:
        selected_indices: List of indices of selected samples
    """
    unique_labels = np.unique(labels)
    selected_indices = []
    
    print(f"Selecting {n_samples_per_class} representative samples for each of {len(unique_labels)} classes...")
    
    # Calculate distance matrix once
    distances = euclidean_distances(embeddings)
    
    for label in unique_labels:
        if label < 1 or label > 12:  # Skip invalid labels
            continue
            
        # Get indices for this label
        label_indices = np.where(labels == label)[0]
        
        if len(label_indices) <= n_samples_per_class:
            # If we have fewer samples than requested, use all of them
            selected_indices.extend(label_indices)
            print(f"  Class {label}: Using all {len(label_indices)} available samples")
            continue
        
        # Calculate scores for each sample in this class
        scores = []
        
        for idx in label_indices:
            # Get distances to all other samples
            same_class_indices = label_indices[label_indices != idx]
            other_class_indices = np.where(labels != label)[0]
            
            # Calculate mean intra-class distance (to same class)
            intra_class_dist = np.mean(distances[idx, same_class_indices])
            
            # Calculate mean inter-class distance (to other classes)
            inter_class_dist = np.mean(distances[idx, other_class_indices])
            
            # Score = inter_class_dist / intra_class_dist
            # Higher score is better (far from other classes, close to same class)
            score = inter_class_dist / (intra_class_dist + 1e-10)  # Avoid division by zero
            scores.append((idx, score))
        
        # Sort by score (descending) and take top n_samples_per_class
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:n_samples_per_class]]
        selected_indices.extend(top_indices)
        
        print(f"  Class {label}: Selected {len(top_indices)} samples with scores ranging from {scores[0][1]:.2f} to {scores[n_samples_per_class-1][1]:.2f}")
    
    print(f"Total selected samples: {len(selected_indices)}")
    return selected_indices

def visualize_embeddings_with_pca(embeddings, label_data, selected_indices=None, output_path='pretrain_embeddings_pca.pdf'):
    """
    Visualize embeddings using PCA and color points based on labels.
    If selected_indices is provided, highlight those points.
    """
    # Extract labels
    labels = label_data[:, 0, 0].astype(int)
    
    # Define short labels
    short_labels = [
        "Up", "Down", "Left", "Right",
        "CW Rotation", "CCW Rotation",
        "Up Jerk", "Down Jerk", "Left Jerk", "Right Jerk",
        "Square", "Circle"
    ]
    
    # Create color mapping based on gesture type
    label_to_color = {
        1: 'blue',      # Up - directional
        2: 'blue',      # Down - directional
        3: 'blue',      # Left - directional
        4: 'blue',      # Right - directional
        5: 'red',       # CW Rotation - rotational
        6: 'red',       # CCW Rotation - rotational
        7: 'green',     # Up Jerk - complex
        8: 'green',     # Down Jerk - complex
        9: 'green',     # Left Jerk - complex
        10: 'green',    # Right Jerk - complex
        11: 'purple',   # Square - shape
        12: 'purple'    # Circle - shape
    }
    
    # Prepare embeddings for PCA (convert from 3D to 2D if needed)
    if len(embeddings.shape) == 3:
        embeddings_2d_input = prepare_embeddings_for_pca(embeddings)
    else:
        embeddings_2d_input = embeddings
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_2d_input)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create plot
    plt.figure(figsize=(5, 3))
    
    # If selected_indices is None, plot all points
    if selected_indices is None:
        selected_indices = range(len(labels))
        highlight = False
    else:
        # Plot all points first with low alpha as background
        for label in np.unique(labels):
            if label < 1 or label > 12:
                continue
                
            mask = labels == label
            color = label_to_color[label]
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                color=color,
                alpha=0.0,
                s=60
            )
        highlight = True
    
    # Plot selected points
    for label in np.unique(labels[selected_indices]):
        if label < 1 or label > 12:
            continue
            
        indices = [i for i in selected_indices if labels[i] == label]
        label_idx = label - 1  # Convert to 0-based index
        color = label_to_color[label]
        
        scatter = plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1],
            color=color,
            label=f"{label}. {short_labels[label_idx]}",
            s=60 if highlight else 50,
            alpha=1.0 if highlight else 0.0,
            
        )
    
    # Add legend for color meanings
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Directional'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rotational'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Complex'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Shape')
    ]
    
    # Create legend
    plt.legend(handles=legend_elements, loc='lower left', ncol = 4, fontsize=8)
    
    # Add labels
    plt.xlabel(f'PC1')# ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2')# ({pca.explained_variance_ratio_[1]:.2%})')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('pretrain_model_embed_pca.pdf', dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Save individual selected samples
    if highlight:
        # Create a table of selected samples
        data_table = []
        for label in np.unique(labels[selected_indices]):
            if label < 1 or label > 12:
                continue
                
            indices = [i for i in selected_indices if labels[i] == label]
            label_name = short_labels[label - 1]
            for idx in indices:
                data_table.append([idx, label, label_name])
        
        # Save to CSV
        import csv
        csv_path = output_path.replace('.pdf', '_selected_samples.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Label", "Description"])
            writer.writerows(data_table)
        print(f"Selected samples saved to {csv_path}")
    
    return embeddings_2d, pca, selected_indices

def main():
    # Paths to files
    embed_path = 'embed/embed_limu_v1_blind_user_filtered_20_120.npy'
    input_data_path = 'dataset/blind_user_filtered/data_20_120.npy'
    label_data_path = 'dataset/blind_user_filtered/label_20_120.npy'
    
    # Load data
    embeddings, input_data, label_data = load_model_and_data(embed_path, input_data_path, label_data_path)
    
    # Prepare embeddings for selection
    if len(embeddings.shape) == 3:
        embeddings_2d = prepare_embeddings_for_pca(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Extract labels
    labels = label_data[:, 0, 0].astype(int)
    
    # First, visualize all data points
    """embeddings_2d_pca, pca, _ = visualize_embeddings_with_pca(
        embeddings, 
        label_data, 
        selected_indices=None, 
        output_path='all_samples_pca.pdf'
    )"""
    
    # Select representative samples
    selected_indices = select_representative_samples(embeddings_2d, labels, n_samples_per_class=10)
    
    # Visualize selected samples
    _, _, _ = visualize_embeddings_with_pca(
        embeddings, 
        label_data, 
        selected_indices=selected_indices, 
        output_path='pretrain_model_embed_pca.pdf'
    )
    
    # Save selected data for further use
    selected_input_data = input_data[selected_indices]
    selected_label_data = label_data[selected_indices]
    
    # Create output directory if it doesn't exist
    """os.makedirs('selected_samples', exist_ok=True)
    
    # Save selected data
    np.save('selected_samples/selected_input_data.npy', selected_input_data)
    np.save('selected_samples/selected_label_data.npy', selected_label_data)
    np.save('selected_samples/selected_indices.npy', np.array(selected_indices))"""
    
    print("\nCompleted sample selection and visualization.")
    print(f"Selected {len(selected_indices)} representative samples.")
    print("Selected data saved to 'selected_samples' directory.")

if __name__ == "__main__":
    main()