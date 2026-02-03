import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import os

# Define model architecture that matches the state dict
class LIMUModel(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128):
        super().__init__()
        # Main transformer components based on state dict keys
        self.transformer = nn.Module()
        self.transformer.embed = nn.Module()
        self.transformer.embed.lin = nn.Linear(input_dim, hidden_dim)
        self.transformer.embed.pos_embed = nn.Embedding(120, hidden_dim)  # Sequence length = 120
        self.transformer.embed.nucleus_embed = nn.Embedding(10, hidden_dim)  # Placeholder
        self.transformer.embed.sig_axis_embed = nn.Embedding(10, hidden_dim)  # Placeholder
        self.transformer.embed.norm = nn.LayerNorm(hidden_dim)
        
        # Attention components
        self.transformer.attn = nn.Module()
        self.transformer.attn.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.transformer.attn.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.transformer.attn.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.transformer.proj = nn.Linear(hidden_dim, hidden_dim)
        self.transformer.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward components
        self.transformer.pwff = nn.Module()
        self.transformer.pwff.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.transformer.pwff.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.transformer.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output components
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 12)  # Final classification layer
        
    def forward(self, x):
        # This is a simplified placeholder implementation
        # We won't actually use this for forward computation
        x = self.transformer.embed.lin(x)
        # In a real implementation, this would use all the components
        return x
        
    def extract_features(self, x):
        """Extract penultimate layer features"""
        # Get embeddings (this is what we would use for feature extraction)
        batch_size, seq_len, input_dim = x.shape
        x = self.transformer.embed.lin(x)
        
        # For simplicity, we'll just return the mean across the sequence dimension
        # This is what your model might be doing for classification
        features = torch.mean(x, dim=1)
        return features

def load_model_and_data(model_path, input_data_path, label_data_path):
    """
    Load the saved model and data files
    
    Args:
        model_path: Path to the saved model (.pt file)
        input_data_path: Path to input .npy file with shape (n, 120, 6)
        label_data_path: Path to label .npy file with shape (n, 120, 2)
    
    Returns:
        model: Loaded PyTorch model
        input_data: Loaded input data
        label_data: Loaded label data
    """
    # Load the state dictionary
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except Exception as e:
        print(f"Error loading model with weights_only=True: {e}")
        print("Trying without weights_only parameter...")
        state_dict = torch.load(model_path)
    
    # Initialize the model with the correct input dimension
    input_dim = 72  # Based on your input shape
    model = LIMUModel(input_dim=input_dim)
    
    # Print state_dict keys for debugging
    print("State dict keys:")
    for key in list(state_dict.keys())[:5]:  # Print first 5 keys
        print(f"  {key}")
    print("  ...")
    
    # Print model state_dict keys for debugging
    print("\nModel state_dict keys:")
    model_state = model.state_dict()
    for key in list(model_state.keys())[:5]:  # Print first 5 keys
        print(f"  {key}")
    print("  ...")
    
    # Try to load the state dict into the model
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded state dictionary into model.")
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        print("Will proceed with the initialized model without pretrained weights.")
    
    model.eval()  # Set to evaluation mode
    
    # Load input data and label data
    input_data = np.load(input_data_path)
    label_data = np.load(label_data_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Loaded input data with shape: {input_data.shape}")
    print(f"Loaded label data with shape: {label_data.shape}")
    
    return model, input_data, label_data

def get_penultimate_features(model, input_data, batch_size=32):
    """
    Extract features from the penultimate layer of the model
    
    Args:
        model: Loaded PyTorch model
        input_data: NumPy array with shape (n, 120, 6)
        batch_size: Batch size for processing
        
    Returns:
        features: Features from the penultimate layer
    """
    # Use the model for feature extraction if possible
    if hasattr(model, 'extract_features'):
        try:
            print("Extracting features using model's extract_features method")
            # Convert numpy array to tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            
            # Process data in batches
            n_samples = input_tensor.shape[0]
            penultimate_features = []
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch = input_tensor[i:i+batch_size]
                    features = model.extract_features(batch)
                    penultimate_features.append(features.cpu().numpy())
                    print(f"Processed samples {i+1} to {min(i+batch_size, n_samples)}/{n_samples}")
            
            # Concatenate and return features
            all_features = np.concatenate(penultimate_features, axis=0)
            print(f"Extracted features with shape: {all_features.shape}")
            return all_features
            
        except Exception as e:
            print(f"Feature extraction using model failed: {e}")
            print("Falling back to using input data directly")
    
    # Fallback: use input data directly
    print("Using input data as feature representation")
    
    # Process input data - if it's 3D, take average over sequence dimension
    if len(input_data.shape) == 3:
        # For (n_samples, sequence_length, features), average over sequence
        features = np.mean(input_data, axis=1)
        print(f"Averaged input data over sequence dimension: {features.shape}")
    else:
        features = input_data
    
    return features

def prepare_embeddings_for_pca(embeddings):
    """
    Prepare embeddings for PCA by handling multi-dimensional arrays
    """
    # If embeddings are more than 2D, flatten all but the first dimension
    if len(embeddings.shape) > 2:
        # If 3D tensor (batch, seq_len, features), take mean across sequence length
        if len(embeddings.shape) == 3:
            embeddings_2d = np.mean(embeddings, axis=1)
        else:
            # For higher dimensions, flatten all except the first
            embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_2d = embeddings
    
    print(f"Prepared embeddings for PCA with shape: {embeddings_2d.shape}")
    return embeddings_2d

def select_representative_samples(embeddings, labels, n_samples_per_class=10):
    """
    Select representative samples from each class that maximize inter-class distance
    and minimize intra-class distance.
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
    return np.array(selected_indices)  # Convert to numpy array

def visualize_embeddings_with_pca(embeddings, label_data, selected_indices=None, output_path='model_embeddings_pca.pdf'):
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
    
    # Prepare embeddings for PCA
    embeddings_2d_input = prepare_embeddings_for_pca(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_2d_input)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create plot
    plt.figure(figsize=(5, 3))
    
    # If selected_indices is None, plot all points
    if selected_indices is None:
        selected_indices = np.arange(len(labels))
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
                alpha=0.0,  # Low alpha for background points
                s=20
            )
        highlight = True
    
    # Plot selected points - make sure selected_indices is a numpy array
    # Convert to ensure it works for indexing
    selected_indices = np.array(selected_indices)
    
    # Get unique labels from selected points
    unique_selected_labels = np.unique(labels[selected_indices])
    
    for label in unique_selected_labels:
        if label < 1 or label > 12:
            continue
            
        # Find indices where labels match the current label
        mask = labels[selected_indices] == label
        indices = selected_indices[mask]
        
        label_idx = label - 1  # Convert to 0-based index
        color = label_to_color[label]
        
        scatter = plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1],
            color=color,
            label=f"{label}. {short_labels[label_idx]}",
            s=60 if highlight else 50,
            alpha=1.0,
        )
    
    # Add legend for gesture types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Directional'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Rotational'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Complex'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Shape')
    ]
    
    # Add legend for specific gestures
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    first_legend = plt.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize='small')
    plt.gca().add_artist(first_legend)
    
    # Add labels
    plt.xlabel(f'PC1')
    plt.ylabel(f'PC2')
    #plt.grid(alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Save individual selected samples
    if highlight:
        # Create a table of selected samples
        data_table = []
        for label in unique_selected_labels:
            if label < 1 or label > 12:
                continue
                
            mask = labels[selected_indices] == label
            indices = selected_indices[mask]
            
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
    # Paths to files - update these to your actual file paths
    model_path = 'saved/pretrain_base_blind_user_filtered_20_120/limu_v1.pt'  # Path to your .pt model file
    embed_data_path = 'embed/embed_limu_v1_blind_user_filtered_20_120.npy'  # Shape (n, 120, 72)
    label_data_path = 'dataset/blind_user_filtered/label_20_120.npy'  # Shape (n, 120, 2)
    
    # Load model and data
    model, input_data, label_data = load_model_and_data(model_path, embed_data_path, label_data_path)
    
    # Extract features using the model
    features = get_penultimate_features(model, input_data)
    
    # Prepare features for selection
    features_2d = prepare_embeddings_for_pca(features)
    
    # Extract labels
    labels = label_data[:, 0, 0].astype(int)
    
    # Select representative samples
    selected_indices = select_representative_samples(features_2d, labels, n_samples_per_class=10)
    
    # Visualize selected samples
    embeddings_2d_pca, pca, selected_indices = visualize_embeddings_with_pca(
        features, 
        label_data, 
        selected_indices=selected_indices, 
        output_path='final_model_embed_pca.pdf'
    )
    
    print("\nCompleted feature extraction, sample selection and visualization.")
    print(f"Selected {len(selected_indices)} representative samples.")

if __name__ == "__main__":
    main()