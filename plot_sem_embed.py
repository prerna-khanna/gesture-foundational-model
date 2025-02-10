import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

def ensure_symmetric_matrix(matrix):
    """
    Ensure the matrix is symmetric by averaging its upper and lower triangles
    """
    return (matrix + matrix.T) / 2

def plot_bert_gru_comparison(bert_sim, gru_sim, label_names):
    """
    Comprehensive visualization and analysis of BERT and GRU similarities
    
    Parameters:
    - bert_sim: BERT similarity matrix
    - gru_sim: GRU similarity matrix
    - label_names: List of class labels
    """
    # Ensure matrix is symmetric
    bert_sim = ensure_symmetric_matrix(bert_sim)
    
    # Ensure matrices are normalized
    bert_sim_norm = bert_sim / bert_sim.sum(axis=1, keepdims=True)
    
    # Handle PyTorch tensor for GRU similarities
    if torch.is_tensor(gru_sim):
        gru_sim_norm = gru_sim / gru_sim.sum(axis=1, keepdims=True)
        gru_sim_norm = gru_sim_norm.cpu().numpy()
    else:
        gru_sim_norm = gru_sim / gru_sim.sum(axis=1, keepdims=True)
    
    # Ensure GRU matrix is symmetric
    gru_sim_norm = ensure_symmetric_matrix(gru_sim_norm)
    
    # Create a comprehensive figure
    plt.figure(figsize=(25, 25))
    
    # 1. Normalized Similarity Heatmaps
    plt.subplot(321)
    sns.heatmap(bert_sim_norm, xticklabels=label_names, yticklabels=label_names, 
                cmap='viridis', annot=True, fmt='.2f', square=True)
    plt.title('BERT Semantic Similarities (Normalized)')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(322)
    sns.heatmap(gru_sim_norm, xticklabels=label_names, yticklabels=label_names, 
                cmap='viridis', annot=True, fmt='.2f', square=True)
    plt.title('GRU Center Similarities (Normalized)')
    plt.xticks(rotation=45, ha='right')
    
    # 2. Difference and Ratio Heatmaps
    plt.subplot(323)
    diff_matrix = gru_sim_norm - bert_sim_norm
    sns.heatmap(diff_matrix, xticklabels=label_names, yticklabels=label_names,
                cmap='RdBu', annot=True, fmt='.2f', center=0, square=True)
    plt.title('Difference (GRU - BERT)\nRed = GRU higher, Blue = BERT higher')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(324)
    # Add small epsilon to avoid division by zero
    ratio_matrix = gru_sim_norm / (bert_sim_norm + 1e-10)
    sns.heatmap(ratio_matrix, xticklabels=label_names, yticklabels=label_names,
                cmap='RdBu', annot=True, fmt='.2f', center=1, square=True)
    plt.title('Ratio (GRU / BERT)\nRed = GRU relatively higher')
    plt.xticks(rotation=45, ha='right')
    
    # 3. Hierarchical Clustering
    plt.subplot(325)
    bert_dist = 1 - bert_sim_norm
    linkage_bert = hierarchy.linkage(bert_dist, method='ward')
    hierarchy.dendrogram(linkage_bert, labels=label_names, leaf_rotation=45)
    plt.title('BERT Hierarchical Clustering')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(326)
    gru_dist = 1 - gru_sim_norm
    linkage_gru = hierarchy.linkage(gru_dist, method='ward')
    hierarchy.dendrogram(linkage_gru, labels=label_names, leaf_rotation=45)
    plt.title('GRU Hierarchical Clustering')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('dataset/blind_user/limu_gru_v1_comprehensive_comparison.png')
    plt.show()
    
    # 4. Quantitative Analysis
    print("\n--- Similarity Analysis ---")
    print(f"BERT Mean Similarity: {bert_sim_norm.mean():.3f}")
    print(f"GRU Mean Similarity: {gru_sim_norm.mean():.3f}")
    print(f"Max Difference: {diff_matrix.max():.3f}")
    print(f"Min Difference: {diff_matrix.min():.3f}")
    
    # Compute correlation between flattened similarity matrices
    correlation, p_value = pearsonr(bert_sim_norm.flatten(), gru_sim_norm.flatten())
    print(f"\nSimilarity Matrix Correlation:")
    print(f"Pearson Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.4f}")

# Main execution
def main():
    # Load data
    matrices = np.load('dataset/blind_user/limu_gru_v1_matrices.npy', allow_pickle=True).item()
    saved_data = torch.load('dataset/blind_user/limu_gru_v1_embeddings.pt')

    # Get BERT similarities
    bert_sim = matrices['bert_sim']

    # Calculate GRU centers and similarities
    test_embeddings = saved_data['test_embeddings']
    test_labels = saved_data['test_labels']

    # Get unique labels and calculate class centers
    unique_labels = torch.unique(test_labels)
    class_centers = []
    for label in unique_labels:
        mask = test_labels == label
        center = test_embeddings[mask].mean(0)
        class_centers.append(center)
    class_centers = torch.stack(class_centers)

    # Calculate distances and convert to similarities
    center_dists = torch.cdist(class_centers, class_centers, p=2)
    center_sims = 1 / (1 + center_dists)

    # Label names
    label_names = ['up', 'down', 'left', 'right', 'rotate wrist and right', 
                   'rotate wrist and left', 'flick and up', 'flick and down', 
                   'flick and left', 'flick and right', 'square', 'circle', 
                   'triangle', 'question mark', 'infinity']

    # Run comprehensive comparison
    plot_bert_gru_comparison(bert_sim, center_sims, label_names)

if __name__ == "__main__":
    main()