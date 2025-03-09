import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np

class SemanticEmbedder:
    def __init__(
        self, 
        model_name='bert-base-uncased', 
        device=None
    ):
        # Set device
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() else 
            'cpu'
        )
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
    
    def get_embeddings(self, descriptions):
        """
        Get CLS token embeddings for the descriptions.
        
        Args:
            descriptions: List of gesture descriptions
            
        Returns:
            embeddings: Normalized CLS token embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Use only the CLS token [0]
            embeddings = hidden_states[:, 0, :]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity_matrix(self, descriptions):
        """
        Compute similarity matrix for the descriptions.
        
        Args:
            descriptions: List of gesture descriptions
            
        Returns:
            sim_matrix: Similarity matrix
        """
        # Get embeddings
        embeddings = self.get_embeddings(descriptions)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Convert to numpy for easier integration with the rest of the code
        sim_matrix_np = sim_matrix.cpu().numpy()
        sim_matrix_np = sim_matrix_np

        sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min())
        
        # Non-linear transformation to enhance differences
        sim_matrix = torch.pow(sim_matrix, 3)
        
        # Enhance similarities
        very_similar = (sim_matrix > 0.95).float()
        somewhat_similar = ((sim_matrix > 0.85) & (sim_matrix <= 0.95)).float()
        different = (sim_matrix <= 0.85).float()
        
        enhanced_sim_matrix = (
            very_similar * sim_matrix * 1.2 +
            somewhat_similar * sim_matrix * 0.8 +
            different * sim_matrix * 0.5
        )
        
        return enhanced_sim_matrix.cpu().numpy()
    

def visualize_bert_similarity(descriptions, title="BERT Embeddings Similarity"):
    """
    Visualize the similarity matrix for a set of descriptions.
    
    Args:
        descriptions: List of gesture descriptions
        title: Title for the plot
    
    Returns:
        similarity_matrix: The computed similarity matrix
    """
    # Create embedder using CLS token
    embedder = SemanticEmbedder()
    
    # Compute similarity matrix
    sim_matrix = embedder.compute_similarity_matrix(descriptions)
    
    # Plot similarity matrix as heatmap
    
    plt.figure(figsize=(15, 15))
    cax = plt.matshow(sim_matrix, cmap='Blues', fignum=1)
    plt.colorbar(cax)
    
    # Add text annotations
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            plt.text(j, i, f"{sim_matrix[i, j]:.4f}", ha='center', va='center', color='black')
    
    plt.title(title)
    plt.savefig("bert_similarity_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return sim_matrix