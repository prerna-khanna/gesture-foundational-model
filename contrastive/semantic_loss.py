import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel



class SemanticLoss(nn.Module):
    def __init__(self, label_names, device, temperature=0.07, hidden_dim=128):
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Initialize BERT for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.bert.to(device)
        
        # Single projection layer with correct dimensions
        self.semantic_projection = nn.Linear(hidden_dim, 32).to(device)
        
        # Compute semantic similarities between gesture labels
        self.semantic_sims = self.compute_semantic_similarities(label_names)
    
    def compute_semantic_similarities(self, label_names):
        """
        Compute semantic similarity matrix between gesture labels using BERT
        """
        descriptions = [
            f"a {name} gesture with properties: " + 
            f"primary type: {'directional' if name in ['up', 'down', 'left', 'right'] else 'rotational' if 'rotate' in name or name in ['circle'] else 'shape' if name in ['square', 'triangle', 'infinity'] else 'complex'}, " +
            f"direction: {name.split()[0]}, " +
            f"complexity: {'simple' if name in ['up', 'down', 'left', 'right'] else 'complex'}"
            for name in label_names
        ]
        
        with torch.no_grad():
            inputs = self.tokenizer(descriptions, padding=True, return_tensors="pt").to(self.device)
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute similarity matrix with increased contrast
            sim_matrix = torch.matmul(embeddings, embeddings.t())
            sim_matrix = torch.pow(sim_matrix, 3)  # Enhance contrast

            print("Semantic similarity matrix:")
            print(sim_matrix)
            
            return sim_matrix

    def forward(self, features, labels, epoch=0):
        """
        Compute semantic loss for the features
        """
        # Project features to semantic space
        semantic_features = self.semantic_projection(features)
        semantic_features = F.normalize(semantic_features + 1e-8, p=2, dim=1)
        
        # Compute distances and loss
        dists = torch.cdist(semantic_features, semantic_features, p=2)
        loss = 0
        batch_size = semantic_features.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                # Dynamic margin based on similarity
                margin = 1.0 * (2.0 if sem_sim > 0.8 else 
                             1.5 if sem_sim > 0.5 else 1.0)
                
                # Weight loss based on similarity
                weight = torch.pow(sem_sim, 2) + 0.1
                loss += weight * torch.max(torch.tensor(0.0).to(self.device),
                                       margin - dists[i, j])
        
        final_loss = loss / (batch_size * (batch_size - 1))
        return final_loss