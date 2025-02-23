import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SemanticLoss(nn.Module):
    def __init__(self, label_names, descriptions, pooling, device, temperature=0.07, hidden_dim=128):
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.temperature = temperature
        self.pooling = pooling
        
        # Initialize BERT for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.to(device)
        
        # Single projection layer with correct dimensions
        self.semantic_projection = nn.Linear(hidden_dim, 32).to(device)
        
        # Compute semantic similarities between gesture labels
        self.semantic_sims = self.compute_semantic_similarities(label_names, descriptions)
    
    def compute_semantic_similarities(self, label_names, descriptions):

        """descriptions = [
            f"a {name} gesture with properties: " + 
            f"primary type: {'directional' if name in ['up', 'down', 'left', 'right'] else 'rotational' if 'rotate' in name or name in ['circle'] else 'shape' if name in ['square', 'triangle', 'infinity'] else 'complex'}, " +
            f"direction: {name.split()[0]}, " +
            f"complexity: {'simple' if name in ['up', 'down', 'left', 'right'] else 'complex'}"
            for name in label_names
        ]"""

        print("description is ", descriptions)
        with torch.no_grad():
            
            inputs = self.tokenizer(descriptions, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Create attention mask to ignore padding
            attention_mask = inputs['attention_mask']

            print("Pooling criterion used: " + self.pooling)

            ## cls
            if self.pooling == "cls":
                embeddings = hidden_states[:, 0, :] 
                
            ## Mean pooling, ignoring padding
            elif self.pooling == "mean":
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    # Select only non-padding tokens
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.mean(dim=0)
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)

            ## Max pooling, ignoring padding
            elif self.pooling == "max":
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    # Select only non-padding tokens
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.max(dim=0)[0]
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)

            
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute similarity matrix with increased contrast
            sim_matrix = torch.matmul(embeddings, embeddings.t())
            sim_matrix = torch.pow(sim_matrix, 3)  # Enhance contrast

            #print("Semantic similarity matrix:")
            #print(sim_matrix)
            
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