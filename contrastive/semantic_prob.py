import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ProbabilisticSemanticLoss(nn.Module):
    def __init__(self, label_names, descriptions, pooling, device, temperature=0.07, hidden_dim=128):
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.temperature = temperature
        
        # Initialize BERT for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.to(device)
        
        # Projection layer
        self.semantic_projection = nn.Linear(hidden_dim, 32).to(device)
        
        # Compute semantic similarities between classes
        self.semantic_sims = self.compute_semantic_similarities(label_names, descriptions)
        
    def compute_semantic_similarities(self, label_names, descriptions):
        """
        Compute pairwise semantic similarities between class descriptions using BERT
        """
        with torch.no_grad():
            # Tokenize descriptions
            inputs = self.tokenizer(descriptions, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply specified pooling strategy
            if self.pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            elif self.pooling == "mean":
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.mean(dim=0)
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
            elif self.pooling == "max":
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.max(dim=0)[0]
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
            
            # Normalize embeddings and compute similarity matrix
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings, embeddings.t())
            sim_matrix = torch.pow(sim_matrix, 3)  # Enhance contrast
            
            return sim_matrix
            
    def compute_expected_similarity(self, probs_i, probs_j):
        """
        Compute expected semantic similarity given two probability distributions over classes
        
        Args:
            probs_i: Probabilities for sample i (shape: num_classes)
            probs_j: Probabilities for sample j (shape: num_classes)
            
        Returns:
            Expected semantic similarity between the samples
        """
        # Outer product of probability distributions
        prob_matrix = torch.outer(probs_i, probs_j)
        
        # Weight semantic similarities by probability pairs
        expected_sim = torch.sum(prob_matrix * self.semantic_sims)
        
        return expected_sim

    def forward(self, features, class_probs, epoch=0):
        """
        Compute semantic loss using class probabilities
        
        Args:
            features: Feature vectors from the model
            class_probs: Class probabilities from model (softmax outputs)
            epoch: Current training epoch
        """
        # Project features to semantic space
        semantic_features = self.semantic_projection(features)
        semantic_features = F.normalize(semantic_features + 1e-8, p=2, dim=1)
        
        # Compute pairwise distances
        dists = torch.cdist(semantic_features, semantic_features, p=2)
        
        loss = 0
        batch_size = semantic_features.size(0)
        
        # Compute loss for all pairs
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Compute expected semantic similarity based on class probabilities
                expected_sim = self.compute_expected_similarity(
                    class_probs[i], 
                    class_probs[j]
                )
                
                # Dynamic margin based on expected similarity
                margin = 1.0 * (2.0 if expected_sim > 0.8 else 
                              1.5 if expected_sim > 0.5 else 1.0)
                
                # Weight based on expected similarity
                weight = torch.pow(expected_sim, 2) + 0.1
                
                # Compute weighted contrastive loss
                pair_loss = weight * torch.max(
                    torch.tensor(0.0).to(self.device),
                    margin - dists[i, j]
                )
                
                loss += pair_loss
        
        # Normalize loss by number of pairs
        final_loss = loss / (batch_size * (batch_size - 1))
        
        return final_loss