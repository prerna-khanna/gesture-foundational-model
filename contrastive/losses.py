import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from contrastive.augmenter import GestureAugmenter
from contrastive.semantic_loss import SemanticLoss


class ContrastiveCombinedLoss(nn.Module):
    def __init__(self, label_names, device, temperature=0.07, hidden_dim=128):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.semantic_criterion = SemanticLoss(label_names, device, hidden_dim=hidden_dim)
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, logits, features, projected, labels, epoch=0):
        """
        Compute combined loss with all components
        
        Args:
            logits: Classification outputs [batch_size, num_classes]
            features: GRU features [batch_size, hidden_dim]
            projected: Contrastive projected features [batch_size, proj_dim]
            labels: Ground truth labels [batch_size]
            epoch: Current training epoch
        """
        
        # Classification loss
        classification_loss = self.classification_criterion(logits, labels)
        
        # Semantic loss
        semantic_output = self.semantic_criterion(features=features, labels=labels, epoch=epoch)
        semantic_loss = semantic_output[0] if isinstance(semantic_output, tuple) else semantic_output
        
        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(projected, labels)
        
        # Dynamic weighting starting with small non-zero values
        w_semantic = max(0.1, min(0.3, (epoch / 10) * 0.3))
        w_contrastive = max(0.1, min(0.5, (epoch / 20) * 0.5))
        
        # Combine all losses
        total_loss = (
            classification_loss + 
            semantic_loss * w_semantic + 
            contrastive_loss * w_contrastive
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item()
        }
    

    def compute_contrastive_loss(self, features, labels):
        # Previous implementation remains the same...
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(mask.shape[0]).to(self.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -mean_log_prob_pos.mean()
    


"""

use this for controlled ratio of positive to negative pairs

def compute_contrastive_loss(self, features, labels):
    # Normalize features as before
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0]
    
    # Compute similarity matrix as before
    similarity_matrix = torch.matmul(features, features.T) / self.temperature
    
    # Create positive pair mask
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(self.device)
    
    # Remove self-comparisons
    diagonal_mask = torch.eye(batch_size).bool().to(self.device)
    positive_mask[diagonal_mask] = 0
    
    # Create negative pair mask
    negative_mask = 1 - positive_mask - torch.eye(batch_size).to(self.device)
    
    # NEW: Control ratio by sampling negatives
    num_positives = positive_mask.sum(dim=1)
    desired_negatives = (num_positives / self.pos_to_neg_ratio).int()
    
    # NEW: Sample negative pairs based on desired ratio
    for i in range(batch_size):
        negative_indices = torch.where(negative_mask[i] > 0)[0]
        if len(negative_indices) > desired_negatives[i]:
            # Randomly select subset of negative pairs
            perm = torch.randperm(len(negative_indices))
            indices_to_zero = negative_indices[perm[desired_negatives[i]:]]
            negative_mask[i, indices_to_zero] = 0
    
    # Compute loss with controlled ratio
    exp_similarities = torch.exp(similarity_matrix)
    pos_loss = torch.sum(positive_mask * similarity_matrix, dim=1)
    neg_loss = torch.log(
        torch.sum(negative_mask * exp_similarities, dim=1) + 1e-8
    )
    
    loss = -(pos_loss - neg_loss)
    non_zero_elements = (positive_mask.sum(dim=1) > 0).float()
    loss = (loss * non_zero_elements).sum() / (non_zero_elements.sum() + 1e-8)
    
    return loss


"""