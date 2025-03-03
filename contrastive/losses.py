import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from contrastive.augmenter import GestureAugmenter
from contrastive.semantic_loss import SemanticLoss


class ContrastiveCombinedLoss(nn.Module):
    def __init__(self, label_names, descriptions, pooling, device, temperature=0.07, hidden_dim=128):
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.temperature = temperature
        self.semantic_criterion = SemanticLoss(label_names, descriptions, pooling, device, hidden_dim=hidden_dim)
        self.classification_criterion = nn.CrossEntropyLoss()

    def compute_contrastive_loss(self, features, labels):
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
        
        w_classification = 1
        """w_semantic = 1
        w_contrastive = 0.5"""
        
        
        # Combine all losses
        total_loss = (
            classification_loss * w_classification + 
            semantic_loss * w_semantic + 
            contrastive_loss * w_contrastive
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item()
        }
    
class SVMHingeLoss(nn.Module):
    """
    SVM Hinge Loss implementation for multi-class classification
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, logits, labels):
        batch_size = logits.size(0)
        # One-hot encode the labels
        y_one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        
        # Get the prediction for the correct class
        correct_logits = torch.sum(logits * y_one_hot, dim=1)
        
        # Calculate the hinge loss
        margins = torch.clamp(self.margin - correct_logits.unsqueeze(1) + logits, min=0)
        # Zero out the margin for the correct class
        margins = margins * (1 - y_one_hot)
        
        # Sum over classes and average over the batch
        loss = torch.sum(margins) / batch_size
        return loss

    
class ContrastiveSVMLoss(nn.Module):
    def __init__(self, label_names, descriptions, pooling, device, temperature=0.07, hidden_dim=128, margin=1.0):
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.temperature = temperature
        self.svm_criterion = SVMHingeLoss(margin=margin)
        self.semantic_criterion = SemanticLoss(label_names, descriptions, pooling, device, hidden_dim=hidden_dim)

    def compute_contrastive_loss(self, features, labels):
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

    def forward(self, logits, features, projected, labels, epoch=0):
        # SVM classification loss
        classification_loss = self.svm_criterion(logits, labels)
        
        # Semantic loss
        semantic_output = self.semantic_criterion(features=features, labels=labels, epoch=epoch)
        semantic_loss = semantic_output[0] if isinstance(semantic_output, tuple) else semantic_output
        
        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(projected, labels)
        
        # Dynamic weighting
        w_semantic = max(0.1, min(0.3, (epoch / 10) * 0.3))
        w_contrastive = max(0.1, min(0.5, (epoch / 20) * 0.5))
        w_classification = 1
        
        # Combine all losses
        total_loss = (
            classification_loss * w_classification + 
            semantic_loss * w_semantic + 
            contrastive_loss * w_contrastive
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item()
        }