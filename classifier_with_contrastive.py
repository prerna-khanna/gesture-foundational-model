#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import math

from contrastive.augmenter import GestureAugmenter
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.models import ContrastiveGRUClassifier

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset

class SemanticLoss(nn.Module):
    def __init__(self, label_names, device, temperature=0.07):
        """
        Initialize SemanticLoss with both semantic and contrastive learning capabilities
        
        Args:
            label_names: List of gesture label names
            device: Device to run computations on
            temperature: Temperature parameter for contrastive loss scaling
        """
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Initialize BERT for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.bert.to(device)
        
        # Projection networks for different spaces
        self.semantic_projection = nn.Linear(20, 32).to(device)  # For semantic space
        self.contrastive_projection = nn.Sequential(  # For contrastive learning
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).to(device)
        
        # Compute semantic similarities between gesture labels
        self.semantic_sims = self._compute_semantic_sims(label_names)
        
    def _compute_semantic_sims(self, label_names):
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
        
        return sim_matrix
    
    def semantic_loss(self, embeddings, labels, base_margin=1.0):
        """
        Compute semantic loss with dynamic margins based on gesture similarity
        """
        embeddings = self.semantic_projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        dists = torch.cdist(embeddings, embeddings, p=2)
        loss = 0
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                
                # Dynamic margin based on semantic similarity
                margin = base_margin * (2.0 if sem_sim > 0.8 else 
                                      1.5 if sem_sim > 0.5 else 1.0)
                
                # Weight loss based on similarity
                weight = torch.pow(sem_sim, 2) + 0.1
                loss += weight * torch.max(torch.tensor(0.0).to(self.device),
                                         margin - dists[i, j])
                    
        return loss / (batch_size * (batch_size - 1))
    
    def contrastive_loss(self, embeddings, labels):
        """
        Compute NT-Xent contrastive loss
        """
        # Project and normalize features
        features = self.contrastive_projection(embeddings)
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
            0
        )
        
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        return -mean_log_prob_pos.mean()
    
    def forward(self, embeddings, labels, epoch=0):
        """
        Combine semantic and contrastive losses with dynamic weighting
        """
        sem_loss = self.semantic_loss(embeddings, labels)
        cont_loss = self.contrastive_loss(embeddings, labels)
        
        # Dynamic weighting based on training epoch
        semantic_weight = min(0.3, (epoch / 10) * 0.3)  # Gradually increase to 0.3
        contrastive_weight = min(0.5, (epoch / 20) * 0.5)  # Gradually increase to 0.5
        
        total_loss = sem_loss * semantic_weight + cont_loss * contrastive_weight
        
        # Return both total loss and components for logging
        return total_loss, {
            'semantic_loss': sem_loss.item(),
            'contrastive_loss': cont_loss.item(),
            'total_loss': total_loss.item()
        }
    

def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    try:
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
        
        print(f"Number of classes: {label_num}")
        print(f"Label names: {label_names}")
        
        # Calculate hidden dimension if not in config
        # Using 128 as default hidden dim - this is a common choice for gesture recognition
        hidden_dim = getattr(model_cfg, 'hidden_dim', 128)
        
        # Prepare data
        data_train, label_train, data_vali, label_vali, data_test, label_test = \
            prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                     label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                     balance=balance)
        
        # Create datasets with augmentation only for training
        augmenter = GestureAugmenter()
        data_set_train = IMUDataset(data_train, label_train, pipeline=[augmenter.augment])
        data_set_vali = IMUDataset(data_vali, label_vali)  # No augmentation for validation
        data_set_test = IMUDataset(data_test, label_test)  # No augmentation for testing
        
        # Create dataloaders
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

        # Initialize model with the calculated hidden dimension
        model = ContrastiveGRUClassifier(
            input_dim=data_train.shape[-1],  # 72 from your data
            hidden_dim=hidden_dim,           # Using our default or config value
            num_classes=label_num
        ).to(device)
        
        print(f"Model architecture initialized with hidden dimension: {hidden_dim}")
        
        # Initialize combined loss (classification + semantic + contrastive)
        criterion = ContrastiveCombinedLoss(
            label_names=label_names, 
            device=device,
            hidden_dim=hidden_dim  # Pass the hidden dimension
        )
        
        # Setup optimizer and trainer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

        def func_loss(model, batch, current_epoch=0):
            inputs, label = batch
            
            # Get all outputs from model
            logits, features, projected = model(inputs, True)
            
            # Pass all necessary components to criterion
            total_loss, loss_dict = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=label,
                epoch=current_epoch
            )
            
            print(f"Epoch {current_epoch} - Loss Components:")
            for loss_name, loss_value in loss_dict.items():
                print(f"  {loss_name}: {loss_value:.4f}")
            
            return total_loss, loss_dict

        def func_forward(model, batch):
            inputs, label = batch
            # For inference, we only need classification output
            logits = model(inputs, False)
            return logits, label

        def func_evaluate(label, predicts):
            # Compute accuracy and F1 score
            stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
            return stat

        # Train the model
        trainer.train(func_loss, func_forward, func_evaluate,
                     data_loader_train, data_loader_test, data_loader_vali)
        
        # Final evaluation on test set
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
        
        print("Training completed successfully")
        return label_test, label_estimate_test
        
    except Exception as e:
        print(f"Error in classify_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        # Set training parameters
        training_rate = 0.8  # 80% for training
        label_rate = 0.1    # 10% for validation
        balance = True      # Balance dataset across classes

        # Set model configuration
        mode = "contrastive"  # Changed from "base" to reflect new approach
        method = "gru"
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        
        # Load data
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("Data dimensions:", embedding.shape, "Label dimensions:", labels.shape)

        # Train and evaluate model
        label_test, label_estimate_test = classify_embeddings(args, embedding, labels, args.label_index,
                                                            training_rate, label_rate, balance=balance, method=method)

        # Generate evaluation metrics and visualization
        if label_test is not None and label_estimate_test is not None:
            label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            matrix_norm = plot_matrix(matrix, label_names)
        else:
            print("Error: classify_embeddings returned None")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()