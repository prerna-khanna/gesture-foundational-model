#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ablation study version of classifier_with_contrastive.py
# Allows toggling semantic and contrastive loss components

"""
# Classification only
python classifier_with_contrastive_ablation.py v2 hhar 20_120 -f limu_v1_no_nucleus -s cls_only -l 2 --use_semantic false --use_contrastive false

# Classification + Semantic
python classifier_with_contrastive_ablation.py v2 hhar 20_120 -f limu_v1_no_nucleus -s cls_semantic -l 2 --use_semantic true --use_contrastive false

# Classification + Contrastive
python classifier_with_contrastive_ablation.py v2 hhar 20_120 -f limu_v1_no_nucleus -s cls_contrastive -l 2 --use_semantic false --use_contrastive true

# All three (same as original)
python classifier_with_contrastive_ablation.py v2 hhar 20_120 -f limu_v1_no_nucleus -s cls_sem_con -l 2 --use_semantic true --use_contrastive true
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import datetime
import pandas as pd
import math
import argparse
import copy
import json
import os
import sys
import numpy as np
from sklearn.model_selection import ParameterGrid

from contrastive.augmenter import GestureAugmenter
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.models import ContrastiveGRUClassifier, ContrastiveLSTMClassifier, ContrastiveTransformerClassifier, ContrastiveBiGRUClassifier, ContrastiveBiLSTMAttentionClassifier, ContrastiveTCNClassifier
from contrastive.models import ContrastiveSVMClassifier
from contrastive.losses import ContrastiveSVMLoss

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset


class ClassificationOnlyLoss(nn.Module):
    """Classification loss only (no semantic or contrastive)"""
    def __init__(self, num_classes, device):
        super().__init__()
        self.device = device
        self.classification_criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, features, projected, labels, epoch=0):
        classification_loss = self.classification_criterion(logits, labels.long())
        
        total_loss = classification_loss
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': 0.0,
            'contrastive_loss': 0.0,
            'total_loss': total_loss.item()
        }


class ClassificationSemanticLoss(nn.Module):
    """Classification + Semantic loss (no contrastive)"""
    def __init__(self, label_names, descriptions, pooling, device, hidden_dim=128):
        super().__init__()
        self.device = device
        self.classification_criterion = nn.CrossEntropyLoss()
        self.temperature = 0.07
        
        # Import SemanticLoss from contrastive module
        from contrastive.semantic_loss import SemanticLoss
        self.semantic_criterion = SemanticLoss(label_names, descriptions, pooling, device, hidden_dim=hidden_dim)
    
    def forward(self, logits, features, projected, labels, epoch=0):
        # Classification loss
        classification_loss = self.classification_criterion(logits, labels.long())
        
        # Semantic loss
        semantic_output = self.semantic_criterion(features=features, labels=labels, epoch=epoch)
        semantic_loss = semantic_output[0] if isinstance(semantic_output, tuple) else semantic_output
        
        # Dynamic weighting for semantic loss (matching original)
        w_classification = 1.0
        w_semantic = max(0.1, min(0.3, (epoch / 10) * 0.3))
        
        # Combine losses
        total_loss = (
            classification_loss * w_classification + 
            semantic_loss * w_semantic
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'contrastive_loss': 0.0,
            'total_loss': total_loss.item()
        }


class ClassificationContrastiveLoss(nn.Module):
    """Classification + Contrastive loss (no semantic)"""
    def __init__(self, num_classes, device, hidden_dim=128):
        super().__init__()
        self.device = device
        self.classification_criterion = nn.CrossEntropyLoss()
        self.temperature = 0.07
    
    def compute_contrastive_loss(self, features, labels):
        """Compute contrastive loss (NT-Xent) matching original implementation"""
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
        # Classification loss
        classification_loss = self.classification_criterion(logits, labels.long())
        
        # Contrastive loss (use projected features)
        contrastive_loss = self.compute_contrastive_loss(projected, labels)
        
        # Dynamic weighting for contrastive loss (matching original)
        w_classification = 1.0
        w_contrastive = max(0.1, min(0.5, (epoch / 20) * 0.5))
        
        # Combine losses
        total_loss = (
            classification_loss * w_classification + 
            contrastive_loss * w_contrastive
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'semantic_loss': 0.0,
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': total_loss.item()
        }


def create_config_with_params(original_cfg, params):
    """
    Creates a new configuration by updating the original config with new parameters.
    """
    config_dict = {
        'seed': original_cfg.seed,
        'batch_size': params.get('batch_size', original_cfg.batch_size),
        'lr': params.get('lr', original_cfg.lr),
        'n_epochs': params.get('n_epochs', original_cfg.n_epochs),
        'warmup': params.get('warmup', original_cfg.warmup),
        'save_steps': original_cfg.save_steps,
        'total_steps': original_cfg.total_steps,
        'lambda1': original_cfg.lambda1,
        'lambda2': params.get('lambda2', original_cfg.lambda2),
        'pooling': getattr(original_cfg, 'pooling', 'mean')
    }
    return type(original_cfg)(**config_dict)


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None, params=None, 
                       use_semantic=True, use_contrastive=True):
    """
    Classification with ablation support for semantic and contrastive losses.
    
    Args:
        use_semantic: Whether to use semantic loss
        use_contrastive: Whether to use contrastive loss
    """
    try:
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        
        # Update configuration with custom parameters if provided
        if params:
            train_cfg = create_config_with_params(train_cfg, params)
            print(f"Using custom parameters: {params}")
        
        label_names, label_num, descriptions = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
        
        print(f"Number of classes: {label_num}")
        print(f"Label names: {label_names}")
        print(f"Descriptions available: {descriptions is not None}")
        if descriptions is None:
            print("Warning: No descriptions found in dataset config, using labels as descriptions")
            descriptions = [f"{name} gesture" for name in label_names]
        
        # Calculate hidden dimension if not in config
        hidden_dim = getattr(model_cfg, 'hidden_dim', 128)
        
        # Prepare data
        data_train, label_train, data_vali, label_vali, data_test, label_test = \
            prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                     label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                     balance=balance)
        
        # Create datasets with augmentation only for training
        augmenter = GestureAugmenter()
        data_set_train = IMUDataset(data_train, label_train, pipeline=[augmenter.augment])
        data_set_vali = IMUDataset(data_vali, label_vali)
        data_set_test = IMUDataset(data_test, label_test)
        
        # Create dataloaders
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

        # Initialize model
        model = ContrastiveTransformerClassifier(
            input_dim=data_train.shape[-1],
            hidden_dim=hidden_dim,
            num_classes=label_num
        ).to(device)
        
        print(f"Model architecture initialized with hidden dimension: {hidden_dim}")
        
        # Initialize loss based on ablation configuration
        if use_semantic and use_contrastive:
            criterion = ContrastiveCombinedLoss(
                label_names=label_names, 
                descriptions=descriptions,
                pooling=getattr(train_cfg, 'pooling', 'cls'),
                device=device,
                hidden_dim=hidden_dim
            )
            loss_config = "Classification + Semantic + Contrastive"
        elif use_semantic:
            criterion = ClassificationSemanticLoss(
                label_names=label_names,
                descriptions=descriptions,
                pooling=getattr(train_cfg, 'pooling', 'cls'),
                device=device,
                hidden_dim=hidden_dim
            )
            loss_config = "Classification + Semantic"
        elif use_contrastive:
            criterion = ClassificationContrastiveLoss(
                num_classes=label_num,
                device=device,
                hidden_dim=hidden_dim
            )
            loss_config = "Classification + Contrastive"
        else:
            criterion = ClassificationOnlyLoss(
                num_classes=label_num,
                device=device
            )
            loss_config = "Classification Only"
        
        # Setup optimizer and trainer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        print(f"Optimizer initialized with learning rate: {train_cfg.lr}")
        
        # Create a unique save path
        save_path = args.save_path
        if params:
            param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
            save_path = f"{args.save_path}_{param_str}"
        
        trainer = train.Trainer(train_cfg, model, optimizer, save_path, device)

        def func_loss(model, batch, current_epoch=0):
            inputs, label = batch
            
            # Get all outputs from model
            logits, features, projected = model(inputs, True)
            
            # Pass to criterion
            total_loss, loss_dict = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=label,
                epoch=current_epoch
            )
            
            return total_loss, loss_dict

        def func_forward(model, batch):
            inputs, label = batch
            logits = model(inputs, False)
            return logits, label

        def func_evaluate(label, predicts):
            stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
            return stat

        # Train the model
        trainer.train(func_loss, func_forward, func_evaluate,
                     data_loader_train, data_loader_test, data_loader_vali)
        
        # Final evaluation on test set
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
        
        # Calculate final metrics
        acc, matrix, f1 = stat_results(label_test, label_estimate_test)
        
        print(f"Training completed successfully. Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(f"Loss configuration used: {loss_config}\n")
        return label_test, label_estimate_test, acc, f1
        
    except Exception as e:
        print(f"Error in classify_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    try:
        training_rate = 0.8
        label_rate = 0.1
        balance = True
        
        mode = "contrastive"
        method = "gru"
        
        # Parse ablation parameters BEFORE handle_argv
        use_semantic = True
        use_contrastive = True
        
        # Check command line arguments for ablation settings
        ablation_parser = argparse.ArgumentParser(add_help=False)
        ablation_parser.add_argument('--use_semantic', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
        ablation_parser.add_argument('--use_contrastive', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
        
        try:
            ablation_args, remaining_args = ablation_parser.parse_known_args()
            use_semantic = ablation_args.use_semantic
            use_contrastive = ablation_args.use_contrastive
            # Replace sys.argv with remaining args for handle_argv
            sys.argv = [sys.argv[0]] + remaining_args
        except:
            pass
        
        # Check if grid search options are in command line arguments
        grid_search_enabled = "--grid_search" in sys.argv
        grid_config_path = None
        
        # Check for grid config path
        if "--grid_config" in sys.argv:
            try:
                idx = sys.argv.index("--grid_config")
                if idx + 1 < len(sys.argv):
                    grid_config_path = sys.argv[idx + 1]
                    sys.argv.remove("--grid_config")
                    sys.argv.remove(grid_config_path)
            except:
                pass
                
        # Remove grid_search flag if present
        if grid_search_enabled:
            sys.argv.remove("--grid_search")
        
        # Now handle the standard arguments
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        
        print(f"\n{'='*80}")
        print("CLASSIFIER ABLATION CONFIGURATION")
        print(f"{'='*80}")
        print(f"Semantic loss: {use_semantic}")
        print(f"Contrastive loss: {use_contrastive}")
        print(f"{'='*80}\n")
        
        # Load data
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("Data dimensions:", embedding.shape, "Label dimensions:", labels.shape)

        # Run normal training with ablation configuration
        print("Running classification with ablation configuration...")
        label_test, label_estimate_test, acc, f1 = classify_embeddings(
            args, embedding, labels, args.label_index,
            training_rate, label_rate, balance, method,
            use_semantic=use_semantic,
            use_contrastive=use_contrastive
        )

        save_path = input("Enter save path for model with dataset and label rate (blind_user_filtered_10): ")
        
        now = datetime.datetime.now()
        save_path = save_path + "_" + now.strftime("%m_%d_%Y_%H_%M")
        save_path = os.path.join("results", "final_results", save_path)
        os.makedirs(save_path, exist_ok=True)

        results_df = pd.DataFrame({'true_label': label_test})
        if len(label_estimate_test.shape) > 1 and label_estimate_test.shape[1] > 1:
            predicted_labels = np.argmax(label_estimate_test, axis=1)
            results_df['predicted_label'] = predicted_labels
            
            for i in range(label_estimate_test.shape[1]):
                results_df[f'prob_class_{i}'] = label_estimate_test[:, i]
        else:
            results_df['predicted_label'] = label_estimate_test

        results_df['correct'] = (results_df['true_label'] == results_df['predicted_label']).astype(int)

        results_df.to_csv(os.path.join(save_path, "results.csv"), index=False)
        print(f"Results saved to {os.path.join(save_path, 'results.csv')}")

        # Plot confusion matrix
        if label_test is not None:
            label_names, label_num, descriptions = load_dataset_label_names(args.dataset_cfg, args.label_index)
            
            if descriptions is None:
                print("Warning: No descriptions found in dataset config")
                descriptions = [f"{name} gesture" for name in label_names]

            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            print(f"Final - Accuracy: {acc:.4f}, F1 score: {f1:.4f}")
            matrix_norm = plot_matrix(matrix, label_names)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
