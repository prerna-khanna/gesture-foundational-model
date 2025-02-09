#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import train_save_gru_embed
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset
from train_save_gru_embed import compute_similarity_matrix, compute_distance_matrix
import os


def compare_embeddings(save_path, trainer, label_names, device):
    """Compare GRU and BERT embeddings
    
    Args:
        save_path (str): Base path where files are saved
        trainer (Trainer): Trainer instance
        label_names (list): List of class label names
        device (torch.device): Device to run computations on
    """
    # Load saved embeddings using same base path as model.pt
    embeddings_path = save_path + '_embeddings.pt'
    saved_data = torch.load(embeddings_path)
    test_embeddings = saved_data['test_embeddings'].to(device)
    test_labels = saved_data['test_labels'].to(device)
    
    # Create BERT semantic loss object to get BERT embeddings
    semantic_loss = SemanticLoss(label_names, device)
    bert_sim_matrix = semantic_loss.semantic_sims
    
    # Compute GRU similarity and distance matrices
    gru_sim_matrix = torch.matmul(F.normalize(test_embeddings, p=2, dim=1), 
                                 F.normalize(test_embeddings, p=2, dim=1).t())
    
    # Compute and normalize distance matrix
    dist_matrix = torch.cdist(test_embeddings, test_embeddings, p=2)
    gru_dist_matrix = dist_matrix / dist_matrix.max()
    
    # Save matrices for visualization
    matrices = {
        'bert_sim': bert_sim_matrix.cpu().numpy(),
        'gru_sim': gru_sim_matrix.cpu().numpy(),
        'gru_dist': gru_dist_matrix.cpu().numpy(),
        'labels': test_labels.cpu().numpy()
    }
    
    matrices_path = save_path + '_matrices.npy'
    np.save(matrices_path, matrices)
    print(f"Saved matrices to {matrices_path}")
    return matrices


class SemanticLoss(nn.Module):
    def __init__(self, label_names, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.bert.to(device)
        
        # Change projection input dimension to match GRU output (20)
        self.projection = nn.Linear(20, 32).to(device)  # Changed from 72 to 20
        self.semantic_sims = self._compute_semantic_sims(label_names)
        
    def forward(self, embeddings, labels, margin=1.0):
        # Project to lower dimension
        embeddings = self.projection(embeddings)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        loss = 0
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                if sem_sim > 0.7:
                    loss += torch.max(torch.tensor(0.0).to(self.device),
                                    margin - dists[i, j])
                    
        return loss / (batch_size * (batch_size - 1))
        
    def _compute_semantic_sims(self, label_names):
        """Compute semantic similarities between gesture labels using BERT"""
        # Add context to gesture names for better semantic understanding
        descriptions = [f"a gesture movement in the {name} direction" for name in label_names]
        
        # Get BERT embeddings
        with torch.no_grad():
            inputs = self.tokenizer(descriptions, padding=True, return_tensors="pt").to(self.device)
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        #print("Semantic similarity matrix shape:", sim_matrix)
        return sim_matrix
    
def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    try:
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
    

        save_dir = os.path.join('dataset', args.dataset)
        os.makedirs(save_dir, exist_ok=True)

        # Override any complex path structure from args
        args.save_path = os.path.join(save_dir, 'limu_gru_v1')  # Direct simple path

    
        print(f"Number of classes: {label_num}")
        print(f"Label names: {label_names}")
        
        # Prepare data
        data_train, label_train, data_vali, label_vali, data_test, label_test = \
            prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                     label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                     balance=balance)
        
        # Create datasets and dataloaders
        data_set_train = IMUDataset(data_train, label_train)
        data_set_vali = IMUDataset(data_vali, label_vali)
        data_set_test = IMUDataset(data_test, label_test)
        
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

        # Initialize losses
        criterion = nn.CrossEntropyLoss()
        semantic_criterion = SemanticLoss(label_names, device).to(device)
        
        # Initialize model
        model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
        model = model.to(device)
        print("Model structure:", model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        
        trainer = train_save_gru_embed.Trainer(train_cfg, model, optimizer, args.save_path, device)

        def func_loss(model, batch, epoch=None):
            inputs, label = batch
            logits = model(inputs, True)
            class_loss = criterion(logits, label)
            semantic_features = model.gru0(inputs)[0][:, -1, :]
            sem_loss = semantic_criterion(semantic_features, label)
            total_loss = class_loss + 0.1 * sem_loss
            
            # Return both loss and dictionary of loss components
            return total_loss, {
                'classification_loss': class_loss.item(),
                'semantic_loss': sem_loss.item(),
                'total_loss': total_loss.item()
            }

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
        
        # Get predictions for test set
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
        
        # Compare embeddings after training is complete
        print("Comparing GRU and BERT embeddings...")
        matrices = compare_embeddings(
            save_path=args.save_path,
            trainer=trainer,
            label_names=label_names,
            device=device
        )
        print("Embedding comparison completed and saved")
        
        print("Training completed successfully")
        return label_test, label_estimate_test, matrices
        
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

        mode = "base"
        method = "gru"
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("size of embedding: ", embedding.shape, "size of labels: ", labels.shape)

        label_test, label_estimate_test, comparison_matrices = classify_embeddings(
            args, embedding, labels, args.label_index,
            training_rate, label_rate, balance=balance, method=method
        )

        if label_test is not None and label_estimate_test is not None:
            label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            matrix_norm = plot_matrix(matrix, label_names)
            
            # Now you can also analyze the embedding comparison matrices
            print("\nEmbedding comparison matrices saved. You can load them using:")
            print("matrices = np.load(args.save_path + '_matrices.npy', allow_pickle=True).item()")
        else:
            print("Error: classify_embeddings returned None")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()