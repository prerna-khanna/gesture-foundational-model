#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset

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
        return sim_matrix
    
def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    try:
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
        
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
        model = model.to(device)  # Ensure model is on correct device
        print("Model structure:", model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

        def func_loss(model, batch):
            inputs, label = batch
            
            # Get logits for classification loss
            logits = model(inputs, True)
            class_loss = criterion(logits, label)
            
            # Get GRU features for semantic loss
            # Run inputs through first GRU layer only
            semantic_features = model.gru0(inputs)[0][:, -1, :]  # Shape will be [batch_size, 20]
            sem_loss = semantic_criterion(semantic_features, label)
            
            total_loss = class_loss + 0.1 * sem_loss
            return total_loss

        def func_forward(model, batch):
            inputs, label = batch
            # Let model handle the full forward pass
            logits = model(inputs, False)
            return logits, label

        def func_evaluate(label, predicts):
            stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
            return stat

        trainer.train(func_loss, func_forward, func_evaluate,
                     data_loader_train, data_loader_test, data_loader_vali)
        
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
        training_rate = 0.8
        label_rate = 0.1
        balance = True

        mode = "base"
        method = "gru"
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("size of embedding: ", embedding.shape, "size of labels: ", labels.shape)

        label_test, label_estimate_test = classify_embeddings(args, embedding, labels, args.label_index,
                                                            training_rate, label_rate, balance=balance, method=method)

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