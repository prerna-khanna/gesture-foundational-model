#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        'pooling': params.get('pooling', getattr(original_cfg, 'pooling', 'mean')),
        'weight_decay': params.get('weight_decay', getattr(original_cfg, 'weight_decay', 0.0)),
        'grad_clip_norm': params.get('grad_clip_norm', getattr(original_cfg, 'grad_clip_norm', 1.0))
    }
    return type(original_cfg)(**config_dict)


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None, params=None):
    # contrastive + semantic learning
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
        model = ContrastiveTransformerClassifier(
            input_dim=data_train.shape[-1],  # Feature dimension
            hidden_dim=hidden_dim,           # Using our default or config value
            num_classes=label_num
        ).to(device)
        
        print(f"Model architecture initialized with hidden dimension: {hidden_dim}")
        
        # Initialize combined loss (classification + semantic + contrastive)
        criterion = ContrastiveCombinedLoss(
            label_names=label_names, 
            descriptions=descriptions,
            pooling=getattr(train_cfg, 'pooling', 'cls'),  # Default to 'mean' if not present
            device=device,
            hidden_dim=hidden_dim  # Pass the hidden dimension
        )

        # Only when using SVM classifier
        """criterion = ContrastiveSVMLoss(
            label_names=label_names, 
            descriptions=descriptions,
            pooling=getattr(train_cfg, 'pooling', 'cls'),
            device=device,
            hidden_dim=hidden_dim
        )"""
        
        # Setup optimizer and trainer
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=train_cfg.lr,
            weight_decay=getattr(train_cfg, 'weight_decay', 0.0)
        )
        print(
            f"Optimizer initialized with learning rate: {train_cfg.lr}, "
            f"weight decay: {getattr(train_cfg, 'weight_decay', 0.0)}"
        )
        
        # Create a unique save path for this parameter configuration if doing grid search
        save_path = args.save_path
        if params:
            param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
            save_path = f"{args.save_path}_{param_str}"
        
        trainer = train.Trainer(train_cfg, model, optimizer, save_path, device)

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
        
        # Calculate final metrics
        acc, matrix, f1 = stat_results(label_test, label_estimate_test)
        
        print(f"Training completed successfully. Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return label_test, label_estimate_test, acc, f1
        
    except Exception as e:
        print(f"Error in classify_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e


def run_grid_search(args, embedding, labels, label_index, training_rate, label_rate, balance=True, method="gru", param_grid=None):
    """
    Run grid search over the parameter grid to find the best hyperparameters.
    
    Args:
        args: Command line arguments
        embedding: Embedding data
        labels: Label data
        label_index: Index of labels to use
        training_rate: Rate of training data to use
        label_rate: Rate of labeled data to use
        balance: Whether to balance classes
        method: Method to use (e.g., "gru")
        param_grid: Dictionary of parameter grids to search over
    
    Returns:
        best_params: Best parameters found
        best_acc: Best accuracy found
        best_f1: Best F1 score found
        results: All results from grid search
    """
    if param_grid is None:
        print("No parameter grid provided, using default parameters.")
        return None, None, None, None
    
    # Create a grid of all parameter combinations
    grid = list(ParameterGrid(param_grid))
    print(f"Running grid search with {len(grid)} parameter combinations.")
    
    # Create directory for grid search results
    results_dir = os.path.join('results', f'grid_search_{args.dataset}_{args.dataset_version}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_acc = 0
    best_f1 = 0
    best_params = None
    results = []
    
    # Results file
    results_file = os.path.join(results_dir, f'grid_search_results_new_{args.save_model}.csv')
    
    # Write header to results file
    with open(results_file, 'w') as f:
        params_header = ','.join(param_grid.keys())
        f.write(f"{params_header},accuracy,f1_score\n")
    
    # Run each parameter combination
    for i, params in enumerate(grid):
        print(f"\n\n===== Running parameter combination {i+1}/{len(grid)} =====")
        print(f"Parameters: {params}")
        
        try:
            # Modify args to create a unique save path for this run
            param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
            run_args = copy.deepcopy(args)
            run_args.save_model = f"{args.save_model}_{param_str}"
            
            # Run training with these parameters
            _, _, acc, f1 = classify_embeddings(
                run_args, embedding, labels, label_index, 
                training_rate, label_rate, balance, method, params
            )
            
            # Track results
            result = {**params, 'accuracy': acc, 'f1_score': f1}
            results.append(result)
            
            # Write results to file
            with open(results_file, 'a') as f:
                params_values = ','.join([str(params[k]) for k in param_grid.keys()])
                f.write(f"{params_values},{acc},{f1}\n")
            
            # Update best parameters if this combination is better
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_params = params
                print(f"New best: Accuracy = {best_acc:.4f}, F1 = {best_f1:.4f}")
                print(f"Best parameters: {best_params}")
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            # Write error to results file
            with open(results_file, 'a') as f:
                params_values = ','.join([str(params[k]) for k in param_grid.keys()])
                f.write(f"{params_values},ERROR,ERROR\n")
    
    # Write final results summary
    summary_file = os.path.join(results_dir, f'grid_search_summary_{args.save_model}.json')
    with open(summary_file, 'w') as f:
        summary = {
            'best_params': best_params,
            'best_accuracy': float(best_acc),
            'best_f1': float(best_f1),
            'all_results': results
        }
        json.dump(summary, f, indent=2)
    
    print("\n===== Grid Search Complete =====")
    print(f"Best Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Results saved to {results_dir}")
    
    return best_params, best_acc, best_f1, results


if __name__ == "__main__":
    try:
        training_rate = 0.8
        label_rate = 0.2
        balance = True
        
        mode = "contrastive"
        method = "gru"
        
        # Modify the argument parser to include grid search options
        # This needs to be done before handle_argv is called
        import sys
        
        # Check if grid search options are in command line arguments
        grid_search_enabled = "--grid_search" in sys.argv
        grid_config_path = None
        
        # Check for grid config path
        if "--grid_config" in sys.argv:
            try:
                idx = sys.argv.index("--grid_config")
                if idx + 1 < len(sys.argv):
                    grid_config_path = sys.argv[idx + 1]
                    # Remove these arguments so they don't interfere with handle_argv
                    sys.argv.remove("--grid_config")
                    sys.argv.remove(grid_config_path)
            except:
                pass
                
        # Remove grid_search flag if present
        if grid_search_enabled:
            sys.argv.remove("--grid_search")
        
        # Now handle the standard arguments
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        
        # Default parameter grid
        param_grid = {
            'batch_size': [64],
            'lr': [1e-4, 1e-3, 5e-3],
            'n_epochs': [500, 1000],
            #'warmup': [0.1, 0.2],
            'lambda2': [0.001, 0.005, 0.01]
        }
        
        # Load custom parameter grid if specified
        if grid_config_path:
            try:
                with open(grid_config_path, 'r') as f:
                    param_grid = json.load(f)
                print(f"Loaded parameter grid from {grid_config_path}")
            except Exception as e:
                print(f"Error loading grid config: {str(e)}")
                print("Using default parameter grid.")
        
        # Load data
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("Data dimensions:", embedding.shape, "Label dimensions:", labels.shape)

        # Run grid search if enabled
        if grid_search_enabled:
            print("Running grid search...")
            best_params, best_acc, best_f1, _ = run_grid_search(
                args, embedding, labels, args.label_index,
                training_rate, label_rate, balance, method, param_grid
            )
            
            # Train final model with best parameters if grid search was successful
            if best_params:
                print("Training final model with best parameters...")
                final_args = copy.deepcopy(args)
                final_args.save_model = f"{args.save_model}_best"
                label_test, label_estimate_test, _, _ = classify_embeddings(
                    final_args, embedding, labels, args.label_index,
                    training_rate, label_rate, balance, method, best_params
                )
                
                # Plot confusion matrix for best model
                label_names, label_num, descriptions = load_dataset_label_names(args.dataset_cfg, args.label_index)
                acc, matrix, f1 = stat_results(label_test, label_estimate_test)
                print(f"Final model - Accuracy: {acc:.4f}, F1 score: {f1:.4f}")
                matrix_norm = plot_matrix(matrix, label_names)
            
        else:
            # Run normal training without grid search
            print("Running normal training...")
            label_test, label_estimate_test, acc, f1 = classify_embeddings(
                args, embedding, labels, args.label_index,
                training_rate, label_rate, balance, method
            )

            save_path = input("Enter save path for model with dataset and label rate (blind_user_filtered_10): ")
            
            now = datetime.datetime.now()
            save_path = save_path + "_" + now.strftime("%m_%d_%Y_%H_%M")
            save_path = os.path.join("results", "final_results", save_path)
            os.makedirs(save_path, exist_ok=True)

            results_df = pd.DataFrame({'true_label': label_test})
            if len(label_estimate_test.shape) > 1 and label_estimate_test.shape[1] > 1:
                # Get the predicted class (argmax along axis 1)
                predicted_labels = np.argmax(label_estimate_test, axis=1)
                results_df['predicted_label'] = predicted_labels
                
                # Add probability columns for each class
                for i in range(label_estimate_test.shape[1]):
                    results_df[f'prob_class_{i}'] = label_estimate_test[:, i]
            else:
                # If label_estimate_test already contains class predictions
                results_df['predicted_label'] = label_estimate_test

            # Add a column for correct/incorrect predictions
            results_df['correct'] = (results_df['true_label'] == results_df['predicted_label']).astype(int)

            # Save to CSV
            results_df.to_csv(os.path.join(save_path, "results.csv"), index=False)
            print(f"Results saved to {os.path.join(save_path, 'results.csv')}")
            print(f"Shape of true labels: {label_test.shape}, predicted: {label_estimate_test.shape}")

            # Plot confusion matrix
            if label_test is not None:
                label_names, label_num, descriptions = load_dataset_label_names(args.dataset_cfg, args.label_index)
                
                if descriptions is None:
                    print("Warning: No descriptions found in dataset config")
                    descriptions = [f"{name} gesture" for name in label_names]

                acc, matrix, f1 = stat_results(label_test, label_estimate_test)
                print(f"Normal training - Accuracy: {acc:.4f}, F1 score: {f1:.4f}")
                matrix_norm = plot_matrix(matrix, label_names)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()