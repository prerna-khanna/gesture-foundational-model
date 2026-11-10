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
        'pooling': getattr(original_cfg, 'pooling', 'mean')  # Default to 'mean' if not present
    }
    return type(original_cfg)(**config_dict)


LAST_TEST_USER_ROWS = None    # user ID of each test piece from the latest LOSO/user-split call
LAST_TEST_WINDOW_ROWS = None  # original window index of each test piece


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None, params=None, test_user=None, adapt_shots=0, adapt_frac=0.0):
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
        if test_user is None:
            # original random split (unchanged)
            data_train, label_train, data_vali, label_vali, data_test, label_test = \
                prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                         label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                         balance=balance)
        else:
            # LOSO: train/validate on all other users with the original function, test on the held-out user
            users = labels[:, 0, dataset_cfg.user_label_index]
            test_mask = np.isin(users, np.atleast_1d(test_user))
            data_rest, labels_rest = data[~test_mask], labels[~test_mask]
            data_train, label_train, data_vali, label_vali, _, _ = \
                prepare_classifier_dataset(data_rest, labels_rest, label_index=label_index, training_rate=0.9,
                                         label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                         balance=balance)
            # same label offset the original function applies (min label of the data it was given)
            offset = np.min(labels_rest[:, :, label_index])
            win_data = data[test_mask]                                   # (n_windows, 120, F)
            win_label = (labels[test_mask][:, 0, label_index] - offset).astype(np.int64)
            win_user = labels[test_mask][:, 0, dataset_cfg.user_label_index].astype(int)
            assert win_label.min() >= 0, "held-out user has a class no other user performed"

            # User adaptation: choose WHOLE windows of each held-out user to move into training
            # (adapt_shots = k windows per gesture per user; adapt_frac = fraction of that user's windows).
            # At least one window of every gesture always stays in the test set.
            move = np.array([], dtype=int)
            if adapt_shots > 0 or adapt_frac > 0:
                rng = np.random.RandomState(train_cfg.seed)
                mv = []
                for tu in np.unique(win_user):
                    rows_u = np.where(win_user == tu)[0]
                    if adapt_shots > 0:
                        for c in np.unique(win_label[rows_u]):
                            rows_uc = rows_u[win_label[rows_u] == c]
                            k = min(adapt_shots, len(rows_uc) - 1)
                            if k > 0:
                                mv += rng.choice(rows_uc, k, replace=False).tolist()
                    else:
                        k = min(max(1, int(round(adapt_frac * len(rows_u)))), len(rows_u) - 1)
                        mv += rng.choice(rows_u, k, replace=False).tolist()
                move = np.array(sorted(mv), dtype=int)
            keep = np.setdiff1d(np.arange(len(win_label)), move)

            # cut windows into seq_len pieces exactly like prepare_classifier_dataset does (merge=seq_len)
            merge = model_cfg.seq_len
            n_sub = win_data.shape[1] // merge
            def to_pieces(x):
                return x[:, :n_sub * merge].reshape(-1, merge, x.shape[-1])

            if len(move):
                data_train = np.concatenate([data_train, to_pieces(win_data[move]).astype(data_train.dtype)])
                label_train = np.concatenate([label_train, np.repeat(win_label[move], n_sub).astype(label_train.dtype)])
                print(f"Adaptation: moved {len(move)} windows ({len(move) * n_sub} pieces) of the held-out user(s) into training")

            data_test = to_pieces(win_data[keep])
            label_test = np.repeat(win_label[keep], n_sub)
            global LAST_TEST_USER_ROWS, LAST_TEST_WINDOW_ROWS
            LAST_TEST_USER_ROWS = np.repeat(win_user[keep], n_sub)
            LAST_TEST_WINDOW_ROWS = np.repeat(np.where(test_mask)[0][keep], n_sub)   # original window index
            print(f"Held-out test: {len(keep)} windows -> {len(data_test)} pieces of length {merge}")
            print(f"User split: test users {np.atleast_1d(test_user).astype(int).tolist()} | train {len(data_train)} | vali {len(data_vali)} | test {len(data_test)}")
        
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
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        print(f"Optimizer initialized with learning rate: {train_cfg.lr}")
        
        # Create a unique save path for this parameter configuration if doing grid search
        save_path = args.save_path
        if test_user is not None:
            save_path = f"{args.save_path}_loso_u" + "-".join(str(int(x)) for x in np.atleast_1d(test_user))
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
        label_rate = 0.1
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

        # Leave-one-subject-out flag
        loso_enabled = "--loso" in sys.argv
        if loso_enabled:
            sys.argv.remove("--loso")

        # Repeated user split: --user_split <test fraction>, e.g. --user_split 0.3 (5 random user splits)
        user_split_frac = None
        if "--user_split" in sys.argv:
            idx = sys.argv.index("--user_split")
            user_split_frac = float(sys.argv[idx + 1])
            del sys.argv[idx:idx + 2]
        n_user_repeats = 5

        # User adaptation: --adapt_shots k (per gesture) or --adapt_frac f (fraction of the new user's windows)
        adapt_shots, adapt_frac = 0, 0.0
        if "--adapt_shots" in sys.argv:
            idx = sys.argv.index("--adapt_shots"); adapt_shots = int(sys.argv[idx + 1]); del sys.argv[idx:idx + 2]
        if "--adapt_frac" in sys.argv:
            idx = sys.argv.index("--adapt_frac"); adapt_frac = float(sys.argv[idx + 1]); del sys.argv[idx:idx + 2]
        
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

        # Leave-one-subject-out, or repeated train-on-70%/test-on-30% users
        if loso_enabled or user_split_frac is not None:
            from sklearn.metrics import accuracy_score, f1_score
            user_index = args.dataset_cfg.user_label_index
            user_ids = np.unique(labels[:, 0, user_index])
            if loso_enabled:
                folds = [np.array([u]) for u in user_ids]
                print(f"Running LOSO over {len(user_ids)} users: {user_ids.astype(int).tolist()}")
            else:
                n_test = max(1, int(round(user_split_frac * len(user_ids))))
                rng, folds, seen = np.random.RandomState(0), [], set()
                while len(folds) < n_user_repeats:          # distinct random splits
                    f = tuple(np.sort(rng.choice(user_ids, n_test, replace=False)))
                    if f not in seen:
                        seen.add(f); folds.append(np.array(f))
                print(f"Running {n_user_repeats} random user splits, {n_test} of {len(user_ids)} users held out each time")

            tag = ("loso" if loso_enabled else f"usersplit{int(user_split_frac*100)}") + \
                  (f"_adapt{adapt_shots}shot" if adapt_shots else "") + (f"_adapt{adapt_frac}" if adapt_frac else "")
            out_dir = os.path.join("results", tag,
                                   f"{args.dataset}_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M')}")
            os.makedirs(out_dir, exist_ok=True)
            per_user, all_preds = [], []
            for u in folds:
                label_test, label_estimate_test, acc, f1 = classify_embeddings(
                    args, embedding, labels, args.label_index,
                    training_rate, label_rate, balance, method, test_user=u,
                    adapt_shots=adapt_shots, adapt_frac=adapt_frac)
                pred = np.argmax(label_estimate_test, axis=1)
                test_users_str = "-".join(str(int(x)) for x in u)
                per_user.append({'user': test_users_str, 'n_test': len(label_test),
                                 'n_classes': len(np.unique(label_test)), 'accuracy': acc, 'f1_score': f1})
                all_preds.append(pd.DataFrame({'user': LAST_TEST_USER_ROWS, 'window': LAST_TEST_WINDOW_ROWS, 'fold': test_users_str, 'true_label': label_test, 'predicted_label': pred}))
                print(f"===== test users {test_users_str}: acc={acc:.4f} f1={f1:.4f} =====")
                # save after every user so nothing is lost if a run crashes
                pd.DataFrame(per_user).to_csv(os.path.join(out_dir, "per_user.csv"), index=False)
                pd.concat(all_preds).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

            df, preds = pd.DataFrame(per_user), pd.concat(all_preds)
            # per individual user (also meaningful for the 70/30 splits)
            by_user = preds.groupby('user').apply(lambda g: accuracy_score(g.true_label, g.predicted_label))
            by_user.rename('accuracy').to_csv(os.path.join(out_dir, "per_individual_user.csv"))
            pooled_acc = accuracy_score(preds.true_label, preds.predicted_label)
            # one decision per gesture: majority vote over the pieces of each window
            vote = preds.groupby(['fold', 'window']).agg(true_label=('true_label', 'first'),
                                                         predicted_label=('predicted_label', lambda x: x.mode().iloc[0]))
            window_acc = accuracy_score(vote.true_label, vote.predicted_label)
            window_f1 = f1_score(vote.true_label, vote.predicted_label, average='macro')
            pooled_f1 = f1_score(preds.true_label, preds.predicted_label, average='macro')
            mode_name = "LOSO" if loso_enabled else f"user split (test {user_split_frac:.0%}, {n_user_repeats} repeats)"
            if adapt_shots: mode_name += f" + {adapt_shots}-shot/gesture adaptation"
            if adapt_frac: mode_name += f" + {adapt_frac:.0%} adaptation"
            summary = (f"{mode_name} {args.dataset}: mean over folds acc={df.accuracy.mean():.4f} ± {df.accuracy.std():.4f}, "
                       f"f1={df.f1_score.mean():.4f} ± {df.f1_score.std():.4f} | "
                       f"pooled acc={pooled_acc:.4f}, pooled macro-F1={pooled_f1:.4f} | "
                       f"per-gesture (majority vote over pieces) acc={window_acc:.4f}, macro-F1={window_f1:.4f} | "
                       f"worst user {by_user.idxmin()} ({by_user.min():.4f})")
            print(summary)
            with open(os.path.join(out_dir, "summary.txt"), "w") as f:
                f.write(summary + "\n")
            print(f"Results saved to {out_dir}")

        # Run grid search if enabled
        elif grid_search_enabled:
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