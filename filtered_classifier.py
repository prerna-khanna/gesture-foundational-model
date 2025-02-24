#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from embedding import load_embedding_label
from contrastive.augmenter import GestureAugmenter
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.models import ContrastiveGRUClassifier

import train
from config import load_dataset_label_names
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, IMUDataset, load_classifier_config, prepare_classifier_dataset, handle_argv


def filter_and_remap_labels(data, labels, descriptions, label_names, exclude_labels=[9, 11]):
    """
    Filter out specified labels and remap remaining labels to be consecutive
    """
    # Get the first label for each sequence (since labels are repeated)
    if len(labels.shape) == 3:
        label_index = labels[:, 0, 0]  # Take first timestep of activity labels
    else:
        label_index = labels
        
    # Create a mask for samples we want to keep
    keep_mask = np.ones(len(label_index), dtype=bool)
    for label in exclude_labels:
        keep_mask &= (label_index != label)
    
    # Filter data and labels
    filtered_data = data[keep_mask]
    if len(labels.shape) == 3:
        filtered_labels = labels[keep_mask]
    else:
        filtered_labels = label_index[keep_mask]
    
    # Create label mapping (excluding specified labels)
    unique_labels = sorted(list(set(label_index[keep_mask])))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    # Apply remapping
    if len(labels.shape) == 3:
        # For 3D labels, remap each timestep
        remapped_labels = np.zeros_like(filtered_labels)
        for i in range(filtered_labels.shape[0]):
            for j in range(filtered_labels.shape[1]):
                old_label = filtered_labels[i, j, 0]
                remapped_labels[i, j, 0] = label_map[old_label]
                remapped_labels[i, j, 1] = filtered_labels[i, j, 1]  # Preserve user labels
    else:
        remapped_labels = np.array([label_map[l] for l in filtered_labels])
    
    # Filter descriptions and label names
    filtered_descriptions = [desc for i, desc in enumerate(descriptions) if i not in exclude_labels]
    filtered_label_names = [name for i, name in enumerate(label_names) if i not in exclude_labels]
    
    return filtered_data, remapped_labels, filtered_descriptions, filtered_label_names, len(unique_labels)

def classify_filtered_embeddings(args, data, labels, label_index, training_rate, label_rate, 
                               balance=False, method=None, exclude_labels=[9, 11]):
    try:
        # Load configurations
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        label_names, label_num, descriptions = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
        
        if descriptions is None:
            descriptions = [f"{name} gesture" for name in label_names]
        
        # Filter and remap data
        filtered_data, filtered_labels, filtered_descriptions, filtered_label_names, new_label_num = \
            filter_and_remap_labels(data, labels, descriptions, label_names, exclude_labels)
        
        print(f"Original number of samples: {len(data)}")
        print(f"Filtered number of samples: {len(filtered_data)}")
        print(f"Number of classes after filtering: {new_label_num}")
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Filtered labels shape: {filtered_labels.shape}")
        
        # Calculate hidden dimension
        hidden_dim = getattr(model_cfg, 'hidden_dim', 128)
        
        # Prepare datasets with filtered data
        data_train, label_train, data_vali, label_vali, data_test, label_test = \
            prepare_classifier_dataset(filtered_data, filtered_labels, label_index=0,
                                    training_rate=training_rate, label_rate=label_rate,
                                    merge=model_cfg.seq_len, seed=train_cfg.seed,
                                    balance=balance)
        
        # Flatten data for training if needed
        if len(data_train.shape) == 3:
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_vali = data_vali.reshape(data_vali.shape[0], -1)
            data_test = data_test.reshape(data_test.shape[0], -1)
        
        # Create datasets with augmentation
        augmenter = GestureAugmenter()
        data_set_train = IMUDataset(data_train, label_train, pipeline=[augmenter.augment])
        data_set_vali = IMUDataset(data_vali, label_vali)
        data_set_test = IMUDataset(data_test, label_test)
        
        # Create dataloaders
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
        
        # Initialize model
        model = ContrastiveGRUClassifier(
            input_dim=data_train.shape[-1],
            hidden_dim=hidden_dim,
            num_classes=new_label_num
        ).to(device)
        
        # Initialize combined loss
        criterion = ContrastiveCombinedLoss(
            label_names=filtered_label_names,
            descriptions=filtered_descriptions,
            pooling=train_cfg.pooling,
            device=device,
            hidden_dim=hidden_dim
        )
        
        # Setup optimizer and trainer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)
        
        def func_loss(model, batch, current_epoch=0):
            inputs, label = batch
            logits, features, projected = model(inputs, True)
            total_loss, loss_dict = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=label,
                epoch=current_epoch
            )
            
            if current_epoch % 5 == 0:
                print(f"Epoch {current_epoch} - Loss Components:")
                for loss_name, loss_value in loss_dict.items():
                    print(f"  {loss_name}: {loss_value:.4f}")
            
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
        
        # Final evaluation
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
        
        print("Training completed successfully")
        return label_test, label_estimate_test, filtered_label_names
        
    except Exception as e:
        print(f"Error in classify_filtered_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        training_rate = 0.8
        label_rate = 0.1
        balance = True
        exclude_labels = [9, 11]
        
        mode = "contrastive"
        method = "gru"
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        
        # Load embeddings and labels
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("Data dimensions:", embedding.shape, "Label dimensions:", labels.shape)
        
        # Train classifier with filtered data
        label_test, label_estimate_test, filtered_label_names = classify_filtered_embeddings(
            args, embedding, labels, args.label_index,
            training_rate, label_rate, balance=balance,
            method=method, exclude_labels=exclude_labels
        )
        
        if label_test is not None:
            # Calculate and plot results
            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            print(f"Final Test Accuracy: {acc:.4f}")
            print(f"Final Test F1 Score: {f1:.4f}")
            matrix_norm = plot_matrix(matrix, filtered_label_names)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()