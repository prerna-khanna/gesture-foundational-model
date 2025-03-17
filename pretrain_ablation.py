#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from original pretrain.py for ablation study
# This version allows for selective ablation of nucleus and significant axis detection

import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig, load_model_config, load_dataset_stats
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization, Preprocess4Mask


# Create a flexible transformer that can selectively use or ignore nucleus and sig_axis
class TransformerFlexible(nn.Module):
    """ Transformer with Self-Attentive Blocks - Allows for selective feature ablation"""
    def __init__(self, cfg, use_nucleus=True, use_sig_axis=True):
        super().__init__()
        self.embed = models.Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = models.MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = models.LayerNorm(cfg)
        self.pwff = models.PositionWiseFeedForward(cfg)
        self.norm2 = models.LayerNorm(cfg)
        
        # Control which features to use
        self.use_nucleus = use_nucleus
        self.use_sig_axis = use_sig_axis

    def forward(self, x, nucleus_mask=None, sig_axis_mask=None):
        # Selectively use or ignore nucleus and sig_axis
        h = self.embed(
            x, 
            nucleus_mask=nucleus_mask if self.use_nucleus else None,
            sig_axis_mask=sig_axis_mask if self.use_sig_axis else None
        )

        for _ in range(self.n_layers):
            # The attn function can handle None for sig_axis_mask
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


# Flexible BERT model for ablation studies
class LIMUBertAblation(nn.Module):
    def __init__(self, cfg, use_nucleus=True, use_sig_axis=True, output_embed=False):
        super().__init__()
        self.transformer = TransformerFlexible(cfg, use_nucleus, use_sig_axis) # encoder with flexible features
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed
        
        # Store the ablation settings
        self.use_nucleus = use_nucleus
        self.use_sig_axis = use_sig_axis

    def forward(self, input_seqs, masked_pos=None, nucleus_mask=None, sig_axis_mask=None):
        # Selectively pass nucleus and sig_axis masks based on ablation settings
        h_masked = self.transformer(
            input_seqs, 
            nucleus_mask=nucleus_mask if self.use_nucleus else None,
            sig_axis_mask=sig_axis_mask if self.use_sig_axis else None
        )

        if self.output_embed:
            return h_masked

        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm


def load_data_and_config(args):
    """Load data and model configuration from args or from files"""
    # Check if model configuration is already loaded
    if not hasattr(args, 'model_cfg'):
        args.model_cfg = load_model_config('pretrain', 'base', args.model_version)
    
    # Check if dataset configuration is already loaded
    if not hasattr(args, 'dataset_cfg'):
        args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    
    # Load data files
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    
    # Load training and mask configuration
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    
    return data, labels, train_cfg, args.model_cfg, mask_cfg, args.dataset_cfg


def main(args, training_rate):
    # Load data and configurations
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_data_and_config(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    # Parse the ablation mode
    ablation_features = args.ablation.split(',') if args.ablation else []
    
    # Determine which features to use
    use_nucleus = 'nucleus' not in ablation_features
    use_sig_axis = 'sig_axis' not in ablation_features
    
    # Print ablation status
    print(f"Ablation Settings:")
    print(f"  Use Nucleus: {use_nucleus}")
    print(f"  Use Significant Axis: {use_sig_axis}")
    
    # Use the flexible ablation model
    model = LIMUBertAblation(model_cfg, use_nucleus=use_nucleus, use_sig_axis=use_sig_axis)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        loss_lm = criterion(seq_recon, seqs)
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    print(f"Training model: {args.save_path}")
    
    if hasattr(args, 'pretrain_model') and args.pretrain_model:
        print(f"Loading pretrained model from: {args.pretrain_model}")
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=args.pretrain_model)
    else:
        print("Training from scratch")
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)
    
    print(f"Training completed: {args.save_path}")


if __name__ == "__main__":
    # Create custom argument parser to add ablation options
    parser = argparse.ArgumentParser(description='Run pretrain with selective feature ablation')
    parser.add_argument('model_version', type=str, help='Model config, e.g. v1')
    parser.add_argument('dataset', type=str, choices=['hhar', 'motion', 'uci', 'shoaib'], help='Dataset name')
    parser.add_argument('dataset_version', type=str, help='Dataset version, e.g. 20_120')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='pretrain.json', help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='config/mask.json', help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1, help='Label Index setting the task')
    parser.add_argument('-s', '--save_model', type=str, default='model', help='The saved model name')
    parser.add_argument('--ablation', type=str, default='', 
                      help='Comma-separated list of features to ablate: "nucleus,sig_axis" or "nucleus" or "sig_axis"')
    
    args = parser.parse_args()
    
    # Create ablation suffix for the model name
    if args.ablation:
        ablation_suffix = '_no_' + '_'.join(args.ablation.split(','))
        args.save_model = args.save_model + ablation_suffix
    
    # Set up directory for saving
    mode = "base"
    save_dir = f"saved/pretrain_{mode}_{args.dataset}_{args.dataset_version}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    args.save_path = os.path.join(save_dir, args.save_model)
    
    # If model_file is specified, set up pretrain_model path
    if args.model_file:
        args.pretrain_model = os.path.join(save_dir, args.model_file)
    else:
        args.pretrain_model = None
    
    # Set data paths
    args.data_path = f"dataset/{args.dataset}/data_{args.dataset_version}.npy"
    args.label_path = f"dataset/{args.dataset}/label_{args.dataset_version}.npy"
    
    # Load configurations
    args.model_cfg = load_model_config('pretrain', 'base', args.model_version)
    args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    
    print(f"Model configuration loaded: {args.model_version}")
    
    # Run pretraining
    training_rate = 0.8
    main(args, training_rate)