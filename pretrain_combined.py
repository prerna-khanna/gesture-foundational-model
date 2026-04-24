#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device, LIBERTDataset4Pretrain, \
    prepare_pretrain_dataset, Preprocess4Normalization, Preprocess4Mask
from combined_dataset import load_combined_datasets

def main(args, training_rate, nucleus_prob=0.8):
    # Load the combined dataset
    data, labels = load_combined_datasets(dataset_version='20_120', seed=42)
    
    # Load configurations
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    model_cfg = args.model_cfg
    
    # Prepare the dataset with nucleus-aware masking
    pipeline = [
        Preprocess4Normalization(model_cfg.feature_num), 
        Preprocess4Mask(mask_cfg, nucleus_aware=True, nucleus_prob=nucleus_prob)
    ]
    
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(
        data, labels, training_rate, seed=train_cfg.seed
    )
    
    # Create data loaders
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(
        data_set_train, 
        shuffle=True, 
        batch_size=train_cfg.batch_size
    )
    data_loader_test = DataLoader(
        data_set_test, 
        shuffle=False, 
        batch_size=train_cfg.batch_size
    )
    
    # Initialize model, criterion, and optimizer
    model = LIMUBertModel4Pretrain(model_cfg)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    
    # Initialize trainer
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)
    
    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
        seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        loss_lm = criterion(seq_recon, seqs)
        return loss_lm
    
    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
        seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        return seq_recon, seqs
    
    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()
    
    # Train the model
    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(
            func_loss, func_forward, func_evaluate,
            data_loader_train, data_loader_test,
            model_file=args.pretrain_model
        )
    else:
        trainer.pretrain(
            func_loss, func_forward, func_evaluate,
            data_loader_train, data_loader_test
        )

if __name__ == "__main__":
    import sys
    from utils import handle_argv
    
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    
    # Check if nucleus_prob is provided as command-line argument
    nucleus_prob = 0.8  # Default value
    if len(sys.argv) > 1:
        # Try to extract nucleus_prob from save model name
        for i, arg in enumerate(sys.argv):
            if 'nucleus_' in arg:
                try:
                    nucleus_prob = float(arg.split('nucleus_')[-1])
                except:
                    nucleus_prob = 0.8
    
    main(args, training_rate, nucleus_prob=nucleus_prob)