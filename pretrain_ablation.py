#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ablation study version of pretrain.py
# Allows toggling nucleus and sig_axis embeddings

"""
python pretrain_ablation.py v1 hhar 20_120 -s limu_v1 --use_nucleus true --use_sig_axis true --nucleus_prob 0.8
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


def main(args, training_rate, nucleus_prob=0.8, use_nucleus=True, use_sig_axis=True):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    # Use nucleus-aware masking with specified nucleus probability
    pipeline = [Preprocess4Normalization(model_cfg.feature_num), 
                Preprocess4Mask(mask_cfg, nucleus_aware=True, nucleus_prob=nucleus_prob)]
    
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    # Use custom dataset that supports embedding ablation
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline, 
                                            use_nucleus=use_nucleus, use_sig_axis=use_sig_axis)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline,
                                           use_nucleus=use_nucleus, use_sig_axis=use_sig_axis)
    
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        if use_nucleus and use_sig_axis:
            # Both embeddings
            mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        elif use_nucleus:
            # Only nucleus
            mask_seqs, masked_pos, seqs, nucleus_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=None)
        elif use_sig_axis:
            # Only sig_axis
            mask_seqs, masked_pos, seqs, sig_axis_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=None, sig_axis_mask=sig_axis_mask)
        else:
            # Neither (baseline BERT)
            mask_seqs, masked_pos, seqs = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=None, sig_axis_mask=None)
        
        loss_lm = criterion(seq_recon, seqs)
        return loss_lm
    
    def func_forward(model, batch):
        if use_nucleus and use_sig_axis:
            mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        elif use_nucleus:
            mask_seqs, masked_pos, seqs, nucleus_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=None)
        elif use_sig_axis:
            mask_seqs, masked_pos, seqs, sig_axis_mask = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=None, sig_axis_mask=sig_axis_mask)
        else:
            mask_seqs, masked_pos, seqs = batch
            seq_recon = model(mask_seqs, masked_pos, nucleus_mask=None, sig_axis_mask=None)
        
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)

if __name__ == "__main__":
    mode = "base"
    training_rate = 0.8
    
    # Parse additional ablation parameters BEFORE handle_argv
    nucleus_prob = 0.8
    use_nucleus = True
    use_sig_axis = True
    
    # Check command line arguments for ablation settings
    ablation_parser = argparse.ArgumentParser(add_help=False)
    ablation_parser.add_argument('--nucleus_prob', type=float, default=0.8)
    ablation_parser.add_argument('--use_nucleus', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
    ablation_parser.add_argument('--use_sig_axis', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
    
    try:
        ablation_args, remaining_args = ablation_parser.parse_known_args()
        nucleus_prob = ablation_args.nucleus_prob
        use_nucleus = ablation_args.use_nucleus
        use_sig_axis = ablation_args.use_sig_axis
        # Replace sys.argv with remaining args for handle_argv
        sys.argv = [sys.argv[0]] + remaining_args
    except:
        pass
    
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    
    print(f"\n{'='*80}")
    print("ABLATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Nucleus embedding: {use_nucleus}")
    print(f"Sig_axis embedding: {use_sig_axis}")
    print(f"Nucleus masking probability: {nucleus_prob}")
    print(f"{'='*80}\n")
    
    main(args, training_rate, nucleus_prob=nucleus_prob, use_nucleus=use_nucleus, use_sig_axis=use_sig_axis)
