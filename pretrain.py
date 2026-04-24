#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
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


def main(args, training_rate, nucleus_prob=0.8):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    # Use nucleus-aware masking with specified nucleus probability
    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg, nucleus_aware=True, nucleus_prob=nucleus_prob)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)
    #model = LIMUBertModel4Pretrain(model_cfg, use_conformer=model_cfg.use_conformer)

    criterion = nn.MSELoss(reduction='none')
    #criterion = FrequencyDomainLoss(alpha=train_cfg.freq_loss_alpha, reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    #device = torch.device("mps")
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
        seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        loss_lm = criterion(seq_recon, seqs) # for masked LM
        return loss_lm
    
    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs, nucleus_mask, sig_axis_mask = batch
        seq_recon = model(mask_seqs, masked_pos, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
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
    import sys
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    
    # Check if nucleus_prob is provided as command-line argument
    nucleus_prob = 0.8  # Default value
    if len(sys.argv) > 1:
        # Try to extract nucleus_prob from save model name
        # Format: python pretrain.py v1 hhar 20_120 -s limu_v1_nucleus_0.8
        for i, arg in enumerate(sys.argv):
            if 'nucleus_' in arg:
                try:
                    nucleus_prob = float(arg.split('nucleus_')[-1])
                except:
                    nucleus_prob = 0.8
    
    main(args, training_rate, nucleus_prob=nucleus_prob)
