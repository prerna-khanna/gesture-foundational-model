#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from original pretrain.py for ablation study
# This version removes nucleus and significant axis detection from LIMU-BERT

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
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


# Create a modified version of LIMUBertModel4Pretrain that doesn't use nucleus or sig_axis
class LIMUBertModel4PretrainAblation(nn.Module):
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = models.Transformer(cfg) # encoder with modified forward
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        # Call transformer without nucleus_mask and sig_axis_mask
        h_masked = self.transformer(input_seqs)

        if self.output_embed:
            return h_masked

        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm


# Also modify the Transformer class to not use nucleus or sig_axis
class TransformerAblation(nn.Module):
    """ Transformer with Self-Attentive Blocks - Ablation version that doesn't use nucleus or sig_axis"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = models.Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = models.MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = models.LayerNorm(cfg)
        self.pwff = models.PositionWiseFeedForward(cfg)
        self.norm2 = models.LayerNorm(cfg)

    def forward(self, x):
        # Only use basic embedding without nucleus_mask or sig_axis_mask
        h = self.embed(x)

        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    # Use the ablation model instead of the original
    model = LIMUBertModel4PretrainAblation(model_cfg)

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

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)

if __name__ == "__main__":
    mode = "base"  # changed from "base" to "ablation"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)