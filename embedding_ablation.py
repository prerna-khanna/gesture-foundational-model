# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from original embedding.py for ablation study
# This version removes nucleus and significant axis detection from embedding generation

import os

import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader

import train
from config import load_dataset_label_names
from plot import plot_reconstruct_sensor, plot_embedding
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset

# Define the ablation model (matching the pretraining one)
class LIMUBertModel4PretrainAblation(nn.Module):
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        # Import here to avoid circular imports
        from models import Transformer, gelu, LayerNorm
        
        self.transformer = TransformerAblation(cfg) # encoder with modified forward
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
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


# Simplified Transformer without nucleus or sig_axis
class TransformerAblation(nn.Module):
    """ Transformer with Self-Attentive Blocks - Ablation version """
    def __init__(self, cfg):
        super().__init__()
        # Import here to avoid circular imports
        from models import Embeddings, MultiHeadedSelfAttention, LayerNorm, PositionWiseFeedForward
        
        self.embed = Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        # Only use basic embedding without nucleus_mask or sig_axis_mask
        h = self.embed(x)

        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


def fetch_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    
    # Use the ablation model
    model = LIMUBertModel4PretrainAblation(model_cfg, output_embed=output_embed)
    
    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, model, criterion, train_cfg


def generate_embedding_or_output(args, save=False, output_embed=True):
    data, labels, data_loader, model, criterion, train_cfg = fetch_setup(args, output_embed)

    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        device = next(model.parameters()).device
        seqs, label = batch
        seqs = seqs.to(device)

        # Simply pass the sequences without nucleus or sig_axis masks
        embed = model(seqs)
        return embed, label

    output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

    if save:
        save_name = 'embed_ablation_' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        np.save(os.path.join('embed', save_name + '.npy'), output)

    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version, is_ablation=True):
    prefix = 'embed_ablation_' if is_ablation else 'embed_'
    embed_name = prefix + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    embed = np.load(os.path.join('embed', embed_name + '.npy')).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels


if __name__ == "__main__":
    save = True
    mode = "base"  # changed from "base" to "ablation"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    data, output, labels = generate_embedding_or_output(args=args, output_embed=True, save=save)

    label_index = 0  # put activity_label_index from data_config.json here
    label_names, label_num, _ = load_dataset_label_names(args.dataset_cfg, label_index)
    data_tsne, labels_tsne = plot_embedding(output, labels, label_index=label_index, reduce=1000, label_names=label_names)