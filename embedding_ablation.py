# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from original embedding.py for ablation study
# This version allows for selective ablation of nucleus and significant axis detection

import os
import argparse
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from features import detect_nucleus, compute_energy, calculate_significant_axis

import train
from config import load_dataset_label_names, load_model_config, load_dataset_stats
from plot import plot_reconstruct_sensor, plot_embedding
from utils import LIBERTDataset4Pretrain, get_device, \
    Preprocess4Normalization, IMUDataset


# Define the flexible transformer that can selectively use or ignore nucleus and sig_axis
class TransformerFlexible(nn.Module):
    """ Transformer with Self-Attentive Blocks - Allows for selective feature ablation"""
    def __init__(self, cfg, use_nucleus=True, use_sig_axis=True):
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
        from models import gelu, LayerNorm
        
        self.transformer = TransformerFlexible(cfg, use_nucleus, use_sig_axis)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
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


def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """
    Generate a binary mask for the nucleus for each sequence in the batch.

    Args:
        seq_len: Length of each sequence
        batch_nucleus_points: List of lists, where each inner list contains start and end points for the nucleus of a sequence

    Returns:
        A binary mask tensor with shape (batch_size, seq_len), where 1 indicates the nucleus region.
    """
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1  # Mark nucleus region with 1

    return nucleus_mask


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
    
    # Load training configuration
    from config import TrainConfig, MaskConfig
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    
    return data, labels, train_cfg, args.model_cfg, mask_cfg, args.dataset_cfg


def fetch_setup(args, output_embed, use_nucleus=True, use_sig_axis=True):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_data_and_config(args)
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    
    # Use the ablation model
    model = LIMUBertAblation(model_cfg, use_nucleus=use_nucleus, use_sig_axis=use_sig_axis, output_embed=output_embed)
    
    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, model, criterion, train_cfg


def generate_embedding_or_output(args, save=False, output_embed=True, use_nucleus=True, use_sig_axis=True):
    """Generate embeddings with specified ablation settings"""
    data, labels, data_loader, model, criterion, train_cfg = fetch_setup(
        args, output_embed, use_nucleus=use_nucleus, use_sig_axis=use_sig_axis
    )

    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        device = next(model.parameters()).device
        seqs, label = batch
        seqs = seqs.to(device)

        # Compute energy and detect nucleus/sig_axis only if needed
        nucleus_mask = None
        sig_axis_mask = None
        
        if use_nucleus:
            # Compute the energy for each sequence in the batch
            energy = compute_energy(seqs)
            # Detect nucleus points for each sequence in the batch
            batch_nucleus_points = detect_nucleus(energy)
            # Generate the batched nucleus mask
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(device)
        
        if use_sig_axis:
            # Calculate significant axis mask
            sig_axis = calculate_significant_axis(seqs)
            # Create mask as Long/Int type instead of Float
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).long()  # Change to long() instead of float()
            sig_axis_mask = sig_axis_mask.to(device)
            
        # Pass sequences with appropriate masks based on ablation settings
        embed = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        return embed, label
        device = next(model.parameters()).device
        seqs, label = batch
        seqs = seqs.to(device)

        # Compute energy and detect nucleus/sig_axis only if needed
        nucleus_mask = None
        sig_axis_mask = None
        
        if use_nucleus:
            # Compute the energy for each sequence in the batch
            energy = compute_energy(seqs)
            # Detect nucleus points for each sequence in the batch
            batch_nucleus_points = detect_nucleus(energy)
            # Generate the batched nucleus mask
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            nucleus_mask = nucleus_mask.to(device)
        
        if use_sig_axis:
            # Calculate significant axis mask
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()  # Shape: (batch_size, seq_len)
            sig_axis_mask = sig_axis_mask.to(device)
            
        # Pass sequences with appropriate masks based on ablation settings
        embed = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        return embed, label

    output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

    if save:
        # Determine the ablation suffix
        ablation_parts = []
        if not use_nucleus:
            ablation_parts.append("no_nucleus")
        if not use_sig_axis:
            ablation_parts.append("no_sig_axis")
            
        ablation_suffix = "_" + "_".join(ablation_parts) if ablation_parts else ""
        
        save_name = f'embed{ablation_suffix}_{args.model_file}_{args.dataset}_{args.dataset_version}'
        
        # Create embed directory if it doesn't exist
        os.makedirs('embed', exist_ok=True)
        
        save_path = os.path.join('embed', save_name + '.npy')
        np.save(save_path, output)
        print(f"Embeddings saved to: {save_path}")

    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version, ablation_suffix=""):
    """Load embeddings with specified ablation suffix"""
    prefix = 'embed' + ablation_suffix + '_'
    embed_name = prefix + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    
    embed_path = os.path.join('embed', embed_name + '.npy')
    print(f"Loading embeddings from: {embed_path}")
    
    embed = np.load(embed_path).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels

def get_device(gpu):
    if gpu == -1:
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}" if gpu is not None else "cuda:0")
        else:
            print("Warning: CUDA not available, using CPU instead")
            return torch.device("cpu")
        
if __name__ == "__main__":
    # Create custom argument parser to add ablation options
    parser = argparse.ArgumentParser(description='Generate embeddings with selective feature ablation')
    parser.add_argument('model_version', type=str, help='Model config, e.g. v1')
    parser.add_argument('dataset', type=str, choices=['hhar', 'motion', 'uci', 'shoaib'], help='Dataset name')
    parser.add_argument('dataset_version', type=str, help='Dataset version, e.g. 20_120')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, required=True, help='Pretrained model filename (without .pt)')
    parser.add_argument('-a', '--mask_cfg', type=str, default='config/mask.json', help='Mask strategy json file path')
    parser.add_argument('-t', '--train_cfg', type=str, default='pretrain.json', help='Training config json file path')
    parser.add_argument('-l', '--label_index', type=int, default=0, help='Label Index for visualization')
    parser.add_argument('--ablation', type=str, default='', 
                      help='Comma-separated list of features to ablate: "nucleus,sig_axis" or "nucleus" or "sig_axis"')
    parser.add_argument('--no_save', action='store_true', help='Do not save embeddings to file')
    parser.add_argument('--no_plot', action='store_true', help='Do not generate t-SNE plot')
    
    args = parser.parse_args()
    
    # Determine which features to use
    ablation_features = args.ablation.split(',') if args.ablation else []
    use_nucleus = 'nucleus' not in ablation_features
    use_sig_axis = 'sig_axis' not in ablation_features
    
    # Print ablation status
    print(f"Ablation Settings:")
    print(f"  Use Nucleus: {use_nucleus}")
    print(f"  Use Significant Axis: {use_sig_axis}")
    
    # Set up paths
    mode = "base"
    save_dir = f"saved/pretrain_{mode}_{args.dataset}_{args.dataset_version}"
    args.save_path = os.path.join(save_dir, args.model_file)
    args.pretrain_model = os.path.join(save_dir, args.model_file)
    
    # Set data paths
    args.data_path = f"dataset/{args.dataset}/data_{args.dataset_version}.npy"
    args.label_path = f"dataset/{args.dataset}/label_{args.dataset_version}.npy"
    
    # Load configurations
    args.model_cfg = load_model_config('pretrain', 'base', args.model_version)
    args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    
    print(f"Model configuration loaded: {args.model_version}")
    print(f"Using pretrained model from: {args.pretrain_model}")
    
    # Generate embeddings
    save = not args.no_save
    data, output, labels = generate_embedding_or_output(
        args=args, 
        output_embed=True, 
        save=save,
        use_nucleus=use_nucleus,
        use_sig_axis=use_sig_axis
    )

    # Generate t-SNE visualization
    if not args.no_plot:
        label_names, label_num, _ = load_dataset_label_names(args.dataset_cfg, args.label_index)
        data_tsne, labels_tsne = plot_embedding(output, labels, label_index=args.label_index, reduce=1000, label_names=label_names)