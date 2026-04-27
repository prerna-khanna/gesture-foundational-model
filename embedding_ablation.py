# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : embedding_ablation.py
# @Description : generate embeddings using pretrained LIMU-BERT models with ablation support

"""
# Nucleus only
python embedding_ablation.py v1 hhar 20_120 -f limu_v1_no_nucleus --use_nucleus false --use_sig_axis true

# Sig_axis only
python embedding_ablation.py v1 hhar 20_120 -f limu_v1_no_sig_axis --use_nucleus true --use_sig_axis false

# Both (default)
python embedding_ablation.py v1 hhar 20_120 -f limu_v1 --use_nucleus true --use_sig_axis true

# Baseline (neither)
python embedding_ablation.py v1 hhar 20_120 -f limu_v1_baseline --use_nucleus false --use_sig_axis false
"""

import os
import argparse
import sys

import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from features import detect_nucleus, compute_energy, calculate_significant_axis

import train
from config import load_dataset_label_names
from models import LIMUBertModel4Pretrain
from plot import plot_reconstruct_sensor, plot_embedding
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset


def fetch_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)
    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, model, criterion, train_cfg

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


from features import detect_nucleus, compute_energy  # Import both nucleus detection and energy computation

def generate_embedding_or_output(args, save=False, output_embed=True, use_nucleus=True, use_sig_axis=True):
    """
    Generate embeddings with ablation support for nucleus and sig_axis embeddings.
    
    Args:
        args: Command line arguments
        save: Whether to save embeddings to file
        output_embed: Whether to output embeddings
        use_nucleus: Whether to use nucleus mask embeddings
        use_sig_axis: Whether to use significant axis mask embeddings
    """
    data, labels, data_loader, model, criterion, train_cfg = fetch_setup(args, output_embed)

    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        
        device = next(model.parameters()).device
        seqs, label = batch
        seqs = seqs.to(device)

        # Prepare masks based on ablation settings
        nucleus_mask = None
        sig_axis_mask = None
        
        if use_nucleus:
            # Compute the energy for each sequence in the batch
            energy = compute_energy(seqs)
            # Detect nucleus points for each sequence in the batch
            batch_nucleus_points = detect_nucleus(energy)
            # Generate the batched nucleus mask
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
        
        if use_sig_axis:
            # Calculate significant axis mask
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).long()  # Shape: (batch_size, seq_len)

        # Pass sequences and masks to the model based on ablation configuration
        embed = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        return embed, label

    output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

    if save:
        # Save with standard format (compatible with classifier)
        save_name = 'embed_' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        np.save(os.path.join('embed', save_name + '.npy'), output)
        
        # Print ablation configuration used
        ablation_config = ""
        if use_nucleus and use_sig_axis:
            ablation_config = "both nucleus and sig_axis embeddings"
        elif use_nucleus:
            ablation_config = "nucleus embedding only"
        elif use_sig_axis:
            ablation_config = "sig_axis embedding only"
        else:
            ablation_config = "neither nucleus nor sig_axis embeddings (baseline)"
        
        print(f"\n{'='*80}")
        print(f"Embeddings saved to: embed/{save_name}.npy")
        print(f"Generated with: {ablation_config}")
        print(f"{'='*80}\n")

    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version, ablation_suffix=""):
    """
    Load embeddings with optional ablation suffix.
    
    Args:
        model_file: Model file name
        dataset: Dataset name
        dataset_version: Dataset version
        ablation_suffix: Ablation suffix (e.g., "_nucleus_only", "_sig_axis_only", "_baseline", "")
    """
    embed_name = 'embed_' + model_file + ablation_suffix + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    embed = np.load(os.path.join('embed', embed_name + '.npy')).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels


if __name__ == "__main__":
    save = True
    mode = "base"
    
    # Parse ablation parameters BEFORE handle_argv
    use_nucleus = True
    use_sig_axis = True
    
    # Check command line arguments for ablation settings
    ablation_parser = argparse.ArgumentParser(add_help=False)
    ablation_parser.add_argument('--use_nucleus', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
    ablation_parser.add_argument('--use_sig_axis', type=lambda x: x.lower() in ('true', '1', 't', 'yes'), default=True)
    
    try:
        ablation_args, remaining_args = ablation_parser.parse_known_args()
        use_nucleus = ablation_args.use_nucleus
        use_sig_axis = ablation_args.use_sig_axis
        # Replace sys.argv with remaining args for handle_argv
        sys.argv = [sys.argv[0]] + remaining_args
    except:
        pass
    
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    
    print(f"\n{'='*80}")
    print("EMBEDDING ABLATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Nucleus embedding: {use_nucleus}")
    print(f"Sig_axis embedding: {use_sig_axis}")
    print(f"{'='*80}\n")
    
    data, output, labels = generate_embedding_or_output(args=args, output_embed=True, save=save, 
                                                        use_nucleus=use_nucleus, use_sig_axis=use_sig_axis)

    label_index = 0  #put activity_label_index from data_config.json here
    label_names, label_num,_ = load_dataset_label_names(args.dataset_cfg, label_index)
    data_tsne, labels_tsne = plot_embedding(output, labels, label_index=label_index, reduce=1000, label_names=label_names)
