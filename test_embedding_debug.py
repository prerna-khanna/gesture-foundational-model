#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug version of embedding generation with step-by-step output
"""

import sys
import os
sys.path.insert(0, '/home/prerna/LIMU-BERT-blind-users')

from utils import load_pretrain_data_config, get_device, handle_argv, Preprocess4Normalization, IMUDataset
from models import LIMUBertModel4Pretrain
from torch.utils.data import DataLoader
import torch
from torch import nn
import train
from features import detect_nucleus, compute_energy, calculate_significant_axis

print("[1] Parsing arguments...")
mode = "base"
args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
print(f"    Model version: {args.model_version}")
print(f"    Dataset: {args.dataset}")

print("\n[2] Loading config...")
data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
print(f"    Data shape: {data.shape}")
print(f"    Labels shape: {labels.shape}")

print("\n[3] Creating pipeline...")
pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
data_set = IMUDataset(data, labels, pipeline=pipeline)
print(f"    Dataset size: {len(data_set)}")

print("\n[4] Creating data loader...")
data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
print(f"    Batch size: {train_cfg.batch_size}")
print(f"    Number of batches: {len(data_loader)}")

print("\n[5] Creating model...")
model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
print(f"    Model created")

print("\n[6] Loading pretrained weights...")
model_file = f"saved/pretrain_base_{args.dataset}_{args.dataset_version}/limu_v1_ablation_both"
print(f"    Model file: {model_file}.pt")
state_dict = torch.load(model_file + '.pt', map_location='cpu')
print(f"    State dict keys: {len(state_dict)} items")
model.load_state_dict(state_dict)
print(f"    Weights loaded")

print("\n[7] Creating trainer...")
optimizer = None
device = get_device(args.gpu)
print(f"    Device: {device}")
trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)
print(f"    Trainer created")

print("\n[8] Processing batches...")
batch_count = 0
def func_forward(model, batch):
    global batch_count
    batch_count += 1
    
    device = next(model.parameters()).device
    seqs, label = batch
    seqs = seqs.to(device)
    
    print(f"    Batch {batch_count}: seqs shape = {seqs.shape}")
    
    # Compute energy
    energy = compute_energy(seqs)
    print(f"    -> Energy computed")
    
    # Detect nucleus
    batch_nucleus_points = detect_nucleus(energy)
    print(f"    -> Nucleus detected: {batch_nucleus_points[:2]}")
    
    # Embedding forward pass
    embed = model(seqs, nucleus_mask=None, sig_axis_mask=None)
    print(f"    -> Forward pass done: embed shape = {embed.shape}")
    
    return embed, label

print("\n[9] Running inference on data_loader...")
results = []
labels_list = []
start_time = __import__('time').time()

try:
    for i, batch in enumerate(data_loader):
        batch = [t.to(trainer.device) for t in batch]
        with torch.no_grad():
            result, label = func_forward(model, batch)
            results.append(result)
            labels_list.append(label)
        
        if (i + 1) % 5 == 0:
            elapsed = __import__('time').time() - start_time
            print(f"\n    Processed {i+1}/{len(data_loader)} batches in {elapsed:.1f}s")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print(f"\n[10] Done! Processed {batch_count} batches")
final_embeddings = torch.cat(results, 0).cpu().numpy()
print(f"     Final embeddings shape: {final_embeddings.shape}")
print(f"     Embedding generation SUCCESSFUL")
