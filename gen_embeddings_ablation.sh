#!/bin/bash
# Generate embeddings for all 4 ablation models (skip visualization)

cd /home/prerna/LIMU-BERT-blind-users

echo "===== GENERATING EMBEDDINGS FOR ABLATION MODELS ====="
echo ""

DATASET="hhar"
VERSION="20_120"
MODEL_VERSION="v1"

# Create embed directory if it doesn't exist
mkdir -p embed

# Function to generate embeddings
generate_embed() {
    local suffix=$1
    echo "[EMBED] Generating embeddings for: limu_v1_ablation_${suffix}"
    
    # Run directly with timeout and capture output
    timeout 300 conda run -n limu-bert-env python -c "
import sys
sys.path.insert(0, '/home/prerna/LIMU-BERT-blind-users')

from utils import load_pretrain_data_config, get_device, handle_argv, Preprocess4Normalization, IMUDataset
from models import LIMUBertModel4Pretrain
from torch.utils.data import DataLoader
import torch
from torch import nn
import train
from features import detect_nucleus, compute_energy, calculate_significant_axis
import numpy as np
import os

mode = 'base'
sys.argv = ['embedding.py', '$MODEL_VERSION', '$DATASET', '$VERSION', '-f', 'limu_v1_ablation_${suffix}']

args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
data_set = IMUDataset(data, labels, pipeline=pipeline)
data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)

optimizer = None
device = get_device(args.gpu)
trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

def func_forward(model, batch):
    device = next(model.parameters()).device
    seqs, label = batch
    seqs = seqs.to(device)
    
    energy = compute_energy(seqs)
    batch_nucleus_points = detect_nucleus(energy)
    nucleus_mask = None
    
    sig_axis = calculate_significant_axis(seqs)
    sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).long()
    
    embed = model(seqs, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
    return embed, label

output = trainer.run(func_forward, None, data_loader, args.pretrain_model)

# Save embeddings
save_name = 'embed_limu_v1_ablation_${suffix}_${DATASET}_${VERSION}'
np.save(os.path.join('embed', save_name + '.npy'), output)
print(f'[OK] Embeddings saved: embed/{save_name}.npy')
print(f'[OK] Shape: {output.shape}')
" 2>&1 | tail -20
}

# Generate embeddings for all 4 models
generate_embed "both"
echo ""
generate_embed "nucleus_only"
echo ""
generate_embed "sig_axis_only"
echo ""
generate_embed "baseline"

echo ""
echo "===== EMBEDDING GENERATION COMPLETE ====="
ls -lh embed/embed_limu_v1_ablation*.npy 2>/dev/null
