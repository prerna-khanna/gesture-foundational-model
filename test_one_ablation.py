#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test single ablation config: Train for 10 epochs + embeddings
Debug version
"""

import subprocess
import os

BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
DATASET = "hhar"
VERSION = "20_120"
MODEL_VERSION = "v1"

print("\n" + "="*80)
print("SINGLE ABLATION TEST")
print("="*80)
print(f"Training: limu_v1_ablation_both")
print(f"Dataset: {DATASET}, Version: {VERSION}")
print(f"="*80 + "\n")

# Step 1: Train with 10-epoch config
print("[1] Starting training (10 epochs)...")
train_cmd = f"cd {BASE_PATH} && conda run -n limu-bert-env python pretrain.py {MODEL_VERSION} {DATASET} {VERSION} -s limu_v1_ablation_both -t config/pretrain_10epochs.json"

result = subprocess.run(train_cmd, shell=True)

if result.returncode == 0:
    print("\n[OK] Training completed\n")
else:
    print(f"\n[ERROR] Training failed")
    exit(1)

# Check model exists
model_path = f"{BASE_PATH}/saved/pretrain_base_{DATASET}_{VERSION}/limu_v1_ablation_both.pt"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"[OK] Model exists: {size_mb:.1f} MB\n")
else:
    print(f"[ERROR] Model not found")
    exit(1)

# Step 2: Generate embeddings
print("[2] Generating embeddings...")
embed_cmd = f"cd {BASE_PATH} && conda run -n limu-bert-env python embedding.py {MODEL_VERSION} {DATASET} {VERSION} -f limu_v1_ablation_both"

result = subprocess.run(embed_cmd, shell=True, timeout=600)

if result.returncode == 0:
    print("\n[OK] Embeddings completed\n")
else:
    print(f"\n[WARNING] Embeddings had issues")

print("="*80)
print("TEST COMPLETE")
print("="*80)
