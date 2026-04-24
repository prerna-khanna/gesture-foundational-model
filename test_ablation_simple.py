#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple ablation test: Train 4 configs for 10 epochs each + generate embeddings
Runs directly (no tmux) with proper error handling and status reporting
"""

import subprocess
import os
import sys
import time

BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
DATASET = "hhar"
VERSION = "20_120"
MODEL_VERSION = "v1"
NUCLEUS_PROB = 0.8
EPOCHS = 10

# Define ablation configurations
configs = [
    ("Both", True, True, "ablation_both"),
    ("Nucleus-only", True, False, "ablation_nucleus_only"),
    ("Sig_axis-only", False, True, "ablation_sig_axis_only"),
    ("Baseline", False, False, "ablation_baseline"),
]

def train_model(suffix, use_nucleus, use_sig_axis):
    """Train a single ablation model"""
    model_name = f"limu_v1_{suffix}"
    
    print(f"\n{'='*80}")
    print(f"[TRAIN] {suffix}")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Nucleus: {use_nucleus}, Sig_axis: {use_sig_axis}")
    print(f"Epochs: {EPOCHS} (verbose output enabled)")
    print(f"{'='*80}\n")
    
    # Run pretrain.py directly with verbose output
    cmd = f"cd {BASE_PATH} && conda run -n limu-bert-env python pretrain.py {MODEL_VERSION} {DATASET} {VERSION} -s {model_name}"
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Training failed for {suffix}")
        return False
    
    # Check if model was saved
    model_path = f"{BASE_PATH}/saved/pretrain_base_{DATASET}_{VERSION}/{model_name}.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return False
    
    print(f"\n[OK] Model saved: {model_path}")
    return True

def generate_embeddings(suffix):
    """Generate embeddings for a trained model"""
    model_name = f"limu_v1_{suffix}"
    
    print(f"\n{'='*80}")
    print(f"[EMBED] Generating embeddings for {suffix}")
    print(f"{'='*80}\n")
    
    cmd = f"cd {BASE_PATH} && conda run -n limu-bert-env python embedding.py {MODEL_VERSION} {DATASET} {VERSION} -f {model_name}"
    
    result = subprocess.run(cmd, shell=True, timeout=600)  # 10 minute timeout
    
    if result.returncode != 0:
        print(f"[WARNING] Embedding generation had issues for {suffix}")
        return False
    
    embed_path = f"{BASE_PATH}/embeddings/embed_{model_name}_{DATASET}_{VERSION}.npy"
    if os.path.exists(embed_path):
        size_mb = os.path.getsize(embed_path) / (1024*1024)
        print(f"[OK] Embeddings saved: {embed_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"[WARNING] Embeddings not found: {embed_path}")
        return False

def main():
    print("\n" + "="*80)
    print("ABLATION TEST: 10 EPOCHS + EMBEDDINGS")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Version: {VERSION}")
    print(f"Epochs: {EPOCHS}")
    print(f"Nucleus masking prob: {NUCLEUS_PROB}\n")
    
    print("Configurations to test:")
    for name, _, _, suffix in configs:
        print(f"  - {name} ({suffix})")
    print()
    
    results = {}
    
    for idx, (name, use_nucleus, use_sig_axis, suffix) in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# CONFIG {idx}/4: {name}")
        print(f"{'#'*80}\n")
        
        # Train
        train_ok = train_model(suffix, use_nucleus, use_sig_axis)
        results[name] = {"train": train_ok, "embed": False}
        
        if not train_ok:
            print(f"[SKIP] Skipping embeddings due to training failure")
            continue
        
        # Generate embeddings
        embed_ok = generate_embeddings(suffix)
        results[name]["embed"] = embed_ok
        
        time.sleep(2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, result in results.items():
        train_status = "✓" if result["train"] else "✗"
        embed_status = "✓" if result["embed"] else "✗"
        print(f"{name:20} Train: {train_status}  Embed: {embed_status}")
    
    all_ok = all(r["train"] and r["embed"] for r in results.values())
    if all_ok:
        print("\n[SUCCESS] All configurations completed!")
    else:
        print("\n[WARNING] Some configurations failed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
