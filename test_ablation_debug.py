#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple debug script to test ablation training and embedding generation
No tmux - runs directly in terminal for easy debugging
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*80}")
    print(f"[RUN] {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] {description} completed")
        return True

def check_model_exists(model_path):
    """Check if model file exists"""
    full_path = model_path + '.pt'
    exists = os.path.exists(full_path)
    print(f"Checking model: {full_path}")
    print(f"Model exists: {exists}")
    return exists

def main():
    BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
    DATASET = "hhar"
    VERSION = "20_120"
    MODEL_VERSION = "v1"
    
    print("\n" + "="*80)
    print("ABLATION DEBUG - SIMPLE TEST (Direct pretrain.py + embedding.py)")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Version: {VERSION}\n")
    
    # Step 1: Train a simple model using pretrain.py with nucleus masking
    model_save_name = "limu_v1_test_nucleus_0.8"
    
    train_cmd = (
        f"cd {BASE_PATH} && "
        f"conda run -n limu-bert-env python pretrain.py {MODEL_VERSION} {DATASET} {VERSION} "
        f"-s {model_save_name}"
    )
    
    success = run_command(train_cmd, f"Train model with default settings")
    
    if not success:
        print("\n[FATAL] Training failed, skipping embedding generation")
        return
    
    # Step 2: Check if model exists
    model_path = f"{BASE_PATH}/saved/pretrain_base_{DATASET}_{VERSION}/{model_save_name}"
    if not check_model_exists(model_path):
        print("\n[ERROR] Model file not found after training!")
        print(f"Expected: {model_path}.pt")
        print("\nLet's check what files were created:")
        os.system(f"ls -la {BASE_PATH}/saved/pretrain_base_{DATASET}_{VERSION}/ | tail -20")
        return
    
    # Step 3: Generate embeddings
    embedding_cmd = (
        f"cd {BASE_PATH} && "
        f"conda run -n limu-bert-env python embedding.py {MODEL_VERSION} {DATASET} {VERSION} -f {model_save_name}"
    )
    
    success = run_command(embedding_cmd, f"Generate embeddings")
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] Full test completed successfully!")
        print(f"Model: {model_path}.pt")
        print(f"Embeddings: {BASE_PATH}/embeddings/{model_save_name}_embed.npy")
        print("="*80)
    else:
        print("\n[ERROR] Embedding generation failed")

if __name__ == "__main__":
    main()
