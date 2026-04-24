#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of ablation study with 10 epochs and embedding generation
"""

import subprocess
import time
import sys

def run_ablation_test_quick(session_name="ablation_test_quick"):
    """Run ablation study with 10 epochs for quick testing"""
    
    # Kill existing session if it exists
    subprocess.run(['tmux', 'kill-session', '-t', session_name], 
                   check=False, capture_output=True)
    time.sleep(0.5)
    
    # Create new session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name], 
                   check=False, capture_output=True)
    print(f"Created tmux session: {session_name}")
    time.sleep(0.5)
    
    DATASET = "hhar"
    VERSION = "20_120"
    MODEL_VERSION = "v1"
    BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
    NUCLEUS_PROB = 0.8
    EPOCHS = 10  # Quick test with 10 epochs
    
    # Define ablation configurations
    # Format: (config_name, include_nucleus, include_sig_axis, model_suffix)
    ablations = [
        ("Both (nucleus + sig_axis)", True, True, "both"),
        ("Nucleus only", True, False, "nucleus_only"),
        ("Sig_axis only", False, True, "sig_axis_only"),
        ("Neither (baseline BERT)", False, False, "baseline"),
    ]
    
    print("\n" + "="*80)
    print("QUICK TEST: ABLATION STUDY WITH 10 EPOCHS + EMBEDDING GENERATION")
    print("="*80 + "\n")
    print("Workflow: Train Model 1 -> Generate Embeddings 1 -> Train Model 2 -> ... and so on\n")
    
    # Build the full command with all experiments in sequence
    # Interleave training and embedding generation: train model -> generate embeddings -> next model
    full_command = "cd " + BASE_PATH + " && conda activate limu-bert-env && "
    
    for idx, (config_name, use_nucleus, use_sig_axis, suffix) in enumerate(ablations):
        model_save_name = f"limu_v1_ablation_{suffix}"
        epoch_arg = f"--epoch {EPOCHS}"
        
        if use_nucleus and use_sig_axis:
            flag_args = "--use_nucleus True --use_sig_axis True"
        elif use_nucleus and not use_sig_axis:
            flag_args = "--use_nucleus True --use_sig_axis False"
        elif not use_nucleus and use_sig_axis:
            flag_args = "--use_nucleus False --use_sig_axis True"
        else:  # neither
            flag_args = "--use_nucleus False --use_sig_axis False"
        
        # Train the model
        train_cmd = f"python pretrain_ablation.py {MODEL_VERSION} {DATASET} {VERSION} -s {model_save_name} {epoch_arg} {flag_args} --nucleus_prob {NUCLEUS_PROB}"
        
        # Generate embeddings for this model
        embedding_cmd = f"python embedding.py {MODEL_VERSION} {DATASET} {VERSION} -f {model_save_name}"
        
        # Chain them together: train -> wait -> generate embeddings
        full_command += f"{train_cmd} && sleep 5 && echo 'Generating embeddings for {suffix}...' && {embedding_cmd} && sleep 3 && "
    
    # Add final echo to mark completion
    full_command += "echo 'Ablation test complete! All embeddings generated.'"
    
    # Send command to tmux session
    subprocess.run(['tmux', 'send-keys', '-t', session_name, full_command, 'Enter'],
                   check=False)
    
    print(f"\n[✓] Submitted ablation test to tmux session: {session_name}\n")
    print(f"To monitor progress, attach to the session:")
    print(f"    tmux attach-session -t {session_name}\n")
    print(f"To detach from tmux session, press: Ctrl+b then d\n")
    
    print("="*80)
    print("QUICK TEST DETAILS")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Version: {VERSION}")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Nucleus Masking Prob: {NUCLEUS_PROB} (80% masking inside nucleus)")
    print(f"Epochs: {EPOCHS} (for quick testing)")
    print(f"Base Path: {BASE_PATH}\n")
    
    print("Ablation Configurations:")
    for config_name, use_nucleus, use_sig_axis, suffix in ablations:
        print(f"  - {config_name}")
        print(f"    Model: saved/pretrain_base_{DATASET}_{VERSION}/limu_v1_ablation_{suffix}.pt")
        print(f"    Embeddings: embeddings/limu_v1_ablation_{suffix}_embed.npy\n")
    
    print("="*80)
    print(f"Estimated time: ~30-45 minutes (10 epochs × 4 configurations + embeddings)")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_ablation_test_quick()
