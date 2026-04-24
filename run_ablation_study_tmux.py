#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation study for different embedding configurations:
1. Both nucleus and sig_axis (baseline) - 80% masking in nucleus
2. Only nucleus (no sig_axis) - 80% masking in nucleus
3. Only sig_axis (no nucleus) - 80% masking in nucleus
4. Neither (baseline BERT) - 80% masking in nucleus (for reference)
"""

import subprocess
import time

def create_and_run_ablation_tmux(session_name):
    """Create a tmux session and run ablation experiments sequentially"""
    
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
    
    # Define ablation configurations
    # Format: (config_name, include_nucleus, include_sig_axis, model_suffix)
    ablations = [
        ("Both (nucleus + sig_axis)", True, True, "both"),
        ("Nucleus only", True, False, "nucleus_only"),
        ("Sig_axis only", False, True, "sig_axis_only"),
        ("Neither (baseline BERT)", False, False, "baseline"),
    ]
    
    print("\n" + "="*80)
    print("SETTING UP ABLATION STUDY FOR EMBEDDING CONFIGURATIONS")
    print("="*80 + "\n")
    
    # Build the full command with all experiments in sequence
    full_command = "cd " + BASE_PATH + " && conda activate limu-bert-env && echo 'Starting ablation study experiments...' && "
    
    for idx, (config_name, use_nucleus, use_sig_axis, suffix) in enumerate(ablations):
        model_name = f"limu_v1_ablation_{suffix}"
        
        # Create command with ablation flags
        command = f"python pretrain_ablation.py {MODEL_VERSION} {DATASET} {VERSION} -s {model_name} --use_nucleus {use_nucleus} --use_sig_axis {use_sig_axis} --nucleus_prob {NUCLEUS_PROB}"
        
        full_command += f"echo '\\n{'='*80}' && "
        full_command += f"echo '[{idx + 1}/{len(ablations)}] {config_name}' && "
        full_command += f"echo '  Config: nucleus={use_nucleus}, sig_axis={use_sig_axis}' && "
        full_command += f"echo '  Model: {model_name}' && "
        full_command += f"echo '{'='*80}\\n' && "
        full_command += command
        
        # Add separator between experiments (except for the last one)
        if idx < len(ablations) - 1:
            full_command += " && echo '\\nWaiting 5 seconds before next experiment...' && sleep 5 && "
    
    # Add completion message
    full_command += f" && echo '\\n{'='*80}' && echo 'All ablation experiments completed!' && echo '{'='*80}'"
    
    # Send the entire command to tmux
    subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}', full_command, 'C-m'], 
                   check=False, capture_output=True)
    
    print(f"[✓] Submitted ablation study to tmux session: {session_name}")
    print(f"\nTo monitor progress, attach to the session:")
    print(f"    tmux attach-session -t {session_name}")
    print(f"\nTo detach from tmux session, press: Ctrl+b then d")
    print(f"\nTo kill the session when done:")
    print(f"    tmux kill-session -t {session_name}")
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY DETAILS")
    print(f"{'='*80}")
    print(f"Dataset: {DATASET}")
    print(f"Version: {VERSION}")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Nucleus Masking Prob: {NUCLEUS_PROB} (80% masking inside nucleus)")
    print(f"Base Path: {BASE_PATH}")
    
    print(f"\nAblation Configurations:")
    for idx, (config_name, use_nucleus, use_sig_axis, suffix) in enumerate(ablations):
        print(f"  {idx + 1}. {config_name}")
        print(f"     - Use nucleus embedding: {use_nucleus}")
        print(f"     - Use sig_axis embedding: {use_sig_axis}")
        print(f"     - Model: saved/pretrain_base_{DATASET}_{VERSION}/limu_v1_ablation_{suffix}.pt")
    
    print(f"\nEstimated time:")
    print(f"    ~{len(ablations) * 2}-3 hours (depending on dataset size and hardware)")

if __name__ == "__main__":
    create_and_run_ablation_tmux("pretrain_ablation_study")
