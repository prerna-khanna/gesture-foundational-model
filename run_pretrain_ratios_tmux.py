#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run pretraining with different nucleus masking ratios SEQUENTIALLY in tmux
Usage: python run_pretrain_ratios_tmux.py
"""

import subprocess
import time

def create_and_run_sequential_tmux(session_name):
    """Create a tmux session and run experiments sequentially"""
    
    # Kill existing session if it exists
    subprocess.run(['tmux', 'kill-session', '-t', session_name], 
                   check=False, capture_output=True)
    time.sleep(0.5)
    
    # Create new session
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name], 
                   check=False, capture_output=True)
    print(f"Created tmux session: {session_name}")
    time.sleep(0.5)
    
    RATIOS = [0.5, 0.7, 0.8, 0.9]
    DATASET = "hhar"
    VERSION = "20_120"
    MODEL_VERSION = "v1"
    BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
    
    print("\n" + "="*80)
    print("SETTING UP SEQUENTIAL PRETRAINING IN TMUX SESSION")
    print("="*80 + "\n")
    
    # Build the full command with all experiments in sequence
    full_command = "cd " + BASE_PATH + " && conda activate limu-bert-env && echo 'Starting sequential pretraining experiments...' && "
    
    for idx, ratio in enumerate(RATIOS):
        model_name = f"limu_v1_nucleus_{ratio}"
        command = f"python pretrain.py {MODEL_VERSION} {DATASET} {VERSION} -s {model_name}"
        
        full_command += f"echo '\\n{'='*80}' && "
        full_command += f"echo '[{idx + 1}/{len(RATIOS)}] Running pretraining with nucleus_prob={ratio}' && "
        full_command += f"echo '{'='*80}\\n' && "
        full_command += command
        
        # Add separator between experiments (except for the last one)
        if idx < len(RATIOS) - 1:
            full_command += " && echo '\\nWaiting 5 seconds before next experiment...' && sleep 5 && "
    
    # Add completion message
    full_command += f" && echo '\\n{'='*80}' && echo 'All pretraining experiments completed!' && echo '{'='*80}'"
    
    # Send the entire command to tmux
    subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}', full_command, 'C-m'], 
                   check=False, capture_output=True)
    
    print(f"[✓] Submitted sequential pretraining job to tmux session: {session_name}")
    print(f"\nTo monitor progress, attach to the session:")
    print(f"    tmux attach-session -t {session_name}")
    print(f"\nTo detach from tmux session, press: Ctrl+b then d")
    print(f"\nTo kill the session when done:")
    print(f"    tmux kill-session -t {session_name}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT DETAILS")
    print(f"{'='*80}")
    print(f"Ratios: {RATIOS}")
    print(f"Dataset: {DATASET}")
    print(f"Version: {VERSION}")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Base Path: {BASE_PATH}")
    
    print(f"\nModels will be saved as:")
    for ratio in RATIOS:
        print(f"    - saved/pretrain_base_{DATASET}_{VERSION}/limu_v1_nucleus_{ratio}.pt")
    
    print(f"\nEstimated time:")
    print(f"    ~{len(RATIOS) * 2}-3 hours (depending on dataset size and hardware)")

if __name__ == "__main__":
    create_and_run_sequential_tmux("pretrain_nucleus_ratios")
