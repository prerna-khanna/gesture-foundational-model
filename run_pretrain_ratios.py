#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run pretraining with different nucleus masking ratios SEQUENTIALLY
Usage: python run_pretrain_ratios.py
"""

import subprocess
import time
import sys

def run_pretraining_experiments():
    """Run pretraining experiments with different nucleus ratios SEQUENTIALLY"""
    
    RATIOS = [0.5, 0.7, 0.8, 0.9]
    DATASET = "hhar"
    VERSION = "20_120"
    MODEL_VERSION = "v1"
    BASE_PATH = "/home/prerna/LIMU-BERT-blind-users"
    
    print("\n" + "="*80)
    print("STARTING SEQUENTIAL PRETRAINING WITH DIFFERENT NUCLEUS MASKING RATIOS")
    print("="*80 + "\n")
    
    successful_runs = []
    failed_runs = []
    
    for idx, ratio in enumerate(RATIOS):
        model_name = f"limu_v1_nucleus_{ratio}"
        
        # Construct the command
        command = f"cd {BASE_PATH} && conda activate limu-bert-env && python pretrain.py {MODEL_VERSION} {DATASET} {VERSION} -s {model_name}"
        
        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{len(RATIOS)}] Starting pretraining with nucleus_prob={ratio}")
        print(f"{'='*80}")
        print(f"Model name: {model_name}")
        print(f"Command: {command}\n")
        
        # Run the command synchronously
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=BASE_PATH,
                check=False
            )
            
            if result.returncode == 0:
                print(f"\n✓ Successfully completed pretraining for nucleus_prob={ratio}")
                successful_runs.append(ratio)
            else:
                print(f"\n✗ Failed pretraining for nucleus_prob={ratio} (exit code: {result.returncode})")
                failed_runs.append(ratio)
        except Exception as e:
            print(f"\n✗ Error running pretraining for nucleus_prob={ratio}: {e}")
            failed_runs.append(ratio)
        
        # Add delay between runs (5 seconds) to let GPU cool down if needed
        if idx < len(RATIOS) - 1:
            print(f"\nWaiting 5 seconds before next experiment...")
            time.sleep(5)
    
    # Print summary
    print("\n\n" + "="*80)
    print("PRETRAINING SUMMARY")
    print("="*80)
    print(f"\nSuccessful runs ({len(successful_runs)}):")
    for ratio in successful_runs:
        print(f"    ✓ nucleus_prob={ratio}")
    
    if failed_runs:
        print(f"\nFailed runs ({len(failed_runs)}):")
        for ratio in failed_runs:
            print(f"    ✗ nucleus_prob={ratio}")
    
    print(f"\nExperiment details:")
    print(f"    Ratios: {RATIOS}")
    print(f"    Dataset: {DATASET}")
    print(f"    Version: {VERSION}")
    print(f"    Model Version: {MODEL_VERSION}")
    print(f"    Base Path: {BASE_PATH}")
    
    print(f"\nModels saved as:")
    for ratio in RATIOS:
        print(f"    - saved/pretrain_base_{DATASET}_{VERSION}/limu_v1_nucleus_{ratio}.pt")
    
    print(f"\nTo compare model performance:")
    print(f"    python benchmark.py -m limu_v1_nucleus_0.5 -m limu_v1_nucleus_0.7 -m limu_v1_nucleus_0.8 -m limu_v1_nucleus_0.9")

if __name__ == "__main__":
    run_pretraining_experiments()
