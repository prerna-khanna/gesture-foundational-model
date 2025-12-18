"""
Test script to verify that nucleus-aware masking is working correctly.
Compares mask positions with and without nucleus-aware masking.
"""

import numpy as np
import torch
from utils import LIBERTDataset4Pretrain, Preprocess4Mask, load_pretrain_data_config
from features import compute_energy, detect_nucleus
import argparse

def test_nucleus_masking(args):
    print("="*80)
    print("Testing Nucleus-Aware Masking")
    print("="*80)
    
    # Load data directly
    data_path = f'dataset/{args.data_name}/data_{args.data_version}.npy'
    labels_path = f'dataset/{args.data_name}/label_{args.data_version}.npy'
    
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    print(f"\nLoaded data shape: {data.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    # Sample a few instances
    n_samples = 5
    sample_indices = np.random.choice(len(data), n_samples, replace=False)
    
    print(f"\nAnalyzing {n_samples} random samples...")
    print("-"*80)
    
    nucleus_mask_overlap = []
    random_mask_overlap = []
    
    for idx in sample_indices:
        instance = data[idx]
        
        # Detect nucleus
        instance_tensor = torch.from_numpy(instance).unsqueeze(0)
        energy = compute_energy(instance_tensor)
        nucleus_points = detect_nucleus(energy)
        
        if not nucleus_points or len(nucleus_points[0]) != 2:
            print(f"Sample {idx}: No nucleus detected, skipping...")
            continue
            
        nucleus_start, nucleus_end = nucleus_points[0]
        nucleus_len = nucleus_end - nucleus_start
        seq_len = instance.shape[0]
        
        # Create a dummy mask config
        class MaskConfig:
            mask_ratio = 0.15
            mask_alpha = 6
            max_gram = 5
            mask_prob = 0.8
            replace_prob = 0.1
        
        mask_cfg = MaskConfig()
        n_pred = max(1, int(round(seq_len * mask_cfg.mask_ratio)))
        
        # Test nucleus-aware masking
        nucleus_aware_preprocessor = Preprocess4Mask(mask_cfg, nucleus_aware=True, nucleus_prob=0.8)
        mask_seq, masked_pos, seq = nucleus_aware_preprocessor(instance.copy(), nucleus_start, nucleus_end)
        
        # Count how many masked positions fall in nucleus
        nucleus_overlap = sum(1 for pos in masked_pos if nucleus_start <= pos < nucleus_end)
        nucleus_overlap_pct = (nucleus_overlap / len(masked_pos)) * 100
        nucleus_mask_overlap.append(nucleus_overlap_pct)
        
        # Test random masking (for comparison)
        random_preprocessor = Preprocess4Mask(mask_cfg, nucleus_aware=False)
        mask_seq_random, masked_pos_random, seq_random = random_preprocessor(instance.copy())
        
        # Count how many random masked positions fall in nucleus
        random_overlap = sum(1 for pos in masked_pos_random if nucleus_start <= pos < nucleus_end)
        random_overlap_pct = (random_overlap / len(masked_pos_random)) * 100
        random_mask_overlap.append(random_overlap_pct)
        
        print(f"\nSample {idx}:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Nucleus region: [{nucleus_start}, {nucleus_end}) = {nucleus_len} positions ({nucleus_len/seq_len*100:.1f}%)")
        print(f"  Number of masked positions: {len(masked_pos)}")
        print(f"  Nucleus-aware masking:")
        print(f"    - Positions in nucleus: {nucleus_overlap}/{len(masked_pos)} ({nucleus_overlap_pct:.1f}%)")
        print(f"  Random masking (baseline):")
        print(f"    - Positions in nucleus: {random_overlap}/{len(masked_pos_random)} ({random_overlap_pct:.1f}%)")
        print(f"  Improvement: {nucleus_overlap_pct - random_overlap_pct:+.1f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Nucleus-aware masking (nucleus overlap): {np.mean(nucleus_mask_overlap):.1f}% ± {np.std(nucleus_mask_overlap):.1f}%")
    print(f"Random masking (nucleus overlap):       {np.mean(random_mask_overlap):.1f}% ± {np.std(random_mask_overlap):.1f}%")
    print(f"Improvement:                            {np.mean(nucleus_mask_overlap) - np.mean(random_mask_overlap):+.1f}%")
    print("\nInterpretation:")
    print("  - Nucleus-aware masking should have HIGHER overlap with nucleus zones")
    print("  - Random masking overlap should be close to the nucleus size percentage (~15-30%)")
    print("  - With nucleus_prob=0.8, expect ~60-80% of masks in nucleus zones")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', type=str, help='Dataset name (e.g., hhar, blind_user)')
    parser.add_argument('data_version', type=str, help='Data version (e.g., 20_120)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    test_nucleus_masking(args)
