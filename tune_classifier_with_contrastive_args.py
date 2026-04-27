#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Hyperparameter tuning for classifier_with_contrastive with command line arguments.

Usage:
    python tune_classifier_with_contrastive.py <dataset> <dataset_version> <model_file> [--label-index 0] [--lr 1e-4,3e-4] [--lambda2 0.001,0.005]
    
Examples:
    python tune_classifier_with_contrastive.py umahand 20_120 limu_v1_umahand
    python tune_classifier_with_contrastive.py INF2018 20_120 limu_v1_inf2018 --label-index 0
    python tune_classifier_with_contrastive.py hhar 20_120 limu_v1 --lr 1e-4,5e-4 --lambda2 0.001,0.01
"""

import argparse
import sys
from types import SimpleNamespace

from config import create_io_config, load_model_config, load_dataset_stats
from embedding import load_embedding_label
from classifier_with_contrastive import run_grid_search


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for classifier_with_contrastive')
    
    # Positional arguments
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., umahand, INF2018, hhar)')
    parser.add_argument('dataset_version', type=str, help='Dataset version (e.g., 20_120)')
    parser.add_argument('model_file', type=str, help='Pretrained model file name')
    
    # Optional arguments
    parser.add_argument('--label-index', type=int, default=0, help='Label index (default: 0)')
    parser.add_argument('--model-version', type=str, default='v2', help='Model version (default: v2)')
    parser.add_argument('--training-rate', type=float, default=0.8, help='Training data rate (default: 0.8)')
    parser.add_argument('--label-rate', type=float, default=0.2, help='Labeled data rate (default: 0.2)')
    parser.add_argument('--balance', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help='Balance classes (default: True)')
    parser.add_argument('--gpu', type=str, default=None, help='GPU device (e.g., 0)')
    
    # Hyperparameter grids (comma-separated values)
    parser.add_argument('--lr', type=str, default='1e-4,3e-4', help='Learning rates to search (comma-separated)')
    parser.add_argument('--weight-decay', type=str, default='1e-4', help='Weight decay values (comma-separated)')
    parser.add_argument('--lambda2', type=str, default='0.001,0.005', help='Lambda2 values (comma-separated)')
    parser.add_argument('--grad-clip-norm', type=str, default='0.5', help='Gradient clip norm values (comma-separated)')
    
    return parser.parse_args()


def parse_param_list(param_str):
    """Parse comma-separated parameter string to list of floats or ints"""
    try:
        # Try to parse as floats in scientific notation
        return [float(x.strip()) for x in param_str.split(',')]
    except ValueError:
        # If that fails, try as regular numbers
        return [float(x.strip()) for x in param_str.split(',')]


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING FOR CLASSIFIER")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset Version: {args.dataset_version}")
    print(f"Model File: {args.model_file}")
    print(f"Label Index: {args.label_index}")
    print(f"{'='*80}\n")
    
    # Parse hyperparameter grids
    param_grid = {
        'lr': parse_param_list(args.lr),
        'weight_decay': parse_param_list(args.weight_decay),
        'lambda2': parse_param_list(args.lambda2),
        'grad_clip_norm': parse_param_list(args.grad_clip_norm),
    }
    
    print("Hyperparameter Grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()
    
    # Load configurations
    model_cfg = load_model_config(f'classifier_contrastive_gru', 'gru', args.model_version)
    if model_cfg is None:
        raise SystemExit("Unable to find corresponding model config!")

    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        raise SystemExit("Unable to find corresponding dataset config!")

    # Create args namespace
    tune_args = SimpleNamespace(
        model_version=args.model_version,
        dataset=args.dataset,
        dataset_version=args.dataset_version,
        gpu=args.gpu,
        model_file=args.model_file,
        train_cfg="./config/train.json",
        mask_cfg="./config/mask.json",
        label_index=args.label_index,
        save_model=f"tune_{args.dataset.lower()}_{args.dataset_version}",
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
    )
    tune_args = create_io_config(tune_args, args.dataset, args.dataset_version, 
                                 pretrain_model=tune_args.model_file, 
                                 target='classifier_contrastive_gru')

    # Load embeddings
    embedding, labels = load_embedding_label(tune_args.model_file, tune_args.dataset, tune_args.dataset_version)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Labels shape: {labels.shape}\n")
    
    # Run grid search
    best_params, best_acc, best_f1, _ = run_grid_search(
        tune_args,
        embedding,
        labels,
        args.label_index,
        args.training_rate,
        args.label_rate,
        balance=args.balance,
        method='gru',
        param_grid=param_grid,
    )

    print("\n" + "="*80)
    print("BEST TUNING RESULT")
    print("="*80)
    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1: {best_f1:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
