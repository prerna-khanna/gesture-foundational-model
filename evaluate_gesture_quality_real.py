#!/usr/bin/env python3
"""
Evaluate gesture quality using the REAL classifier_with_contrastive model.

This script trains the actual classifier_with_contrastive on the dataset
with REDUCED epochs (30-50) for quick assessment, then evaluates recognizability
of each gesture class based on:
- Precision: How often when we predict this gesture, we're correct
- Recall: How often we correctly identify this gesture
- Confusion: Which other gestures it gets confused with

This is NOT cheating because:
1. We use pre-computed LIMU-BERT embeddings (not trained on new data)
2. We train ONLY the classifier head, not the embeddings
3. We use the SAME architecture as the full pipeline
4. Training happens within this evaluation (not using pre-trained classifier)

Usage:
    python evaluate_gesture_quality_real.py --dataset Edery --version 20_120 \
        --embedding_model limu_v1 --quick_epochs 30
"""

import numpy as np
import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from embedding import load_embedding_label
from config import load_dataset_label_names
from utils import get_device, IMUDataset
from contrastive.augmenter import GestureAugmenter
from contrastive.models import ContrastiveTransformerClassifier
from contrastive.losses import ContrastiveCombinedLoss
import train as train_module


def extract_gesture_info(labels):
    """Extract gesture IDs from labels."""
    if len(labels.shape) == 3:
        gesture_ids = labels[:, 0, 0].astype(int)
    else:
        gesture_ids = labels.astype(int)
    
    return gesture_ids


def train_real_classifier_and_evaluate(embeddings, raw_data, labels, dataset, n_epochs=30, test_size=0.2, label_rate=0.5):
    """
    Train the actual ContrastiveTransformerClassifier with reduced epochs
    and evaluate each gesture's recognizability.
    
    Args:
        embeddings: Pre-computed LIMU-BERT embeddings (N, 120, 72)
        raw_data: Raw IMU data for augmentation (N, 120, 6)
        labels: Label array (N, 120, 2)
        dataset: Dataset name for loading config
        n_epochs: Number of epochs for quick training (default: 30)
        test_size: Test set fraction (default: 0.2)
        label_rate: Fraction of training data to use as labeled (default: 0.5, semi-supervised)
    """
    
    gesture_ids = extract_gesture_info(labels)
    unique_gestures = np.unique(gesture_ids)
    
    # Map gesture IDs to 0-indexed
    unique_sorted = np.sort(unique_gestures)
    gesture_to_idx = {int(g): i for i, g in enumerate(unique_sorted)}
    gesture_labels_indexed = np.array([gesture_to_idx[int(g)] for g in gesture_ids])
    
    # Use EXACT same split as classifier_with_contrastive.py
    # training_rate=0.7, vali_rate=0.1, test=0.2
    training_rate = 0.7
    vali_rate = 0.1
    test_rate = 1.0 - training_rate - vali_rate  # 0.2
    
    # Shuffle indices
    arr = np.arange(len(embeddings))
    np.random.seed(42)
    np.random.shuffle(arr)
    
    # Split indices
    train_num = int(len(embeddings) * training_rate)
    vali_num = int(len(embeddings) * vali_rate)
    
    train_idx = arr[:train_num]
    val_idx = arr[train_num:train_num + vali_num]
    test_idx = arr[train_num + vali_num:]
    
    # Prepare data for training
    raw_train = raw_data[train_idx]
    raw_test = raw_data[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    
    # Extract gesture IDs from labels (labels are (N, 120, 2) with [gesture_id, user_id])
    # Convert to just gesture IDs for loss computation
    if len(labels_train.shape) == 3:
        # Take first gesture ID from each sample (should be same for all timesteps)
        gesture_labels_train_orig = labels_train[:, 0, 0].astype(np.int64)
        gesture_labels_test_orig = labels_test[:, 0, 0].astype(np.int64)
    else:
        gesture_labels_train_orig = labels_train.astype(np.int64)
        gesture_labels_test_orig = labels_test.astype(np.int64)
    
    # Convert gesture IDs (1-15) to 0-indexed (0-14) for PyTorch
    gesture_labels_train = gesture_labels_train_orig - 1
    gesture_labels_test = gesture_labels_test_orig - 1
    
    # Apply label_rate for semi-supervised learning (0.5 = 50% labeled)
    # Remaining 50% are unlabeled
    labeled_count = int(len(raw_train) * label_rate)
    labeled_idx = np.random.choice(len(raw_train), labeled_count, replace=False)
    
    # Create masks for labeled/unlabeled
    labeled_mask = np.zeros(len(raw_train), dtype=bool)
    labeled_mask[labeled_idx] = True
    
    raw_train_labeled = raw_train[labeled_idx]
    gesture_labels_train_labeled = gesture_labels_train[labeled_idx]
    raw_train_unlabeled = raw_train[~labeled_mask]
    class GestureDataset:
        def __init__(self, raw_data, gesture_labels, labeled_mask=None, augment=False):
            self.raw_data = raw_data
            self.gesture_labels = gesture_labels
            self.labeled_mask = labeled_mask if labeled_mask is not None else np.ones(len(raw_data), dtype=bool)
            self.augment = augment
            self.augmenter = GestureAugmenter() if augment else None
        
        def __len__(self):
            return len(self.raw_data)
        
        def __getitem__(self, idx):
            # Normalize idx to plain Python int to handle numpy scalars/ndarrays
            try:
                idx = int(idx)
            except Exception:
                # fallback: try to extract scalar from numpy types
                idx = int(np.asarray(idx).item())
            
            data = self.raw_data[idx].copy()
            if self.augment:
                data = self.augmenter.augment(data)
            label = torch.tensor(int(self.gesture_labels[idx]), dtype=torch.long)
            is_labeled = torch.tensor(bool(self.labeled_mask[idx]), dtype=torch.bool)
            return torch.from_numpy(data).float(), label, is_labeled
    
    # Apply label_rate for semi-supervised learning (split training data into labeled/unlabeled)
    labeled_count = int(len(raw_train) * label_rate)
    labeled_indices = np.random.choice(len(raw_train), labeled_count, replace=False)
    labeled_mask = np.zeros(len(raw_train), dtype=bool)
    labeled_mask[labeled_indices] = True
    
    # Create datasets with proper gesture labels (0-indexed) and labeled/unlabeled mask
    train_dataset = GestureDataset(raw_train, gesture_labels_train, labeled_mask=labeled_mask, augment=True)
    test_dataset = GestureDataset(raw_test, gesture_labels_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Setup device and model
    device = get_device(gpu=0) if torch.cuda.is_available() else torch.device('cpu')
    
    # Load config for descriptions
    try:
        dataset_cfg = None  # Use default
        label_names, label_num, descriptions = load_dataset_label_names(
            dataset_cfg, label_index=0, dataset=dataset
        )
    except:
        label_num = len(unique_gestures)
        label_names = [f"Gesture {i+1}" for i in range(label_num)]
        descriptions = [f"{name} gesture" for name in label_names]
    
    # Initialize model
    input_dim = raw_train.shape[-1]  # 6 for IMU
    hidden_dim = 128
    
    model = ContrastiveTransformerClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=label_num
    ).to(device)
    
    # Initialize loss and optimizer
    criterion = ContrastiveCombinedLoss(
        label_names=label_names,
        descriptions=descriptions,
        pooling='mean',
        device=device,
        hidden_dim=hidden_dim
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"  Training real ContrastiveTransformer: {label_num} classes, {input_dim} features")
    print(f"  Epochs: {n_epochs}, Batch size: 16")
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, batch_labels, is_labeled = batch
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            is_labeled = is_labeled.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (use is_labeled to separate labeled/unlabeled in loss)
            logits, features, projected = model(inputs, return_features=True)
            
            # Compute loss
            total_loss, loss_dict = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=batch_labels,
                epoch=epoch
            )
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, batch_labels, _ = batch  # Ignore is_labeled for test set
            inputs = inputs.to(device)
            
            logits, _, _ = model(inputs, return_features=True)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    y_pred_idx = np.array(all_preds)
    y_test_idx = np.array(all_labels)
    
    # Convert predictions back to original gesture IDs (1-indexed)
    y_pred_gestures = y_pred_idx + 1  # Convert from 0-indexed to 1-indexed
    y_test_gestures = gesture_labels_test_orig  # Already has original gesture IDs
    
    # Compute metrics
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test_gestures, y_pred_gestures, labels=unique_sorted
    )
    
    cm = confusion_matrix(y_test_gestures, y_pred_gestures, labels=unique_sorted)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    stats = {}
    for i, g_id in enumerate(unique_sorted):
        g_id = int(g_id)
        
        # Find top confusions
        row = cm_normalized[i]
        row_copy = row.copy()
        row_copy[i] = -1
        
        top_confusion_idx = np.argmax(row_copy)
        top_confusion_rate = row_copy[top_confusion_idx]
        top_confused_with = int(unique_sorted[top_confusion_idx]) if top_confusion_rate > 0 else None
        
        stats[g_id] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(fscore[i]),
            'support': int(support[i]),
            'correct': int(cm[i, i]),
            'total': int(cm[i].sum()),
            'top_confusion_gesture': top_confused_with,
            'top_confusion_rate': float(max(0, top_confusion_rate))
        }
    
    return stats, cm, cm_normalized, y_test_gestures, y_pred_gestures


def identify_quality_issues(stats, gesture_names, min_recall=0.7, max_confusion=0.3):
    """Identify gestures with quality issues."""
    quality_report = {}
    
    for g_id, metric in stats.items():
        issues = []
        
        if metric['recall'] < min_recall:
            issues.append(f"Low recall: {metric['recall']:.1%} (misses {100-metric['recall']*100:.0f}% of samples)")
        
        if metric['top_confusion_rate'] > max_confusion:
            confused_g = metric['top_confusion_gesture']
            confused_name = gesture_names.get(confused_g, f"Gesture {confused_g}")
            issues.append(f"Confused with '{confused_name}': {metric['top_confusion_rate']:.1%}")
        
        quality_report[g_id] = {
            'name': gesture_names.get(g_id, f"Gesture {g_id}"),
            'issues': issues,
            'quality': 'PROBLEMATIC' if issues else 'GOOD',
            'metrics': metric
        }
    
    return quality_report


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate gesture quality using REAL classifier_with_contrastive'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., Alexandra, Julius, Turiya)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='20_120',
        help='Dataset version (default: 20_120)'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='limu_v1',
        help='Embedding model name (default: limu_v1)'
    )
    parser.add_argument(
        '--quick_epochs',
        type=int,
        default=10,
        help='Number of epochs for quick training (default: 10)'
    )
    parser.add_argument(
        '--min_recall',
        type=float,
        default=0.70,
        help='Minimum acceptable recall (default: 0.70)'
    )
    parser.add_argument(
        '--max_confusion',
        type=float,
        default=0.30,
        help='Maximum acceptable confusion with one gesture (default: 0.30)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gesture_quality',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GESTURE QUALITY EVALUATION (REAL CLASSIFIER)")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Version: {args.version}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Quick Training Epochs: {args.quick_epochs}")
    print(f"Quality Thresholds:")
    print(f"  Min Recall: {args.min_recall:.1%}")
    print(f"  Max Confusion: {args.max_confusion:.1%}")
    
    # Load raw data
    print(f"\nLoading dataset...")
    try:
        # Load embeddings
        embeddings, labels = load_embedding_label(args.embedding_model, args.dataset, args.version)
        
        # Load raw IMU data for training
        raw_data_path = os.path.join('dataset', args.dataset, f'data_{args.version}.npy')
        if os.path.exists(raw_data_path):
            raw_data = np.load(raw_data_path).astype(np.float32)
        else:
            print(f"  Warning: Raw data not found at {raw_data_path}")
            print(f"  Using embeddings as proxy (less ideal)")
            raw_data = embeddings
        
        gesture_ids = extract_gesture_info(labels)
        unique_gestures = np.unique(gesture_ids)
        
        print(f"  ✓ Loaded {len(embeddings)} samples")
        print(f"  ✓ Raw data shape: {raw_data.shape}")
        print(f"  ✓ Embedding shape: {embeddings.shape}")
        print(f"  ✓ {len(unique_gestures)} gesture classes")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load gesture names
    try:
        dataset_cfg = None
        label_names, _, _ = load_dataset_label_names(dataset_cfg, label_index=0, dataset=args.dataset)
        # Map label names to the actual gesture IDs present in the dataset.
        # Use unique_sorted (actual gesture IDs) to avoid off-by-one or gaps.
        unique_sorted = np.sort(unique_gestures)
        if len(label_names) >= len(unique_sorted):
            gesture_names = {int(g): (label_names[i] if i < len(label_names) else f"Gesture {int(g)}")
                             for i, g in enumerate(unique_sorted)}
        else:
            # fallback: label_names shorter than IDs → name by ID
            gesture_names = {int(g): f"Gesture {int(g)}" for g in unique_sorted}
    except Exception:
        gesture_names = {int(g): f"Gesture {int(g)}" for g in unique_gestures}
    
    # Train and evaluate
    print(f"\nTraining classifier and evaluating gestures...")
    try:
        stats, cm, cm_normalized, y_test, y_pred = train_real_classifier_and_evaluate(
            embeddings, raw_data, labels, args.dataset, 
            n_epochs=args.quick_epochs, 
            test_size=0.2
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    overall_accuracy = np.mean(y_test == y_pred)
    print(f"  ✓ Overall test accuracy: {overall_accuracy:.1%}")
    
    # Identify quality issues
    quality_report = identify_quality_issues(
        stats, gesture_names,
        min_recall=args.min_recall,
        max_confusion=args.max_confusion
    )
    
    # Display results
    print("\n" + "="*80)
    print("PER-GESTURE QUALITY ANALYSIS")
    print("="*80)
    
    results_df = []
    for g_id in sorted(unique_gestures):
        g_id = int(g_id)
        report = quality_report[g_id]
        metric = stats[g_id]
        
        status = "❌ PROBLEMATIC" if report['quality'] == 'PROBLEMATIC' else "✓ GOOD"
        
        results_df.append({
            'ID': g_id,
            'Name': report['name'][:25],
            'Precision': f"{metric['precision']:.1%}",
            'Recall': f"{metric['recall']:.1%}",
            'F1': f"{metric['f1']:.1%}",
            'Correct': f"{metric['correct']}/{metric['total']}",
            'Status': status
        })
    
    df = pd.DataFrame(results_df)
    print("\n" + df.to_string(index=False))
    
    # Detailed issues
    problematic = [g for g, r in quality_report.items() if r['quality'] == 'PROBLEMATIC']
    if problematic:
        print("\n" + "="*80)
        print(f"PROBLEMATIC GESTURES ({len(problematic)} found)")
        print("="*80)
        
        for g_id in sorted(problematic):
            report = quality_report[g_id]
            print(f"\n❌ {g_id}: {report['name']}")
            print(f"   Metrics: P={stats[g_id]['precision']:.1%}, R={stats[g_id]['recall']:.1%}, F1={stats[g_id]['f1']:.1%}")
            for issue in report['issues']:
                print(f"   • {issue}")
    
    # Summary
    good_count = sum(1 for r in quality_report.values() if r['quality'] == 'GOOD')
    problem_count = sum(1 for r in quality_report.values() if r['quality'] == 'PROBLEMATIC')
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Good quality gestures: {good_count}/{len(unique_gestures)}")
    print(f"❌ Problematic gestures: {problem_count}/{len(unique_gestures)}")
    print(f"Overall dataset accuracy: {overall_accuracy:.1%}")
    
    if problem_count == 0:
        print("\n✅ All gestures passed quality checks!")
        print("This is a robust gesture set suitable for training.")
    else:
        print(f"\n⚠️  {problem_count} gesture(s) need attention:")
        for g_id in sorted(problematic):
            print(f"   • {quality_report[g_id]['name']}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_file = os.path.join(args.output_dir, f'{args.dataset}_quality_report_real.json')
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'version': args.version,
            'method': 'real_classifier',
            'quick_epochs': args.quick_epochs,
            'overall_accuracy': float(overall_accuracy),
            'per_gesture_stats': {str(k): v for k, v in stats.items()},
            'quality_report': {str(k): {
                'name': v['name'],
                'quality': v['quality'],
                'issues': v['issues']
            } for k, v in quality_report.items()},
            'problematic_gestures': [int(g) for g in problematic],
            'good_gestures': [int(g) for g in sorted(unique_gestures) if int(g) not in problematic]
        }, f, indent=2)
    
    csv_file = os.path.join(args.output_dir, f'{args.dataset}_quality_report_real.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\n" + "="*80)
    print(f"Results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  CSV:  {csv_file}")
    print("="*80)


if __name__ == '__main__':
    main()
