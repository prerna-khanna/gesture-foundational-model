#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test exact pipeline on full Sony Watch dataset
"""

import sys
sys.path.insert(0, '/home/prerna/LIMU-BERT-blind-users')

import numpy as np
from inference_exact import ExactSonyWatchPipeline
from tqdm import tqdm

def main():
    print("="*80)
    print("Full Dataset Evaluation - Exact Pipeline")
    print("="*80)
    
    # Paths
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    data_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/data_20_120.npy"
    label_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/label_20_120.npy"
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = ExactSonyWatchPipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path
    )
    
    # Load data
    print("\nLoading dataset...")
    data = np.load(data_path)
    labels = np.load(label_path)
    print(f"Dataset size: {len(data)} samples")
    
    # Evaluate on full dataset using batches
    print("\nEvaluating on full dataset...")
    batch_size = 128
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    all_predictions = []
    all_true_labels = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Get predictions
        results = pipeline.predict_batch(batch_data)
        
        # Extract predictions and true labels
        for i, result in enumerate(results):
            predicted_class = result['predicted_class']
            true_label_value = int(batch_labels[i][0, 0])
            true_label_idx = true_label_value - 1
            
            all_predictions.append(predicted_class)
            all_true_labels.append(true_label_idx)
    
    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    correct = (all_predictions == all_true_labels).sum()
    total = len(all_true_labels)
    accuracy = correct / total * 100
    
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*80}")
    
    # Per-class accuracy
    print(f"\n{'='*80}")
    print(f"PER-CLASS ACCURACY")
    print(f"{'='*80}")
    
    for class_idx in range(20):
        class_mask = all_true_labels == class_idx
        if class_mask.sum() > 0:
            class_correct = ((all_predictions == all_true_labels) & class_mask).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total * 100
            class_name = pipeline.config['activity_label'][class_idx]
            print(f"{class_name:30s} {class_acc:6.2f}% ({class_correct}/{class_total})")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
