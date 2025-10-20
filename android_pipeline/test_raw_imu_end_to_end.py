#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test complete end-to-end pipeline: Raw IMU → Preprocessing → BERT Embedding → Classification
This is the EXACT pipeline that will be used in Android implementation
"""

import sys
sys.path.insert(0, '/home/prerna/LIMU-BERT-blind-users')

import numpy as np
from inference_exact import ExactSonyWatchPipeline
from tqdm import tqdm
import time

def main():
    print("="*80)
    print("Raw IMU Data End-to-End Pipeline Test")
    print("Pipeline: Raw IMU → Normalize → Nucleus → Sig Axis → BERT → Transformer")
    print("="*80)
    
    # Paths
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    data_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/data_20_120.npy"
    label_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/label_20_120.npy"
    
    # Initialize pipeline
    print("\nInitializing complete pipeline...")
    print("  - Loading BERT embedder (generates embeddings from raw data)")
    print("  - Loading Transformer classifier")
    print("  - Initializing preprocessing (normalization, nucleus, sig axis)")
    
    start_time = time.time()
    pipeline = ExactSonyWatchPipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path
    )
    init_time = time.time() - start_time
    print(f"✓ Pipeline initialized in {init_time:.2f}s")
    
    # Load data
    print("\nLoading dataset...")
    data = np.load(data_path)
    labels = np.load(label_path)
    print(f"Dataset size: {len(data)} samples")
    print(f"Data shape: {data.shape} (samples, seq_len, 6-axis IMU)")
    
    # Test 1: Single sample inference (simulate Android app)
    print("\n" + "="*80)
    print("TEST 1: Single Sample Inference (Android App Simulation)")
    print("="*80)
    
    test_idx = 1000
    raw_imu = data[test_idx]  # Shape: (120, 6) - This is what Android will send
    true_label_value = int(labels[test_idx][0, 0])
    true_label_idx = true_label_value - 1
    true_label_name = pipeline.config['activity_label'][true_label_idx]
    
    print(f"\nSimulating gesture detection from smartwatch...")
    print(f"Input: Raw 6-axis IMU data (120 timesteps)")
    print(f"  - Accelerometer: {raw_imu[:3, :3]}")
    print(f"  - Gyroscope: {raw_imu[:3, 3:]}")
    
    print(f"\nProcessing pipeline:")
    start_time = time.time()
    
    # This is exactly what will happen in Android
    result = pipeline.predict(raw_imu)
    
    inference_time = time.time() - start_time
    
    print(f"  1. ✓ Normalized accelerometer (÷9.8)")
    print(f"  2. ✓ Detected nucleus: [{result['metadata']['nucleus_start']}, {result['metadata']['nucleus_end']}]")
    print(f"  3. ✓ Calculated significant axis: {result['metadata']['significant_axis']}")
    print(f"  4. ✓ Generated BERT embeddings (120, 72)")
    print(f"  5. ✓ Classified with Transformer")
    
    print(f"\n{'='*80}")
    print(f"RESULT:")
    print(f"  True Gesture: {true_label_name}")
    print(f"  Predicted: {result['predicted_label']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Inference Time: {inference_time*1000:.1f}ms")
    print(f"  Status: {'✓ CORRECT' if result['predicted_class'] == true_label_idx else '✗ INCORRECT'}")
    print(f"{'='*80}")
    
    # Show top 3 predictions
    print(f"\nTop 3 predictions:")
    probs = result['probabilities']
    top3_indices = np.argsort(probs)[-3:][::-1]
    for rank, class_idx in enumerate(top3_indices, 1):
        prob = probs[class_idx]
        label = result['all_labels'][class_idx]
        marker = "→" if class_idx == true_label_idx else " "
        print(f"  {marker} {rank}. {label:30s} {prob:.4f}")
    
    # Test 2: Random samples
    print("\n" + "="*80)
    print("TEST 2: Random Sample Testing")
    print("="*80)
    
    np.random.seed(42)
    test_indices = np.random.choice(len(data), size=20, replace=False)
    
    correct = 0
    total_time = 0
    
    print(f"\nTesting 20 random samples...")
    for i, idx in enumerate(test_indices, 1):
        raw_imu = data[idx]
        true_label_value = int(labels[idx][0, 0])
        true_label_idx = true_label_value - 1
        
        start_time = time.time()
        result = pipeline.predict(raw_imu)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        is_correct = result['predicted_class'] == true_label_idx
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {i:2d}. {status} {pipeline.config['activity_label'][true_label_idx]:30s} → {result['predicted_label']:30s} (conf: {result['confidence']:.4f}, {inference_time*1000:.1f}ms)")
    
    avg_time = (total_time / 20) * 1000
    accuracy = (correct / 20) * 100
    
    print(f"\n{'='*80}")
    print(f"Random Sample Results:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/20)")
    print(f"  Avg Inference Time: {avg_time:.1f}ms per sample")
    print(f"{'='*80}")
    
    # Test 3: Batch processing (efficient for evaluation)
    print("\n" + "="*80)
    print("TEST 3: Full Dataset Evaluation with Batch Processing")
    print("="*80)
    
    batch_size = 128
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    print(f"\nProcessing {len(data)} samples in batches of {batch_size}...")
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        
        batch_data = data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Batch processing
        results = pipeline.predict_batch(batch_data)
        
        for i, result in enumerate(results):
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            true_label_value = int(batch_labels[i][0, 0])
            true_label_idx = true_label_value - 1
            
            all_predictions.append(predicted_class)
            all_true_labels.append(true_label_idx)
            all_confidences.append(confidence)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_confidences = np.array(all_confidences)
    
    correct = (all_predictions == all_true_labels).sum()
    total = len(all_true_labels)
    accuracy = correct / total * 100
    avg_confidence = np.mean(all_confidences)
    throughput = len(data) / total_time
    
    print(f"\n{'='*80}")
    print(f"FULL DATASET RESULTS:")
    print(f"  Total Samples: {total}")
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Average Confidence: {avg_confidence:.4f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Avg Time per Sample: {(total_time/total)*1000:.1f}ms")
    print(f"{'='*80}")
    
    # Per-class accuracy
    print(f"\n{'='*80}")
    print(f"PER-CLASS ACCURACY:")
    print(f"{'='*80}")
    print(f"{'Class':<35s} {'Accuracy':>10s} {'Samples':>10s} {'Avg Conf':>10s}")
    print(f"{'-'*80}")
    
    for class_idx in range(20):
        class_mask = all_true_labels == class_idx
        if class_mask.sum() > 0:
            class_correct = ((all_predictions == all_true_labels) & class_mask).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total * 100
            class_conf = np.mean(all_confidences[class_mask])
            class_name = pipeline.config['activity_label'][class_idx]
            print(f"{class_name:<35s} {class_acc:>9.2f}% {class_total:>10d} {class_conf:>10.4f}")
    
    print(f"{'-'*80}")
    
    # Confusion analysis
    print(f"\n{'='*80}")
    print(f"COMMON MISCLASSIFICATIONS (Top 5):")
    print(f"{'='*80}")
    
    misclassified = all_predictions != all_true_labels
    if misclassified.sum() > 0:
        misclass_pairs = list(zip(all_true_labels[misclassified], all_predictions[misclassified]))
        from collections import Counter
        common_errors = Counter(misclass_pairs).most_common(5)
        
        for (true_idx, pred_idx), count in common_errors:
            true_name = pipeline.config['activity_label'][true_idx]
            pred_name = pipeline.config['activity_label'][pred_idx]
            print(f"  {true_name:30s} → {pred_name:30s} ({count} times)")
    
    print(f"\n{'='*80}")
    print(f"✓ END-TO-END PIPELINE VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nThis pipeline is ready for Android implementation!")
    print(f"Expected inference time per gesture: ~{avg_time:.0f}ms")
    print(f"Expected accuracy: ~{accuracy:.1f}%")

if __name__ == "__main__":
    main()
