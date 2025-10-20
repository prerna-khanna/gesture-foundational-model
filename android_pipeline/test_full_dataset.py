#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Sony Watch Inference Pipeline on Full Dataset
Evaluates accuracy on the entire dataset
"""

import numpy as np
import json
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sony_watch_inference_pipeline import SonyWatchInferencePipeline


def load_sony_watch_data():
    """Load Sony Watch dataset"""
    data_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/data_20_120.npy"
    label_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/label_20_120.npy"
    
    data = np.load(data_path)
    labels = np.load(label_path)
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return data, labels


def load_config():
    """Load Sony Watch configuration"""
    config_path = "/home/prerna/LIMU-BERT-blind-users/dataset/data_config.json"
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    return all_configs['sony_watch_20_120']


def evaluate_full_dataset(pipeline, data, labels, config, batch_size=100):
    """
    Evaluate pipeline on full dataset
    
    Args:
        pipeline: Inference pipeline
        data: Full dataset (N, 120, 6)
        labels: Full labels (N, 120, 2)
        config: Dataset configuration
        batch_size: Number of samples to evaluate at once (for progress display)
    
    Returns:
        Dictionary with evaluation results
    """
    activity_labels = config['activity_label']
    num_samples = len(data)
    
    print(f"\n{'='*80}")
    print(f"Evaluating on full dataset: {num_samples} samples")
    print(f"{'='*80}\n")
    
    # Initialize counters
    total_correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predictions = defaultdict(lambda: defaultdict(int))  # confusion matrix
    
    # Process all samples
    start_time = time.time()
    
    for idx in tqdm(range(num_samples), desc="Evaluating", ncols=100):
        # Get sample
        imu_data = data[idx]  # (120, 6)
        label_array = labels[idx][:, 0]  # Get activity labels
        true_label_idx = int(label_array[0])
        
        # Run inference
        try:
            result = pipeline.predict(imu_data)
            predicted_class = result['predicted_class']
            
            # Update counters
            class_total[true_label_idx] += 1
            class_predictions[true_label_idx][predicted_class] += 1
            
            if predicted_class == true_label_idx:
                total_correct += 1
                class_correct[true_label_idx] += 1
                
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    overall_accuracy = (total_correct / num_samples) * 100 if num_samples > 0 else 0
    
    # Per-class metrics
    class_metrics = {}
    for class_idx in range(len(activity_labels)):
        total = class_total.get(class_idx, 0)
        correct = class_correct.get(class_idx, 0)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        class_metrics[class_idx] = {
            'label': activity_labels[class_idx],
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
    
    # Calculate average per-class accuracy
    avg_class_accuracy = np.mean([m['accuracy'] for m in class_metrics.values()])
    
    results = {
        'num_samples': num_samples,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy,
        'avg_class_accuracy': avg_class_accuracy,
        'class_metrics': class_metrics,
        'confusion_matrix': dict(class_predictions),
        'elapsed_time': elapsed_time,
        'samples_per_second': num_samples / elapsed_time
    }
    
    return results


def print_results(results, config):
    """Print evaluation results"""
    activity_labels = config['activity_label']
    
    print(f"\n{'='*80}")
    print("FULL DATASET EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total samples: {results['num_samples']}")
    print(f"Correct predictions: {results['total_correct']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Average Per-Class Accuracy: {results['avg_class_accuracy']:.2f}%")
    print(f"\nEvaluation time: {results['elapsed_time']:.2f} seconds")
    print(f"Throughput: {results['samples_per_second']:.2f} samples/second")
    
    # Per-class results
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY")
    print(f"{'='*80}")
    print(f"{'Class':<5s} {'Label':<30s} {'Samples':<10s} {'Correct':<10s} {'Accuracy':<10s}")
    print(f"{'-'*80}")
    
    class_metrics = results['class_metrics']
    for class_idx in sorted(class_metrics.keys()):
        metrics = class_metrics[class_idx]
        label = metrics['label']
        total = metrics['total']
        correct = metrics['correct']
        accuracy = metrics['accuracy']
        
        print(f"{class_idx:<5d} {label:<30s} {total:<10d} {correct:<10d} {accuracy:>8.2f}%")
    
    # Find most confused classes
    print(f"\n{'='*80}")
    print("MOST COMMON CONFUSIONS (Top 10)")
    print(f"{'='*80}")
    print(f"{'True Label':<30s} → {'Predicted Label':<30s} {'Count':<10s}")
    print(f"{'-'*80}")
    
    confusion_matrix = results['confusion_matrix']
    confusions = []
    
    for true_idx in confusion_matrix:
        for pred_idx, count in confusion_matrix[true_idx].items():
            if true_idx != pred_idx:  # Only misclassifications
                confusions.append((true_idx, pred_idx, count))
    
    # Sort by count
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    for true_idx, pred_idx, count in confusions[:10]:
        true_label = activity_labels[true_idx]
        pred_label = activity_labels[pred_idx]
        print(f"{true_label:<30s} → {pred_label:<30s} {count:<10d}")
    
    # Find best and worst performing classes
    print(f"\n{'='*80}")
    print("BEST AND WORST PERFORMING CLASSES")
    print(f"{'='*80}")
    
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nBest 5 classes:")
    for class_idx, metrics in sorted_classes[:5]:
        if metrics['total'] > 0:
            print(f"  {metrics['label']:<30s} {metrics['accuracy']:>6.2f}% ({metrics['correct']}/{metrics['total']})")
    
    print("\nWorst 5 classes:")
    for class_idx, metrics in sorted_classes[-5:]:
        if metrics['total'] > 0:
            print(f"  {metrics['label']:<30s} {metrics['accuracy']:>6.2f}% ({metrics['correct']}/{metrics['total']})")


def save_results(results, output_path):
    """Save results to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    print("="*80)
    print("Sony Watch Inference Pipeline - Full Dataset Evaluation")
    print("="*80)
    
    # Model paths
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    
    print(f"\nEmbedder model: {embedder_path}")
    print(f"Classifier model: {classifier_path}")
    
    # Check if models exist
    if not os.path.exists(embedder_path):
        print(f"Error: Embedder model not found at {embedder_path}")
        return
    
    if not os.path.exists(classifier_path):
        print(f"Error: Classifier model not found at {classifier_path}")
        return
    
    # Initialize pipeline
    print("\nInitializing inference pipeline...")
    import torch
    pipeline = SonyWatchInferencePipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("Pipeline initialized successfully!")
    
    # Load data
    print("\nLoading Sony Watch dataset...")
    data, labels = load_sony_watch_data()
    config = load_config()
    
    print(f"\nDataset info:")
    print(f"  - Number of samples: {len(data)}")
    print(f"  - Sequence length: {data.shape[1]}")
    print(f"  - Number of axes: {data.shape[2]}")
    print(f"  - Number of classes: {config['activity_label_size']}")
    
    # Evaluate on full dataset
    results = evaluate_full_dataset(pipeline, data, labels, config)
    
    # Print results
    print_results(results, config)
    
    # Save results
    output_path = "/home/prerna/LIMU-BERT-blind-users/android_pipeline/full_dataset_results.json"
    save_results(results, output_path)
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
