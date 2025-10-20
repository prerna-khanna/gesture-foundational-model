"""
Test Sony Watch Inference Pipeline
======================        true_label_name = activity_labels[true_label_idx]
        
        # Prepare sample (NO batch dimension - predict() adds it)
        sample_data = imu_data  # shape: (120, 6)
        true_label = true_label_name
        
        print(f"{'='*80}")======
This script tests the end-to-end inference pipeline on real Sony Watch samples
and compares predictions with ground truth labels.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sony_watch_inference_pipeline import SonyWatchInferencePipeline
import json
import random

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

def test_random_samples(pipeline, data, labels, config, num_samples=20):
    """Test pipeline on random samples"""
    
    activity_labels = config['activity_label']
    descriptions = config['descriptions']
    
    print("\n" + "="*80)
    print(f"Testing {num_samples} random samples from Sony Watch dataset")
    print("="*80)
    
    # Get random indices
    indices = random.sample(range(len(data)), num_samples)
    
    correct = 0
    total = num_samples
    
    results = []
    
    for i, idx in enumerate(indices):
        # Get raw IMU data and true label
        imu_data = data[idx]  # shape: (120, 6)
        # Labels shape is (num_samples, 120, 2), get the first label in the sequence
        label_array = labels[idx][:, 0]  # Get activity labels for all timesteps
        true_label_idx = int(label_array[0])  # Get the first label
        true_label_name = activity_labels[true_label_idx]
        
        # Prepare sample (NO batch dimension - predict() handles that)
        sample_data = imu_data  # shape: (120, 6)
        true_label = true_label_name
        
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples} (Index: {idx})")
        print(f"{'='*80}")
        print(f"Data shape: {sample_data.shape}")
        print(f"True Label: {true_label} (class {true_label_idx})")
        
        # Run inference
        result = pipeline.predict(sample_data)
        
        predicted_class = result['predicted_class']
        predicted_label = activity_labels[predicted_class]
        confidence = result['confidence']
        
        print(f"\nPredicted Label: {predicted_label} (class {predicted_class})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Match: {'✓ CORRECT' if predicted_class == true_label_idx else '✗ INCORRECT'}")
        
        # Show top 5 predictions
        print("\nTop 5 predictions:")
        probabilities = result['probabilities']
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        for rank, class_idx in enumerate(top5_indices, 1):
            prob = probabilities[class_idx]
            label = activity_labels[class_idx]
            marker = "→" if class_idx == true_label_idx else " "
            print(f"  {marker} {rank}. {label:25s} {prob:.4f}")
        
        # Show metadata info
        metadata = result.get('metadata', {})
        if 'nucleus_start' in metadata:
            print(f"\nNucleus: [{metadata['nucleus_start']}, {metadata['nucleus_end']}]")
        if 'significant_axis' in metadata:
            sig_axis = metadata['significant_axis']
            print(f"Significant axis: {sig_axis.get('primary_axis', 'N/A')} (energy: {sig_axis.get('max_energy', 0):.2f})")
        
        # Check if correct
        if predicted_class == true_label_idx:
            correct += 1
        
        results.append({
            'index': idx,
            'true_label': true_label,
            'true_class': true_label_idx,
            'predicted_label': predicted_label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct': predicted_class == true_label_idx
        })
    
    # Print summary
    accuracy = correct / total * 100
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Show confusion patterns
    print("\nIncorrect predictions:")
    for r in results:
        if not r['correct']:
            print(f"  - True: {r['true_label']:25s} → Predicted: {r['predicted_label']:25s} (conf: {r['confidence']:.4f})")
    
    return results, accuracy

def test_per_class_accuracy(pipeline, data, labels, config, samples_per_class=5):
    """Test pipeline on samples from each class"""
    
    activity_labels = config['activity_label']
    num_classes = config['activity_label_size']
    
    print("\n" + "="*80)
    print(f"Testing {samples_per_class} samples per class ({num_classes} classes)")
    print("="*80)
    
    class_results = {}
    
    for class_idx in range(num_classes):
        class_label = activity_labels[class_idx]
        
        # Find samples of this class
        class_indices = np.where(labels[:, 0] == class_idx)[0]
        
        if len(class_indices) == 0:
            print(f"\nClass {class_idx} ({class_label}): No samples found")
            continue
        
        # Sample random indices from this class
        num_to_sample = min(samples_per_class, len(class_indices))
        sampled_indices = random.sample(list(class_indices), num_to_sample)
        
        print(f"\n{'-'*80}")
        print(f"Class {class_idx}: {class_label}")
        print(f"Available samples: {len(class_indices)}, Testing: {num_to_sample}")
        print(f"{'-'*80}")
        
        correct = 0
        confidences = []
        
        for idx in sampled_indices:
            sample_data = data[idx]
            result = pipeline.predict(sample_data)
            
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            confidences.append(confidence)
            
            if predicted_class == class_idx:
                correct += 1
                status = "✓"
            else:
                status = "✗"
                predicted_label = activity_labels[predicted_class]
                print(f"  {status} Sample {idx}: Predicted as '{predicted_label}' (conf: {confidence:.4f})")
        
        accuracy = correct / num_to_sample * 100
        avg_confidence = np.mean(confidences)
        
        class_results[class_label] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': num_to_sample,
            'avg_confidence': avg_confidence
        }
        
        print(f"Accuracy: {correct}/{num_to_sample} = {accuracy:.2f}%")
        print(f"Average confidence: {avg_confidence:.4f}")
    
    # Print class-wise summary
    print(f"\n{'='*80}")
    print("PER-CLASS ACCURACY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Class':<30s} {'Accuracy':<15s} {'Avg Confidence':<15s}")
    print(f"{'-'*80}")
    
    for class_label, results in class_results.items():
        print(f"{class_label:<30s} {results['accuracy']:>6.2f}% ({results['correct']}/{results['total']})  {results['avg_confidence']:>6.4f}")
    
    overall_accuracy = sum(r['correct'] for r in class_results.values()) / sum(r['total'] for r in class_results.values()) * 100
    print(f"{'-'*80}")
    print(f"{'Overall':<30s} {overall_accuracy:>6.2f}%")
    
    return class_results

def main():
    print("="*80)
    print("Sony Watch Inference Pipeline - End-to-End Test")
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
    print(f"  - Classes: {', '.join(config['activity_label'][:5])}...")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test 1: Random samples
    print("\n" + "="*80)
    print("TEST 1: Random Samples")
    print("="*80)
    random_results, random_accuracy = test_random_samples(pipeline, data, labels, config, num_samples=10)
    
    # Test 2: Per-class accuracy
    print("\n" + "="*80)
    print("TEST 2: Per-Class Accuracy")
    print("="*80)
    class_results = test_per_class_accuracy(pipeline, data, labels, config, samples_per_class=3)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Random sample accuracy: {random_accuracy:.2f}%")
    overall_accuracy = sum(r['correct'] for r in class_results.values()) / sum(r['total'] for r in class_results.values()) * 100
    print(f"Per-class average accuracy: {overall_accuracy:.2f}%")
    print("\nPipeline validation complete!")

if __name__ == "__main__":
    main()
