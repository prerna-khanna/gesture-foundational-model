#!/usr/bin/env python3
"""
Baseline SVM Classifier for Turiya Dataset

This script trains a Support Vector Machine (SVM) classifier on the Turiya dataset
and evaluates it on the test set.

Usage:
    python svm_baseline.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


def load_dataset(dataset_name='Alexandra', merge=20, label_rate=0.08, stratified_split=True):
    """
    Load dataset and apply same preprocessing as classifier_with_contrastive.
    
    Args:
        dataset_name: Name of user dataset (e.g., 'Turiya', 'blind_user', etc.)
        merge: Merge factor for windowing (from classifier.json, default 20)
        label_rate: Fraction of training data to label (default 0.08)
        stratified_split: Use stratified split to ensure class distribution
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    data_path = f'dataset/{dataset_name}/data_20_120.npy'
    label_path = f'dataset/{dataset_name}/label_20_120.npy'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    data = np.load(data_path)
    labels = np.load(label_path)
    
    print(f"Loaded {dataset_name} dataset: {data.shape}, {labels.shape}")
    
    # Get original gesture IDs (one per sample in first dimension)
    original_gesture_ids = labels[:, 0, 0]
    
    if stratified_split:
        # Step 1: Stratified 80/20 split for train/test
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(original_gesture_ids))
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=0.1, 
            stratify=original_gesture_ids,
            random_state=42
        )
        
        # Step 2: Stratified split of training data for train/val
        train_gesture_ids = original_gesture_ids[train_indices]
        train_indices_2, val_indices_2 = train_test_split(
            train_indices,
            test_size=0.111,  # 10/90 ≈ 0.111
            stratify=train_gesture_ids,
            random_state=42
        )
        
        data_train = data[train_indices_2]
        data_val = data[val_indices_2]
        data_test = data[test_indices]
        
        label_train = labels[train_indices_2, :, 0]
        label_val = labels[val_indices_2, :, 0]
        label_test = labels[test_indices, :, 0]
    else:
        # Original sequential split
        total = data.shape[0]
        train_idx = int(total * 0.8)
        val_idx = int(total * 0.1)
        
        data_train = data[:train_idx]
        data_val = data[train_idx:train_idx+val_idx]
        data_test = data[train_idx+val_idx:]
        
        label_train = labels[:train_idx, :, 0]
        label_val = labels[train_idx:train_idx+val_idx, :, 0]
        label_test = labels[train_idx+val_idx:, :, 0]
    
    print(f"After stratified 80/10/10 split: train={data_train.shape[0]}, val={data_val.shape[0]}, test={data_test.shape[0]}")
    print(f"  Train classes: {sorted(np.unique(label_train))}")
    print(f"  Val classes: {sorted(np.unique(label_val))}")
    print(f"  Test classes: {sorted(np.unique(label_test))}")
    
    # Step 2: Reshape with merge=20 (windowing)
    def reshape_data(data, merge):
        # (N, 120, features) -> (N*120//merge, merge, features)
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])
    
    def reshape_label(labels, merge):
        # (N, 120) -> (N*120//merge, merge)
        return labels.reshape(labels.shape[0] * labels.shape[1] // merge, merge)
    
    X_train = reshape_data(data_train, merge)
    X_val = reshape_data(data_val, merge)
    X_test = reshape_data(data_test, merge)
    
    y_train = reshape_label(label_train, merge)
    y_val = reshape_label(label_val, merge)
    y_test = reshape_label(label_test, merge)
    
    # Use first label in each window (all should be same class)
    y_train = y_train[:, 0]
    y_val = y_val[:, 0]
    y_test = y_test[:, 0]
    
    print(f"After windowing (merge={merge}): train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    # Step 3: Flatten each window to a feature vector
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Feature vectors: train={X_train_flat.shape}, val={X_val_flat.shape}, test={X_test_flat.shape}")
    print(f"Unique classes in training: {sorted(np.unique(y_train))}")
    print(f"Unique classes in test: {sorted(np.unique(y_test))}")
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test


def train_svm(X_train, y_train, X_val, y_val, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train SVM classifier with standardization.
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        kernel: SVM kernel ('linear', 'rbf', 'poly')
        C: Regularization parameter
        gamma: Kernel coefficient
    
    Returns:
        svm_model, scaler
    """
    print(f"\nTraining SVM with kernel='{kernel}', C={C}, gamma={gamma}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train SVM
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, verbose=1)
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    y_val_pred = svm_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1 (weighted): {val_f1:.4f}")
    
    return svm_model, scaler


def evaluate_svm(svm_model, scaler, X_test, y_test, gesture_ids=None):
    """
    Evaluate SVM on test set.
    
    Args:
        svm_model: Trained SVM model
        scaler: StandardScaler used for training
        X_test, y_test: Test data and labels
        gesture_ids: List of gesture IDs for labels
    
    Returns:
        Dictionary with metrics
    """
    print(f"\nEvaluating on test set...")
    
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = svm_model.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(f"{'='*70}")
    
    # Classification report - use only the classes that appear in the test set
    test_classes = sorted(np.unique(y_test))
    
    print(f"\nClassification Report (test classes only: {test_classes}):")
    print(classification_report(y_test, y_test_pred, labels=test_classes, target_names=[str(int(g)) for g in test_classes]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=test_classes)
    
    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'predictions': y_test_pred,
        'confusion_matrix': cm,
        'gesture_ids': test_classes
    }


def plot_confusion_matrix(cm, gesture_ids, save_path='results/svm_confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    """
    os.makedirs('results', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(int(g)) for g in gesture_ids])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title('SVM Classifier - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.close()


def main():
    print("="*70)
    print("SVM Baseline Classifier")
    print("="*70)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        dataset_name='Alexandra',
        merge=20,
        label_rate=0.08
    )
    
    gesture_ids = sorted(np.unique(y_test))
    num_classes = len(gesture_ids)
    feature_dim = X_train.shape[1]
    
    print(f"\nDataset Summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Train SVM
    svm_model, scaler = train_svm(
        X_train, y_train, X_val, y_val,
        kernel='rbf',
        C=1.0,
        gamma='scale'
    )
    
    # Evaluate
    results = evaluate_svm(svm_model, scaler, X_test, y_test, gesture_ids)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], gesture_ids)
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'svm_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SVM Baseline Classifier\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test F1 (weighted): {results['f1']:.4f}\n")
        f.write(f"\nDataset:\n")
        f.write(f"  Training samples: {X_train.shape[0]}\n")
        f.write(f"  Validation samples: {X_val.shape[0]}\n")
        f.write(f"  Test samples: {X_test.shape[0]}\n")
        f.write(f"  Number of classes: {num_classes}\n")
        f.write(f"  Feature dimension: {feature_dim}\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    print(f"\n{'='*70}")
    print(f"SVM Baseline Summary")
    print(f"{'='*70}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1 Score: {results['f1']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
