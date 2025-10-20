#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exact End-to-End Inference Pipeline for Sony Watch
Uses the EXACT same preprocessing and models as training
"""

import sys
import os
sys.path.insert(0, '/home/prerna/LIMU-BERT-blind-users')

import torch
import numpy as np
import json
from typing import Dict, List, Tuple
import torch.nn.functional as F

# Import exact training modules
from models import LIMUBertModel4Pretrain
from contrastive.models import ContrastiveGRUClassifier
from features import compute_energy, detect_nucleus, calculate_significant_axis
from embedding import load_embedding_label


class ExactSonyWatchPipeline:
    """
    Exact inference pipeline matching training code
    """
    
    def __init__(
        self,
        embedder_path: str,
        classifier_path: str,
        config_path: str = "/home/prerna/LIMU-BERT-blind-users/dataset/data_config.json",
        device: str = 'cuda'
    ):
        """
        Initialize pipeline with saved models
        
        Args:
            embedder_path: Path to pre-trained BERT embedder (.pt file)
            classifier_path: Path to trained classifier (.pt file)
            config_path: Path to dataset config
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
        self.config = all_configs['sony_watch_20_120']
        
        # Load embedder
        print("\nLoading BERT embedder...")
        self.embedder = self._load_embedder(embedder_path)
        print("✓ BERT embedder loaded")
        
        # Load classifier
        print("\nLoading classifier...")
        self.classifier = self._load_classifier(classifier_path)
        print("✓ Classifier loaded")
        
        print(f"\n✓ Pipeline ready to recognize {self.config['activity_label_size']} gestures")
    
    def _load_embedder(self, model_path: str):
        """Load pre-trained BERT embedder"""
        # Create model configuration (same as training)
        class Config:
            hidden = 72
            hidden_ff = 144
            n_layers = 4
            n_heads = 4
            seq_len = 120
            feature_num = 6
            emb_norm = True
        
        # Create model with output_embed=True to get embeddings
        config = Config()
        model = LIMUBertModel4Pretrain(config, output_embed=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_classifier(self, model_path: str):
        """Load trained classifier - exact same as training"""
        # Check what type of classifier was saved
        # Load checkpoint first to inspect
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine classifier type from state dict keys
        if 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict:
            # It's a Transformer classifier
            from contrastive.models import ContrastiveTransformerClassifier
            input_dim = 72
            hidden_dim = 128
            num_classes = self.config['activity_label_size']
            num_heads = 4
            num_layers = 2
            proj_dim = 128
            dropout = 0.4
            
            classifier = ContrastiveTransformerClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                num_layers=num_layers,
                proj_dim=proj_dim,
                dropout=dropout
            )
            print(f"Detected Transformer classifier")
        elif 'gru.weight_ih_l0' in state_dict:
            # It's a GRU classifier
            input_dim = 72
            hidden_dim = 128
            num_classes = self.config['activity_label_size']
            proj_dim = 128
            
            classifier = ContrastiveGRUClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                proj_dim=proj_dim
            )
            print(f"Detected GRU classifier")
        else:
            raise ValueError("Unknown classifier type in checkpoint")
        
        # Load weights
        classifier.load_state_dict(state_dict, strict=True)
        classifier.to(self.device)
        classifier.eval()
        
        return classifier
    
    def preprocess(self, raw_data: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess raw IMU data - EXACT same as training
        
        Args:
            raw_data: Raw IMU data of shape (seq_len, 6)
        
        Returns:
            Tuple of (preprocessed_tensor, metadata)
        """
        # Apply normalization (exact same as training)
        # Normalize accelerometer by dividing by 9.8
        preprocessed_data = raw_data.copy()
        preprocessed_data[:, :3] = preprocessed_data[:, :3] / 9.8
        
        # Convert to tensor and add batch dimension
        data_tensor = torch.from_numpy(preprocessed_data).float().unsqueeze(0)  # (1, seq_len, 6)
        data_tensor = data_tensor.to(self.device)
        
        # Compute energy (exact same as training)
        energy = compute_energy(data_tensor)  # (1, seq_len)
        
        # Detect nucleus (exact same as training)
        nucleus_points = detect_nucleus(energy)  # List of [start, end] for each sample
        nucleus_start, nucleus_end = nucleus_points[0]
        
        # Calculate significant axis (exact same as training)
        sig_axis = calculate_significant_axis(data_tensor)  # (1,)
        sig_axis_value = sig_axis.item()
        
        # Create nucleus mask (exact same as training)
        nucleus_mask = torch.zeros(data_tensor.shape[0], data_tensor.shape[1], dtype=torch.long, device=self.device)
        nucleus_mask[0, nucleus_start:nucleus_end] = 1
        
        # Create significant axis mask (exact same as training)
        # According to embedding.py: sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
        sig_axis_mask = (data_tensor.argmax(dim=-1) == sig_axis[:, None]).float()
        
        metadata = {
            'nucleus_start': nucleus_start,
            'nucleus_end': nucleus_end,
            'significant_axis': sig_axis_value,
            'energy': energy.cpu().numpy(),
            'nucleus_mask': nucleus_mask.cpu().numpy(),
            'sig_axis_mask': sig_axis_mask.cpu().numpy()
        }
        
        return data_tensor, nucleus_mask, sig_axis_mask, metadata
    
    def predict(self, raw_data: np.ndarray) -> Dict:
        """
        Perform complete inference - EXACT same pipeline as training
        
        Args:
            raw_data: Raw IMU data of shape (seq_len, 6)
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess (exact same as training)
        data_tensor, nucleus_mask, sig_axis_mask, metadata = self.preprocess(raw_data)
        
        with torch.no_grad():
            # Generate embeddings (exact same as training)
            # The embedder forward takes: input_seqs, masked_pos=None, nucleus_mask=None, sig_axis_mask=None
            embeddings = self.embedder(
                data_tensor, 
                masked_pos=None,
                nucleus_mask=nucleus_mask,
                sig_axis_mask=sig_axis_mask
            )  # Output: (1, seq_len, 72)
            
            # Classifier expects embeddings
            # In training, the classifier is called directly with embeddings
            logits = self.classifier(embeddings, return_features=False)  # (1, num_classes)
            
            # Get predictions
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get label name (remember: labels are 1-indexed in dataset, 0-indexed in array)
        predicted_label = self.config['activity_label'][predicted_class]
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'all_labels': self.config['activity_label'],
            'metadata': metadata
        }
        
        return result
    
    def predict_batch(self, raw_data_batch: np.ndarray) -> List[Dict]:
        """
        Predict on a batch of samples
        
        Args:
            raw_data_batch: Batch of raw IMU data of shape (batch_size, seq_len, 6)
        
        Returns:
            List of prediction dictionaries
        """
        # Apply normalization
        preprocessed_batch = raw_data_batch.copy()
        preprocessed_batch[:, :, :3] = preprocessed_batch[:, :, :3] / 9.8
        
        # Convert to tensor
        data_tensor = torch.from_numpy(preprocessed_batch).float().to(self.device)
        batch_size = data_tensor.shape[0]
        
        # Compute energy for all samples
        energy = compute_energy(data_tensor)
        
        # Detect nucleus for all samples
        nucleus_points = detect_nucleus(energy)
        
        # Calculate significant axis for all samples
        sig_axis = calculate_significant_axis(data_tensor)
        
        # Create masks for all samples
        nucleus_masks = []
        sig_axis_masks = []
        
        # Create nucleus masks
        nucleus_mask_batch = torch.zeros(batch_size, data_tensor.shape[1], dtype=torch.long, device=self.device)
        for i in range(batch_size):
            nucleus_start, nucleus_end = nucleus_points[i]
            nucleus_mask_batch[i, nucleus_start:nucleus_end] = 1
        
        # Create significant axis mask
        sig_axis_mask_batch = (data_tensor.argmax(dim=-1) == sig_axis[:, None]).float()
        
        with torch.no_grad():
            # Generate embeddings
            embeddings = self.embedder(
                data_tensor,
                masked_pos=None,
                nucleus_mask=nucleus_mask_batch,
                sig_axis_mask=sig_axis_mask_batch
            )
            
            # Classify
            logits = self.classifier(embeddings, return_features=False)
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
        
        # Create results
        results = []
        for i in range(batch_size):
            predicted_class = predicted_classes[i].item()
            confidence = probabilities[i, predicted_class].item()
            predicted_label = self.config['activity_label'][predicted_class]
            
            result = {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities[i].cpu().numpy(),
                'metadata': {
                    'nucleus_start': nucleus_points[i][0],
                    'nucleus_end': nucleus_points[i][1],
                    'significant_axis': sig_axis[i].item()
                }
            }
            results.append(result)
        
        return results


def main():
    """Test the exact pipeline"""
    print("="*80)
    print("Sony Watch Exact Inference Pipeline Test")
    print("="*80)
    
    # Paths
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    data_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/data_20_120.npy"
    label_path = "/home/prerna/LIMU-BERT-blind-users/dataset/sony_watch/label_20_120.npy"
    
    # Initialize pipeline
    pipeline = ExactSonyWatchPipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load data
    print("\nLoading dataset...")
    data = np.load(data_path)
    labels = np.load(label_path)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test on a few samples
    print("\n" + "="*80)
    print("Testing on 10 random samples")
    print("="*80)
    
    np.random.seed(42)
    test_indices = np.random.choice(len(data), size=10, replace=False)
    
    correct = 0
    for idx in test_indices:
        raw_data = data[idx]  # (120, 6)
        true_label_value = int(labels[idx][0, 0])  # Labels are 1-indexed
        true_label_idx = true_label_value - 1  # Convert to 0-indexed
        true_label_name = pipeline.config['activity_label'][true_label_idx]
        
        # Predict
        result = pipeline.predict(raw_data)
        
        is_correct = result['predicted_class'] == true_label_idx
        if is_correct:
            correct += 1
        
        print(f"\nSample {idx}:")
        print(f"  True: {true_label_name} (class {true_label_idx})")
        print(f"  Predicted: {result['predicted_label']} (class {result['predicted_class']})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print(f"  Nucleus: [{result['metadata']['nucleus_start']}, {result['metadata']['nucleus_end']}]")
        print(f"  Sig Axis: {result['metadata']['significant_axis']}")
    
    print(f"\n{'='*80}")
    print(f"Accuracy: {correct}/10 = {correct*10.0:.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
