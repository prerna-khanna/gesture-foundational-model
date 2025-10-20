#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sony Watch Real-time Inference Pipeline
This script provides a complete inference pipeline matching the exact training preprocessing.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
import json

# Import the necessary modules from the project
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LIMUBertModel4Pretrain
from contrastive.models import ContrastiveGRUClassifier
from features import detect_nucleus, compute_energy, calculate_significant_axis


class SonyWatchInferencePipeline:
    """
    Complete inference pipeline for Sony Watch IMU data.
    Matches the exact preprocessing and inference flow used during training.
    """
    
    def __init__(
        self,
        embedder_path: str,
        classifier_path: str,
        config_path: str = None,
        device: str = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            embedder_path: Path to the pretrained LIMU-BERT embedder model
            classifier_path: Path to the trained GRU classifier model
            config_path: Path to configuration JSON (optional)
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
        """
        self.device = self._setup_device(device)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize preprocessing parameters (matching training)
        self.seq_len = self.config.get('seq_len', 120)
        self.feature_num = self.config.get('feature_num', 6)  # 3 accel + 3 gyro
        self.acc_norm = 9.8  # Gravity normalization
        self.eps = 1e-5
        
        # Load models
        self.embedder = self._load_embedder(embedder_path)
        self.classifier = self._load_classifier(classifier_path)
        
        # Set models to evaluation mode
        self.embedder.eval()
        self.classifier.eval()
        
        # Load label names
        self.label_names = self.config.get('label_names', [])
        
        print(f"✓ Inference pipeline initialized on {self.device}")
        print(f"✓ Sequence length: {self.seq_len}")
        print(f"✓ Feature dimensions: {self.feature_num}")
        print(f"✓ Number of classes: {len(self.label_names)}")
    
    def _setup_device(self, device: str = None) -> torch.device:
        """Setup the computation device."""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)
        
        print(f"Using device: {device}")
        return device
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from JSON file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded configuration from {config_path}")
        else:
            # Default configuration for Sony Watch dataset
            config = {
                'seq_len': 120,
                'feature_num': 6,
                'hidden_dim': 128,
                'num_classes': None,  # Will be inferred from model
                'label_names': []
            }
            print("⚠ Using default configuration")
        
        return config
    
    def _load_embedder(self, model_path: str) -> nn.Module:
        """Load the pretrained LIMU-BERT embedder model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        if 'model_cfg' in checkpoint:
            model_cfg = checkpoint['model_cfg']
        else:
            # Create default model config
            from config import ModelConfig
            model_cfg = ModelConfig(
                vocab_size=None,
                dim=120,
                depth=4,
                heads=4,
                feature_num=self.feature_num,
                seq_len=self.seq_len
            )
        
        # Initialize model
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
        
        # Load state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        print(f"✓ Loaded embedder from {model_path}")
        
        return model
    
    def _load_classifier(self, model_path: str) -> nn.Module:
        """Load the trained GRU classifier model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract configuration
        if 'model_cfg' in checkpoint or 'config' in checkpoint:
            config = checkpoint.get('model_cfg', checkpoint.get('config'))
            hidden_dim = getattr(config, 'hidden_dim', 128)
            num_classes = getattr(config, 'num_classes', None)
        else:
            hidden_dim = self.config.get('hidden_dim', 128)
            num_classes = self.config.get('num_classes', None)
        
        # Infer input dimension from embedder
        # The embedder output dimension is typically the model's hidden dimension
        input_dim = 120  # LIMU-BERT embedding dimension
        
        # If num_classes is not in config, infer from classifier weights
        if num_classes is None and 'model' in checkpoint:
            # Look for classifier layer in state dict
            for key in checkpoint['model'].keys():
                if 'classifier' in key and 'weight' in key:
                    num_classes = checkpoint['model'][key].shape[0]
                    break
        
        if num_classes is None:
            raise ValueError("Could not determine number of classes from checkpoint")
        
        # Update config
        self.config['num_classes'] = num_classes
        
        # Initialize classifier
        model = ContrastiveGRUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        # Load state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        print(f"✓ Loaded classifier from {model_path}")
        print(f"  Input dim: {input_dim}, Hidden dim: {hidden_dim}, Classes: {num_classes}")
        
        return model
    
    def preprocess_normalization(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize IMU data (matching Preprocess4Normalization from utils.py).
        
        Args:
            data: Raw IMU data of shape (seq_len, feature_num)
                  First 3 channels: accelerometer (m/s²)
                  Last 3 channels: gyroscope (rad/s)
        
        Returns:
            Normalized data
        """
        data_normalized = data.copy()[:, :self.feature_num]
        
        # Normalize accelerometer by gravity (9.8 m/s²)
        if data_normalized.shape[1] >= 6:
            data_normalized[:, :3] = data_normalized[:, :3] / self.acc_norm
        
        return data_normalized
    
    def generate_nucleus_mask(
        self,
        seq_len: int,
        batch_nucleus_points: List[List[int]]
    ) -> torch.Tensor:
        """
        Generate binary mask for nucleus regions (matching embedding.py).
        
        Args:
            seq_len: Sequence length
            batch_nucleus_points: List of [start, end] points for each sequence
        
        Returns:
            Binary mask tensor of shape (batch_size, seq_len)
        """
        batch_size = len(batch_nucleus_points)
        nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for i, nucleus_points in enumerate(batch_nucleus_points):
            if len(nucleus_points) == 2:
                start, end = nucleus_points
                nucleus_mask[i, start:end] = 1
        
        return nucleus_mask.to(self.device)
    
    def generate_sig_axis_mask(
        self,
        seqs: torch.Tensor,
        sig_axis: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate significant axis mask (matching embedding.py).
        
        Args:
            seqs: Input sequences of shape (batch_size, seq_len, feature_num)
            sig_axis: Significant axis indices of shape (batch_size,)
        
        Returns:
            Significant axis mask of shape (batch_size, seq_len)
        """
        # Create mask where positions match the significant axis
        sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
        return sig_axis_mask
    
    def preprocess_single_window(
        self,
        data: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess a single window of IMU data.
        
        Args:
            data: Raw IMU data of shape (seq_len, feature_num)
        
        Returns:
            Tuple of (normalized_data, nucleus_mask, sig_axis_mask)
        """
        # Step 1: Normalize the data
        data_normalized = self.preprocess_normalization(data)
        
        # Step 2: Convert to tensor and add batch dimension
        data_tensor = torch.from_numpy(data_normalized).float().unsqueeze(0)
        data_tensor = data_tensor.to(self.device)
        
        # Step 3: Compute energy
        energy = compute_energy(data_tensor)
        
        # Step 4: Detect nucleus points
        batch_nucleus_points = detect_nucleus(energy)
        nucleus_mask = self.generate_nucleus_mask(data_tensor.size(1), batch_nucleus_points)
        
        # Step 5: Calculate significant axis
        sig_axis = calculate_significant_axis(data_tensor)
        sig_axis_mask = self.generate_sig_axis_mask(data_tensor, sig_axis)
        
        return data_tensor, nucleus_mask, sig_axis_mask
    
    def preprocess_batch(
        self,
        data_batch: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess a batch of IMU data windows.
        
        Args:
            data_batch: Batch of raw IMU data of shape (batch_size, seq_len, feature_num)
        
        Returns:
            Tuple of (normalized_data, nucleus_mask, sig_axis_mask)
        """
        # Normalize all samples
        batch_normalized = np.array([
            self.preprocess_normalization(sample) 
            for sample in data_batch
        ])
        
        # Convert to tensor
        data_tensor = torch.from_numpy(batch_normalized).float().to(self.device)
        
        # Compute energy
        energy = compute_energy(data_tensor)
        
        # Detect nucleus points
        batch_nucleus_points = detect_nucleus(energy)
        nucleus_mask = self.generate_nucleus_mask(data_tensor.size(1), batch_nucleus_points)
        
        # Calculate significant axis
        sig_axis = calculate_significant_axis(data_tensor)
        sig_axis_mask = self.generate_sig_axis_mask(data_tensor, sig_axis)
        
        return data_tensor, nucleus_mask, sig_axis_mask
    
    @torch.no_grad()
    def generate_embedding(
        self,
        data: np.ndarray,
        batch_mode: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings from raw IMU data using the LIMU-BERT model.
        
        Args:
            data: Raw IMU data
                  Single window: shape (seq_len, feature_num)
                  Batch: shape (batch_size, seq_len, feature_num)
            batch_mode: Whether data is a batch or single window
        
        Returns:
            Embeddings of shape (embedding_dim,) or (batch_size, embedding_dim)
        """
        # Preprocess
        if batch_mode:
            data_tensor, nucleus_mask, sig_axis_mask = self.preprocess_batch(data)
        else:
            data_tensor, nucleus_mask, sig_axis_mask = self.preprocess_single_window(data)
        
        # Generate embeddings
        embeddings = self.embedder(
            data_tensor,
            nucleus_mask=nucleus_mask,
            sig_axis_mask=sig_axis_mask
        )
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        if not batch_mode:
            embeddings_np = embeddings_np.squeeze(0)
        
        return embeddings_np
    
    @torch.no_grad()
    def classify_embedding(
        self,
        embedding: np.ndarray,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Classify an embedding using the GRU classifier.
        
        Args:
            embedding: Embedding vector of shape (embedding_dim,) or (batch_size, embedding_dim)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary containing:
                - 'class_idx': Predicted class index
                - 'class_name': Predicted class name
                - 'confidence': Confidence score
                - 'probabilities': Class probabilities (if return_probabilities=True)
        """
        # Convert to tensor and add sequence dimension
        if len(embedding.shape) == 1:
            # Single embedding: (embedding_dim,) -> (1, 1, embedding_dim)
            embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0).unsqueeze(0)
        else:
            # Batch of embeddings: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
            embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(1)
        
        embedding_tensor = embedding_tensor.to(self.device)
        
        # Get logits from classifier
        logits = self.classifier(embedding_tensor, return_features=False)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        # Convert to numpy
        predicted_class = predicted_class.cpu().numpy()
        confidence = confidence.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        # Prepare result
        if len(embedding.shape) == 1:
            # Single prediction
            result = {
                'class_idx': int(predicted_class[0]),
                'class_name': self.label_names[int(predicted_class[0])] if self.label_names else f"Class_{int(predicted_class[0])}",
                'confidence': float(confidence[0])
            }
            if return_probabilities:
                result['probabilities'] = probabilities_np[0].tolist()
        else:
            # Batch predictions
            result = {
                'class_idx': predicted_class.tolist(),
                'class_name': [self.label_names[idx] if self.label_names else f"Class_{idx}" for idx in predicted_class],
                'confidence': confidence.tolist()
            }
            if return_probabilities:
                result['probabilities'] = probabilities_np.tolist()
        
        return result
    
    @torch.no_grad()
    def predict(
        self,
        data: np.ndarray,
        batch_mode: bool = False,
        return_probabilities: bool = True
    ) -> Dict:
        """
        End-to-end prediction from raw IMU data to classification result.
        
        Args:
            data: Raw IMU data
                  Single window: shape (seq_len, feature_num)
                  Batch: shape (batch_size, seq_len, feature_num)
            batch_mode: Whether data is a batch or single window
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary containing prediction results
        """
        # Generate embedding
        embedding = self.generate_embedding(data, batch_mode=batch_mode)
        
        # Classify
        result = self.classify_embedding(embedding, return_probabilities=return_probabilities)
        
        return result
    
    def predict_stream(
        self,
        data_stream: np.ndarray,
        window_size: int = None,
        stride: int = None,
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Process a continuous stream of IMU data using sliding windows.
        
        Args:
            data_stream: Continuous IMU data of shape (stream_len, feature_num)
            window_size: Size of sliding window (default: self.seq_len)
            stride: Stride for sliding window (default: window_size // 2)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            List of prediction results for each window
        """
        if window_size is None:
            window_size = self.seq_len
        
        if stride is None:
            stride = window_size // 2
        
        results = []
        stream_len = data_stream.shape[0]
        
        # Process sliding windows
        for start_idx in range(0, stream_len - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_data = data_stream[start_idx:end_idx]
            
            # Predict for this window
            result = self.predict(
                window_data,
                batch_mode=False,
                return_probabilities=return_probabilities
            )
            
            # Add window position info
            result['window_start'] = start_idx
            result['window_end'] = end_idx
            
            results.append(result)
        
        return results


class RealTimeBuffer:
    """
    Buffer for real-time streaming data from Sony Watch.
    Manages sliding windows and triggers inference when ready.
    """
    
    def __init__(
        self,
        window_size: int = 120,
        stride: int = 60,
        sampling_rate: int = 50
    ):
        """
        Initialize the real-time buffer.
        
        Args:
            window_size: Number of samples per window
            stride: Number of samples to slide the window
            sampling_rate: IMU sampling rate in Hz
        """
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        
        self.buffer = []
        self.sample_count = 0
        
        print(f"✓ Real-time buffer initialized")
        print(f"  Window size: {window_size} samples ({window_size/sampling_rate:.2f}s)")
        print(f"  Stride: {stride} samples ({stride/sampling_rate:.2f}s)")
        print(f"  Sampling rate: {sampling_rate} Hz")
    
    def add_sample(self, sample: np.ndarray) -> bool:
        """
        Add a new IMU sample to the buffer.
        
        Args:
            sample: IMU sample of shape (feature_num,) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            True if a complete window is ready for inference
        """
        self.buffer.append(sample)
        self.sample_count += 1
        
        # Check if we have a complete window
        if len(self.buffer) >= self.window_size:
            return True
        
        return False
    
    def get_window(self) -> np.ndarray:
        """
        Get the current window for inference.
        
        Returns:
            Window data of shape (window_size, feature_num)
        """
        if len(self.buffer) < self.window_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} samples, need {self.window_size}")
        
        # Extract window
        window = np.array(self.buffer[:self.window_size])
        
        # Slide the buffer
        self.buffer = self.buffer[self.stride:]
        
        return window
    
    def reset(self):
        """Reset the buffer."""
        self.buffer = []
        self.sample_count = 0


# Example usage and testing functions
def test_inference_pipeline():
    """Test the inference pipeline with synthetic data."""
    print("\n" + "="*60)
    print("Testing Sony Watch Inference Pipeline")
    print("="*60 + "\n")
    
    # Paths to models
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    
    # Initialize pipeline
    pipeline = SonyWatchInferencePipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path
    )
    
    # Test 1: Single window prediction
    print("\n--- Test 1: Single Window Prediction ---")
    seq_len = 120
    feature_num = 6
    
    # Generate synthetic IMU data
    synthetic_data = np.random.randn(seq_len, feature_num).astype(np.float32)
    synthetic_data[:, :3] *= 9.8  # Scale accelerometer
    synthetic_data[:, 3:] *= 2.0  # Scale gyroscope
    
    print(f"Input shape: {synthetic_data.shape}")
    
    # Predict
    result = pipeline.predict(synthetic_data, batch_mode=False)
    
    print(f"Predicted class: {result['class_name']} (idx: {result['class_idx']})")
    print(f"Confidence: {result['confidence']:.4f}")
    if 'probabilities' in result:
        print(f"Top 3 classes:")
        probs = np.array(result['probabilities'])
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            class_name = pipeline.label_names[idx] if pipeline.label_names else f"Class_{idx}"
            print(f"  {class_name}: {probs[idx]:.4f}")
    
    # Test 2: Batch prediction
    print("\n--- Test 2: Batch Prediction ---")
    batch_size = 4
    batch_data = np.random.randn(batch_size, seq_len, feature_num).astype(np.float32)
    batch_data[:, :, :3] *= 9.8
    batch_data[:, :, 3:] *= 2.0
    
    print(f"Batch shape: {batch_data.shape}")
    
    result_batch = pipeline.predict(batch_data, batch_mode=True)
    print(f"Predicted classes: {result_batch['class_name']}")
    print(f"Confidences: {[f'{c:.4f}' for c in result_batch['confidence']]}")
    
    # Test 3: Streaming prediction
    print("\n--- Test 3: Streaming Prediction ---")
    stream_len = 500
    stream_data = np.random.randn(stream_len, feature_num).astype(np.float32)
    stream_data[:, :3] *= 9.8
    stream_data[:, 3:] *= 2.0
    
    print(f"Stream shape: {stream_data.shape}")
    
    results_stream = pipeline.predict_stream(stream_data, window_size=120, stride=60)
    print(f"Processed {len(results_stream)} windows")
    print(f"First window prediction: {results_stream[0]['class_name']} (confidence: {results_stream[0]['confidence']:.4f})")
    print(f"Last window prediction: {results_stream[-1]['class_name']} (confidence: {results_stream[-1]['confidence']:.4f})")
    
    # Test 4: Real-time buffer
    print("\n--- Test 4: Real-time Buffer ---")
    buffer = RealTimeBuffer(window_size=120, stride=60, sampling_rate=50)
    
    print(f"Simulating real-time streaming...")
    predictions = []
    
    for i in range(stream_len):
        sample = stream_data[i]
        
        if buffer.add_sample(sample):
            window = buffer.get_window()
            result = pipeline.predict(window, batch_mode=False, return_probabilities=False)
            predictions.append(result)
            print(f"  Sample {i}: Predicted {result['class_name']} (confidence: {result['confidence']:.4f})")
    
    print(f"\nTotal predictions made: {len(predictions)}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    test_inference_pipeline()
