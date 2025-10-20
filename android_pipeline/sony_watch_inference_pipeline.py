#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sony Watch Inference Pipeline
================================
Complete implementation of the inference pipeline for Sony Watch dataset including:
- Nucleus detection and significant axis calculation
- BERT embeddings generation
- GRU classification with semantic descriptors
- Exact same preprocessing as training pipeline

Usage:
    python sony_watch_inference_pipeline.py --data_path <path_to_raw_imu_data>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import json
from typing import Tuple, List, Dict, Optional


class NucleusDetector:
    """
    Detects nucleus (most significant segment) in IMU data
    Based on embedding.py implementation
    """
    
    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Size of the sliding window for nucleus detection
        """
        self.window_size = window_size
    
    def detect_nucleus(self, data: np.ndarray) -> Tuple[int, int]:
        """
        Detect the most significant segment (nucleus) in the IMU data
        
        Args:
            data: IMU data of shape (seq_len, 6) - 6 axis (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        
        Returns:
            Tuple of (start_idx, end_idx) for the nucleus segment
        """
        seq_len = data.shape[0]
        
        # Calculate magnitude for all 6 axes combined
        magnitude = np.linalg.norm(data, axis=1)
        
        # Find the segment with maximum cumulative magnitude
        max_energy = -np.inf
        nucleus_start = 0
        nucleus_end = self.window_size
        
        for i in range(seq_len - self.window_size + 1):
            segment_energy = np.sum(magnitude[i:i + self.window_size])
            if segment_energy > max_energy:
                max_energy = segment_energy
                nucleus_start = i
                nucleus_end = i + self.window_size
        
        return nucleus_start, nucleus_end
    
    def extract_nucleus_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from the nucleus segment
        
        Args:
            data: IMU data of shape (seq_len, 6)
        
        Returns:
            Feature vector of shape (36,) - 6 features per axis
        """
        start_idx, end_idx = self.detect_nucleus(data)
        nucleus_segment = data[start_idx:end_idx]
        
        features = []
        for axis in range(6):
            axis_data = nucleus_segment[:, axis]
            features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.max(axis_data),
                np.min(axis_data),
                np.median(axis_data),
                np.ptp(axis_data)  # peak-to-peak (max - min)
            ])
        
        return np.array(features)


class SignificantAxisCalculator:
    """
    Calculate the most significant axis in IMU data
    Based on embedding.py implementation
    """
    
    def __init__(self):
        pass
    
    def calculate_significant_axis(self, data: np.ndarray) -> Dict[str, any]:
        """
        Calculate the most significant axis and related statistics
        
        Args:
            data: IMU data of shape (seq_len, 6)
        
        Returns:
            Dictionary containing:
                - sig_axis: Index of most significant axis (0-5)
                - sig_axis_name: Name of the axis
                - variance_ratio: Variance of sig axis / total variance
                - axis_variances: Variance for each axis
        """
        axis_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Calculate variance for each axis
        axis_variances = np.var(data, axis=0)
        
        # Find the most significant axis
        sig_axis = np.argmax(axis_variances)
        total_variance = np.sum(axis_variances)
        variance_ratio = axis_variances[sig_axis] / total_variance if total_variance > 0 else 0
        
        return {
            'sig_axis': sig_axis,
            'sig_axis_name': axis_names[sig_axis],
            'variance_ratio': variance_ratio,
            'axis_variances': axis_variances.tolist()
        }
    
    def get_axis_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix between all axes
        
        Args:
            data: IMU data of shape (seq_len, 6)
        
        Returns:
            Correlation matrix of shape (6, 6)
        """
        return np.corrcoef(data.T)


class IMUPreprocessor:
    """
    Preprocess raw IMU data with normalization and augmentation
    Exactly matches the training pipeline
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to apply z-score normalization
        """
        self.normalize = normalize
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess IMU data
        
        Args:
            data: Raw IMU data of shape (seq_len, 6)
        
        Returns:
            Preprocessed data of same shape
        """
        # Apply normalization per axis
        if self.normalize:
            data = self._normalize_per_axis(data)
        
        return data
    
    def _normalize_per_axis(self, data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization per axis
        
        Args:
            data: IMU data of shape (seq_len, 6)
        
        Returns:
            Normalized data
        """
        normalized = np.zeros_like(data)
        for axis in range(data.shape[1]):
            mean = np.mean(data[:, axis])
            std = np.std(data[:, axis])
            if std > 0:
                normalized[:, axis] = (data[:, axis] - mean) / std
            else:
                normalized[:, axis] = data[:, axis] - mean
        
        return normalized
    
    def segment_data(self, data: np.ndarray, seq_len: int = 120, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment continuous IMU data into fixed-length windows
        
        Args:
            data: Continuous IMU data of shape (total_len, 6)
            seq_len: Length of each segment
            overlap: Overlap ratio between segments (0-1)
        
        Returns:
            List of segments, each of shape (seq_len, 6)
        """
        segments = []
        stride = int(seq_len * (1 - overlap))
        
        for start in range(0, len(data) - seq_len + 1, stride):
            segment = data[start:start + seq_len]
            segments.append(segment)
        
        return segments


class BERTEmbedder(nn.Module):
    """
    BERT-based embedder model for IMU data
    Loads the pre-trained LIMU-BERT model
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Path to the pre-trained BERT model (.pt file)
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        
        # Load the pre-trained model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Reconstruct the model (you may need to adjust based on your model architecture)
        self.model = self._build_model_from_checkpoint(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded BERT embedder from {model_path}")
    
    def _build_model_from_checkpoint(self, state_dict):
        """
        Build model architecture from checkpoint state dict
        This is a placeholder - adjust based on your actual model architecture
        """
        # Import your actual BERT model class
        from models import LIMUBertModel4Pretrain
        
        # You'll need to infer the config from state_dict or load it separately
        # For now, using common configuration
        class Config:
            hidden = 72
            hidden_ff = 144
            n_layers = 4
            n_heads = 4
            seq_len = 120
            feature_num = 6
            emb_norm = True
        
        config = Config()
        # Set output_embed=True to get embeddings instead of reconstruction
        model = LIMUBertModel4Pretrain(config, output_embed=True)
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings for input IMU data
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 6)
        
        Returns:
            Embeddings of shape (batch_size, hidden_dim)
        """
        with torch.no_grad():
            # Get BERT output
            output = self.model(x)
            
            # Extract embeddings (use CLS token or mean pooling)
            if isinstance(output, tuple):
                embeddings = output[0]  # Usually the first output
            else:
                embeddings = output
            
            # If output is (batch, seq_len, hidden), take mean or CLS
            if len(embeddings.shape) == 3:
                # Use CLS token (first position) or mean pooling
                embeddings = embeddings[:, 0, :]  # CLS token
                # Or use: embeddings = torch.mean(embeddings, dim=1)  # Mean pooling
        
        return embeddings


class SemanticDescriptorEncoder:
    """
    Encodes gesture descriptors using BERT for semantic understanding
    Used in the contrastive loss during training
    """
    
    def __init__(self, label_names: List[str], descriptions: List[str], 
                 pooling: str = 'cls', device: str = 'cuda'):
        """
        Args:
            label_names: List of gesture label names
            descriptions: List of detailed descriptions for each gesture
            pooling: Pooling strategy ('cls', 'mean', or 'max')
            device: Device to run on
        """
        self.label_names = label_names
        self.descriptions = descriptions
        self.pooling = pooling
        self.device = device
        
        # Initialize BERT for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.to(device)
        self.bert_model.eval()
        
        # Precompute semantic embeddings for all gestures
        self.semantic_embeddings = self._compute_semantic_embeddings()
        self.semantic_similarity_matrix = self._compute_similarity_matrix()
        
        print(f"Initialized semantic encoder with {len(label_names)} gestures")
        print(f"Using pooling strategy: {pooling}")
    
    def _compute_semantic_embeddings(self) -> torch.Tensor:
        """
        Compute BERT embeddings for all gesture descriptions
        
        Returns:
            Tensor of shape (num_classes, bert_dim)
        """
        with torch.no_grad():
            inputs = self.tokenizer(self.descriptions, padding=True, 
                                   return_tensors="pt").to(self.device)
            outputs = self.bert_model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            if self.pooling == "cls":
                # Use CLS token
                embeddings = hidden_states[:, 0, :]
                
            elif self.pooling == "mean":
                # Mean pooling, ignoring padding
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.mean(dim=0)
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
                
            elif self.pooling == "max":
                # Max pooling, ignoring padding
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.max(dim=0)[0]
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _compute_similarity_matrix(self) -> torch.Tensor:
        """
        Compute semantic similarity matrix between all gestures
        
        Returns:
            Similarity matrix of shape (num_classes, num_classes)
        """
        # Compute cosine similarity
        sim_matrix = torch.matmul(self.semantic_embeddings, 
                                 self.semantic_embeddings.t())
        
        # Enhance contrast
        sim_matrix = torch.pow(sim_matrix, 3)
        
        return sim_matrix
    
    def get_gesture_embedding(self, label_idx: int) -> torch.Tensor:
        """
        Get semantic embedding for a specific gesture
        
        Args:
            label_idx: Index of the gesture
        
        Returns:
            Semantic embedding vector
        """
        return self.semantic_embeddings[label_idx]
    
    def get_similarity(self, label_idx1: int, label_idx2: int) -> float:
        """
        Get semantic similarity between two gestures
        
        Args:
            label_idx1: Index of first gesture
            label_idx2: Index of second gesture
        
        Returns:
            Similarity score (0-1)
        """
        return self.semantic_similarity_matrix[label_idx1, label_idx2].item()


class ContrastiveGRUClassifier(nn.Module):
    """
    GRU-based classifier with contrastive learning
    Matches the training architecture
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension (BERT embedding size)
            hidden_dim: Hidden dimension for GRU
            num_classes: Number of gesture classes
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # Project to contrastive space
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"Initialized GRU classifier: input={input_dim}, hidden={hidden_dim}, "
              f"classes={num_classes}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features=False: logits of shape (batch_size, num_classes)
            If return_features=True: (logits, features, projected)
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # GRU forward
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        if self.num_layers == 1:
            features = hidden.squeeze(0)
        else:
            features = hidden[-1]  # Last layer's hidden state
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            # Projection for contrastive learning
            projected = self.projection(features)
            return logits, features, projected
        
        return logits


class SonyWatchInferencePipeline:
    """
    Complete inference pipeline for Sony Watch gesture recognition
    """
    
    def __init__(self, 
                 embedder_path: str,
                 classifier_path: str,
                 config_path: str = None,
                 device: str = None):
        """
        Args:
            embedder_path: Path to pre-trained BERT embedder model
            classifier_path: Path to pre-trained GRU classifier model
            config_path: Path to dataset configuration file
            device: Device to run on (cuda/cpu)
        """
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.preprocessor = IMUPreprocessor(normalize=True)
        self.nucleus_detector = NucleusDetector(window_size=20)
        self.sig_axis_calculator = SignificantAxisCalculator()
        
        # Load models
        print("\nLoading embedder model...")
        self.embedder = self._load_embedder(embedder_path)
        
        print("\nLoading classifier model...")
        self.classifier = self._load_classifier(classifier_path)
        
        # Initialize semantic descriptor encoder
        if 'descriptions' in self.config:
            print("\nInitializing semantic descriptor encoder...")
            self.semantic_encoder = SemanticDescriptorEncoder(
                label_names=self.config['activity_label'],
                descriptions=self.config['descriptions'],
                pooling='cls',
                device=self.device
            )
        else:
            print("\nWarning: No descriptions found, semantic features disabled")
            self.semantic_encoder = None
        
        print("\nPipeline initialized successfully!")
        print(f"Ready to recognize {len(self.config['activity_label'])} gestures:")
        for i, label in enumerate(self.config['activity_label']):
            print(f"  {i}: {label}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load dataset configuration"""
        if config_path is None:
            config_path = 'dataset/data_config.json'
        
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
        
        # Load Sony Watch configuration
        if 'sony_watch_20_120' in all_configs:
            return all_configs['sony_watch_20_120']
        else:
            raise ValueError("Sony Watch configuration not found!")
    
    def _load_embedder(self, model_path: str) -> BERTEmbedder:
        """Load BERT embedder model"""
        embedder = BERTEmbedder(model_path, device=self.device)
        return embedder
    
    def _load_classifier(self, model_path: str) -> ContrastiveGRUClassifier:
        """Load GRU classifier model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Infer model dimensions from state dict
        # This is a heuristic - adjust based on your actual model
        input_dim = 72  # BERT hidden dimension
        hidden_dim = 128  # GRU hidden dimension
        num_classes = self.config['activity_label_size']
        
        # Create model
        classifier = ContrastiveGRUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=2,
            dropout=0.3
        )
        
        # Load weights
        classifier.load_state_dict(state_dict, strict=False)
        classifier.to(self.device)
        classifier.eval()
        
        return classifier
    
    def preprocess_raw_data(self, raw_data: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess raw IMU data
        
        Args:
            raw_data: Raw IMU data of shape (seq_len, 6)
        
        Returns:
            Tuple of (preprocessed_tensor, metadata_dict)
        """
        # Detect nucleus
        nucleus_start, nucleus_end = self.nucleus_detector.detect_nucleus(raw_data)
        nucleus_features = self.nucleus_detector.extract_nucleus_features(raw_data)
        
        # Calculate significant axis
        sig_axis_info = self.sig_axis_calculator.calculate_significant_axis(raw_data)
        
        # Preprocess data
        preprocessed = self.preprocessor.preprocess(raw_data)
        
        # Convert to tensor
        tensor = torch.from_numpy(preprocessed).float().unsqueeze(0)  # Add batch dim
        tensor = tensor.to(self.device)
        
        # Metadata
        metadata = {
            'nucleus_start': nucleus_start,
            'nucleus_end': nucleus_end,
            'nucleus_features': nucleus_features,
            'significant_axis': sig_axis_info,
            'preprocessed_shape': preprocessed.shape
        }
        
        return tensor, metadata
    
    def predict(self, raw_data: np.ndarray, return_probabilities: bool = True) -> Dict:
        """
        Perform complete inference on raw IMU data
        
        Args:
            raw_data: Raw IMU data of shape (seq_len, 6)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary containing:
                - predicted_class: Index of predicted gesture
                - predicted_label: Name of predicted gesture
                - confidence: Confidence score (0-1)
                - probabilities: Class probabilities (if requested)
                - metadata: Additional information about processing
        """
        # Preprocess
        preprocessed, metadata = self.preprocess_raw_data(raw_data)
        
        # Generate BERT embeddings
        with torch.no_grad():
            embeddings = self.embedder(preprocessed)
        
        # The GRU classifier expects (batch, seq, features), but BERT outputs (batch, features)
        # Add sequence dimension
        embeddings = embeddings.unsqueeze(1)  # Shape: (batch, 1, hidden_dim)
        
        # Classify
        with torch.no_grad():
            logits = self.classifier(embeddings, return_features=False)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get label name
        predicted_label = self.config['activity_label'][predicted_class]
        
        # Prepare result
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'metadata': metadata
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities.cpu().numpy()[0]
            result['all_labels'] = self.config['activity_label']
        
        # Add semantic information if available
        if self.semantic_encoder is not None:
            result['description'] = self.config['descriptions'][predicted_class]
            result['semantic_embedding'] = self.semantic_encoder.get_gesture_embedding(
                predicted_class).cpu().numpy()
        
        return result
    
    def predict_batch(self, raw_data_list: List[np.ndarray]) -> List[Dict]:
        """
        Perform inference on a batch of raw IMU data
        
        Args:
            raw_data_list: List of raw IMU data arrays
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for raw_data in raw_data_list:
            result = self.predict(raw_data)
            results.append(result)
        return results
    
    def predict_streaming(self, raw_data: np.ndarray, window_size: int = 120, 
                         overlap: float = 0.5) -> List[Dict]:
        """
        Perform inference on streaming/continuous IMU data
        
        Args:
            raw_data: Continuous IMU data of shape (total_len, 6)
            window_size: Size of sliding window
            overlap: Overlap ratio between windows
        
        Returns:
            List of predictions for each window
        """
        # Segment data
        segments = self.preprocessor.segment_data(raw_data, window_size, overlap)
        
        # Predict for each segment
        results = []
        for i, segment in enumerate(segments):
            result = self.predict(segment)
            result['window_index'] = i
            result['window_start'] = int(i * window_size * (1 - overlap))
            results.append(result)
        
        return results


def demo_inference():
    """
    Demo function showing how to use the inference pipeline
    """
    print("Sony Watch Inference Pipeline Demo")
    print("=" * 50)
    
    # Paths to models
    embedder_path = "/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt"
    classifier_path = "/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt"
    config_path = "/home/prerna/LIMU-BERT-blind-users/dataset/data_config.json"
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = SonyWatchInferencePipeline(
        embedder_path=embedder_path,
        classifier_path=classifier_path,
        config_path=config_path
    )
    
    # Generate dummy data for demo (replace with actual sensor data)
    print("\nGenerating demo data...")
    dummy_data = np.random.randn(120, 6)  # 120 timesteps, 6 axes
    
    # Single prediction
    print("\n" + "=" * 50)
    print("Single Prediction Demo")
    print("=" * 50)
    result = pipeline.predict(dummy_data)
    
    print(f"\nPredicted Gesture: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Class Index: {result['predicted_class']}")
    
    if 'description' in result:
        print(f"Description: {result['description']}")
    
    print("\nMetadata:")
    print(f"  Nucleus: [{result['metadata']['nucleus_start']}, {result['metadata']['nucleus_end']}]")
    print(f"  Significant Axis: {result['metadata']['significant_axis']['sig_axis_name']}")
    print(f"  Variance Ratio: {result['metadata']['significant_axis']['variance_ratio']:.4f}")
    
    if 'probabilities' in result:
        print("\nTop 3 Predictions:")
        probs = result['probabilities']
        top_indices = np.argsort(probs)[-3:][::-1]
        for idx in top_indices:
            print(f"  {result['all_labels'][idx]}: {probs[idx]:.4f}")
    
    # Streaming prediction demo
    print("\n" + "=" * 50)
    print("Streaming Prediction Demo")
    print("=" * 50)
    
    # Generate continuous data
    continuous_data = np.random.randn(500, 6)
    results = pipeline.predict_streaming(continuous_data, window_size=120, overlap=0.5)
    
    print(f"\nProcessed {len(results)} windows")
    print("\nFirst 3 predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\nWindow {i} (start: {result['window_start']}):")
        print(f"  Gesture: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Sony Watch Inference Pipeline'
    )
    parser.add_argument('--embedder_path', type=str,
                       default='/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt',
                       help='Path to BERT embedder model')
    parser.add_argument('--classifier_path', type=str,
                       default='/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt',
                       help='Path to GRU classifier model')
    parser.add_argument('--config_path', type=str,
                       default='dataset/data_config.json',
                       help='Path to dataset configuration')
    parser.add_argument('--data_path', type=str,
                       help='Path to input IMU data (.npy file)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with dummy data')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference()
    elif args.data_path:
        # Load and process real data
        print(f"Loading data from {args.data_path}")
        data = np.load(args.data_path)
        
        # Initialize pipeline
        pipeline = SonyWatchInferencePipeline(
            embedder_path=args.embedder_path,
            classifier_path=args.classifier_path,
            config_path=args.config_path
        )
        
        # Run prediction
        if len(data.shape) == 2:
            # Single sample
            result = pipeline.predict(data)
            print(f"\nPredicted: {result['predicted_label']} "
                  f"(confidence: {result['confidence']:.4f})")
        else:
            # Multiple samples
            results = pipeline.predict_batch([data[i] for i in range(len(data))])
            for i, result in enumerate(results):
                print(f"Sample {i}: {result['predicted_label']} "
                      f"(confidence: {result['confidence']:.4f})")
    else:
        print("Please provide --data_path or use --demo")
        parser.print_help()


if __name__ == '__main__':
    main()
