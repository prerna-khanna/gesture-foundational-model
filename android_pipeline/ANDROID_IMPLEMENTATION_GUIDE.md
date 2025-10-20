# Android Implementation Guide for Sony Watch Gesture Recognition

## Overview

This guide provides a complete implementation pipeline for real-time gesture recognition on Sony smartwatches using the LIMU-BERT model with contrastive learning and semantic descriptors.

## Key Features

- **BERT-based Embeddings**: Pre-trained LIMU-BERT model for robust feature extraction
- **Contrastive Learning**: GRU classifier trained with contrastive and semantic losses
- **Semantic Descriptors**: Natural language descriptions of gestures for better semantic understanding
- **Nucleus Detection**: Identifies the most significant segment in IMU data
- **Significant Axis Calculation**: Determines the primary movement axis
- **Real-time Processing**: Optimized for on-device inference

## System Architecture

```
┌─────────────────┐
│  Sony Watch     │
│  (6-axis IMU)   │
└────────┬────────┘
         │ Bluetooth
         ▼
┌─────────────────┐
│  Android Phone  │
│                 │
│  ┌───────────┐  │
│  │ IMU Data  │  │
│  │ Streaming │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │Preprocessing│ 
│  │(Normalize)│  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │LIMU-BERT  │  │
│  │ Embedder  │  │
│  │(limu_v1.pt)│ │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │   GRU     │  │
│  │Classifier │  │
│  │(limu_gru_v1)│
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │Prediction │  │
│  │(20 classes)│ │
│  └───────────┘  │
└─────────────────┘
```

## Pipeline Components

### 1. Data Streaming (Watch → Phone)

**Sony Watch Side:**
- Collect 6-axis IMU data (accelerometer + gyroscope)
  - Axes: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
- Sample rate: 20 Hz
- Buffer data in 120-sample windows (6 seconds)
- Data shape: (120, 6)

**Communication:**
- Use Bluetooth Low Energy (BLE) for efficient data transfer
- Send data as byte arrays to minimize latency
- Implement sliding window with 50% overlap for continuous recognition

### 2. Nucleus Detection

**Purpose:** Identify the most significant segment of the gesture

**Implementation:**
```python
class NucleusDetector:
    def detect_nucleus(self, data, window_size=20):
        # Calculate magnitude across all 6 axes
        magnitude = np.linalg.norm(data, axis=1)
        
        # Find segment with maximum cumulative energy
        max_energy = -inf
        nucleus_start, nucleus_end = 0, window_size
        
        for i in range(len(data) - window_size + 1):
            segment_energy = sum(magnitude[i:i + window_size])
            if segment_energy > max_energy:
                max_energy = segment_energy
                nucleus_start = i
                nucleus_end = i + window_size
        
        return nucleus_start, nucleus_end
    
    def extract_nucleus_features(self, data):
        # Extract 6 statistical features per axis (36 total)
        start, end = self.detect_nucleus(data)
        nucleus = data[start:end]
        
        features = []
        for axis in range(6):
            axis_data = nucleus[:, axis]
            features.extend([
                mean(axis_data),
                std(axis_data),
                max(axis_data),
                min(axis_data),
                median(axis_data),
                peak_to_peak(axis_data)
            ])
        
        return features  # Shape: (36,)
```

**Output:** 
- Nucleus start/end indices
- 36 statistical features (6 per axis)

### 3. Significant Axis Calculation

**Purpose:** Determine which axis contains the most information about the gesture

**Implementation:**
```python
class SignificantAxisCalculator:
    def calculate_significant_axis(self, data):
        axis_names = ['accel_x', 'accel_y', 'accel_z', 
                     'gyro_x', 'gyro_y', 'gyro_z']
        
        # Calculate variance for each axis
        axis_variances = np.var(data, axis=0)
        
        # Find most significant axis
        sig_axis = np.argmax(axis_variances)
        total_variance = np.sum(axis_variances)
        variance_ratio = axis_variances[sig_axis] / total_variance
        
        return {
            'sig_axis': sig_axis,
            'sig_axis_name': axis_names[sig_axis],
            'variance_ratio': variance_ratio,
            'axis_variances': axis_variances
        }
```

**Output:**
- Most significant axis index and name
- Variance ratio (contribution to total variance)
- Variance per axis

### 4. Preprocessing
```python
# 20 gesture classes for Sony Watch dataset
gestures = [
    "horizontal right",           # 0 - swipe right horizontally
    "horizontal left",            # 1 - swipe left horizontally
    "vertical up",                # 2 - swipe up vertically
    "vertical down",              # 3 - swipe down vertically
    "clockwise circle",           # 4 - circular motion clockwise
    "counter-clockwise circle",   # 5 - circular motion counter-clockwise
    "counter-clockwise square",   # 6 - square motion counter-clockwise
    "clockwise square",           # 7 - square motion clockwise
    "right diagonal",             # 8 - diagonal motion to right
    "left diagonal",              # 9 - diagonal motion to left
    "vertical down double",       # 10 - double swipe down
    "horizontal right double",    # 11 - double swipe right
    "V-shape down-up",            # 12 - V gesture downward then up
    "V-shape up-down",            # 13 - V gesture upward then down
    "triangle upward",            # 14 - triangular motion upward
    "triangle downward",          # 15 - triangular motion downward
    "S-curve leftward",           # 16 - S-shaped motion to left
    "S-curve rightward",          # 17 - S-shaped motion to right
    "wave leftward",              # 18 - waving motion left
    "wave rightward"              # 19 - waving motion right
]
```

## Preprocessing Pipeline

### Step 1: Data Normalization
```python
class Preprocess4Normalization:
    def __init__(self, feature_len=6, norm_acc=True, norm_mag=False):
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.acc_norm = 9.8  # Gravity constant
        self.eps = 1e-5
    
    def normalize(self, instance):
        """
        Normalize IMU data
        Args:
            instance: numpy array of shape (seq_len, 6)
                     First 3 columns: accelerometer (m/s²)
                     Last 3 columns: gyroscope (rad/s)
        Returns:
            Normalized data of same shape
        """
        instance_new = instance.copy()[:, :self.feature_len]
        
        # Normalize accelerometer data by gravity
        if self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        
        return instance_new
```

### Step 2: Feature Extraction (Nucleus Detection & Significant Axis)
```python
import numpy as np
import torch

def compute_energy(seqs):
    """
    Compute energy of IMU sequence
    Args:
        seqs: Tensor (batch_size, seq_len, 6)
    Returns:
        energy: Tensor (batch_size, seq_len)
    """
    energy = torch.sqrt((seqs ** 2).sum(dim=-1))
    return energy

def detect_nucleus(energy, min_nucleus_width=15, max_nucleus_width=40):
    """
    Detect the nucleus (most active part) of the gesture
    Args:
        energy: Tensor (batch_size, seq_len)
    Returns:
        batch_nucleus_points: list of [start, end] for each sequence
    """
    batch_nucleus_points = []
    
    for sequence_energy in energy:
        if isinstance(sequence_energy, torch.Tensor):
            sequence_energy = sequence_energy.cpu().numpy()
        
        seq_len = len(sequence_energy)
        energy_min = np.min(sequence_energy)
        energy_max = np.max(sequence_energy)
        
        # Handle flat energy case
        if energy_max - energy_min < 1e-6:
            mid_point = seq_len // 2
            nucleus_points = [
                max(0, mid_point - min_nucleus_width//2),
                min(seq_len, mid_point + min_nucleus_width//2)
            ]
            batch_nucleus_points.append(nucleus_points)
            continue
        
        # Normalize energy
        norm_energy = (sequence_energy - energy_min) / (energy_max - energy_min)
        
        # Find active region (energy > 50% of max)
        active_indices = np.where(norm_energy > 0.5)[0]
        
        if len(active_indices) > 0:
            start = max(0, active_indices[0])
            end = min(seq_len, active_indices[-1] + 1)
            
            # Ensure minimum width
            if end - start < min_nucleus_width:
                mid = (start + end) // 2
                start = max(0, mid - min_nucleus_width // 2)
                end = min(seq_len, mid + min_nucleus_width // 2)
            
            # Limit maximum width
            if end - start > max_nucleus_width:
                mid = (start + end) // 2
                start = max(0, mid - max_nucleus_width // 2)
                end = min(seq_len, mid + max_nucleus_width // 2)
            
            nucleus_points = [start, end]
        else:
            # Default to middle
            mid_point = seq_len // 2
            nucleus_points = [
                max(0, mid_point - min_nucleus_width//2),
                min(seq_len, mid_point + min_nucleus_width//2)
            ]
        
        batch_nucleus_points.append(nucleus_points)
    
    return batch_nucleus_points

def calculate_significant_axis(seqs):
    """
    Calculate which axis has maximum rotational activity
    Args:
        seqs: Tensor (batch_size, seq_len, 6)
    Returns:
        sig_axis: Tensor (batch_size,) with values 0, 1, or 2
    """
    abs_rotations = torch.abs(seqs[:, :, 3:6])  # Last 3 features are gyroscope
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)
    return sig_axis

def generate_nucleus_mask(seq_len, batch_nucleus_points):
    """
    Generate binary mask for nucleus region
    Args:
        seq_len: Length of sequence (120)
        batch_nucleus_points: List of [start, end] for each sequence
    Returns:
        nucleus_mask: Tensor (batch_size, seq_len) with 0s and 1s
    """
    batch_size = len(batch_nucleus_points)
    nucleus_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    for i, nucleus_points in enumerate(batch_nucleus_points):
        if len(nucleus_points) == 2:
            start, end = nucleus_points
            nucleus_mask[i, start:end] = 1
    
    return nucleus_mask
```

## Model Architecture

### 1. LIMU-BERT Embedder
The embedder is a transformer-based model that generates contextualized embeddings from IMU sequences.

**Model Path**: `/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt`

**Configuration**:
- Input: (batch_size, 120, 6)
- Hidden dim: Defined in model config
- Output: (batch_size, 120, hidden_dim) - sequence embeddings

### 2. GRU Classifier
The classifier takes BERT embeddings and predicts gesture class.

**Model Path**: `/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt`

**Architecture**:
```python
class ContrastiveGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
    
    def forward(self, x, return_features=False):
        gru_out, _ = self.gru(x)
        features = gru_out[:, -1, :]  # Take last hidden state
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
```

## Complete Inference Pipeline

### Python Implementation
```python
import torch
import numpy as np
from models import LIMUBertModel4Pretrain
from contrastive.models import ContrastiveGRUClassifier
from config import PretrainModelConfig, ClassifierModelConfig

class GestureRecognitionPipeline:
    def __init__(self, 
                 embedder_path,
                 classifier_path,
                 device='cpu'):
        """
        Initialize the complete pipeline
        """
        self.device = torch.device(device)
        
        # Load embedder
        self.embedder = self.load_embedder(embedder_path)
        self.embedder.eval()
        
        # Load classifier
        self.classifier = self.load_classifier(classifier_path)
        self.classifier.eval()
        
        # Initialize preprocessor
        self.preprocessor = Preprocess4Normalization(feature_len=6)
        
        # Gesture names
        self.gesture_names = [
            "horizontal right", "horizontal left", "vertical up", "vertical down",
            "clockwise circle", "counter-clockwise circle", 
            "counter-clockwise square", "clockwise square",
            "right diagonal", "left diagonal", 
            "vertical down double", "horizontal right double",
            "V-shape down-up", "V-shape up-down", 
            "triangle upward", "triangle downward",
            "S-curve leftward", "S-curve rightward", 
            "wave leftward", "wave rightward"
        ]
    
    def load_embedder(self, model_path):
        """Load LIMU-BERT embedder"""
        # Load model config (you need to read from config files)
        model_cfg = PretrainModelConfig(
            hidden=128,
            hidden_ff=256,
            feature_num=6,
            n_layers=4,
            n_heads=4,
            seq_len=120,
            emb_norm=True
        )
        
        model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def load_classifier(self, model_path):
        """Load GRU classifier"""
        model = ContrastiveGRUClassifier(
            input_dim=128,  # Should match embedder hidden_dim
            hidden_dim=128,
            num_classes=20,
            proj_dim=128
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def preprocess(self, raw_data):
        """
        Preprocess raw IMU data
        Args:
            raw_data: numpy array (seq_len, 6) or (batch_size, seq_len, 6)
        Returns:
            Preprocessed tensor
        """
        if len(raw_data.shape) == 2:
            raw_data = raw_data[np.newaxis, ...]  # Add batch dimension
        
        # Normalize
        batch_normalized = []
        for sample in raw_data:
            normalized = self.preprocessor.normalize(sample)
            batch_normalized.append(normalized)
        
        normalized_data = np.stack(batch_normalized)
        return torch.from_numpy(normalized_data).float().to(self.device)
    
    def predict(self, raw_imu_data):
        """
        Complete prediction pipeline
        Args:
            raw_imu_data: numpy array (120, 6) - single gesture sequence
        Returns:
            predicted_class: int
            predicted_gesture: str
            confidence: float
        """
        with torch.no_grad():
            # Step 1: Preprocess
            seqs = self.preprocess(raw_imu_data)
            
            # Step 2: Compute energy and features
            energy = compute_energy(seqs)
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            
            # Step 3: Generate BERT embeddings
            nucleus_mask = nucleus_mask.to(self.device)
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            embeddings = self.embedder(
                seqs, 
                nucleus_mask=nucleus_mask,
                sig_axis_mask=sig_axis_mask
            )
            
            # Step 4: Classify
            logits = self.classifier(embeddings)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Step 5: Get prediction
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            predicted_gesture = self.gesture_names[predicted_class]
            
            return predicted_class, predicted_gesture, confidence
    
    def predict_batch(self, raw_imu_batch):
        """
        Batch prediction
        Args:
            raw_imu_batch: numpy array (batch_size, 120, 6)
        Returns:
            predictions: list of (class, gesture_name, confidence)
        """
        with torch.no_grad():
            # Preprocess
            seqs = self.preprocess(raw_imu_batch)
            
            # Compute features
            energy = compute_energy(seqs)
            batch_nucleus_points = detect_nucleus(energy)
            nucleus_mask = generate_nucleus_mask(seqs.size(1), batch_nucleus_points)
            sig_axis = calculate_significant_axis(seqs)
            sig_axis_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()
            
            # Generate embeddings
            nucleus_mask = nucleus_mask.to(self.device)
            sig_axis_mask = sig_axis_mask.to(self.device)
            
            embeddings = self.embedder(
                seqs,
                nucleus_mask=nucleus_mask,
                sig_axis_mask=sig_axis_mask
            )
            
            # Classify
            logits = self.classifier(embeddings)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get predictions
            predicted_classes = torch.argmax(probabilities, dim=-1)
            confidences = torch.max(probabilities, dim=-1)[0]
            
            predictions = []
            for i in range(len(predicted_classes)):
                pred_class = predicted_classes[i].item()
                confidence = confidences[i].item()
                gesture_name = self.gesture_names[pred_class]
                predictions.append((pred_class, gesture_name, confidence))
            
            return predictions

# Usage Example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GestureRecognitionPipeline(
        embedder_path="/home/prerna/LIMU-BERT-blind-users/saved/pretrain_base_sony_watch_20_120/limu_v1.pt",
        classifier_path="/home/prerna/LIMU-BERT-blind-users/saved/classifier_contrastive_gru_sony_watch_20_120/limu_gru_v1.pt",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load sample data (replace with real streaming data)
    sample_data = np.random.randn(120, 6).astype(np.float32)
    
    # Make prediction
    pred_class, gesture, confidence = pipeline.predict(sample_data)
    print(f"Predicted: {gesture} (class {pred_class}) with confidence {confidence:.2%}")
```

## Android Implementation Strategy

### 1. Model Conversion
Convert PyTorch models to TorchScript or ONNX for mobile deployment:

```python
# Convert embedder to TorchScript
traced_embedder = torch.jit.trace(
    embedder,
    (sample_input, sample_nucleus_mask, sample_sig_axis_mask)
)
traced_embedder.save("limu_embedder_mobile.pt")

# Convert classifier to TorchScript
traced_classifier = torch.jit.trace(
    classifier,
    sample_embeddings
)
traced_classifier.save("gru_classifier_mobile.pt")
```

### 2. Android Data Flow

```
1. Bluetooth Connection
   ├─ Connect to Sony Watch
   ├─ Subscribe to IMU sensor notifications
   └─ Buffer incoming data

2. Data Collection (Sliding Window)
   ├─ Maintain circular buffer of 120 samples
   ├─ Update buffer at 20 Hz
   └─ Trigger inference when buffer is full

3. Preprocessing (Native/Java)
   ├─ Normalize accelerometer data (divide by 9.8)
   ├─ Keep gyroscope data as is
   └─ Convert to float array

4. Feature Extraction (PyTorch Mobile)
   ├─ Compute energy
   ├─ Detect nucleus
   └─ Calculate significant axis

5. Inference (PyTorch Mobile)
   ├─ Run LIMU-BERT embedder
   ├─ Run GRU classifier
   └─ Get prediction + confidence

6. Post-processing
   ├─ Apply confidence threshold
   ├─ Smooth predictions (optional)
   └─ Display result to user
```

### 3. Key Android Dependencies
```gradle
dependencies {
    implementation 'org.pytorch:pytorch_android:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'
    implementation 'com.google.android.gms:play-services-wearable:18.0.0'
}
```

## Performance Considerations

### Latency Budget
- **IMU Sampling**: 50ms per sample @ 20 Hz
- **Buffer Fill**: 6 seconds (120 samples)
- **Preprocessing**: < 10ms
- **BERT Embedding**: 50-100ms (on mobile GPU)
- **GRU Classification**: 10-20ms
- **Total Inference**: ~100ms

### Optimization Strategies
1. **Quantization**: Convert models to INT8 for faster inference
2. **Pruning**: Remove unnecessary weights
3. **Batch Processing**: Process multiple gestures in parallel
4. **Sliding Window**: Use overlap to reduce latency
5. **GPU Acceleration**: Use mobile GPU (if available)

## Testing & Validation

### Unit Tests
1. Test preprocessing normalization
2. Test nucleus detection on known patterns
3. Test model loading and inference
4. Test end-to-end pipeline with saved data

### Integration Tests
1. Test Bluetooth streaming
2. Test real-time buffer management
3. Test inference on live data
4. Measure latency and throughput

### Validation Metrics
- **Accuracy**: Match training accuracy (target: >90%)
- **Latency**: Inference time < 100ms
- **Energy**: Battery consumption per hour
- **Robustness**: Handle dropped packets, noise

## Troubleshooting Guide

### Common Issues

1. **Bluetooth Connection Drops**
   - Implement reconnection logic
   - Buffer data locally during disconnection

2. **High Latency**
   - Use quantized models
   - Reduce batch size
   - Profile bottlenecks

3. **Low Accuracy**
   - Verify preprocessing matches training
   - Check normalization constants
   - Validate nucleus detection

4. **Memory Issues**
   - Use smaller batch sizes
   - Release tensors after inference
   - Monitor memory usage

## Next Steps

1. **Export Models**: Convert to mobile-friendly format
2. **Build Android App**: Implement Bluetooth + inference
3. **Test on Device**: Validate with real Sony Watch
4. **Optimize**: Profile and improve performance
5. **Deploy**: Package and distribute app

## References
- LIMU-BERT Paper: [Link to paper]
- PyTorch Mobile: https://pytorch.org/mobile/
- Sony Watch SDK: [Link to SDK documentation]
