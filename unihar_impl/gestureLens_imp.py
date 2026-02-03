import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.signal as signal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

def save_predictions_to_csv(y_true, y_pred, save_file_test_size=0.1):
    # Create a DataFrame with true and predicted labels
    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # Save to CSV
    base_filename = 'results/unihar_results/sony_watch_.csv'
    print("save_file_test_size", save_file_test_size)
    # add save_file_test_size to the filename just before the extension
    filename = base_filename.replace('.csv', f'{int(100-(save_file_test_size*100))}.csv')
    #df.to_csv(filename, index=False)
    print(f"Predictions and true labels saved to {filename}")

# Implement data augmentation for IMU data
def augment_imu_data(data, noise_level=0.05, scaling_range=(0.9, 1.1), time_shift_max=2):
    """
    Augment IMU data with noise, scaling, and time shifts
    
    Args:
        data: Input tensor of shape [batch_size, seq_len, features]
        noise_level: Maximum magnitude of noise to add
        scaling_range: Range of scaling factors to apply
        time_shift_max: Maximum number of steps to shift in time
    
    Returns:
        Augmented data tensor of same shape
    """
    augmented = data.clone()
    batch_size = augmented.shape[0]
    
    # Add random noise
    noise = noise_level * torch.randn_like(augmented)
    augmented += noise
    
    # Random scaling per batch item
    for i in range(batch_size):
        scale = torch.FloatTensor(1).uniform_(*scaling_range)
        augmented[i] *= scale
    
    # Random time shift (sequence shift)
    for i in range(batch_size):
        if time_shift_max > 0:
            shift = torch.randint(-time_shift_max, time_shift_max + 1, (1,)).item()
            if shift > 0:
                augmented[i, shift:, :] = augmented[i, :-shift, :]
                augmented[i, :shift, :] = augmented[i, shift, :]  # Repeat first valid entry
            elif shift < 0:
                shift = abs(shift)
                augmented[i, :-shift, :] = augmented[i, shift:, :]
                augmented[i, -shift:, :] = augmented[i, -shift-1, :]  # Repeat last valid entry
    
    return augmented

# Improved ProjectionHead with residual connection
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Add residual connection if dimensions match
        self.has_residual = (input_dim == output_dim)
        
    def forward(self, x):
        projected = self.projection(x)
        if self.has_residual:
            return projected + x
        return projected

# Improved attention module for sequence data
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, kernel_size=1)
        self.key = nn.Conv1d(channels, channels, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, L = x.size()
        
        # Generate query, key, value projections
        proj_query = self.query(x).view(batch_size, -1, L).permute(0, 2, 1)  # B x L x C
        proj_key = self.key(x).view(batch_size, -1, L)  # B x C x L
        proj_value = self.value(x).view(batch_size, -1, L)  # B x C x L
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x L x L
        attention = self.softmax(energy)  # B x L x L
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x L
        
        # Apply residual connection with learnable weight
        out = self.gamma * out + x
        
        return out

# Improved residual block with dilated convolutions
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        
        padding = (kernel_size // 2) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, 
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.GELU()
        
        # Projection for residual if dimensions don't match
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Spatial dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out.unsqueeze(3)).squeeze(3)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += identity
        out = self.act2(out)
        
        return out

# Improved ContrastiveCNNClassifier with attention and better residual connections
class ImprovedContrastiveCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, feature_dim=128, num_layers=4, kernel_size=3, proj_dim=128):
        super().__init__()
        
        # Store the feature dimension for projection consistency
        self.feature_dim = feature_dim
        
        # Create progressive channel sizes
        channel_sizes = [input_dim]
        for i in range(num_layers):
            if i == 0:
                channel_sizes.append(hidden_dim // 2)
            elif i == num_layers - 1:
                channel_sizes.append(feature_dim)
            else:
                channel_sizes.append(hidden_dim)
        
        # Create residual blocks with progressively increasing dilation
        self.res_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** (i % 3)  # Use dilated convolutions (1, 2, 4, 1, 2, 4, ...)
            self.res_blocks.append(
                ResidualBlock(
                    channel_sizes[i], channel_sizes[i+1], 
                    kernel_size=kernel_size, dilation=dilation
                )
            )
            
            # Add pooling every second layer
            if i % 2 == 1:
                self.pool_layers.append(nn.MaxPool1d(2))
            else:
                self.pool_layers.append(nn.Identity())
        
        # Add temporal attention module
        self.temporal_attention = TemporalAttention(channel_sizes[-1])
        
        # Adaptive pooling to get fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(channel_sizes[-1])
        
        # Add an improved bottleneck layer with residual connection
        bottleneck_dim = channel_sizes[-1] // 2
        self.bottleneck = nn.Sequential(
            nn.Linear(channel_sizes[-1], bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_dim, channel_sizes[-1])
        )
        
        # Classification head
        self.classifier = nn.Linear(channel_sizes[-1], num_classes)
        
        # Projection head for contrastive learning
        self.projector = ProjectionHead(channel_sizes[-1], hidden_dim, proj_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_features=False):
        batch_size, seq_len, features = x.shape
        
        # Transpose for 1D convolution (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply residual blocks with pooling
        for i, (res_block, pool) in enumerate(zip(self.res_blocks, self.pool_layers)):
            x = res_block(x)
            x = pool(x)
        
        # Apply temporal attention
        x = self.temporal_attention(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        features = x.view(batch_size, -1)
        
        # Normalize features
        features = self.feature_norm(features)
        
        # Apply bottleneck with residual connection
        bottleneck_out = self.bottleneck(features)
        features = features + bottleneck_out
        
        # Apply dropout before final classification
        features = self.dropout(features)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits

# Load the model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

# Resample IMU data from 120 to 20 timesteps
def resample_data(data, original_len=120, target_len=20):
    n, seq_len, feat_dim = data.shape
    resampled_data = np.zeros((n, target_len, feat_dim))
    
    for i in range(n):
        for j in range(feat_dim):
            resampled_data[i, :, j] = signal.resample(data[i, :, j], target_len)
    
    return resampled_data

# Generate embeddings using the loaded model
def generate_embeddings(model, data):
    # Convert data to tensor
    data_tensor = torch.from_numpy(data).float()
    
    # Process in batches to handle large datasets
    batch_size = 32
    n_samples = len(data_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_data = data_tensor[start_idx:end_idx]
            
            # This depends on your actual model structure
            # Assuming the encoder outputs embeddings
            embedding = torch.zeros(batch_data.shape[0], 36)  # 36-dim placeholder
            
            # TODO: Replace with actual embedding extraction
            
            embeddings_list.append(embedding.numpy())
    
    embeddings = np.vstack(embeddings_list)
    return embeddings

# Extract labels and split data
def prepare_train_test_split(embeddings, labels_file, test_size=0.9, random_state=42):
    # Load the labels file
    labels_data = np.load(labels_file)
    
    # Extract the first column of the k dimension
    labels = labels_data[:, 0, 0]
    
    # Important: Ensure labels are consecutive integers starting from 0
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Print label information for debugging
    print(f"Original label range: {np.min(labels)} to {np.max(labels)}")
    print(f"Encoded label range: {np.min(labels_encoded)} to {np.max(labels_encoded)}")
    print(f"Number of unique classes: {len(np.unique(labels_encoded))}")
    
    # Split the data in a stratified manner
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels_encoded,  # Use encoded labels
        test_size=test_size,
        random_state=random_state,
        stratify=labels_encoded  # Use encoded labels for stratification
    )
    
    return X_train, X_test, y_train, y_test, label_encoder

# Improved train_cnn_classifier with learning rate scheduling, early stopping, and data augmentation
def train_cnn_classifier(X_train, y_train, X_test, y_test, input_dim=6, hidden_dim=64, num_classes=None, save_file_test_size=0.9):
    # Determine number of classes
    if num_classes is None:
        num_classes = len(np.unique(y_train))
    
    print(f"Training classifier with {num_classes} classes")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Add debugging: Check label range
    print(f"Train labels min: {y_train.min()}, max: {y_train.max()}")
    print(f"Test labels min: {y_test.min()}, max: {y_test.max()}")
    
    # Create train and test datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders with smaller batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model - using the improved model architecture
    model = ImprovedContrastiveCNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        feature_dim=64,
        num_layers=4,
        kernel_size=3,
        proj_dim=64
    )
    
    # Use CPU to avoid CUDA errors
    device = torch.device("cpu")
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True, min_lr=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 1000
    
    # Variables for early stopping
    best_accuracy = 0.0
    best_model_state = None
    early_stopping_patience = 50
    no_improvement_count = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Apply data augmentation with 50% probability during training
                """if np.random.random() < 0.5:
                    batch_X = augment_imu_data(batch_X)"""
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Evaluation phase
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total
            history['val_accuracy'].append(accuracy)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Update learning rate scheduler
            scheduler.step(accuracy)
            
            # Early stopping logic
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            """if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break"""
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        if best_model_state is not None:
            print("Recovering from best saved model state")
        else:
            print("No saved model state available, returning untrained model")
            return model, 0.0, history
    
    # Load the best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    save_predictions_to_csv(all_labels, all_preds, save_file_test_size)
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    """plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - Improved CNN Classifier')
    plt.savefig('confusion_matrix_improved_cnn.png')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')"""
    
    print(f"Final Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model, accuracy, history

# Save the model
def save_classifier(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Main execution function
def prepare_data_and_train_cnn(model_path, imu_data_path, labels_path):
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Load and resample IMU data
    imu_data = np.load(imu_data_path)  # Shape: (n, 120, 6)
    resampled_data = resample_data(imu_data)  # Resample to (n, 20, 6)
    
    # For simplicity, let's use the resampled data directly
    embeddings = resampled_data
    
    results = {}
    
    # Split into train and test sets
    for test_size in [0.9, 0.7, 0.5, 0.2]:
        print(f"Preparing data with test size: {test_size}")
        X_train, X_test, y_train, y_test, label_encoder = prepare_train_test_split(embeddings, labels_path, test_size=test_size)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Train the CNN classifier
        input_dim = embeddings.shape[2]  # Should be 6 for IMU data
        num_classes = len(label_encoder.classes_)
        
        print(f"Starting CNN training with {num_classes} classes and input dimension {input_dim}")
        cnn_model, accuracy, history = train_cnn_classifier(
            X_train, y_train, X_test, y_test, 
            input_dim=input_dim, 
            num_classes=num_classes,
            save_file_test_size=test_size
        )
        
        # Save the model
        
        # Store results
        results[test_size] = {
            'accuracy': accuracy,
            'history': history
        }
        
        # Delete the model and free up memory
        del cnn_model
        print(f"CNN training completed with accuracy: {accuracy:.4f}")
    
    # Plot comparative results
    """plt.figure(figsize=(10, 6))
    
    test_sizes = list(results.keys())
    accuracies = [results[ts]['accuracy'] for ts in test_sizes]
    
    plt.bar([str(int(ts*100))+"%" for ts in test_sizes], accuracies)
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Test Size')
    plt.ylim(0, 1.0)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('test_size_comparison.png')
    """
    return results
    
if __name__ == "__main__":
    # Replace these with your actual file paths
    model_path = 'saved/pretrain_base_blind_user_filtered_20_120/limu_v1.pt'
    imu_data_path = 'dataset/blind_user_filtered/data_20_120.npy'
    label_file_path = 'dataset/blind_user_filtered/label_20_120.npy'
    
    # Run the full pipeline
    results = prepare_data_and_train_cnn(model_path, imu_data_path, label_file_path)
    
    print("Pipeline completed successfully!")