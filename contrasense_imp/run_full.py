import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.signal as signal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import traceback


def save_predictions_to_csv(y_true, y_pred, save_file_test_size=0.1):
    import pandas as pd
    
    # Create a DataFrame with true and predicted labels
    df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # Save to CSV
    base_filename = 'results/contrasense_results/sony_watch_.csv'
    print("save_file_test_size", save_file_test_size)
    # add save_file_test_size to the filename just before the extension
    filename = base_filename.replace('.csv', f'{int(100-(save_file_test_size*100))}.csv')
    df.to_csv(filename, index=False)
    print(f"Predictions and true labels saved to {filename}")


# ProjectionHead class needed for Contrastive models
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.projection(x)

# ContrastiveBiLSTMAttentionClassifier model
class ContrastiveBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Attention mechanism
        att_weights = self.attention(lstm_out)  # [batch, seq, 1]
        att_weights = F.softmax(att_weights, dim=1)
        features = torch.sum(lstm_out * att_weights, dim=1)  # [batch, hidden]
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits

# 1. Load the model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

# 2. Resample IMU data from 120 to 20 timesteps
def resample_data(data, original_len=120, target_len=20):
    n, seq_len, feat_dim = data.shape
    resampled_data = np.zeros((n, target_len, feat_dim))
    
    for i in range(n):
        for j in range(feat_dim):
            resampled_data[i, :, j] = signal.resample(data[i, :, j], target_len)
    
    return resampled_data

# 3. Generate embeddings using the loaded model
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

# 4. Extract labels and split data
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

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, best_accuracy, label_encoder, filename):
    """
    Save model checkpoint including model state, optimizer state, epoch, and best accuracy
    """
    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir) and checkpoint_dir:
        os.makedirs(checkpoint_dir)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'label_encoder_classes': label_encoder.classes_
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

# Load checkpoint
def load_checkpoint(model, optimizer, filename, label_encoder=None):
    """
    Load model checkpoint including model state, optimizer state, epoch, and best accuracy
    """
    if not os.path.isfile(filename):
        print(f"No checkpoint found at {filename}")
        return model, optimizer, 0, 0.0, label_encoder
    
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    
    # Restore label encoder if provided
    if label_encoder is not None and 'label_encoder_classes' in checkpoint:
        label_encoder.classes_ = checkpoint['label_encoder_classes']
    
    print(f"Loaded checkpoint from {filename} (epoch {epoch})")
    return model, optimizer, epoch, best_accuracy, label_encoder

# Load a pre-trained model following ContrastSense structure
def load_pretrained_model(model, pretrained_path, model_type='ContrastSense', classifier_name=None):
    """
    Load weights from a pre-trained model with handling for ContrastSense model structure
    
    Args:
        model: The model to load weights into
        pretrained_path: Path to the checkpoint file
        model_type: Model type to determine key renaming strategy
        classifier_name: List of classifier keys to verify missing keys
    """
    if not os.path.isfile(pretrained_path):
        print(f"No pre-trained model found at {pretrained_path}")
        return model
    
    try:
        # Load checkpoint
        print(f"Loading pre-trained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            print("Checkpoint format doesn't contain 'state_dict' key")
            # Try other common formats or use the checkpoint directly
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
        
        # Debug: Print keys before modification
        print(f"Original state_dict keys: {list(state_dict.keys())[:5]}...")
        
        # Handle ContrastSense specific key renaming
        if model_type == 'ContrastSense':
            new_state_dict = {}
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q'):
                    # remove prefix
                    new_key = k[len("encoder_q."):]
                    new_state_dict[new_key] = state_dict[k]
                # Don't delete original keys yet as we're building a new dict
            
            if new_state_dict:  # Only replace if we found encoder_q keys
                print(f"Renamed {len(new_state_dict)} keys from encoder_q prefix")
                state_dict = new_state_dict
        
        # Load state dict with strict=False to allow missing keys
        log = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained model from {pretrained_path}")
        print(f"Missing keys: {log.missing_keys}")
        print(f"Unexpected keys: {log.unexpected_keys}")
        
        # Verify missing keys match classifier name if provided
        if classifier_name and log.missing_keys:
            if set(log.missing_keys) == set(classifier_name):
                print("Missing keys match expected classifier keys")
            else:
                print("WARNING: Missing keys don't match expected classifier keys")
                print(f"Expected: {classifier_name}")
                print(f"Actual missing: {log.missing_keys}")
        
        return model
    
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        traceback.print_exc()
        return model

# 5. Train the BiLSTM classifier with checkpointing
def train_lstm_classifier(X_train, y_train, X_test, y_test, input_dim=6, hidden_dim=64, num_classes=None, 
                         save_file_test_size=0.9, checkpoint_dir='checkpoints',
                         resume_from_checkpoint=None, checkpoint_freq=10, model_best_path='model_best.pth'):
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Determine number of classes
    if num_classes is None:
        num_classes = len(np.unique(y_train))
    
    print(f"Training BiLSTM classifier with {num_classes} classes")
    
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
    
    # Initialize model - ContrastiveBiLSTMAttentionClassifier
    model = ContrastiveBiLSTMAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Use CPU to avoid CUDA errors
    device = torch.device("cpu")
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Setup for training with checkpoints
    start_epoch = 0
    best_accuracy = 0.0
    
    # Create a dummy label encoder for checkpointing
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Load checkpoint if resuming training
    if resume_from_checkpoint:
        if os.path.isfile(resume_from_checkpoint):
            model, optimizer, start_epoch, best_accuracy, label_encoder = load_checkpoint(
                model, optimizer, resume_from_checkpoint, label_encoder
            )
            print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")
        else:
            print(f"No checkpoint found at {resume_from_checkpoint}, starting from scratch")
    
    # Training loop
    num_epochs = 100
    best_model_state = None
    
    try:
        for epoch in range(start_epoch, num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            epoch_start_time = time.time()
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            epoch_time = time.time() - epoch_start_time
            
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
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_accuracy, label_encoder, checkpoint_path)
            
            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                
                # Save best model checkpoint
                save_checkpoint(model, optimizer, epoch + 1, best_accuracy, label_encoder, model_best_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.1f}s")
    
    except Exception as e:
        print(f"Error during training: {e}")
        # Save emergency checkpoint if exception occurs
        emergency_path = os.path.join(checkpoint_dir, 'emergency_checkpoint.pth')
        save_checkpoint(model, optimizer, epoch + 1, best_accuracy, label_encoder, emergency_path)
        print(f"Emergency checkpoint saved to {emergency_path}")
        
        if best_model_state is not None:
            print("Recovering from best saved model state")
            model.load_state_dict(best_model_state)
        else:
            print("No saved model state available, returning untrained model")
            return model, 0.0
    
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - BiLSTM Attention Classifier')
    plt.savefig('confusion_matrix_lstm.png')
    
    print(f"Final Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model, accuracy

# Main execution function
def prepare_data_and_train_lstm(imu_data_path, labels_path, resume_from=None, model_best_path='model_best.pth', 
                              use_pretrained=True, pretrained_model_type='ContrastSense'):
    """
    Prepare data and train BiLSTM with optional pre-trained model loading
    
    Args:
        imu_data_path: Path to IMU data file
        labels_path: Path to labels file
        resume_from: Path to checkpoint to resume from (can be None)
        model_best_path: Path to save best model
        use_pretrained: Whether to use pre-trained weights
        pretrained_model_type: Type of pre-trained model for key renaming
    """
    # Load and resample IMU data
    print("Loading and resampling IMU data...")
    imu_data = np.load(imu_data_path)  # Shape: (n, 120, 6)
    resampled_data = resample_data(imu_data)  # Resample to (n, 20, 6)
    
    # For simplicity, let's use the resampled data directly
    embeddings = resampled_data
    
    # Split into train and test sets
    for test_size in [0.9, 0.7, 0.5, 0.2]:
        print(f"\n==========================================")
        print(f"Preparing data with test size: {test_size}")
        print(f"==========================================")
        
        checkpoint_dir = f'checkpoints/test_size_{int(test_size*100)}'
        test_specific_model_best = f'model_best_test_size_{int(test_size*100)}.pth'
        
        # Create test-size specific resume path if provided
        test_specific_resume = None
        if resume_from:
            test_specific_resume = os.path.join(checkpoint_dir, os.path.basename(resume_from))
            if not os.path.exists(test_specific_resume):
                test_specific_resume = resume_from
        
        X_train, X_test, y_train, y_test, label_encoder = prepare_train_test_split(
            embeddings, labels_path, test_size=test_size
        )

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Train the BiLSTM classifier
        input_dim = embeddings.shape[2]  # Should be 6 for IMU data
        num_classes = len(label_encoder.classes_)
        
        # Initialize the model for this test size
        lstm_model = ContrastiveBiLSTMAttentionClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=num_classes
        )
        
        # Load pre-trained weights if specified
        if use_pretrained and os.path.isfile(model_best_path):
            print(f"Loading pre-trained weights from {model_best_path}...")
            # Define classifier keys for verification
            classifier_keys = ['classifier.weight', 'classifier.bias']
            
            # Load pre-trained weights
            lstm_model = load_pretrained_model(
                lstm_model,
                model_best_path,
                model_type=pretrained_model_type,
                classifier_name=classifier_keys
            )
        
        print(f"Starting BiLSTM training with {num_classes} classes and input dimension {input_dim}")
        
        # Now train with the pre-loaded model
        trained_model, accuracy = train_lstm_classifier(
            X_train, y_train, X_test, y_test, 
            input_dim=input_dim, 
            hidden_dim=64,
            num_classes=num_classes,
            save_file_test_size=test_size,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=test_specific_resume,
            model_best_path=test_specific_model_best
        )
        
        print(f"BiLSTM training completed with accuracy: {accuracy:.4f}")
        
        # Delete the model and free up memory
        del lstm_model
        del trained_model
        
        # Force garbage collection
        import gc
        gc.collect()

    print("All training runs completed!")

if __name__ == "__main__":
    # Update these paths to your actual file locations
    imu_data_path = 'dataset/sony_watch/data_20_120.npy'
    label_file_path = 'dataset/sony_watch/label_20_120.npy'
    
    # Path to your existing model_best.pth file
    pretrained_model_path = 'model_best.pth'
    
    # Option 1: Initialize a new model with pre-trained weights
    print("\nOption 1: Initialize a new model with pre-trained weights")
    # Create a new model with appropriate dimensions
    input_dim = 6  # For IMU data with 6 features
    hidden_dim = 64
    num_classes = 10  # Change this to match your dataset
    
    model = ContrastiveBiLSTMAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Define classifier keys for verification
    classifier_keys = ['classifier.weight', 'classifier.bias']
    
    # Load pre-trained weights using ContrastSense key renaming
    model = load_pretrained_model(
        model,
        pretrained_model_path,
        model_type='ContrastSense',
        classifier_name=classifier_keys
    )
    
    print(f"Successfully initialized model with pre-trained weights from {pretrained_model_path}")
    
    # Option 2: Run full pipeline with pre-trained weights
    print("\nOption 2: Run full training pipeline")
    prepare_data_and_train_lstm(
        imu_data_path, 
        label_file_path,
        resume_from=None,  # Start from scratch but with pre-trained weights
        model_best_path=pretrained_model_path,
        use_pretrained=True,
        pretrained_model_type='ContrastSense'
    )
    
    print("Pipeline completed successfully!")