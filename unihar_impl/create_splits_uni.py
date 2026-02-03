import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.signal as signal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def save_predictions_to_csv(y_true, y_pred, save_file_test_size=0.1):
    import pandas as pd
    
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
    df.to_csv(filename, index=False)
    print(f"Predictions and true labels saved to {filename}")


# ProjectionHead class needed for ContrastiveCNNClassifier
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

# ContrastiveCNNClassifier from the provided code
class ContrastiveCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, feature_dim=128, num_layers=3, kernel_size=3, proj_dim=128):
        super().__init__()
        
        # Store the feature dimension for projection consistency
        self.feature_dim = feature_dim
        
        # Reduce model capacity by decreasing channel growth rate
        channel_sizes = [input_dim, hidden_dim//2, hidden_dim, feature_dim]
        
        # Conv layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Add main conv block
            self.conv_layers.append(
                nn.Conv1d(channel_sizes[i], channel_sizes[i+1], 
                         kernel_size=kernel_size, padding=kernel_size//2)
            )
            # Use Group Normalization instead of Batch Normalization
            self.bn_layers.append(nn.GroupNorm(8, channel_sizes[i+1]))
            
            # Add downsampling layer (pool every other layer)
            if i % 2 == 1:
                self.downsample_layers.append(nn.MaxPool1d(2))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # Adaptive pooling to get fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(channel_sizes[-1])
        
        # Add an additional bottleneck layer before classification
        self.bottleneck = nn.Sequential(
            nn.Linear(channel_sizes[-1], channel_sizes[-1]//2),
            #nn.ReLU(),
            #nn.LayerNorm(channel_sizes[-1]//2),
            nn.Linear(channel_sizes[-1]//2, channel_sizes[-1])
        )
        
        # Classification head
        self.classifier = nn.Linear(channel_sizes[-1], num_classes)
        
        # Projection head for contrastive learning
        self.projector = ProjectionHead(channel_sizes[-1], hidden_dim, proj_dim)
        
        # Increase dropout for stronger regularization
        self.dropout = nn.Dropout(0.5)
        
        # Add spatial dropout for feature maps
        self.spatial_dropout = nn.Dropout2d(0.3)
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, features = x.shape
        
        # Transpose for 1D convolution (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers with residual connections where possible
        for i, (conv, bn, downsample) in enumerate(zip(self.conv_layers, self.bn_layers, self.downsample_layers)):
            # Store input for residual connection
            residual = x if x.size(1) == conv.out_channels else None
            
            # Apply convolution
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            
            # Apply spatial dropout to feature maps
            if i < len(self.conv_layers) - 1:  # Skip last layer
                x = self.spatial_dropout(x.unsqueeze(3)).squeeze(3)
            
            # Apply residual connection if available
            if residual is not None:
                x = x + residual
            
            # Apply downsampling and dropout
            x = downsample(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        features = x.view(batch_size, -1)
        
        # Normalize features
        features = self.feature_norm(features)
        
        # Apply bottleneck with skip connection for better feature learning
        features = features + self.bottleneck(features)
        
        # Apply dropout before final classification
        features = self.dropout(features)
        
        # Classification
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
    batch_size = 16
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

# 5. Train the CNN classifier
# 5. Train the CNN classifier
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
    
    # Initialize model
    model = ContrastiveCNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        feature_dim=64,
        num_layers=3,
        kernel_size=3,
        proj_dim=64
    )
    
    # Use CPU to avoid CUDA errors
    device = torch.device("cpu")
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 100
    
    best_accuracy = 0.0
    best_model_state = None
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
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
            
            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        if best_model_state is not None:
            print("Recovering from best saved model state")
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
    plt.title('Confusion Matrix - CNN Classifier')
    plt.savefig('confusion_matrix_cnn.png')
    
    print(f"Final Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model, accuracy  # Make sure this return statement is included



# 6. Save the model
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
    
    # Split into train and test sets
    for test_size in [0.9, 0.7, 0.5, 0.2]:
        print(f"Preparing data with test size: {test_size}")
        X_train, X_test, y_train, y_test, label_encoder = prepare_train_test_split(embeddings, labels_path, test_size=test_size)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Train the CNN classifier
        input_dim = embeddings.shape[2]  # Should be 6 for IMU data
        num_classes = len(label_encoder.classes_)
        
        print(f"Starting CNN training with {num_classes} classes and input dimension {input_dim}")
        cnn_model, accuracy = train_cnn_classifier(X_train, y_train, X_test, y_test, 
                                                input_dim=input_dim, 
                                                num_classes=num_classes,
                                                save_file_test_size=test_size)
        # delete the model and free up memory to remmove variables from the memory
        del cnn_model


        print(f"CNN training completed with accuracy: {accuracy:.4f}")
        
    #return cnn_model, (X_train, X_test, y_train, y_test)

    
if __name__ == "__main__":
    # Replace these with your actual file paths
    model_path = 'unihar_impl/unihar_bert_fed_d1.pt'
    imu_data_path = 'dataset/sony_watch/data_20_120.npy'
    label_file_path = 'dataset/sony_watch/label_20_120.npy'

    
    # Run the full pipeline
    prepare_data_and_train_cnn(model_path, imu_data_path, label_file_path)
    
    
    print("Pipeline completed successfully!")
