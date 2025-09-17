#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

class UserEmbeddingDataset(Dataset):
    """Dataset for user embeddings with label normalization"""
    def __init__(self, embeddings, labels, transform=None):
        # Ensure labels are contiguous integers starting from 0
        self.label_encoder = LabelEncoder()
        self.original_labels = labels
        self.normalized_labels = self.label_encoder.fit_transform(labels)
        
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.normalized_labels, dtype=torch.long)
        self.transform = transform
        
        # Print label mapping for reference
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Label mapping (original → normalized):")
        for i, original in enumerate(self.label_encoder.classes_):
            print(f"  {original} → {i}")
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
            
        return embedding, label

class UserClassifier(nn.Module):
    """Simple classifier for user authentication"""
    def __init__(self, input_dim, hidden_dim, num_users, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_users)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_user_classifier(embeddings_file, labels_file, output_dir='auth_model', 
                         batch_size=32, epochs=50, learning_rate=0.001, 
                         hidden_dim=128, dropout=0.3, gpu=0):
    """Train a user authentication model using pre-generated embeddings"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load embeddings and labels
    try:
        embeddings = np.load(embeddings_file)
        labels = np.load(labels_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print(f"Make sure the path is correct: {embeddings_file}")
        return None, None, None, None, None
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Loaded labels with shape: {labels.shape}")
    
    # Check for any NaN or infinity values in embeddings
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print("Warning: Embeddings contain NaN or infinity values. Replacing with zeros.")
        embeddings = np.nan_to_num(embeddings)
    
    # Create dataset with normalized labels
    dataset = UserEmbeddingDataset(embeddings, labels)
    num_users = dataset.num_classes
    print(f"Using {num_users} normalized user classes")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Use fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create model
    input_dim = embeddings.shape[1]  # Embedding dimension
    model = UserClassifier(input_dim, hidden_dim, num_users, dropout=dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    train_losses = []
    val_accuracies = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (embeddings, labels) in enumerate(train_loader):
            try:
                embeddings, labels = embeddings.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Print occasional batch updates
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
                print(f"Labels: {labels}")
                continue
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                
                outputs = model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pt'))
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Calculate precision, recall, and F1 score with zero_division=0 to avoid warnings
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Get unique classes in the test set
    unique_test_classes = np.unique(all_labels)
    
    # Use only the classes that appear in the test set for the plot
    num_test_classes = len(unique_test_classes)
    tick_marks = np.arange(num_test_classes)
    plt.xticks(tick_marks, unique_test_classes)
    plt.yticks(tick_marks, unique_test_classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True User')
    plt.xlabel('Predicted User')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save results to text file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        # Calculate per-class metrics with zero_division=0
        try:
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=None, zero_division=0
            )
            
            # Only report metrics for classes that actually appear in the test set
            f.write("Per-class metrics (classes present in test set):\n")
            for idx, class_idx in enumerate(unique_test_classes):
                # Map back to original label for reporting
                original_label = dataset.label_encoder.inverse_transform([class_idx])[0]
                f.write(f"User {original_label}:\n")
                f.write(f"  Precision: {class_precision[idx]:.4f}\n")
                f.write(f"  Recall: {class_recall[idx]:.4f}\n")
                f.write(f"  F1 Score: {class_f1[idx]:.4f}\n\n")
        except Exception as e:
            f.write(f"Error calculating per-class metrics: {e}\n")
    
    print(f"Results saved to {output_dir}")
    
    return model, test_acc, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train user authentication model from embeddings')
    
    parser.add_argument('--embeddings_file', type=str, required=True,
                        help='Path to embeddings file (.npy)')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to labels file (.npy)')
    parser.add_argument('--output_dir', type=str, default='auth_model',
                        help='Directory to save model and results (default: auth_model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    train_user_classifier(
        embeddings_file=args.embeddings_file,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gpu=args.gpu
    )