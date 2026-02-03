import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import os
import json
from collections import namedtuple, defaultdict
import random
from typing import Dict, List, Tuple

# Define the models
class ClassifierLSTM(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('lstm' + str(i), 
                                nn.LSTM(input, cfg.rnn_io[i][1], 
                                       num_layers=cfg.num_layers[i], 
                                       batch_first=True))
            else:
                self.__setattr__('lstm' + str(i),
                                nn.LSTM(cfg.rnn_io[i][0], cfg.rnn_io[i][1], 
                                       num_layers=cfg.num_layers[i],
                                       batch_first=True))
            self.__setattr__('bn' + str(i), nn.BatchNorm1d(cfg.seq_len))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear
    
    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            lstm = self.__getattr__('lstm' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h, _ = lstm(h)
            if self.activ:
                h = torch.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = torch.nn.functional.dropout(h, 0.5, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = torch.relu(h)
        return h

class ClassifierGRU(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i), 
                                nn.GRU(input, cfg.rnn_io[i][1], 
                                       num_layers=cfg.num_layers[i], 
                                       batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], 
                                       num_layers=cfg.num_layers[i],
                                       batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear
    
    def forward(self, input_seqs, training=False):
        h = input_seqs
        for i in range(self.num_rnn):
            gru = self.__getattr__('gru' + str(i))
            h, _ = gru(h)
            if self.activ:
                h = torch.relu(h)
        h = h[:, -1, :]
        if self.dropout:
            h = torch.nn.functional.dropout(h, 0.5, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = torch.relu(h)
        return h


class BenchmarkDCNN(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(50, 40, (5, 1))
        self.bn2 = nn.BatchNorm2d(40)
        if cfg.seq_len <= 20:
            self.conv3 = nn.Conv2d(40, 20, (2, 1))
        else:
            self.conv3 = nn.Conv2d(40, 20, (3, 1))
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d((2, 1))
        
        # We'll use adaptive pooling to get a fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, input))
        self.lin1 = nn.Linear(20 * input, 400)
        self.lin2 = nn.Linear(400, output)

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        h = torch.relu(torch.tanh(self.conv1(h)))
        h = self.bn1(self.pool(h))
        h = torch.relu(torch.tanh(self.conv2(h)))
        h = self.bn2(self.pool(h))
        h = torch.relu(torch.tanh(self.conv3(h)))
        
        # Use adaptive pooling to get fixed size regardless of input
        h = self.adaptive_pool(h)
        
        # Flatten to fixed size
        h = h.view(h.size(0), -1)
        h = self.lin1(h)
        h = torch.relu(torch.tanh(h))
        h = self.lin2(h)
        return h

class BenchmarkDeepSense(nn.Module):
    def __init__(self, cfg, input=None, output=None, num_filter=8):
        super().__init__()
        self.sensor_num = input // 3
        for i in range(self.sensor_num):
            self.__setattr__('conv' + str(i) + "_1", nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__('conv' + str(i) + "_2", nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__('conv' + str(i) + "_3", nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__('bn' + str(i) + "_1", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_2", nn.BatchNorm2d(num_filter))
            self.__setattr__('bn' + str(i) + "_3", nn.BatchNorm2d(num_filter))
        self.conv1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, (2, 1))
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.flatten = nn.Flatten()
        
        # Add adaptive pooling for fixed output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        
        self.lin1 = nn.Linear(128 * num_filter, 12)
        self.lin2 = nn.Linear(12, output)

    def forward(self, input_seqs, training=False):
        # Reshape handling for any sequence length
        if input_seqs.size(2) % 3 != 0:
            # Pad if needed
            pad_size = 3 - (input_seqs.size(2) % 3)
            input_seqs = torch.cat([input_seqs, 
                                  torch.zeros(input_seqs.size(0), input_seqs.size(1), pad_size, 
                                             device=input_seqs.device)], dim=2)
        
        self.sensor_num = input_seqs.size(2) // 3
        h = input_seqs.view(input_seqs.size(0), input_seqs.size(1), self.sensor_num, 3)
        
        hs = []
        for i in range(self.sensor_num):
            if i >= self.sensor_num:
                break
            t = h[:, :, i, :]
            t = torch.unsqueeze(t, 1)
            for j in range(3):
                cv = self.__getattr__('conv' + str(i % 2) + "_" + str(j + 1))  # Reuse weights beyond 2 sensors
                bn = self.__getattr__('bn' + str(i % 2) + "_" + str(j + 1))
                t = bn(torch.relu(cv(t)))
            hs.append(self.flatten(t)[:, :, None])
        
        h = torch.cat(hs, dim=2)
        h = h.unsqueeze(1)
        h = self.bn1(torch.relu(self.conv1(h)))
        h = self.bn2(torch.relu(self.conv2(h)))
        h = self.bn3(torch.relu(self.conv3(h)))
        
        # Use adaptive pooling to ensure fixed size output
        h = h.view(h.size(0), h.size(1), -1)
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        
        h = self.lin1(h)
        h = torch.relu(h)
        h = self.lin2(h)
        return h
    
class ClassifierAttn(nn.Module):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        self.embd = nn.Embedding(cfg.seq_len, input)
        self.proj_q = nn.Linear(input, cfg.atten_hidden)
        self.proj_k = nn.Linear(input, cfg.atten_hidden)
        self.proj_v = nn.Linear(input, cfg.atten_hidden)
        self.attn = nn.MultiheadAttention(cfg.atten_hidden, cfg.num_head)
        
        # Calculate linear layer input size based on sequence length
        linear_input_size = cfg.seq_len * cfg.atten_hidden
        
        # Adjust linear layers to match the actual input size
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(linear_input_size, output))
            else:
                if i == 0:
                    self.__setattr__('lin' + str(i), nn.Linear(linear_input_size, cfg.linear_io[i][1]))
                else:
                    self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i-1][1], cfg.linear_io[i][1]))
        
        self.flatten = nn.Flatten()
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_linear = cfg.num_linear
        self.seq_len = cfg.seq_len

    def forward(self, input_seqs, training=False):
        seq_len = input_seqs.size(1)
        
        # Handle sequence length mismatches
        if seq_len > self.seq_len:
            # If input is longer than expected, truncate
            input_seqs = input_seqs[:, :self.seq_len, :]
            seq_len = self.seq_len
        
        pos = torch.arange(seq_len, dtype=torch.long, device=input_seqs.device)
        pos = pos.unsqueeze(0).expand(input_seqs.size(0), seq_len)  # (S,) -> (B, S)
        h = input_seqs + self.embd(pos)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        h, weights = self.attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        h = h.transpose(0, 1)
        
        if self.dropout:
            h = torch.nn.functional.dropout(h, 0.5, training=training)
        
        h = self.flatten(h)
        
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = torch.relu(h)
        
        return h
    

# Convert dictionary to object for easier access
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Function to load and preprocess data
def load_and_preprocess_data(data_path, labels_path):
    print(f"Loading data from {data_path}")
    # Load data and labels
    X = np.load(data_path)
    y = np.load(labels_path)
    X = X[:,:,:6]
    
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Extract only the first column of labels
    y_processed = y[:, 0, 0]  # Assuming first column for all 120 timesteps has same label
    unique_labels = np.unique(y_processed)
    label_map = {old_label: i for i, old_label in enumerate(unique_labels)}
    y_processed = np.array([label_map[label] for label in y_processed])
    
    print(f"Mapped labels from {unique_labels} to range(0, {len(unique_labels)})")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y_processed)
    

    return X_tensor, y_tensor


# Function to split data for stratified sampling
def stratified_sample_with_validation(X, y, train_ratio):
    # Group indices by class
    class_indices = defaultdict(list)
    for i, label in enumerate(y.numpy()):
        class_indices[label].append(i)
    
    # Calculate number of samples per class for training (including validation)
    n_classes = len(class_indices)
    min_samples = min(len(indices) for indices in class_indices.values())
    samples_per_class = int(min_samples * train_ratio)
    
    # Calculate validation size (10% of training data)
    val_samples_per_class = max(1, int(samples_per_class * 0.1))
    actual_train_samples_per_class = samples_per_class - val_samples_per_class
    
    print(f"Sampling {actual_train_samples_per_class} samples per class for training")
    print(f"Sampling {val_samples_per_class} samples per class for validation")
    print(f"Remaining samples will be used for testing")
    
    # Select samples for training, validation and testing
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        # Shuffle indices
        indices_copy = indices.copy()
        random.shuffle(indices_copy)
        
        # Select samples for training
        train_indices.extend(indices_copy[:actual_train_samples_per_class])
        
        # Select samples for validation
        val_indices.extend(indices_copy[actual_train_samples_per_class:samples_per_class])
        
        # Rest for testing
        test_indices.extend(indices_copy[samples_per_class:])
    
    return train_indices, val_indices, test_indices



# Function to train model
def train_model(model, train_loader, val_loader, num_epochs=30, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, training=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Function to evaluate model
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return acc, report, all_preds, all_labels

# Main function to run experiments
def run_experiments(data_paths, configs, output_dir, train_ratios=(0.1, 0.3, 0.5, 0.8), 
                    batch_size=32, num_epochs=30, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configs
    config_objects = {name: Config(cfg) for name, cfg in configs.items()}
    
    # Initialize models
    models = {
        'lstm_v1': lambda cfg, inp, out: ClassifierLSTM(cfg, inp, out),
        'gru_v1': lambda cfg, inp, out: ClassifierGRU(cfg, inp, out),
        'gru_v2': lambda cfg, inp, out: ClassifierGRU(cfg, inp, out),
        'dcnn_v1': lambda cfg, inp, out: BenchmarkDCNN(cfg, inp, out),
        'deepsense_v1': lambda cfg, inp, out: BenchmarkDeepSense(cfg, inp, out),
        'attn_v1': lambda cfg, inp, out: ClassifierAttn(cfg, inp, out)
    }
    
    # Process each dataset
    results = []
    
    for data_name, paths in data_paths.items():
        print(f"\n=== Processing dataset: {data_name} ===")
        
        X, y = load_and_preprocess_data(paths['data'], paths['labels'])
        
        # Calculate number of classes
        num_classes = len(torch.unique(y))
        print(f"Number of classes: {num_classes}")
        
        # Create full dataset
        full_dataset = TensorDataset(X, y)
        
        # Experiment with different training ratios
        for train_ratio in train_ratios:
            print(f"\n--- Training with {train_ratio*100}% data ---")
            
            # Perform stratified sampling WITH validation split
            train_indices, val_indices, test_indices = stratified_sample_with_validation(X, y, train_ratio)
            
            # Create subsets
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)  # New validation dataset
            test_dataset = Subset(full_dataset, test_indices)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)  # New validation loader
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Train and evaluate each model
            for model_name, config_name in [
                ('lstm', 'lstm_v1'), 
                ('gru', 'gru_v1'), 
                ('gru_dropout', 'gru_v2'),
                ('dcnn', 'dcnn_v1'),
                ('deepsense', 'deepsense_v1'),
                ('attn', 'attn_v1')
            ]:
                # Skip if config not available
                if config_name not in config_objects:
                    print(f"Config {config_name} not found, skipping {model_name}")
                    continue
                
                # Skip if model constructor not available
                if config_name not in models:
                    print(f"Model constructor for {config_name} not found, skipping {model_name}")
                    continue
                
                print(f"\nTraining {model_name} with {train_ratio*100}% data")
                
                cfg = config_objects[config_name]
                
                # Adjust sequence length if needed
                original_seq_len = cfg.seq_len if hasattr(cfg, 'seq_len') else None
                if hasattr(cfg, 'seq_len') and cfg.seq_len != X.shape[1]:
                    print(f"Note: Adjusting sequence length from {cfg.seq_len} to {X.shape[1]}")
                    cfg.seq_len = X.shape[1]
                
                # Create model
                model = models[config_name](cfg, cfg.input, num_classes).to(device)
                
                # Train model - NOW USING SEPARATE VALIDATION LOADER
                model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
                
                # Evaluate model on the TEST set (not used during training)
                acc, report, predictions, true_labels = evaluate_model(model, test_loader, device=device)
                
                print(f"{model_name} Accuracy: {acc:.4f}")
                
                # Save results
                result = {
                    'dataset': data_name,
                    'model': model_name,
                    'train_ratio': train_ratio,
                    'accuracy': acc
                }
                
                # Add detailed metrics
                for cls in report:
                    if cls == 'accuracy':
                        # Handle the case where accuracy is a float
                        result['accuracy'] = report[cls]
                    elif cls == 'macro avg' or cls == 'weighted avg':
                        for metric, value in report[cls].items():
                            result[f'{cls}_{metric}'] = value
                
                results.append(result)
                
                # Save predictions to CSV
                pred_df = pd.DataFrame({
                    'true_label': true_labels,
                    'prediction': predictions
                })
                
                pred_path = os.path.join(output_dir, 
                                        f"{data_name}_{model_name}_{int(train_ratio*100)}pct_predictions.csv")
                pred_df.to_csv(pred_path, index=False)
                
                # Restore original sequence length
                if original_seq_len is not None:
                    cfg.seq_len = original_seq_len
    
    # Save overall results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"\nExperiments completed. Results saved to {output_dir}")
    return results_df


# Example usage
if __name__ == "__main__":
    # Load configs from JSON
    with open('config/classifier.json', 'r') as f:
        configs = json.load(f)
    
    dataset = 'sony_watch'
    # Set paths to datasets
    data_paths = {
        dataset: {
            'data': f'dataset/{dataset}/data_20_120.npy',
            'labels': f'dataset/{dataset}/label_20_120.npy'
        },
        #'dataset2': {
        #    'data': 'path/to/dataset2_data.npy',
        #    'labels': 'path/to/dataset2_labels.npy'
        #}
    }
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir='results/benchmark_results/' + dataset 
 
    # dataset_name to output_dir
    
    # Run experiments
    results = run_experiments(
        data_paths=data_paths,
        configs=configs,
        output_dir= output_dir,
        train_ratios=[0.1],
        batch_size=32,
        num_epochs=50,
        device=device
    )