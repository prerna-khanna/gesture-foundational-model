import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import datetime

# ============ CONFIGURATION SETTINGS ============
# Set these variables to control the script behavior
DATA_PATH = 'embed/embed_limu_v1_sony_watch_20_120.npy'           # Path to your input data file
LABEL_PATH = 'dataset/sony_watch/label_20_120.npy'         # Path to your label file
OUTPUT_DIR = input("Enter the output directory name (baseline 10%): ")  # Directory to save results
now = datetime.datetime.now()
OUTPUT_DIR = OUTPUT_DIR + "_" + now.strftime("%m_%d_%Y_%H_%M")
OUTPUT_DIR = os.path.join("results", "final_results", OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LABEL_RATE = 0.1                 # Percentage of data to use (0.1 = 10%)
MODELS_TO_RUN = ['bilstm', 'cnn', 'svm']  # List of models to run
# ===============================================

def load_data(data_path, label_path):
    """
    Load data and labels from .npy files
    """
    print(f"Loading data from {data_path} and labels from {label_path}")
    X = np.load(data_path)
    y = np.load(label_path)
    
    # Extract first column of labels as mentioned in the requirements
    y = y[:, :, 0]
    
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    return X, y

def prepare_data_for_model(X, y, model_type, label_rate=1.0, test_size=0.2, val_size=0.1):
    """
    Prepare data for different model types with stratified splitting and label rate
    """
    # Reshape data based on model type
    if model_type == 'svm':
        # For SVM, flatten the time series
        X_reshaped = X.reshape(X.shape[0], -1)
        # Get unique classes for stratified splitting
        unique_classes = np.unique(y[:, 0])  # Using first time step for stratification
    else:
        # For neural networks, keep the time series format
        X_reshaped = X
        unique_classes = np.unique(y[:, 0])  # Using first time step for stratification
    
    # Get indices for each class
    class_indices = {c: np.where(y[:, 0] == c)[0] for c in unique_classes}
    
    # Calculate how many samples per class for the given label rate
    samples_per_class = int(len(y) * label_rate / len(unique_classes))
    
    # Select balanced subset of data based on label rate
    train_indices = []
    for c in unique_classes:
        # Take a random subset of this class's indices
        indices = np.random.choice(class_indices[c], min(samples_per_class, len(class_indices[c])), replace=False)
        train_indices.extend(indices)
    
    # Shuffle selected indices
    np.random.shuffle(train_indices)
    
    # Subset data and labels
    X_subset = X_reshaped[train_indices]
    y_subset = y[train_indices]
    
    print(f"Selected {len(X_subset)} samples ({label_rate*100:.1f}% of data)")
    
    # Split data into train, validation, and test sets with stratification
    # For stratification, use the first time step of the labels
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_subset, y_subset, test_size=test_size+val_size, 
        random_state=42, stratify=y_subset[:, 0]
    )
    
    # Split the temporary set into validation and test sets
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, 
        random_state=42, stratify=y_temp[:, 0]
    )
    
    # Further processing based on model type
    if model_type == 'svm':
        # Standardize features for SVM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Use first time step as label for SVM
        y_train_svm = y_train[:, 0]
        y_val_svm = y_val[:, 0]
        y_test_svm = y_test[:, 0]
        
        return X_train, X_val, X_test, y_train_svm, y_val_svm, y_test_svm, scaler
    
    elif model_type in ['bilstm', 'cnn']:
        # For neural networks, we need to one-hot encode the labels
        num_classes = len(unique_classes)
        
        # Process the sequence of labels for each sample
        y_train_seq = [to_categorical(seq, num_classes=num_classes) for seq in y_train]
        y_val_seq = [to_categorical(seq, num_classes=num_classes) for seq in y_val]
        y_test_seq = [to_categorical(seq, num_classes=num_classes) for seq in y_test]
        
        # Convert to numpy arrays
        y_train_nn = np.array(y_train_seq)
        y_val_nn = np.array(y_val_seq)
        y_test_nn = np.array(y_test_seq)
        
        return X_train, X_val, X_test, y_train_nn, y_val_nn, y_test_nn, None

def build_bilstm_model(input_shape, num_classes):
    """
    Build a Bidirectional LSTM model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train an SVM model
    """
    print("Training SVM model...")
    start_time = time.time()
    
    svm = SVC(probability=True, gamma='scale', C=1.0)
    svm.fit(X_train, y_train)
    
    # Validate
    val_accuracy = svm.score(X_val, y_val)
    training_time = time.time() - start_time
    
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Training completed in {training_time:.2f} seconds")
    
    return svm

def train_neural_network(model_type, X_train, y_train, X_val, y_val, input_shape, num_classes):
    """
    Train neural network models (BiLSTM or CNN)
    """
    print(f"Training {model_type} model...")
    start_time = time.time()
    
    if model_type == 'bilstm':
        model = build_bilstm_model(input_shape, num_classes)
    else:  # CNN
        model = build_cnn_model(input_shape, num_classes)
    
    # Train model
    history = model.fit(
        X_train, y_train[:, :, :num_classes],  # Ensure dimensionality matches
        epochs=20,  # Reduced for faster training
        batch_size=32,
        validation_data=(X_val, y_val[:, :, :num_classes]),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history

def get_predictions_and_save_results(model, X_test, y_test, model_type, num_classes, output_dir):
    """
    Get predictions and save results in the required format
    """
    print("Generating predictions...")
    
    if model_type == 'svm':
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities
        y_prob = model.predict_proba(X_test)
        
        # Prepare results dataframe
        results = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
        })
        
        # Add probability columns for each class
        for i in range(num_classes):
            results[f'prob_class_{i}'] = y_prob[:, i]
        
        # Add 'correct' column (1 if prediction is correct, 0 otherwise)
        results['correct'] = (results['true_label'] == results['predicted_label']).astype(int)
        
        # Classification report
        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)
        
    else:  # Neural networks (BiLSTM or CNN)
        # For neural networks, we need to take the max probability class along the sequence
        # Get predictions
        y_prob_seq = model.predict(X_test)
        
        # Get the class with highest probability for each time step
        y_pred_seq = np.argmax(y_prob_seq, axis=2)
        
        # Get true labels
        y_true_seq = np.argmax(y_test, axis=2)
        
        # We'll take the predictions from the first time step for simplicity
        # You can modify this to aggregate predictions across the sequence if needed
        time_step = 0
        
        # Prepare results dataframe
        results = pd.DataFrame({
            'true_label': y_true_seq[:, time_step],
            'predicted_label': y_pred_seq[:, time_step],
        })
        
        # Add probability columns for each class
        for i in range(num_classes):
            results[f'prob_class_{i}'] = y_prob_seq[:, time_step, i]
        
        # Add 'correct' column (1 if prediction is correct, 0 otherwise)
        results['correct'] = (results['true_label'] == results['predicted_label']).astype(int)
        
        # Classification report
        report = classification_report(y_true_seq[:, time_step], y_pred_seq[:, time_step])
        print("Classification Report:\n", report)
    
    # Save results to CSV
    output_file = os.path.join(output_dir, f"{model_type}_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results

def run_all_models():
    """
    Run all specified models with the given configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X, y = load_data(DATA_PATH, LABEL_PATH)
    
    # Number of classes (assuming classes are 0, 1, 2, ..., n)
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    print(f"Using label rate: {LABEL_RATE} ({LABEL_RATE*100}%)")
    
    # Run each model
    for model_type in MODELS_TO_RUN:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_type}")
        print(f"{'='*50}")
        
        # Prepare data based on model type
        if model_type == 'svm':
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_for_model(
                X, y, model_type, LABEL_RATE
            )
            
            # Train SVM
            model = train_svm(X_train, y_train, X_val, y_val)
            
        else:  # Neural network models
            X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data_for_model(
                X, y, model_type, LABEL_RATE
            )
            
            # Define input shape
            input_shape = (X.shape[1], X.shape[2])
            
            # Train neural network
            model, history = train_neural_network(
                model_type, X_train, y_train, X_val, y_val, input_shape, num_classes
            )
        
        # Get predictions and save results
        results = get_predictions_and_save_results(
            model, X_test, y_test, model_type, num_classes, OUTPUT_DIR
        )
        
        print(f"Model {model_type} completed!")

# Main execution
if __name__ == "__main__":
    print("Starting ML model training with multiple models...")
    run_all_models()
    print("\nAll models have been trained and evaluated!")
    print(f"Results are available in the '{OUTPUT_DIR}' directory")