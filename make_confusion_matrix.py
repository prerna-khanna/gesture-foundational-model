import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_matrix(matrix, labels_name=None):
    plt.figure()
    row_sum = matrix.sum(axis=1)
    matrix_per = np.copy(matrix).astype('float')
    for i in range(row_sum.size):
        if row_sum[i] != 0:
            matrix_per[i] = matrix_per[i] / row_sum[i]
    plt.figure(figsize=(15, 10))
    if labels_name is None:
        labels_name = "auto"
    sns.heatmap(matrix_per, annot=True, fmt='.2f', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('plots/confusion_matrix_new_0.png')
    plt.close()
    return matrix

def plot_confusion_matrix(csv_file, output_dir='plots/', filename=None):
    """
    Plots and saves confusion matrix from a CSV file containing prediction results.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file with prediction results
    output_dir : str
        Directory to save the confusion matrix plot
    filename : str, optional
        Custom filename for the saved plot. If None, will use the original filename.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CSV file
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Extract true and predicted labels
    y_true = df['true_label'].astype(int)
    y_pred = df['predicted_label'].astype(int)
    
    # Get number of classes
    num_classes = max(y_true.max(), y_pred.max()) + 1
    class_labels = [f"Class {i}" for i in range(num_classes)]
    
    # Create the confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Use the provided plot_matrix function
    plot_matrix(cm, class_labels)
    
    # Calculate accuracy
    accuracy = df['correct'].mean() * 100
    
    # Normalize the confusion matrix by row (each row sums to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # The original confusion matrix plot still needed for additional metrics
    # Determine output filename
    if filename is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_confusion_matrix_30.png")
    else:
        output_file = os.path.join(output_dir, filename)

    output_file = 'plots/confusion_matrix_new_30.png'
    
    # Copy the generated file to the specified output directory
    if os.path.exists('confusion_matrix_new_30.png'):
        import shutil
        shutil.copy('confusion_matrix_new.png', output_file)
        print(f"Confusion matrix saved to {output_file}")
    
    # Display additional metrics
    print("\nClass-wise Performance:")
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calculate overall metrics
    class_accuracy = np.mean(np.diag(cm_normalized)) * 100
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
    
    print("\nOverall Metrics:")
    print(f"Accuracy (Overall): {overall_accuracy:.2f}%")
    print(f"Accuracy (Class Mean): {class_accuracy:.2f}%")
    
    # Calculate weighted metrics
    support = np.sum(cm, axis=1)
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = 2 * (precision_weighted * recall_weighted) / (precision_weighted + recall_weighted)
    
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall: {recall_weighted:.4f}")
    print(f"F1-Score: {f1_weighted:.4f}")
    
    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    })
    
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"))

    # save confusion matrix
    plt.figure()


    
    return plt.gcf()

def plot_multiple_confusion_matrices(csv_files, output_dir='./plots'):
    """
    Plots and saves confusion matrices for multiple result files.
    
    Parameters:
    -----------
    csv_files : list of str
        List of paths to CSV files with prediction results
    output_dir : str
        Directory to save the confusion matrix plots
    """
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        plot_confusion_matrix(csv_file, output_dir)

if __name__ == "__main__":
    # Example usage
    # Single file
    plot_confusion_matrix("results/final_results/sony_watch_80_03_11_2025_02_10/results.csv")
    
    # Multiple files (uncomment to use)
    # csv_files = ["bilstm_predictions.csv", "cnn_predictions.csv", "svm_predictions.csv"]
    # plot_multiple_confusion_matrices(csv_files)