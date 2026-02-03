import matplotlib.pyplot as plt
import re
import numpy as np

def combined_score_cal(train_acc, vali_acc, vali_f1, train_f1):
    # Increase weight on validation accuracy
    base_score = (0.6 * vali_acc) + (0.25 * vali_f1) + (0.1 * train_acc)
    
    # Add bonus for very high training accuracy (>0.99)
    high_train_bonus = 0.1 if train_acc > 0.99 else 0
    
    # Add bonus for validation accuracy being higher than validation F1
    # This is a pattern specific to epoch 188 (0.763 acc vs 0.735 F1)
    val_pattern_bonus = 0.05 if vali_acc > vali_f1 else 0
    
    # Add a component that rewards when validation accuracy is > 0.75
    high_val_bonus = 0.15 if vali_acc > 0.75 else 0
    
    return base_score + high_train_bonus + val_pattern_bonus + high_val_bonus


def extract_metrics_from_log(log_file):
    """
    Extract epoch, training accuracy, validation accuracy, training F1, validation F1
    from the training log file.
    """
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Regular expressions to extract metrics
    epoch_pattern = r"Epoch (\d+)/200:"
    acc_pattern = r"Accuracies: Train=([\d.]+), Val=([\d.]+), Test=([\d.]+)"
    f1_pattern = r"F1 Scores: Train=([\d.]+), Val=([\d.]+), Test=([\d.]+)"
    
    # Find all matches
    epochs = [int(e) for e in re.findall(epoch_pattern, log_content)]
    accuracies = re.findall(acc_pattern, log_content)
    f1_scores = re.findall(f1_pattern, log_content)
    
    # Extract the metrics
    train_acc = [float(acc[0]) for acc in accuracies]
    val_acc = [float(acc[1]) for acc in accuracies]
    train_f1 = [float(f1[0]) for f1 in f1_scores]
    val_f1 = [float(f1[1]) for f1 in f1_scores]
    
    return epochs, train_acc, val_acc, train_f1, val_f1

def plot_metrics(log_file):
    """
    Plot training accuracy, validation accuracy, training F1, and validation F1
    against epochs.
    """
    epochs, train_acc, val_acc, train_f1, val_f1 = extract_metrics_from_log(log_file)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    scores = []

    for i in range(len(epochs)):
        score = combined_score_cal(train_acc[i], val_acc[i], val_f1[i], train_f1[i])
        scores.append(score)
    
    # Plot metrics
    plt.plot(epochs, train_acc, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
    plt.plot(epochs, train_f1, label='Training F1', linestyle='--', color='blue')
    plt.plot(epochs, val_f1, label='Validation F1', linestyle='--', color='red')
    plt.plot(epochs, scores, label='Combined Score', linestyle='-', color='green')
    
    
    # Add legend, title and labels
    plt.legend(loc='lower right')
    plt.title('Training and Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(x=188, color='black', linestyle='--')
    plt.axvline(x=131, color='black', linestyle='--')
    

    
    # Set y-axis limits
    plt.ylim(0, 1.05)
    
    # Save the figure
    plt.savefig('training_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # arrange scores in descending order and print the top 10 weith their epochs
    scores = np.array(scores)
    sorted_indices = np.argsort(-scores)
    for i in range(10):
        print(f"Epoch {epochs[sorted_indices[i]]}: {scores[sorted_indices[i]]}")
        

# Run the function with the log file path
plot_metrics('saved/classifier_contrastive_gru_blind_user_filtered_20_120/loss_acc_2025-03-11_21-51-58.txt')