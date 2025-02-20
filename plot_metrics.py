import matplotlib.pyplot as plt
import re

def parse_metrics(log_text):
    epochs = []
    metrics = {
        'classification_loss': [],
        'semantic_loss': [],
        'contrastive_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'train_f1': [],
        'val_f1': [],
        'test_f1': []
    }
    
    current_epoch = None
    
    for line in log_text.split('\n'):
        # Parse epoch
        if 'Epoch' in line:
            current_epoch = int(line.split('/')[0].split()[-1])
            epochs.append(current_epoch)
            
        # Parse losses
        elif 'classification_loss:' in line:
            loss = float(line.split(':')[1].strip())
            metrics['classification_loss'].append(loss)
        elif 'semantic_loss:' in line:
            loss = float(line.split(':')[1].strip())
            metrics['semantic_loss'].append(loss)
        elif 'contrastive_loss:' in line:
            loss = float(line.split(':')[1].strip())
            metrics['contrastive_loss'].append(loss)
            
        # Parse accuracies
        elif 'Accuracies:' in line:
            pattern = r'Train=([\d.]+), Val=([\d.]+), Test=([\d.]+)'
            match = re.search(pattern, line)
            if match:
                metrics['train_acc'].append(float(match.group(1)))
                metrics['val_acc'].append(float(match.group(2)))
                metrics['test_acc'].append(float(match.group(3)))
                
        # Parse F1 scores
        elif 'F1 Scores:' in line:
            pattern = r'Train=([\d.]+), Val=([\d.]+), Test=([\d.]+)'
            match = re.search(pattern, line)
            if match:
                metrics['train_f1'].append(float(match.group(1)))
                metrics['val_f1'].append(float(match.group(2)))
                metrics['test_f1'].append(float(match.group(3)))
    
    return epochs, metrics

def plot_metrics(log_text):
    epochs, metrics = parse_metrics(log_text)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    ax1.plot(epochs, metrics['classification_loss'], label='Classification Loss')
    ax1.plot(epochs, metrics['semantic_loss'], label='Semantic Loss')
    ax1.plot(epochs, metrics['contrastive_loss'], label='Contrastive Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies and F1 scores
    ax2.plot(epochs, metrics['train_acc'], label='Train Accuracy', linestyle='-')
    ax2.plot(epochs, metrics['val_acc'], label='Val Accuracy', linestyle='-')
    ax2.plot(epochs, metrics['test_acc'], label='Test Accuracy', linestyle='-')
    ax2.plot(epochs, metrics['train_f1'], label='Train F1', linestyle='--')
    ax2.plot(epochs, metrics['val_f1'], label='Val F1', linestyle='--')
    ax2.plot(epochs, metrics['test_f1'], label='Test F1', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy and F1 Scores')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_plot.png')

# Read the log text from file
with open('saved/classifier_contrastive_gru_blind_user_20_120/loss_acc_2025-02-19_19-14-49.txt', 'r') as f:
    log_text = f.read()

# Create the plots
plot_metrics(log_text)