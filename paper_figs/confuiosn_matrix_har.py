import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

def plot_confusion_matrix(df, labels, ax, title=None, xticklabels=None, annot_kws_size=6, idx_=None):
    """
    Plot a normalized confusion matrix
    """
    # Create confusion matrix
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    # round to 2 decimal places
    cm = np.round(cm, 2)
    
    # Normalize by row (each row sums to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Replace NaN with 0
    cm_normalized = np.nan_to_num(cm_normalized, 0)
    
    annot_kws = {"size": annot_kws_size}
    
    # Reduce size of cells by setting square=True and adjusting fontsize
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        yticklabels=labels,
        annot_kws=annot_kws,
        ax=ax
    )
    
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels(labels)
    
    if title:
        ax.set_xlabel(idx_ + title, fontsize=10, fontweight='bold')
        
    # Remove the legend scale from the heatmap
    ax.collections[0].colorbar.remove()
    
    # Increase label size
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # Rotate x-labels for better fit
    
    # Add box around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)

def main():
    # Define the CSV files
    csv_file1 = 'results/final_results/motion_10_03_10_2025_13_56/results.csv'
    csv_file2 = 'results/final_results/shoaib_03_10_2025_16_36/results.csv'
    csv_file3 = 'results/final_results/uci_10_03_10_2025_04_59/results.csv'
    
    # Load datasets
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df3 = pd.read_csv(csv_file3)
    
    # Define labels for each dataset
    labels1 = ["D", "U", "S", "St", "W", "J"]
    labels2 = ["W", "S", "St", "J", "B", "U", "D"]
    labels3 = ["W", "U", "D", "S", "St", "L"]
    
    # Create a single figure with 3 subplots in one row
    # Adjust figure size to make cells smaller but labels larger
    fig = plt.figure(figsize=(11, 2))
    
    # Set tight spacing between subplots
    gs = gridspec.GridSpec(1, 3, wspace=0.25)
    
    # Create the three plots
    ax1 = plt.subplot(gs[0, 0])
    plot_confusion_matrix(df1, labels1, ax1, title="Motion Dataset", idx_ = '(a)')
    
    ax2 = plt.subplot(gs[0, 1])
    plot_confusion_matrix(df2, labels2, ax2, title="Shoaib Dataset", idx_ = '(b)')
    
    ax3 = plt.subplot(gs[0, 2])
    plot_confusion_matrix(df3, labels3, ax3, title="UCI Dataset", idx_ = '(c)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure with higher DPI to ensure better quality
    plt.savefig('paper_figs/confusion_matrices_har.pdf', bbox_inches='tight', dpi=300)
    print("Confusion matrices saved with improved layout")

if __name__ == "__main__":
    main()