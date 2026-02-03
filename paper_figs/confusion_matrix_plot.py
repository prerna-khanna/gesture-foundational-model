import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# import matplotlib rc settings and set the fint size to 9
plt.rc('font', size=9)

def plot_confusion_matrix(df, labels, ax, xticklabels=None, annot_kws_size = 7):
    """
    Plot a normalized confusion matrix
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing true_label and predicted_label columns
    labels : list
        List of class labels
    title : str
        Title for the plot
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Create confusion matrix
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    # rounbd to 2 decimal places
    cm = np.round(cm, 2)
    
    # Normalize by row (each row sums to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Replace NaN with 0
    cm_normalized = np.nan_to_num(cm_normalized, 0)
    
    if annot_kws_size:
        annot_kws = {"size": annot_kws_size}

    print("using annot_kws_size: ", annot_kws_size)

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
        

    # remvoe the legend scale from the heatmap
    ax.collections[0].colorbar.remove()

    # set x and y ticks font size
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # add box around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    
def main():
    # Load datasets
    # Replace with your actual CSV filenames
    df1 = pd.read_csv('results/final_results/blind_user_filtered_10_03_09_2025_20_06/results.csv')
    df2 = pd.read_csv('results/final_results/earbud_filtered_10_03_10_2025_05_15/results.csv')
    df3 = pd.read_csv('results/final_results/sony_watch_10_03_10_2025_04_23/results.csv')
    
    # Define labels for each dataset
    # Replace with your actual labels
    labels1 = ["Forearm up", "Forearm down", "Forearm left", "Forearm right", "Rotate wrist and\nmove arm right", "Rotate wrist and\nmove arm left", "Flick and\nforearm up", "Flick and\nforearm down", "Flick and\nforearm left", "Flick and\nforearm right", "Square", "Circle"]
    labels1_x = ["FUp", "FDn", "FL", "FR", "RW+\nAR", "RW+\nAL", "FlFUp", "FlFDn", "FlFL", "FlFR", "Sq", "Cir"]
    
    #labels2 = ["Tap", "Double tap", "Swipe up", "Swipe down", "Long press", "Rotate finger", "Tap on \nlower end"]
    labels2 = ["Tap", "DblTap", "SwUp", "SwDn", "LgPress", "Rotate", "Tap\nlower end"]
    
    labels3 = [
            "Right", "Left", "Up", "Down",
            "Clockwise circle", "Anti-clockwise circle", "Clockwise square", "Anti-clockwise square",
            "Right diagonal", "Left diagonal", "Down double", "Right double",
            "V-shape down-up", "V-shape up-down", "Triangle upward", "Triangle downward",
            "S-curve leftward", "S-curve rightward", "Wave left", "Wave right"
        ]
    labels3_short = [
    "Right", "Left", "Up", "Down",
    "CW\nCirc", "CCW\nCirc", "CW\nSq", "CCW\nSq",
    "R\nDiag", "L Diag", "Dn\nDbl", "R\nDbl",
    "V Dn\n-Up", "V Up\n-Dn", "Tri\nUp", "Tri\nDn",
    "S\nLeft", "S\nRight", "Wave\nL", "Wave\nR"
    ]
    
    # Create three separate figures, each 3x3 inches

    plt.figure(figsize=(5.5, 5))
    ax1 = plt.gca()
    plot_confusion_matrix(df1, labels1, ax1, labels1_x)
    plt.tight_layout()
    plt.savefig('paper_figs/confusion_matrix_blind_gest.pdf',bbox_inches='tight')

    plt.figure(figsize=(5.5, 3))
    ax2 = plt.gca()
    plot_confusion_matrix(df2, labels2, ax2, annot_kws_size = 6)
    plt.tight_layout()
    plt.savefig('paper_figs/confusion_matrix_earbud_gest.pdf',bbox_inches='tight')

    plt.figure(figsize=(11, 5))
    ax3 = plt.gca()
    plot_confusion_matrix(df3, labels3, ax3, annot_kws_size = 6, xticklabels=labels3_short)
    plt.tight_layout()
    plt.savefig('paper_figs/confusion_matrix_sony_watch_gest.pdf',bbox_inches='tight')

if __name__ == "__main__":
    main()