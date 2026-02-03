import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
# Set the style
plt.style.use('ggplot')

def calculate_metrics(csv_file):
    """Calculate F1 and accuracy scores from a CSV file with proper std calculation."""
    df = pd.read_csv(csv_file)
    
    # Try to get predicted labels, fall back to 'prediction' if 'predicted_label' not found
    if 'true_label' in df.columns:
            y_true = df['true_label']
    elif 'True_Label' in df.columns:
        y_true = df['True_Label']
    elif 'label' in df.columns:
        y_true = df['label']
    else:
        raise KeyError(f"No column for true labels found in {csv_file}")
        
    # Extract predictions exactly as specified
    if 'predicted_label' in df.columns:
        y_pred = df['predicted_label']
    elif 'Predicted_Label' in df.columns:
        y_pred = df['Predicted_Label']
    elif 'prediction' in df.columns:
        y_pred = df['prediction']
    else:
        raise KeyError(f"Neither 'predicted_label', 'Predicted_Label', nor 'prediction' column found in {csv_file}")
        
    # Calculate F1 and accuracy
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate proper standard deviation using bootstrap method
    n_bootstrap = 100
    f1_samples = []
    acc_samples = []
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(df), len(df), replace=True)
        y_true_sample = y_true.iloc[indices]
        y_pred_sample = y_pred.iloc[indices]
        
        f1_samples.append(f1_score(y_true_sample, y_pred_sample, average='weighted'))
        acc_samples.append(accuracy_score(y_true_sample, y_pred_sample))
    
    # Calculate std from bootstrap samples
    f1_std = np.std(f1_samples)
    acc_std = np.std(acc_samples)
    
    return {
        'F1': f1,
        'F1_std': f1_std,
        'Accuracy': acc,
        'Accuracy_std': acc_std
    }

def calculate_metrics_from_multiple_files(csv_files):
    """Calculate metrics by combining data from multiple CSV files."""
    # Create an empty list to store all dataframes
    all_dfs = []
    
    # Load and combine all CSVs
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to a temporary file
    temp_file = 'temp_combined.csv'
    combined_df.to_csv(temp_file, index=False)
    
    # Calculate metrics using the combined file
    metrics = calculate_metrics(temp_file)
    
    return metrics

def create_bar_plot(file_paths, scenario_names=None, system_names=None, output_file = ""):
    """
    Create a bar plot with 3 subplots, each showing F1 and accuracy metrics
    for 5 systems (your system + 4 baselines).
    
    Args:
        file_paths: Dictionary of dictionaries with scenario and system as keys and file paths as values
                   Example: {'scenario1': {'your_system': 'path/to/file1.csv', 'deepsense': 'path/to/file2.csv', ...}, ...}
        scenario_names: List of names for the 3 scenarios (default: ['Scenario 1', 'Scenario 2', 'Scenario 3'])
        system_names: List of names for the 5 systems (default: ['DeepSense', 'LIMU-BERT', 'UniHAR', 'ContraSense', 'Your System'])
        output_file: Path to save the output figure
    """
    # Default names if not provided
    if scenario_names is None:
        scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3']
    
    if system_names is None:
        # Put "Your System" as the rightmost system, add ContraSense
        system_names = ['DeepSense', 'LIMU-BERT', 'UniHAR', 'ContraSense', 'Your System']
    
    # System keys in the dictionary - reordered to put your_system last and added contrasense
    system_keys = ['deepsense', 'limu_bert', 'unihar', 'contrasense', 'your_system']
    
    # Color scheme - updated to include a new color for ContraSense
    colors = ['#3498db',  # Blue (DeepSense)
              '#9b59b6',  # Purple (LIMU-BERT)
              '#e74c3c',  # Red (UniHAR)
              '#f39c12',  # Orange (ContraSense)
              'green']    # Green (Your System) - now last
    
    # Remove the ggplot style to have a white background
    plt.style.use('default')
    
    # Create the figure with 3 subplots (removing the fourth)
    fig, axes = plt.subplots(1, 3, figsize=(13, 2.5))  # Increased width to accommodate more bars
    
    # Add subplot labels (a, b, c)
    subplot_labels = ['(a) Sighted User Hand Gesture', '(b) Sighted User Earbud Gesture', '(c) Blind User Hand Gesture']
    
    # For each scenario/subplot (limiting to first 3 scenarios)
    scenario_keys = sorted(file_paths.keys())[:3]  # Only take first 3 scenarios
    
    for i, (scenario_key, ax) in enumerate(zip(scenario_keys, axes.flatten())):
        scenario_name = scenario_names[i] if i < len(scenario_names) else f"Scenario {i+1}"
        
        # Store metrics for each system
        f1_scores = []
        f1_errors = []
        acc_scores = []
        acc_errors = []
        
        # Process each system's data for this scenario
        for system_key in system_keys:
            if system_key in file_paths[scenario_key]:
                # Check if this is scenario4 which needs special handling
                if scenario_key == 'scenario4' and isinstance(file_paths[scenario_key][system_key], list):
                    try:
                        # Calculate metrics by combining data from multiple files
                        csv_files = file_paths[scenario_key][system_key]
                        metrics = calculate_metrics_from_multiple_files(csv_files)
                        
                        # Store the metrics
                        f1_scores.append(metrics['F1'])
                        f1_errors.append(metrics['F1_std'])
                        acc_scores.append(metrics['Accuracy'])
                        acc_errors.append(metrics['Accuracy_std'])
                    except Exception as e:
                        print(f"Error processing combined files for {system_key} in {scenario_key}: {e}")
                        # Use placeholder data if error occurs
                        f1_scores.append(0)
                        f1_errors.append(0)
                        acc_scores.append(0)
                        acc_errors.append(0)
                else:
                    # Regular single file processing
                    csv_file = file_paths[scenario_key][system_key]
                    try:
                        # Calculate metrics from the CSV
                        metrics = calculate_metrics(csv_file)
                        
                        # Store the metrics
                        f1_scores.append(metrics['F1'])
                        f1_errors.append(metrics['F1_std'])
                        acc_scores.append(metrics['Accuracy'])
                        acc_errors.append(metrics['Accuracy_std'])
                    except Exception as e:
                        print(f"Error processing file {csv_file}: {e}")
                        # Use placeholder data if file not found or has errors
                        f1_scores.append(0)
                        f1_errors.append(0)
                        acc_scores.append(0)
                        acc_errors.append(0)
            else:
                print(f"Warning: No file specified for {system_key} in {scenario_key}")
                # Use placeholder data if file not specified
                f1_scores.append(0)
                f1_errors.append(0)
                acc_scores.append(0)
                acc_errors.append(0)
        
        # Set positions for the bars
        x = np.arange(len(system_names))
        width = 0.2
        
        # Plot F1 scores
        rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color=colors,
                      yerr=f1_errors, capsize=3)
        
        # Plot Accuracy scores
        rects2 = ax.bar(x + width/2, acc_scores, width, label='Accuracy', color=colors,
                       yerr=acc_errors, capsize=3, alpha=0.8, hatch='////')
        
        # Add labels and title
        # Updated to use the specified titles
        """if i == 0:
            ax.set_title("User Interactions", fontsize=9, fontweight='bold')
        elif i == 1:
            ax.set_title("Cross Device", fontsize=9, fontweight='bold')
        elif i == 2:
            ax.set_title("Accessibility", fontsize=9, fontweight='bold')"""
        
        ax.set_xticks(x)
        ax.set_xticklabels(system_names)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(subplot_labels[i], fontweight='bold')
        # make x label bold
        ax.tick_params(axis='x', labelsize=9)  # Slightly reduced font size to fit all systems
        ax.set_ylim(0, 1)
        
        # Add value labels on the bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 6),  # Increased vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        #add_labels(rects1)
        #add_labels(rects2)
        # Remove the top and right spines of the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add a legend to the figure with transparent background
    from matplotlib.patches import Rectangle, Patch
    
    # Create custom legend elements
    f1_patch = Patch(facecolor='white', edgecolor='black', label='F1 Score')
    
    # Create a hatched patch with border for accuracy
    acc_patch = Patch(facecolor='white', edgecolor='black', hatch='///', label='Accuracy')
    
    # Add the legend with custom handles
    leg = fig.legend([f1_patch, acc_patch], ['F1 Score', 'Accuracy'], 
                    loc='upper right', bbox_to_anchor=(0.81, 0.89),  # Adjusted position
                    frameon=True, facecolor='white', edgecolor='black', fontsize=8, ncol=1)
    
    # Make the legend box background white instead of colored
    for handle in leg.legend_handles:
        handle.set_facecolor('white')
        handle.set_alpha(1.0)  # Make the handles more visible
    
    # Adjust layout and spacing
    plt.tight_layout()
    
    # Save the figure with higher dpi
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    # Example usage with explicit file paths
    file_paths = {
        'scenario1': {
            'your_system': 'results/final_results/sony_watch_10_03_10_2025_04_23/results.csv',
            'deepsense': 'results/benchmark_results/sony_watch/sony_watch_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/sony_watch_10_03_11_2025_19_34/results.csv',
            'unihar': 'results/unihar_results/sony_watch_10.csv',
            'contrasense': 'results/contrasense_results/sony_watch_10.csv'  # Added ContraSense
        },
        'scenario2': {
            'your_system': 'results/final_results/earbud_filtered_10_03_10_2025_05_15/results.csv',
            'deepsense': 'results/benchmark_results/earbud_filtered/earbud_filtered_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/earbud_filtered_10_03_11_2025_20_07/results.csv',
            'unihar': 'results/unihar_results/earbud_filtered_10.csv',
            'contrasense': 'results/contrasense_results/earbud_filtered_10.csv'  # Added ContraSense
        },
        'scenario3': {
            'your_system': 'results/final_results/blind_user_filtered_10_03_09_2025_20_06/results.csv',
            'deepsense': 'results/benchmark_results/blind_user_filtered/blind_user_filtered_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/blind_user_filtered_10_03_11_2025_19_50/results.csv',
            'unihar': 'results/unihar_results/blind_user_filtered_10.csv',
            'contrasense': 'results/contrasense_results/blind_user_filtered_10.csv'  # Added ContraSense
        }
    }
    
    # Updated scenario and system names
    scenario_names = ['Watch Hand Gesture (SU)', 'Earbud Hand Gesture (SU)', 'Blind User Hand Gesture (BU)']
    system_names = ['Deep-\nSense', 'LIMU-\nBERT', 'UniHAR', 'Contrast-\nSense', 'GestureLens']  # Added ContraSense
    
    # Create and show the plot
    fig = create_bar_plot(file_paths, scenario_names, system_names, output_file='paper_figs/new_bar_plot_test_server.pdf')