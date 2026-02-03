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
    
    # Get true labels
    y_true = df['true_label']
    
    # Try to get predicted labels, fall back to 'prediction' if 'predicted_label' not found
    if 'predicted_label' in df.columns:
        y_pred = df['predicted_label']
    elif 'prediction' in df.columns:
        y_pred = df['prediction']
    else:
        raise KeyError("Neither 'predicted_label' nor 'prediction' column found in CSV")
    
    # Calculate F1 and accuracy
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate proper standard deviation using bootstrap method
    n_bootstrap = 100
    f1_samples = []
    acc_samples = []
    
    for _ in range(n_bootstrap):
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

def create_bar_plot(file_paths, scenario_names=None, system_names=None, output_file='performance_comparison.png'):
    """
    Create a bar plot with 4 subplots, each showing F1 and accuracy metrics
    for 4 systems (your system + 3 baselines).
    
    Args:
        file_paths: Dictionary of dictionaries with scenario and system as keys and file paths as values
                   Example: {'scenario1': {'your_system': 'path/to/file1.csv', 'deepsense': 'path/to/file2.csv', ...}, ...}
        scenario_names: List of names for the 4 scenarios (default: ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'])
        system_names: List of names for the 4 systems (default: ['DeepSense', 'LIMU-BERT', 'UniHAR', 'Your System'])
        output_file: Path to save the output figure
    """
    # Default names if not provided
    if scenario_names is None:
        scenario_names = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']
    
    if system_names is None:
        # Put "Your System" as the rightmost system
        system_names = ['Deep-\nSense', 'LIMU-\nBERT', 'UniHAR', 'Our System']
    system_names = ['Deep-\nSense', 'LIMU-\nBERT', 'UniHAR', 'Our System']
    
    # System keys in the dictionary - reordered to put your_system last
    system_keys = ['deepsense', 'limu_bert', 'unihar', 'your_system']
    
    # Color scheme - reordered to match the new system order
    colors = ['#3498db',  # Blue (DeepSense)
              '#9b59b6',  # Purple (LIMU-BERT)
              '#e74c3c',  # Red (UniHAR)
              '#2ecc71']  # Green (Your System) - now last
    
    # Create the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Performance Comparison Across Scenarios', fontsize=16)
    
    # For each scenario/subplot
    for i, (scenario_key, ax) in enumerate(zip(sorted(file_paths.keys()), axes.flatten())):
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
        width = 0.35
        
        # Plot F1 scores
        rects1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color=colors,
                      yerr=f1_errors, capsize=5, alpha=0.8)
        
        # Plot Accuracy scores
        rects2 = ax.bar(x + width/2, acc_scores, width, label='Accuracy', color=colors,
                       yerr=acc_errors, capsize=5, alpha=0.5, hatch='///')
        
        # Add labels and title
        ax.set_title(scenario_name)
        ax.set_xticks(x)
        ax.set_xticklabels(system_names, rotation=45, ha='right')
        
        # Add value labels on the bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        add_labels(rects1)
        add_labels(rects2)
    
    # Add a legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    # Adjust layout and spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Example usage with explicit file paths
    file_paths = {
        'scenario1': {
            'your_system': 'results/final_results/sony_watch_10_03_10_2025_04_23/results.csv',
            'deepsense': 'results/benchmark_results/sony_watch/sony_watch_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/sony_watch_10_03_11_2025_19_34/results.csv',
            'unihar': 'results/limu_bert_results/sony_watch_10_03_11_2025_19_34/results.csv'
        },
        'scenario2': {
            'your_system': 'results/final_results/earbud_filtered_10_03_10_2025_05_15/results.csv',
            'deepsense': 'results/benchmark_results/earbud_filtered/earbud_filtered_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/earbud_filtered_10_03_11_2025_20_07/results.csv',
            'unihar': 'results/limu_bert_results/earbud_filtered_10_03_11_2025_20_07/results.csv'
        },
        'scenario3': {
            'your_system': 'results/final_results/blind_user_filtered_10_03_09_2025_20_06/results.csv',
            'deepsense': 'results/benchmark_results/blind_user_filtered/blind_user_filtered_deepsense_10pct_predictions.csv',
            'limu_bert': 'results/limu_bert_results/blind_user_filtered_10_03_11_2025_19_50/results.csv',
            'unihar': 'results/limu_bert_results/blind_user_filtered_10_03_11_2025_19_50/results.csv'
        },
        'scenario4': {
            # For scenario4, provide lists of files to combine
            'your_system': [
                'results/final_results/motion_10_03_10_2025_13_56/results.csv',
                'results/final_results/shoaib_03_10_2025_16_36/results.csv',
                'results/final_results/uci_10_03_10_2025_04_59/results.csv'
            ],
            'deepsense': [
                'results/benchmark_results/motion/motion_deepsense_50pct_predictions.csv',
                'results/benchmark_results/uci/uci_deepsense_50pct_predictions.csv',
                'results/benchmark_results/shoaib/shoaib_deepsense_50pct_predictions.csv'
            ],
            'limu_bert': [
                'results/limu_bert_results/motion_10_03_11_2025_20_19/results.csv',
                'results/limu_bert_results/shoaib_10_03_11_2025_20_25/results.csv',
                'results/limu_bert_results/uci_10_03_11_2025_20_09/results.csv'
            ],
            'unihar':[
                'results/limu_bert_results/motion_10_03_11_2025_20_19/results.csv',
                'results/limu_bert_results/shoaib_10_03_11_2025_20_25/results.csv',
                'results/limu_bert_results/uci_10_03_11_2025_20_09/results.csv'
            ]
        }
    }
    
    # Optional: customize scenario and system names
    scenario_names = ['Watch', 'Earbud', 'Blind User', 'HAR']
    system_names = ['Deep-\nSense', 'LIMU-\nBERT', 'UniHAR', 'Our System']
    
    # Create and show the plot
    fig = create_bar_plot(file_paths, scenario_names, system_names, output_file='paper_figs/main_bar_plot__.pdf')
    plt.show()