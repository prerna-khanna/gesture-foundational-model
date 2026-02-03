import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

def read_and_get_class_accuracies(file_path):
    """
    Read a CSV file and calculate accuracy for each class.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        class_accuracies: Dictionary mapping class names to their accuracies
        overall_acc: Overall accuracy
        std_dev: Standard deviation for error bars
        total_classes: Total number of unique classes in the dataset
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None, None, None, None
            
        # Read the CSV file
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Extract true labels
        if 'true_label' in df.columns:
            y_true = df['true_label']
        elif 'True_Label' in df.columns:
            y_true = df['True_Label']
        elif 'label' in df.columns:
            y_true = df['label']
        else:
            raise KeyError(f"No column for true labels found in {file_path}")
            
        # Extract predictions
        if 'predicted_label' in df.columns:
            y_pred = df['predicted_label']
        elif 'Predicted_Label' in df.columns:
            y_pred = df['Predicted_Label']
        elif 'prediction' in df.columns:
            y_pred = df['prediction']
        else:
            raise KeyError(f"Neither 'predicted_label', 'Predicted_Label', nor 'prediction' column found in {file_path}")
        
        # Calculate overall accuracy
        overall_acc = accuracy_score(y_true, y_pred)
        print(f"Overall accuracy: {overall_acc:.4f}")
        
        # Get unique classes and count them
        classes = sorted(df['true_label'].unique())
        total_classes = len(classes)
        print(f"Total unique classes: {total_classes}")
        
        # Calculate accuracy for each class
        class_accuracies = {}
        for cls in classes:
            # Get indices for this class
            class_indices = y_true == cls
            
            # If no samples for this class, skip
            if sum(class_indices) == 0:
                continue
            
            # Calculate accuracy for this class
            class_acc = accuracy_score(y_true[class_indices], y_pred[class_indices])
            class_accuracies[cls] = class_acc
            print(f"Class {cls} accuracy: {class_acc:.4f}")
        
        # Calculate standard deviation through bootstrapping
        n_bootstraps = 100
        bootstrapped_scores = []
        
        for i in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            bootstrap_true = y_true.iloc[indices]
            bootstrap_pred = y_pred.iloc[indices]
            
            # Calculate accuracy for this bootstrap sample
            bootstrap_acc = accuracy_score(bootstrap_true, bootstrap_pred)
            bootstrapped_scores.append(bootstrap_acc)
        
        # Standard deviation of bootstrapped accuracies
        std_dev = np.std(bootstrapped_scores)
        
        return class_accuracies, overall_acc, std_dev, total_classes
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None

def create_class_accuracy_plot():
    """
    Create a plot showing accuracy vs number of classes for GestureLens system.
    Only considers the 0.1 label rate for each scenario.
    """
    # Define file paths for each scenario at 0.1 label rate
    file_paths = {
        'scenario1': 'results/final_results/sony_watch_10_03_10_2025_04_23/results.csv',  # SU-Hand
        'scenario2': 'results/final_results/earbud_filtered_10_03_10_2025_05_15/results.csv',  # SU-Earbud
        'scenario3': 'results/final_results/blind_user_filtered_10_03_09_2025_20_06/results.csv'  # BU-Hand
    }
    
    # Define scenario names and colors
    scenario_names = {
        'scenario1': 'Sighted User Hand Gesture',
        'scenario2': 'Sighted User Earbud Gesture',
        'scenario3': 'Blind User Hand Gesture'
    }
    
    colors = {
        'scenario1': 'green',
        'scenario2': 'blue',
        'scenario3': 'red'
    }
    
    # Store data for plotting
    plot_data = {}
    
    # Process each scenario
    for scenario_key, file_path in file_paths.items():
        print(f"\nProcessing {scenario_key} - {file_path}")
        
        # Get class accuracies
        class_accuracies, overall_acc, std_dev, total_classes = read_and_get_class_accuracies(file_path)
        
        if class_accuracies is None or total_classes is None:
            print(f"Could not process {scenario_key}, skipping")
            continue
            
        # Sort classes by accuracy (descending)
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        # Generate x points (number of classes) following the specified rule:
        # Start from 3, increase by 2, and add the total number of classes if it's not already included
        x_points = []
        current = 3
        while current <= total_classes:
            x_points.append(current)
            current += 2
            
        # If the total number of classes is odd and not already included, add it
        if total_classes % 2 == 1 and total_classes not in x_points:
            x_points.append(total_classes)
        # If the total number of classes is even and not already included, add it
        elif total_classes % 2 == 0 and total_classes not in x_points:
            x_points.append(total_classes)
        
        # Sort x_points to ensure they're in ascending order
        x_points = sorted(x_points)
        print(f"X points for {scenario_key}: {x_points}")
        
        # Calculate average accuracy for top N classes
        accuracies = []
        error_bars = []
        
        for n_classes in x_points:
            # Make sure we have enough classes
            if len(sorted_classes) < n_classes:
                print(f"Warning: Not enough classes for {scenario_key}, using all available classes")
                n_classes = len(sorted_classes)
            
            # Get top N classes
            top_n_classes = sorted_classes[:n_classes]
            
            # Calculate average accuracy for top N classes
            avg_acc = sum(acc for _, acc in top_n_classes) / n_classes
            accuracies.append(avg_acc)
            
            # Use the provided std_dev for error bars
            error_bars.append(std_dev)
            
            print(f"{scenario_key} - Top {n_classes} classes avg accuracy: {avg_acc:.4f}")
        
        # Store data for plotting
        plot_data[scenario_key] = {
            'x_points': x_points,
            'accuracies': accuracies,
            'error_bars': error_bars,
            'color': colors[scenario_key],
            'name': scenario_names[scenario_key],
            'total_classes': total_classes
        }
    # check if the acc + error are > 1, if so, set to 1
    for scenario_key, data in plot_data.items():
        for i in range(len(data['accuracies'])):
            if data['accuracies'][i] + data['error_bars'][i] > 1:
                data['accuracies'][i] = 1 - data['error_bars'][i]
    # Create the plot
    plt.figure(figsize=(5, 2.5))
    
    # Plot each scenario
    for scenario_key, data in plot_data.items():
        plt.errorbar(
            data['x_points'],
            data['accuracies'],
            yerr=data['error_bars'],
            fmt='-o',
            capsize=2,
            label=f"{data['name']}",
            color=data['color'],
            linewidth=1.5,
            markersize=4
        )
    
    # Set plot properties
    plt.xlabel('Number of Classes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Find all unique x points across all scenarios to set xticks
    all_x_points = sorted(set([x for data in plot_data.values() for x in data['x_points']]))
    plt.xticks(all_x_points)

    # set size of x and y ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.legend(loc='best', fontsize='small')

    plt.ylim(0.75, 1.01)
    
    # Create directory if it doesn't exist
    os.makedirs('paper_figs', exist_ok=True)
    
    # Save figure
    output_file = 'paper_figs/class_accuracy_comparison.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as '{output_file}'")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    create_class_accuracy_plot()