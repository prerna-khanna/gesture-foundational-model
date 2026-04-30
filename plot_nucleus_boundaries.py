import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from features import detect_nucleus, compute_energy

def load_sample_data(dataset_name, dataset_version="20_120"):
    """Load a random sample from dataset"""
    # Correct path structure: dataset/{DATASET_NAME}/data_{VERSION}.npy
    data_path = f'dataset/{dataset_name}/data_{dataset_version}.npy'
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    print(f"  Loading from: {data_path}")
    data = np.load(data_path)
    print(f"  Loaded data shape: {data.shape}")
    
    # Select random sample
    random_idx = np.random.randint(0, len(data))
    sample = data[random_idx]  # Shape: (seq_len, features)
    
    print(f"  Random sample index: {random_idx}")
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample min/max: {sample.min():.4f} / {sample.max():.4f}")
    
    return sample

def plot_nucleus_boundaries():
    """Plot gyro axis with nucleus boundaries for 4 datasets"""
    
    datasets = [
        'blind_user_filtered',
        'earbud_filtered', 
        'sony_watch',
        'UTD_MHAD'
    ]
    
    dataset_labels = [
        'Hand Gesture (BU)',
        'Earbud Gesture (SU)',
        'Hand Gesture (SU)',
        'Hand Gesture (SU)'
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 3))
    
    for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        print(f"\n{'='*60}")
        print(f"Processing {dataset}...")
        print(f"{'='*60}")
        
        # Load random sample
        sample = load_sample_data(dataset)
        if sample is None:
            print(f"Skipping {dataset} - data not found")
            axes[idx].text(0.5, 0.5, f'{label}\nNOT FOUND', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        try:
            # Extract gyro features (last 3 columns: gyro_x, gyro_y, gyro_z)
            gyro_data = sample[:, -3:]  # Shape: (seq_len, 3)
            print(f"  Gyro data shape: {gyro_data.shape}")
            print(f"  Gyro data min/max: {gyro_data.min():.4f} / {gyro_data.max():.4f}")
            
            # Calculate magnitude of gyro for visualization
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            print(f"  Gyro magnitude min/max: {gyro_magnitude.min():.4f} / {gyro_magnitude.max():.4f}")
            
            # Convert to tensor for energy computation - need 2D tensor (batch, seq_len, features)
            gyro_tensor = torch.from_numpy(gyro_data).unsqueeze(0).float()  # Shape: (1, seq_len, 3)
            
            # Compute energy
            energy = compute_energy(gyro_tensor)  # Shape: (1, seq_len)
            energy_np = energy.numpy().flatten()
            print(f"  Energy shape: {energy_np.shape}")
            print(f"  Energy min/max: {energy_np.min():.4f} / {energy_np.max():.4f}")
            
            # Detect nucleus - need to pass as 2D array (batch, seq_len)
            energy_2d = energy_np.reshape(1, -1)  # Shape: (1, seq_len)
            nucleus_points = detect_nucleus(energy_2d, min_nucleus_width=15, max_nucleus_width=40)
            print(f"  Nucleus points detected: {nucleus_points}")
            
            if nucleus_points is None or len(nucleus_points) == 0:
                print(f"  WARNING: No nucleus detected!")
                nucleus_start = 0
                nucleus_end = len(sample)
            else:
                nucleus_start, nucleus_end = int(nucleus_points[0][0]), int(nucleus_points[0][1])
            
            print(f"  Nucleus boundaries: [{nucleus_start}, {nucleus_end}]")
            
            # Get timesteps
            timesteps = np.arange(len(sample))
            
            # Plot
            ax = axes[idx]
            
            # Plot gyro magnitude
            ax.plot(timesteps, gyro_magnitude, linewidth=2.5, label='Gyro Magnitude', 
                    color='steelblue', alpha=0.8)
            
            # Highlight nucleus region with shading
            ax.axvspan(nucleus_start, nucleus_end, alpha=0.25, color='red', label='Nucleus Region')
            
            # Mark nucleus boundaries with vertical lines
            ax.axvline(nucleus_start, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(nucleus_end, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Set title
            ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Y-axis label only for first plot
            if idx == 0:
                ax.set_ylabel('Gyroscope Magnitude', fontsize=11, fontweight='bold')
            
            # X-axis label only for first plot, add to bottom for all
            ax.set_xlim(0, len(sample))
            
            # Add legend only to last plot
            if idx == 3:
                ax.legend(loc='upper right', fontsize=10)
            
        except Exception as e:
            print(f"  ERROR processing {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            axes[idx].text(0.5, 0.5, f'{label}\nERROR: {str(e)}', 
                          ha='center', va='center', transform=axes[idx].transAxes, fontsize=12)
            continue
    
    # Add shared x-axis label at the bottom
    fig.text(0.5, -0.01, 'Timestep', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig('plots/nucleus_boundaries_comparison.pdf', dpi=300, bbox_inches='tight')
    print("\n" + "="*60)
    print("Plot saved to: plots/nucleus_boundaries_comparison.pdf")
    print("="*60)
    plt.show()

if __name__ == '__main__':
    plot_nucleus_boundaries()