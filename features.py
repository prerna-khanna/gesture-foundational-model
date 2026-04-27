import numpy as np
import torch

def compute_energy(seqs):
    """
    Compute energy of the input sequence.
    Args:
        seqs: Tensor (batch_size, seq_len, feature_size) representing IMU sequences.
    
    Returns:
        energy: A list or tensor representing the energy of the sequence.
    """
    energy = torch.sqrt((seqs ** 2).sum(dim=-1))  # Simple norm-based energy calculation
    return energy


def to_numpy(tensor):
    if tensor.device.type == 'mps':
        return tensor.detach().cpu().numpy()
    elif tensor.is_cuda:
        return tensor.cpu().numpy()
    return tensor.numpy()



def detect_nucleus(energy, window=20, nucleus_thres=8):
   
    batch_nucleus_points = []

    # Loop over each sequence in the batch
    for sequence_energy in energy:
        change_pts = []

        # Convert each sequence to a list of scalars (optional if already 1D)
        sequence_energy = to_numpy(sequence_energy)

        # Sliding window to detect energy changes
        for i in range(len(sequence_energy) - 15):
            if abs(sequence_energy[i + 15] - sequence_energy[i]) > nucleus_thres:
                change_pts.append(i)

        # If no change points are detected, use default nucleus points
        if not change_pts:
            filtered_change_pts = [0, min(len(sequence_energy), window)]
        else:
            # Adjust detected change points
            change_pts = list(map(lambda x: x + window, change_pts))
            
            # Filter close change points
            filtered_change_pts = [change_pts[0]]
            for i in range(1, len(change_pts)):
                if change_pts[i] - filtered_change_pts[-1] >= window:
                    filtered_change_pts.append(change_pts[i])

            filtered_change_pts = filtered_change_pts[:2]

            # Adjust if only one change point detected
            if len(filtered_change_pts) == 1:
                filtered_change_pts.append(change_pts[-1] + 10)

        batch_nucleus_points.append(filtered_change_pts)

    return batch_nucleus_points  # Returns nucleus points for each sequence in the batch """

# def detect_nucleus(energy, min_nucleus_width=15, max_nucleus_width=40):
#     """
#     Adaptive nucleus detection that works across different energy profiles.
    
#     Parameters:
#     - energy: Tensor (batch_size, sequence_length) containing energy values
#     - min_nucleus_width: Minimum width of the nucleus region
#     - max_nucleus_width: Maximum width of the nucleus region
    
#     Returns:
#     - batch_nucleus_points: list of lists, each containing start and end points of the nucleus
#     """
#     batch_nucleus_points = []
    
#     # Loop over each sequence in the batch
#     for sequence_energy in energy:
#         # Convert to numpy for easier manipulation
#         if isinstance(sequence_energy, torch.Tensor):
#             if sequence_energy.is_cuda:
#                 sequence_energy = sequence_energy.cpu().numpy()
#             elif hasattr(sequence_energy, 'device') and sequence_energy.device.type == 'mps':
#                 sequence_energy = sequence_energy.to('cpu').numpy()
#             else:
#                 sequence_energy = sequence_energy.numpy()
        
#         # Get sequence length and normalize energy to [0,1]
#         seq_len = len(sequence_energy)
#         energy_min = np.min(sequence_energy)
#         energy_max = np.max(sequence_energy)
        
#         # Handle flat energy case
#         if energy_max - energy_min < 1e-6:
#             # Default to the middle section if energy is flat
#             mid_point = seq_len // 2
#             nucleus_points = [
#                 max(0, mid_point - min_nucleus_width//2),
#                 min(seq_len, mid_point + min_nucleus_width//2)
#             ]
#             batch_nucleus_points.append(nucleus_points)
#             continue
        
#         # Normalize energy
#         norm_energy = (sequence_energy - energy_min) / (energy_max - energy_min)
        
#         # Compute gradient (first derivative)
#         gradient = np.gradient(norm_energy)
        
#         # Find significant transitions (large gradient values)
#         abs_gradient = np.abs(gradient)
#         gradient_threshold = np.percentile(abs_gradient, 90)  # Adaptive threshold
        
#         # Find indices where gradient exceeds threshold
#         significant_changes = np.where(abs_gradient > gradient_threshold)[0]
        
#         if len(significant_changes) < 2:
#             # Not enough significant changes, use energy-based approach
#             # Find where energy exceeds half of its range
#             active_indices = np.where(norm_energy > 0.5)[0]
            
#             if len(active_indices) > 0:
#                 # Use the active region as nucleus
#                 start = max(0, active_indices[0])
#                 end = min(seq_len, active_indices[-1] + 1)
                
#                 # Ensure minimum width
#                 if end - start < min_nucleus_width:
#                     mid = (start + end) // 2
#                     start = max(0, mid - min_nucleus_width // 2)
#                     end = min(seq_len, mid + min_nucleus_width // 2)
                
#                 # Limit maximum width
#                 if end - start > max_nucleus_width:
#                     mid = (start + end) // 2
#                     start = max(0, mid - max_nucleus_width // 2)
#                     end = min(seq_len, mid + max_nucleus_width // 2)
                
#                 nucleus_points = [start, end]
#             else:
#                 # Default to the middle if no active region
#                 mid_point = seq_len // 2
#                 nucleus_points = [
#                     max(0, mid_point - min_nucleus_width//2),
#                     min(seq_len, mid_point + min_nucleus_width//2)
#                 ]
#         else:
#             # At least two significant changes detected
#             # Use the first and last significant changes as boundaries
#             transitions = []
            
#             # Group consecutive change points
#             current_group = [significant_changes[0]]
#             for i in range(1, len(significant_changes)):
#                 if significant_changes[i] - significant_changes[i-1] <= 3:  # Consider consecutive if within 3 time steps
#                     current_group.append(significant_changes[i])
#                 else:
#                     transitions.append(np.mean(current_group))
#                     current_group = [significant_changes[i]]
            
#             if current_group:
#                 transitions.append(np.mean(current_group))
            
#             # Use the first and last transition points if we have at least two
#             if len(transitions) >= 2:
#                 start = max(0, int(transitions[0]))
#                 end = min(seq_len, int(transitions[-1]))
#             else:
#                 # If only one transition, look at energy levels before and after
#                 transition = int(transitions[0])
#                 if np.mean(norm_energy[:transition]) > np.mean(norm_energy[transition:]):
#                     # Higher energy before transition
#                     start = 0
#                     end = min(transition + min_nucleus_width//2, seq_len)
#                 else:
#                     # Higher energy after transition
#                     start = max(0, transition - min_nucleus_width//2)
#                     end = min(seq_len, seq_len)
            
#             # Ensure minimum width
#             if end - start < min_nucleus_width:
#                 mid = (start + end) // 2
#                 start = max(0, mid - min_nucleus_width // 2)
#                 end = min(seq_len, mid + min_nucleus_width // 2)
            
#             # Limit maximum width
#             if end - start > max_nucleus_width:
#                 mid = (start + end) // 2
#                 start = max(0, mid - max_nucleus_width // 2)
#                 end = min(seq_len, mid + max_nucleus_width // 2)
            
#             nucleus_points = [start, end]
        
#         batch_nucleus_points.append(nucleus_points)
    
#     return batch_nucleus_points

def detect_nucleus(energy, min_nucleus_width=15, max_nucleus_width=40):
    """
    Fixed-center nucleus detector with the same signature as the adaptive version.
    This implementation ignores the min/max args and returns a middle-50-sample
    window (25 left, 25 right) for each sequence in `energy`.

    Args:
        energy: iterable or array of shape (n_sequences, seq_len) or (seq_len,) for single sequence
        min_nucleus_width, max_nucleus_width: accepted for API compatibility but unused

    Returns:
        list of [start, end] pairs for each sequence
    """
    fixed_width = 50
    half = fixed_width // 2
    batch_nucleus_points = []
    for seq in energy:
        # try to coerce to numpy array if needed
        if hasattr(seq, 'cpu') and hasattr(seq, 'numpy'):
            try:
                seq = seq.cpu().numpy()
            except Exception:
                seq = np.asarray(seq)
        elif not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)

        seq_len = len(seq)
        mid = seq_len // 2
        start = max(0, mid - half)
        end = min(seq_len, mid + half)
        batch_nucleus_points.append([int(start), int(end)])

    return batch_nucleus_points

#filtered_change_pts = detect_nucleus(energy)

def calculate_significant_axis(seqs):
    # Calculate the axis with maximum rotational activity (x=0, y=1, z=2)
    abs_rotations = torch.abs(seqs[:, :, 3:6])  # Assumes last three features are rotations
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)  # Shape: (batch_size,)
    return sig_axis
