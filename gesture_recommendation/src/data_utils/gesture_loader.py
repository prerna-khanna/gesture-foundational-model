import numpy as np


def get_gesture_embeddings_from_dataset(dataset_name, target_gesture_id, user_idx=None):
    """
    Get embeddings for a specific gesture from a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to load from
        target_gesture_id: ID of the gesture to analyze
        user_idx: Optional user index to filter by
        
    Returns:
        gesture_embed: Embeddings for the target gesture
        all_embed: All embeddings in the dataset
        all_labels: All labels in the dataset
    """
    # Load dataset
    all_embed, all_labels_raw = load_dataset(dataset_name)
    
    if all_embed is None:
        return None, None, None
    
    # Get all labels (using first frame since all frames have the same label)
    all_labels = all_labels_raw[:, 0, 0]  # First column is gesture ID
    
    # If user_idx provided, also filter by user
    if user_idx is not None:
        user_ids = all_labels_raw[:, 0, 1]  # Second column is user ID
        user_mask = user_ids == user_idx
        
        # Only keep samples from the specified user
        filtered_indices = np.where(user_mask)[0]
        filtered_embed = all_embed[filtered_indices]
        filtered_labels = all_labels[filtered_indices]
    else:
        filtered_embed = all_embed
        filtered_labels = all_labels
    
    # Get indices for the target gesture
    target_indices = np.where(filtered_labels == target_gesture_id)[0]
    
    if len(target_indices) == 0:
        print(f"Warning: Gesture ID {target_gesture_id} not found for user {user_idx} in dataset {dataset_name}")
        return None, filtered_embed, filtered_labels
    
    # Extract embeddings for the target gesture
    gesture_embed = filtered_embed[target_indices]
    
    print(f"Found {len(target_indices)} instances of gesture {target_gesture_id} in {dataset_name}")
    
    return gesture_embed, filtered_embed, filtered_labels



def find_new_gestures_for_user(filtered_dataset, source_dataset, user_idx):
    """
    Find gestures available for a specific user in the source dataset 
    that are not in the filtered dataset.
    
    Args:
        filtered_dataset: name of filtered dataset
        source_dataset: name of source dataset
        user_idx: user index to filter by
        
    Returns:
        new_gestures: list of gesture IDs available for this user in source but not in filtered
    """
    # Load datasets
    filtered_embed, filtered_labels = load_dataset(filtered_dataset)
    source_embed, source_labels = load_dataset(source_dataset)
    
    if filtered_embed is None or source_embed is None:
        print("Error loading datasets")
        return []
    
    # Get unique gestures in filtered dataset (from first frame of each sequence)
    filtered_gestures = set(np.unique(filtered_labels[:, 0, 0]))
    
    # Get gestures for specific user in source dataset (from first frame of each sequence)
    source_user_gestures = []
    for i in range(len(source_labels)):
        if source_labels[i, 0, 1] == user_idx:
            source_user_gestures.append(source_labels[i, 0, 0])
    
    if len(source_user_gestures) == 0:
        print(f"Warning: User {user_idx} not found in source dataset")
        print(f"Available users in source: {np.unique(source_labels[:, 0, 1])}")
        return []
    
    # Get unique gestures for this user
    unique_source_user_gestures = set(np.unique(source_user_gestures))
    
    print(f"User {user_idx} has gestures {sorted(unique_source_user_gestures)} in source dataset")
    
    # Find new gestures (those in source but not in filtered)
    new_gestures = unique_source_user_gestures - filtered_gestures
    
    return sorted(list(new_gestures))


def load_dataset(dataset_name):
    """
    Load embeddings and labels from dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        filtered_embed: Embeddings matrix
        filtered_labels: Labels array
    """
    try:
        # Load dataset embeddings and labels
        filtered_embed = np.load(f'embed/embed_limu_v1_{dataset_name}_20_120.npy') 
        filtered_labels = np.load(f'dataset/{dataset_name}/label_20_120.npy')
        
        # Average embeddings over frames
        filtered_embed = filtered_embed.mean(axis=1)  # avg over 120 frames
        
        print(f"Loaded dataset {dataset_name}: {filtered_embed.shape} embeddings")
        
        return filtered_embed, filtered_labels
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None

def get_gesture_embeddings(filtered_dataset, target_gesture_id, user_idx=None):
    """
    Get embeddings for a specific gesture from the dataset.
    
    Args:
        filtered_dataset: Name of the dataset
        target_gesture_id: ID of the gesture to analyze
        user_idx: Optional user index to filter by
        
    Returns:
        gesture_embed: Embeddings for the target gesture
        all_embed: All embeddings in the dataset
        all_labels: All labels in the dataset
    """
    # Load filtered dataset
    filtered_embed, filtered_labels = load_dataset(filtered_dataset)
    
    if filtered_embed is None or filtered_labels is None:
        return None, None, None
    
    # Get all labels (using first frame since all frames in a sequence have the same label)
    all_labels = filtered_labels[:, 0, 0]  # First column is gesture ID
    
    # If user_idx provided, also filter by user
    if user_idx is not None:
        user_ids = filtered_labels[:, 0, 1]  # Second column is user ID
        user_mask = user_ids == user_idx
        
        # Only keep samples from the specified user
        filtered_embed = filtered_embed[user_mask]
        all_labels = all_labels[user_mask]
    
    # Get indices for the target gesture
    target_indices = np.where(all_labels == target_gesture_id)[0]
    
    if len(target_indices) == 0:
        print(f"Warning: Gesture ID {target_gesture_id} not found in dataset")
        return None, filtered_embed, all_labels
    
    # Extract embeddings for the target gesture
    gesture_embed = filtered_embed[target_indices]
    
    print(f"Found {len(target_indices)} instances of gesture {target_gesture_id}")
    
    return gesture_embed, filtered_embed, all_labels

def calculate_class_distances(embed, label_dict, distance_metric='euclidean'):
    """
    Calculate the average pairwise distance between class instances.
    
    Args:
        embed: embedding matrix (n_samples, n_features)
        label_dict: dictionary mapping labels to indices
        distance_metric: distance metric to use (default: euclidean)
        
    Returns:
        distance_matrix: distance matrix (n_classes, n_classes)
        sorted_labels: list of labels in the order they appear in the distance matrix
    """
    unique_labels = list(label_dict.keys())
    n_classes = len(unique_labels)
    distance_matrix = np.zeros((n_classes, n_classes))
    
    # Calculate distances between classes
    for i, label_i in enumerate(unique_labels):
        indices_i = label_dict[label_i]
        embed_i = embed[indices_i]
        
        for j, label_j in enumerate(unique_labels):
            indices_j = label_dict[label_j]
            embed_j = embed[indices_j]
            
            # Calculate average distance between all pairs of points from both classes
            distances = []
            for idx_i in range(len(indices_i)):
                for idx_j in range(len(indices_j)):
                    # Skip self-comparisons for same class
                    if label_i == label_j and idx_i == idx_j:
                        continue
                    dist = np.linalg.norm(embed_i[idx_i] - embed_j[idx_j])
                    distances.append(dist)
            
            distance_matrix[i, j] = np.mean(distances) if distances else 0
    
    return distance_matrix, unique_labels

def convert_to_similarity(distance_matrix):
    """
    Convert a distance matrix to a similarity matrix.
    
    Args:
        distance_matrix: Distance matrix
        
    Returns:
        similarity_matrix: Similarity matrix (1 - normalized distance)
    """
    # Normalize distances to 0-1 range
    if np.max(distance_matrix) > 0:  # Avoid division by zero
        normalized_dist = distance_matrix / np.max(distance_matrix)
    else:
        normalized_dist = distance_matrix
    
    # Convert distance to similarity (1 - distance)
    similarity_matrix = 1.0 - normalized_dist
    
    return similarity_matrix

def create_label_dict(labels):
    """
    Create a dictionary mapping labels to indices.
    
    Args:
        labels: Array of labels
        
    Returns:
        label_dict: Dictionary mapping labels to indices
    """
    unique_labels = np.unique(labels)
    
    # Create dictionary
    label_dict = {}
    for i in unique_labels:
        label_dict[i] = np.where(labels == i)[0].tolist()
    
    return label_dict

def add_new_gesture_from_dataset(filtered_dataset_name, source_dataset_name, gesture_id, user_idx):
    """
    Adds a new gesture from the source dataset to the filtered dataset and updates the distance matrix.
    Filters by the specified user_idx.
    
    Args:
        filtered_dataset_name: name of the filtered dataset
        source_dataset_name: name of the source dataset
        gesture_id: the gesture ID to add from the source dataset
        user_idx: the user index to filter by
        
    Returns:
        updated_embed: updated embedding matrix including the new gesture
        updated_label_dict: updated dictionary mapping labels to indices
        updated_dist_matrix: updated distance matrix
        updated_labels: list of updated labels
    """
    # Load filtered dataset
    filtered_embed, filtered_labels = load_dataset(filtered_dataset_name)
    
    # Load source dataset
    source_embed, source_labels = load_dataset(source_dataset_name)
    
    if filtered_embed is None or source_embed is None:
        return None, None, None, None
    
    print(f"Looking for gesture ID {gesture_id} for user {user_idx}")
    
    # Get all unique gestures and users from the source dataset
    all_source_gestures = np.unique(source_labels[:, 0, 0])
    all_source_users = np.unique(source_labels[:, 0, 1])
    print(f"Unique users in source: {all_source_users}")
    print(f"Unique gestures in source: {all_source_gestures}")
    
    # Find the indices of samples where the first frame has both the right gesture and user
    gesture_indices = []
    for i in range(len(source_labels)):
        if source_labels[i, 0, 0] == gesture_id and source_labels[i, 0, 1] == user_idx:
            gesture_indices.append(i)
    
    if len(gesture_indices) == 0:
        print(f"Gesture ID {gesture_id} not found for user {user_idx} in source dataset")
        return None, None, None, None
    
    print(f"Found {len(gesture_indices)} instances of gesture {gesture_id} for user {user_idx}")
    
    # Extract embeddings for the selected gesture from the specific user
    gesture_indices = np.array(gesture_indices)
    gesture_embeddings = source_embed[gesture_indices]
    
    # Create new labels for the added gesture
    gesture_labels = np.full(len(gesture_indices), gesture_id)
    
    # Combine with filtered dataset
    updated_embed = np.vstack((filtered_embed, gesture_embeddings))
    updated_label = np.hstack((filtered_labels[:, 0, 0], gesture_labels))
    
    # Create dictionary mapping label to indices
    label_dict = create_label_dict(updated_label)
    
    # Calculate the updated distance matrix
    dist_matrix, sorted_labels = calculate_class_distances(updated_embed, label_dict)
    
    return updated_embed, label_dict, dist_matrix, sorted_labels