import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_existing_matrix(similarity_matrix, class_labels=None, title="Similarity Matrix Analysis"):
    """
    Analyze an existing similarity matrix to identify patterns and potential issues.
    
    Args:
        similarity_matrix: The similarity matrix to analyze
        class_labels: Optional labels for classes
        title: Title for the visualization
    
    Returns:
        results: Dictionary with analysis results
    """
    # Convert similarity to distance if needed
    if np.mean(np.diag(similarity_matrix)) > 0.5:  # Heuristic to detect similarity matrix
        # Higher values on diagonal usually indicate similarity, not distance
        distance_matrix = 1.0 - similarity_matrix
    else:
        distance_matrix = similarity_matrix  # Already a distance matrix
    
    # Calculate class separation metrics
    n_classes = distance_matrix.shape[0]
    
    # Create default labels if none provided
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    
    # Average distance to other classes for each class
    avg_distances = []
    for i in range(n_classes):
        # Exclude self-distance (diagonal)
        other_distances = [distance_matrix[i, j] for j in range(n_classes) if i != j]
        avg_distances.append(np.mean(other_distances))
    
    # Find most and least distinct classes
    most_distinct_idx = np.argmax(avg_distances)
    least_distinct_idx = np.argmin(avg_distances)
    
    # Find pairs of most similar classes
    most_similar_dist = float('inf')
    most_similar_pair = (0, 0)
    
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            if distance_matrix[i, j] < most_similar_dist:
                most_similar_dist = distance_matrix[i, j]
                most_similar_pair = (i, j)
    
    # Return analysis results
    results = {
        "avg_distances": avg_distances,
        "most_distinct_class": {
            "index": most_distinct_idx,
            "label": class_labels[most_distinct_idx] if most_distinct_idx < len(class_labels) else f"Class {most_distinct_idx}",
            "avg_distance": avg_distances[most_distinct_idx]
        },
        "least_distinct_class": {
            "index": least_distinct_idx,
            "label": class_labels[least_distinct_idx] if least_distinct_idx < len(class_labels) else f"Class {least_distinct_idx}",
            "avg_distance": avg_distances[least_distinct_idx]
        },
        "most_similar_pair": {
            "indices": most_similar_pair,
            "labels": (class_labels[most_similar_pair[0]] if most_similar_pair[0] < len(class_labels) else f"Class {most_similar_pair[0]}", 
                      class_labels[most_similar_pair[1]] if most_similar_pair[1] < len(class_labels) else f"Class {most_similar_pair[1]}"),
            "distance": most_similar_dist
        }
    }
    
    return results

def find_similar_gestures(similarity_matrix, gesture_idx, sorted_labels, n_similar=3):
    """
    Find the most similar gestures to a target gesture.
    
    Args:
        similarity_matrix: Similarity matrix
        gesture_idx: Index of the target gesture
        sorted_labels: List of labels corresponding to matrix indices
        n_similar: Number of similar gestures to return
        
    Returns:
        similar_gestures: List of (gesture_id, similarity) tuples
    """
    # Get similarities to target gesture
    similarities = [(sorted_labels[i], similarity_matrix[gesture_idx, i]) 
                    for i in range(len(sorted_labels)) if i != gesture_idx]
    
    # Sort by similarity (descending)
    similar_gestures = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Return top n_similar
    return similar_gestures[:n_similar]

def compare_matrices(gesture_sim_matrix, bert_sim_matrix, sorted_labels, activity_labels):
    """
    Compare gesture similarity matrix with BERT similarity matrix.
    
    Args:
        gesture_sim_matrix: Similarity matrix from gesture embeddings
        bert_sim_matrix: Similarity matrix from BERT embeddings
        sorted_labels: Labels corresponding to gesture_sim_matrix
        activity_labels: Activity labels for gestures
        
    Returns:
        comparison_results: Dictionary with comparison results
    """
    # Only compare gestures that have BERT embeddings
    valid_indices = [i for i, label in enumerate(sorted_labels) if label < len(activity_labels)]
    
    # Calculate differences between matrices
    differences = []
    largest_diff_pair = None
    largest_diff = -1
    
    most_similar_in_both = []
    least_similar_in_both = []
    
    for i in valid_indices:
        for j in valid_indices:
            if i != j:
                # Get the gesture labels
                g_i, g_j = sorted_labels[i], sorted_labels[j]
                
                # Get similarities from both matrices
                gesture_sim = gesture_sim_matrix[i, j]
                bert_sim = bert_sim_matrix[g_i, g_j]
                
                # Calculate difference
                diff = abs(gesture_sim - bert_sim)
                differences.append((g_i, g_j, gesture_sim, bert_sim, diff))
                
                # Track largest difference
                if diff > largest_diff:
                    largest_diff = diff
                    largest_diff_pair = (g_i, g_j)
    
    # Sort by difference (descending)
    differences.sort(key=lambda x: x[4], reverse=True)
    
    # Sort by average similarity (descending)
    avg_similarities = [(g_i, g_j, (gesture_sim + bert_sim) / 2) 
                       for g_i, g_j, gesture_sim, bert_sim, _ in differences]
    avg_similarities.sort(key=lambda x: x[2], reverse=True)
    
    most_similar_in_both = avg_similarities[:3]  # Top 3 most similar pairs
    least_similar_in_both = avg_similarities[-3:]  # Bottom 3 least similar pairs
    
    # Prepare results
    comparison_results = {
        "largest_differences": [(g_i, g_j, gesture_sim, bert_sim, diff) 
                               for g_i, g_j, gesture_sim, bert_sim, diff in differences[:5]],
        "most_similar_in_both": most_similar_in_both,
        "least_similar_in_both": least_similar_in_both,
        "average_difference": np.mean([diff for _, _, _, _, diff in differences])
    }
    
    return comparison_results

def evaluate_gesture_diversity(similarity_matrix, sorted_labels=None):
    """
    Evaluate how diverse the gesture set is.
    
    Args:
        similarity_matrix: Similarity matrix
        sorted_labels: Optional labels for gestures
        
    Returns:
        diversity_metrics: Dictionary with diversity metrics
    """
    # Convert similarity to distance if needed
    if np.mean(np.diag(similarity_matrix)) > 0.5:
        distance_matrix = 1.0 - similarity_matrix
    else:
        distance_matrix = similarity_matrix
    
    # Calculate average inter-class distance
    n_classes = distance_matrix.shape[0]
    
    total_distance = 0
    count = 0
    
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            total_distance += distance_matrix[i, j]
            count += 1
    
    avg_distance = total_distance / count if count > 0 else 0
    
    # Calculate standard deviation of inter-class distances
    std_distance = np.std([distance_matrix[i, j] 
                           for i in range(n_classes) 
                           for j in range(i+1, n_classes)])
    
    # Calculate minimum inter-class distance
    min_distance = np.min([distance_matrix[i, j] 
                          for i in range(n_classes) 
                          for j in range(i+1, n_classes)])
    
    # Find most confusable pair
    min_i, min_j = 0, 0
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            if distance_matrix[i, j] == min_distance:
                min_i, min_j = i, j
    
    # Prepare results
    diversity_metrics = {
        "average_distance": avg_distance,
        "std_distance": std_distance,
        "min_distance": min_distance,
        "most_confusable_pair": (min_i, min_j) if sorted_labels is None else (sorted_labels[min_i], sorted_labels[min_j]),
        "diversity_score": avg_distance / (1.0 + std_distance)  # Higher average and lower std is better
    }
    
    return diversity_metrics

def suggest_gesture_removal(similarity_matrix, sorted_labels=None):
    """
    Suggest which gesture to remove to maximize diversity.
    
    Args:
        similarity_matrix: Similarity matrix
        sorted_labels: Optional labels for gestures
        
    Returns:
        removal_suggestions: List of (index, score) tuples
    """
    n_classes = similarity_matrix.shape[0]
    
    # Try removing each gesture and calculate diversity
    removal_impact = []
    
    for i in range(n_classes):
        # Create a new matrix without this gesture
        reduced_matrix = np.delete(np.delete(similarity_matrix, i, axis=0), i, axis=1)
        
        # Calculate diversity metrics for the reduced matrix
        diversity = evaluate_gesture_diversity(reduced_matrix)
        
        # Record the gesture and the diversity score
        impact = {
            "index": i,
            "label": sorted_labels[i] if sorted_labels is not None else i,
            "diversity_score": diversity["diversity_score"]
        }
        
        removal_impact.append(impact)
    
    # Sort by diversity score (descending - higher is better)
    removal_impact.sort(key=lambda x: x["diversity_score"], reverse=True)
    
    return removal_impact