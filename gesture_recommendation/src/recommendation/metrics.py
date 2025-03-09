import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_distinctiveness(new_gesture_embed, existing_embeds, dist_matrix=None, n_closest=3):
    """
    Calculate how distinct the new gesture is from existing gestures.
    
    Args:
        new_gesture_embed: Embedding of the new gesture
        existing_embeds: Embeddings of existing gestures
        dist_matrix: Optional distance matrix between existing gestures
        n_closest: Number of closest gestures to consider
        
    Returns:
        distinctiveness_score: Higher score means more distinct
    """
    # Calculate distances from new gesture to all existing gestures
    distances = np.array([np.linalg.norm(new_gesture_embed - embed) for embed in existing_embeds])
    
    # Sort distances and get indices of closest N gestures
    closest_indices = np.argsort(distances)[:n_closest]
    closest_distances = distances[closest_indices]
    
    # Basic distinctiveness is the average distance to closest gestures
    basic_distinctiveness = np.mean(closest_distances)
    
    # If we have a distance matrix, we can weight by confusion
    if dist_matrix is not None:
        # For each closest gesture, check how confused it is with others
        confusion_weights = []
        for idx in closest_indices:
            # High similarity (low distance) to other gestures = high confusion
            # We only care about confusion with gestures other than itself
            other_indices = [i for i in range(len(existing_embeds)) if i != idx]
            confusion = 1.0 - np.mean(dist_matrix[idx, other_indices])
            confusion_weights.append(confusion)
        
        # Adjust distinctiveness - multiply by (1 - average confusion)
        # This means if the closest gestures are already confused, we reduce the score
        weighted_distinctiveness = basic_distinctiveness * (1 - np.mean(confusion_weights))
        return weighted_distinctiveness
    
    return basic_distinctiveness

def calculate_coverage_improvement(new_gesture_embed, existing_embeds, 
                                  new_gesture_bert=None, existing_bert_embeds=None):
    """
    Calculate how much the new gesture improves coverage of the gesture space.
    
    Args:
        new_gesture_embed: Embedding of the new gesture
        existing_embeds: Embeddings of existing gestures
        new_gesture_bert: BERT embedding of new gesture description (optional)
        existing_bert_embeds: BERT embeddings of existing gestures (optional)
        
    Returns:
        coverage_score: Higher score means better coverage improvement
    """
    # Calculate distances from new gesture to all existing gestures
    distances = np.array([np.linalg.norm(new_gesture_embed - embed) for embed in existing_embeds])
    
    # Calculate density around the new gesture (inversely proportional to average distance)
    avg_distance = np.mean(distances)
    density = 1.0 / (avg_distance + 1e-6)  # Adding small epsilon to avoid division by zero
    
    # Convert density to coverage score (lower density = higher coverage improvement)
    physical_coverage = 1.0 - density / (1.0 + density)  # Normalized to 0-1 range
    
    # If we have BERT embeddings, calculate semantic diversity
    if new_gesture_bert is not None and existing_bert_embeds is not None:
        bert_distances = np.array([np.linalg.norm(new_gesture_bert - embed) 
                                  for embed in existing_bert_embeds])
        semantic_diversity = np.mean(bert_distances)
        
        # Normalize semantic diversity to 0-1 range
        # This assumes bert_distances are typically in range 0-2 for normalized embeddings
        normalized_semantic_diversity = min(semantic_diversity / 2.0, 1.0)
        
        # Combine physical and semantic scores (can adjust weights)
        coverage_score = 0.6 * physical_coverage + 0.4 * normalized_semantic_diversity
    else:
        coverage_score = physical_coverage
    
    return coverage_score

def calculate_user_performance(gesture_samples):
    """
    Calculate how consistently the user performs the gesture.
    
    Args:
        gesture_samples: Array of embeddings from multiple attempts of the same gesture
        
    Returns:
        performance_score: Higher score means more consistent performance
    """
    if len(gesture_samples) < 2:
        raise ValueError("Need at least 2 samples to calculate user performance")
    
    # Calculate pairwise distances between all samples
    n_samples = len(gesture_samples)
    pairwise_distances = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist = np.linalg.norm(gesture_samples[i] - gesture_samples[j])
            pairwise_distances.append(dist)
    
    # Calculate average distance
    avg_distance = np.mean(pairwise_distances)
    
    # Convert to performance score (lower distance = higher score)
    # Normalize to 0-1 range where 1 is perfect consistency
    performance_score = 1.0 / (1.0 + avg_distance)
    
    return performance_score


