import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns

# Ensure the src directory is in Python's path so imports work properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import necessary modules
from src.recommendation.recommender import GestureRecommender
from src.embedders.bert_embedder import SemanticEmbedder
from src.analysis.matrix_analysis import analyze_existing_matrix, compare_matrices
from src.data_utils.gesture_loader import get_gesture_embeddings, load_dataset, find_new_gestures_for_user

# Define activity labels and descriptions for all gestures
ACTIVITY_LABELS = [
    "up", "down", "left", "right", 
    "rotate wrist and right", "rotate wrist and left", 
    "flick wrist and up", "flick wrist and down", "flick wrist and left", "flick wrist and right", 
    "square", "circle", "triangle", "question mark", "infinity"
]

# BERT descriptions for each gesture - the semantic meaning
DESCRIPTIONS = [
    "a Vertical upward motion gesture with properties: primary type: directional, direction: up, complexity: simple",
    "a Vertical downward motion gesture with properties: primary type: directional, direction: down, complexity: simple",
    "a Horizontal lateral left motion gesture with properties: primary type: directional, direction: left, complexity: simple",
    "a Horizontal lateral right motion gesture with properties: primary type: directional, direction: right, complexity: simple",
    
    "a Clockwise wrist rotation gesture with properties: primary type: rotational, direction: clockwise, complexity: complex",
    "a Anticlockwise wrist rotation gesture with properties: primary type: rotational, direction: anticlockwise, complexity: complex",
    
    "a Sharp upward jerking gesture with properties: primary type: complex, direction: up, complexity: complex",
    "a Sharp downward jerking gesture with properties: primary type: complex, direction: down, complexity: complex",
    "a Sharp leftward jerking gesture with properties: primary type: complex, direction: left, complexity: complex",
    "a Sharp rightward jerking gesture with properties: primary type: complex, direction: right, complexity: complex",
    
    "a Square tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
    "a Circle tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
    "a Triangle tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
    "a Question mark tracing gesture with properties: primary type: complex, direction: mixed, complexity: complex",
    "a Figure eight tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex"
]

def normalize_matrix(matrix):
    """
    Normalize a matrix so each row sums to 1.
    
    Args:
        matrix: Input matrix
        
    Returns:
        normalized_matrix: Normalized matrix where each row sums to 1
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Handle zero rows to avoid division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

def combine_similarity_matrices(phy_dist_matrix, bert_sim_matrix, sorted_labels, weight_physical=0.7):
    """
    Combine physical distance matrix and semantic similarity matrix.
    
    Args:
        phy_dist_matrix: Physical distance matrix (lower = more similar)
        bert_sim_matrix: Semantic similarity matrix (higher = more similar)
        sorted_labels: Labels for the physical similarity matrix
        weight_physical: Weight for physical distance (0-1)
            
    Returns:
        combined_matrix: Combined distance matrix (lower = more similar)
    """
    # Create a new matrix with the same dimensions as the physical distance matrix
    n = phy_dist_matrix.shape[0]
    combined_matrix = np.zeros_like(phy_dist_matrix)
    
    # Weight for semantic distance
    weight_semantic = 1.0 - weight_physical
    
    # Convert BERT similarity to distance (1 - similarity)
    # So now both matrices use the same convention: lower = more similar
    
    for i in range(n):
        for j in range(n):
            # Get the gesture IDs
            gesture_i = sorted_labels[i]
            gesture_j = sorted_labels[j]
            
            # Check if these gesture IDs exist in the BERT matrix
            if gesture_i < len(DESCRIPTIONS) and gesture_j < len(DESCRIPTIONS):
                # Convert BERT similarity to distance and combine with physical distance
                bert_dist = 1.0 - bert_sim_matrix[gesture_i, gesture_j]
                combined_matrix[i, j] = (
                    weight_physical * phy_dist_matrix[i, j] + 
                    weight_semantic * bert_dist
                )
            else:
                # If they don't exist in BERT matrix, use only physical distance
                combined_matrix[i, j] = phy_dist_matrix[i, j]
    
    return combined_matrix

def visualize_matrices(phy_sim_matrix, bert_sim_matrix, combined_matrix, sorted_labels):
    """
    Visualize the original, BERT, and combined similarity matrices.
    
    Args:
        phy_sim_matrix: Physical similarity matrix
        bert_sim_matrix: BERT similarity matrix
        combined_matrix: Combined similarity matrix
        sorted_labels: Labels for the matrices
    """
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Get labels for visualization
    labels = [ACTIVITY_LABELS[i] if i < len(ACTIVITY_LABELS) else f"Gesture {i}" for i in sorted_labels]
    
    # Plot physical similarity matrix
    sns.heatmap(phy_sim_matrix, annot=False, cmap='Blues', xticklabels=labels, 
                yticklabels=labels, ax=axs[0], cbar=True)
    axs[0].set_title('Physical Similarity Matrix')
    
    # Create a version of the BERT matrix with only the gestures in sorted_labels
    valid_indices = [i for i in sorted_labels if i < len(DESCRIPTIONS)]
    bert_subset = np.zeros((len(sorted_labels), len(sorted_labels)))
    
    for i, label_i in enumerate(sorted_labels):
        for j, label_j in enumerate(sorted_labels):
            if label_i < len(DESCRIPTIONS) and label_j < len(DESCRIPTIONS):
                bert_subset[i, j] = bert_sim_matrix[label_i, label_j]
    
    # Plot BERT similarity matrix subset
    sns.heatmap(bert_subset, annot=False, cmap='Greens', xticklabels=labels, 
                yticklabels=labels, ax=axs[1], cbar=True)
    axs[1].set_title('Semantic (BERT) Similarity Matrix')
    
    # Plot combined similarity matrix
    sns.heatmap(combined_matrix, annot=False, cmap='Reds', xticklabels=labels, 
                yticklabels=labels, ax=axs[2], cbar=True)
    axs[2].set_title('Combined Similarity Matrix')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig("similarity_matrices_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # ==== STEP 1: SETUP AND INITIALIZATION ====
    # Define the datasets we're working with
    filtered_dataset = 'blind_user_filtered'  # This is our current dataset
    source_dataset = 'blind_user'             # This is the larger dataset we can pull gestures from
    
    # Set the gesture we want to analyze
    gesture_id = 13  # You can change this to any gesture ID you're interested in
    threshold = 0.4  # Threshold for recommendation
    
    # Weight for physical vs semantic similarity
    physical_weight = 0.7  # 70% physical, 30% semantic

    # List of users to analyze
    # We'll combine results from all these users
    user_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Analyze all users
    
    print(f"=== INITIALIZATION ===")
    print(f"Filtered dataset: {filtered_dataset}")
    print(f"Source dataset: {source_dataset}")
    print(f"Analyzing gesture ID {gesture_id} across users: {user_ids}")
    print(f"Using physical weight: {physical_weight}, semantic weight: {1-physical_weight}")
    
    # ==== STEP 2: PREPARE EMBEDDINGS ====
    # Initialize BERT embedder for semantic analysis
    print("\n=== PREPARING EMBEDDINGS ===")
    print("Initializing BERT embedder...")
    bert_embedder = SemanticEmbedder()
    
    # Calculate similarity matrix from BERT embeddings
    print("Calculating BERT similarity matrix...")
    bert_sim_matrix = bert_embedder.compute_similarity_matrix(DESCRIPTIONS)
    
    # Initialize the gesture recommender
    print(f"Initializing gesture recommender...")
    recommender = GestureRecommender(filtered_dataset, ACTIVITY_LABELS, DESCRIPTIONS, bert_embedder)
    
    # ==== STEP 3: COMBINE SIMILARITY MATRICES ====
    print("\n=== COMBINING SIMILARITY MATRICES ===")
    
    # Get the physical similarity matrix and labels
    physical_sim_matrix = recommender.sim_matrix
    sorted_labels = recommender.sorted_labels
    
    # Combine the matrices
    print(f"Combining physical and semantic similarity matrices with weights: {physical_weight}/{1-physical_weight}...")
    combined_matrix = combine_similarity_matrices(
        physical_sim_matrix, 
        bert_sim_matrix, 
        sorted_labels,
        weight_physical=physical_weight
    )
    
    # Visualize the matrices
    print("Visualizing similarity matrices...")
    visualize_matrices(physical_sim_matrix, bert_sim_matrix, combined_matrix, sorted_labels)
    
    # ==== STEP 4: ANALYZE CURRENT GESTURE SET ====
    print("\n=== ANALYZING CURRENT GESTURE SET ===")
    # Analyze using the combined matrix
    matrix_analysis = analyze_existing_matrix(
        combined_matrix, 
        [ACTIVITY_LABELS[i] if i < len(ACTIVITY_LABELS) else f"Gesture {i}" for i in sorted_labels],
        title="Analysis of Combined Similarity Matrix"
    )
    
    # Print key insights about the current gesture set
    print(f"Most distinct gesture: {matrix_analysis['most_distinct_class']['label']}")
    print(f"Least distinct gesture: {matrix_analysis['least_distinct_class']['label']}")
    print(f"Most similar gesture pair: {matrix_analysis['most_similar_pair']['labels'][0]} and {matrix_analysis['most_similar_pair']['labels'][1]}")
    
    # Override the recommender's similarity matrix with our combined matrix
    # This affects all subsequent analyses
    recommender.sim_matrix = combined_matrix
    
    # ==== STEP 5: ANALYZE GESTURE ACROSS MULTIPLE USERS ====
    gesture_name = ACTIVITY_LABELS[gesture_id] if gesture_id < len(ACTIVITY_LABELS) else f"Gesture {gesture_id}"
    print(f"\n=== ANALYZING GESTURE: {gesture_name} (ID: {gesture_id}) ACROSS MULTIPLE USERS ===")
    
    # Store results for each user
    user_results = {}
    valid_user_count = 0
    
    # Collect aggregate scores
    total_distinctiveness = 0
    total_coverage = 0
    total_performance = 0
    
    # Analyze the gesture for each user
    for user_idx in user_ids:
        print(f"\n--- Analyzing for User {user_idx} ---")
        
        # Check if this user has the gesture in the source dataset
        user_gestures = find_new_gestures_for_user(filtered_dataset, source_dataset, user_idx)
        has_gesture = gesture_id in user_gestures or any(g == gesture_id for g in user_gestures)
        
        if not has_gesture:
            print(f"User {user_idx} doesn't have gesture {gesture_id} in the source dataset. Skipping.")
            continue
        
        # Make recommendation for this user
        recommendation, scores, explanation = recommender.make_recommendation(
            gesture_id,      # The gesture ID to analyze
            user_idx,        # The specific user to analyze
            source_dataset,  # Source dataset to look in
            threshold=threshold  # The threshold for recommendation
        )
        
        # Store the results
        if 'combined' in scores:  # Only store valid results
            user_results[user_idx] = {
                "recommendation": recommendation,
                "scores": scores,
                "explanation": explanation
            }
            
            # Add to totals for aggregation
            total_distinctiveness += scores.get("distinctiveness", 0)
            total_coverage += scores.get("coverage", 0)
            total_performance += scores.get("performance", 0)
            valid_user_count += 1
            
            # Display detailed explanation
            print(f"User {user_idx} analysis:")
            for line in explanation:
                print(f"  {line}")
            
            # Visualize the recommendation for this user
            fig = recommender.visualize_recommendation(scores, recommendation, threshold=threshold)
            output_file = f"gesture_{gesture_id}_user_{user_idx}_recommendation.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    # ==== STEP 6: AGGREGATE RESULTS ACROSS USERS ====
    print(f"\n=== AGGREGATING RESULTS FOR GESTURE {gesture_id} ACROSS {valid_user_count} USERS ===")
    
    if valid_user_count == 0:
        print(f"No valid results for gesture {gesture_id}. No users have this gesture or all analyses failed.")
        return
    
    # Calculate average scores
    avg_distinctiveness = total_distinctiveness / valid_user_count
    avg_coverage = total_coverage / valid_user_count
    avg_performance = total_performance / valid_user_count
    
    # Calculate combined score using the same weights as in the recommender
    weights = (0.4, 0.3, 0.3)  # distinctiveness, coverage, performance
    combined_score = (
        weights[0] * avg_distinctiveness + 
        weights[1] * avg_coverage + 
        weights[2] * avg_performance
    )
    
    # Make overall recommendation
    overall_recommendation = combined_score >= threshold
    
    # Print aggregated results
    print(f"Aggregate scores for gesture {gesture_id} ({gesture_name}):")
    print(f"  - Distinctiveness: {avg_distinctiveness:.2f}")
    print(f"  - Coverage: {avg_coverage:.2f}")
    print(f"  - User Performance: {avg_performance:.2f}")
    print(f"  - Combined Score: {combined_score:.2f}")
    print(f"  - Threshold: {threshold}")
    print(f"  - Overall Recommendation: {'ADD' if overall_recommendation else 'DO NOT ADD'} this gesture")
    
    # Create a visualization of aggregated results
    aggregated_scores = {
        "distinctiveness": avg_distinctiveness,
        "coverage": avg_coverage,
        "performance": avg_performance,
        "combined": combined_score
    }
    
    fig = recommender.visualize_recommendation(aggregated_scores, overall_recommendation, threshold=threshold)
    plt.suptitle(f"Aggregate Results for Gesture {gesture_id} ({gesture_name}) Across {valid_user_count} Users", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"gesture_{gesture_id}_aggregate_recommendation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==== STEP 7: VISUALIZE PER-USER COMPARISON ====
    print(f"\n=== CREATING PER-USER COMPARISON FOR GESTURE {gesture_id} ===")
    
    # Extract data for visualization
    user_indices = list(user_results.keys())
    user_distinctiveness = [user_results[u]["scores"].get("distinctiveness", 0) for u in user_indices]
    user_coverage = [user_results[u]["scores"].get("coverage", 0) for u in user_indices]
    user_performance = [user_results[u]["scores"].get("performance", 0) for u in user_indices]
    user_combined = [user_results[u]["scores"].get("combined", 0) for u in user_indices]
    
    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(user_indices))
    width = 0.2
    
    # Plot bars
    ax.bar(x - width*1.5, user_distinctiveness, width, label='Distinctiveness', color='#3498db')
    ax.bar(x - width/2, user_coverage, width, label='Coverage', color='#2ecc71')
    ax.bar(x + width/2, user_performance, width, label='User Performance', color='#f39c12')
    ax.bar(x + width*1.5, user_combined, width, label='Combined Score', color='#e74c3c')
    
    # Add threshold line
    ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.7)
    ax.text(x[-1] + width*2, threshold, f'Threshold = {threshold}', fontsize=10, va='center')
    
    # Customize chart
    ax.set_ylabel('Score (higher is better)')
    ax.set_title(f'Comparison of {gesture_name} (ID: {gesture_id}) Across Users')
    ax.set_xticks(x)
    ax.set_xticklabels([f"User {u}" for u in user_indices])
    ax.legend()
    
    # Add value labels on top of combined score bars
    for i, score in enumerate(user_combined):
        ax.text(i + width*1.5, score + 0.02, f'{score:.2f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"gesture_{gesture_id}_user_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==== STEP 8: DETAILED USER BREAKDOWN ====
    print(f"\n=== USER BREAKDOWN FOR GESTURE {gesture_id} ===")
    print(f"Total users analyzed: {len(user_ids)}")
    print(f"Users with valid data: {valid_user_count}")
    
    recommended_count = sum(1 for u in user_results if user_results[u]["recommendation"])
    print(f"Users for whom the gesture is recommended: {recommended_count} ({recommended_count/valid_user_count*100:.1f}%)")
    
    print("\nIndividual user recommendations:")
    for user_idx in user_indices:
        rec = user_results[user_idx]["recommendation"]
        score = user_results[user_idx]["scores"].get("combined", 0)
        print(f"  - User {user_idx}: {'RECOMMENDED' if rec else 'NOT RECOMMENDED'} (Score: {score:.2f})")
    
    print("\nAnalysis complete. Saved visualizations to:")
    print("- similarity_matrices_comparison.png")
    print(f"- gesture_{gesture_id}_aggregate_recommendation.png")
    print(f"- gesture_{gesture_id}_user_comparison.png")
    print(f"- Individual user recommendations in gesture_{gesture_id}_user_X_recommendation.png files")

if __name__ == "__main__":
    main()