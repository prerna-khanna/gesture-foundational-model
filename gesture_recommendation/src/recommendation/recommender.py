import numpy as np
import matplotlib.pyplot as plt

class GestureRecommender:
    def __init__(self, dataset_name, activity_labels, descriptions=None, bert_embedder=None):
        """
        Initialize the gesture recommender.
        
        Args:
            dataset_name: Name of the dataset
            activity_labels: List of gesture activity labels
            descriptions: Optional list of gesture descriptions
            bert_embedder: Optional pre-initialized BERT embedder
        """
        from ..embedders.bert_embedder import SemanticEmbedder
        from ..data_utils.gesture_loader import load_dataset, create_label_dict, calculate_class_distances, convert_to_similarity
        
        self.dataset_name = dataset_name
        self.activity_labels = activity_labels
        
        # Initialize BERT embedder if descriptions provided
        if descriptions is not None:
            if bert_embedder is None:
                self.bert_embedder = SemanticEmbedder()
            else:
                self.bert_embedder = bert_embedder
                
            # Calculate BERT similarity matrix once
            self.bert_sim_matrix = self.bert_embedder.compute_similarity_matrix(descriptions)
        else:
            self.bert_embedder = None
            self.bert_sim_matrix = None
            
        # Load all embeddings and labels
        self.all_embed, self.all_labels_raw = load_dataset(dataset_name)
        
        if self.all_embed is None:
            print(f"Error: Could not load dataset {dataset_name}")
            return
            
        # Extract gesture IDs from first frame
        self.all_labels = self.all_labels_raw[:, 0, 0]
        
        # Create label dictionary
        self.label_dict = create_label_dict(self.all_labels)
        
        # Calculate similarity matrix
        self.dist_matrix, self.sorted_labels = calculate_class_distances(self.all_embed, self.label_dict)
        self.sim_matrix = convert_to_similarity(self.dist_matrix)
        
        print(f"Initialized recommender for {dataset_name} with {len(self.sorted_labels)} gestures")
    
    def calculate_distinctiveness(self, new_gesture_embed, existing_indices=None, n_closest=3):
        """
        Calculate how distinct the new gesture is from existing gestures.
        
        Args:
            new_gesture_embed: Embedding of the new gesture (can be average of multiple samples)
            existing_indices: Indices of gestures to compare with (default: all)
            n_closest: Number of closest gestures to consider
            
        Returns:
            distinctiveness_score: Higher score means more distinct (0-1 range)
            closest_gestures: List of (gesture_id, similarity) tuples for closest gestures
        """
        # If existing_indices not provided, use all gestures
        if existing_indices is None:
            existing_indices = list(range(len(self.sorted_labels)))
        
        # Calculate similarity between new gesture and all existing gestures
        similarities = []
        for idx in existing_indices:
            # Get class label
            class_label = self.sorted_labels[idx]
            
            # Get all samples with this label
            class_indices = np.where(self.all_labels == class_label)[0]
            
            if len(class_indices) == 0:
                continue
                
            # Calculate average similarity to this class
            class_similarities = []
            for i in class_indices:
                # Calculate cosine similarity
                similarity = np.dot(new_gesture_embed, self.all_embed[i]) / (
                    np.linalg.norm(new_gesture_embed) * np.linalg.norm(self.all_embed[i])
                )
                class_similarities.append(similarity)
            
            # Use average similarity to this class
            avg_similarity = np.mean(class_similarities)
            similarities.append((class_label, avg_similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get n closest gestures
        closest_gestures = similarities[:n_closest]
        
        # Calculate distinctiveness as 1 - average similarity to closest gestures
        avg_similarity = np.mean([sim for _, sim in closest_gestures])
        distinctiveness = 1.0 - avg_similarity
        
        # Normalize to 0-1 range with a non-linear transformation
        # This makes medium distances more distinct
        normalized_distinctiveness = np.clip(distinctiveness * 1.5, 0, 1)
        
        return normalized_distinctiveness, closest_gestures
    
    def calculate_coverage_improvement(self, new_gesture_embed, new_gesture_id=None):
        """
        Calculate how much the new gesture improves coverage of the gesture space.
        
        Args:
            new_gesture_embed: Embedding of the new gesture
            new_gesture_id: ID of the new gesture (for BERT similarity)
            
        Returns:
            coverage_score: Higher score means better coverage (0-1 range)
        """
        # Calculate physical coverage (density-based)
        # Lower density around the gesture = better coverage
        similarities = []
        for i in range(len(self.sorted_labels)):
            # Get label
            label = self.sorted_labels[i]
            
            # Get indices for this label
            indices = np.where(self.all_labels == label)[0]
            
            # Calculate similarities
            for idx in indices:
                similarity = np.dot(new_gesture_embed, self.all_embed[idx]) / (
                    np.linalg.norm(new_gesture_embed) * np.linalg.norm(self.all_embed[idx])
                )
                similarities.append(similarity)
        
        # Calculate density (average similarity)
        avg_similarity = np.mean(similarities)
        
        # Convert to coverage score (lower similarity = higher coverage)
        physical_coverage = 1.0 - avg_similarity
        
        # Add semantic component if new_gesture_id is provided and BERT embeddings are available
        if new_gesture_id is not None and self.bert_sim_matrix is not None:
            if new_gesture_id < len(self.activity_labels):
                # Calculate average semantic similarity to existing gestures
                semantic_similarities = []
                for i in self.sorted_labels:
                    if i < len(self.activity_labels) and i != new_gesture_id:
                        # Get semantic similarity from BERT
                        semantic_similarities.append(self.bert_sim_matrix[new_gesture_id, i])
                
                # Calculate semantic coverage
                if semantic_similarities:
                    avg_semantic_similarity = np.mean(semantic_similarities)
                    semantic_coverage = 1.0 - avg_semantic_similarity
                    
                    # Combine physical and semantic coverage
                    coverage_score = 0.6 * physical_coverage + 0.4 * semantic_coverage
                else:
                    coverage_score = physical_coverage
            else:
                coverage_score = physical_coverage
        else:
            coverage_score = physical_coverage
        
        # Ensure in 0-1 range
        coverage_score = np.clip(coverage_score, 0, 1)
        
        return coverage_score
    
    def calculate_user_performance(self, gesture_samples):
        """
        Calculate how consistently the user performs the gesture.
        
        Args:
            gesture_samples: List of embeddings for the same gesture
            
        Returns:
            performance_score: Higher score means more consistent (0-1 range)
        """
        if len(gesture_samples) < 2:
            print("Warning: Need at least 2 samples to calculate user performance")
            return 0.5  # Default middle value
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(gesture_samples)):
            for j in range(i+1, len(gesture_samples)):
                dist = np.linalg.norm(gesture_samples[i] - gesture_samples[j])
                distances.append(dist)
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Convert to performance score (lower distance = higher score)
        performance_score = 1.0 / (1.0 + avg_distance * 3)  # Scaling factor to normalize
        
        # Ensure in 0-1 range
        performance_score = np.clip(performance_score, 0, 1)
        
        return performance_score
    
    def make_recommendation(self, new_gesture_id, user_idx=None, 
                      source_dataset=None, weights=(0.4, 0.3, 0.3), threshold=0.4):
        """
        Evaluate whether to add a new gesture.
        
        Args:
            new_gesture_id: ID of the new gesture
            user_idx: Optional user index to filter by
            source_dataset: Optional name of source dataset for new gestures
            weights: Weights for distinctiveness, coverage, performance
            threshold: Threshold for recommendation
            
        Returns:
            recommendation: Boolean (True = add gesture)
            scores: Dictionary of component scores
            explanation: Text explanation of the decision
        """
        from ..data_utils.gesture_loader import get_gesture_embeddings, get_gesture_embeddings_from_dataset
        
        # Try to get gesture samples from filtered dataset first
        gesture_samples, _, _ = get_gesture_embeddings(
            self.dataset_name, new_gesture_id, user_idx
        )
        
        # If not found and source_dataset is provided, try to get from source
        if (gesture_samples is None or len(gesture_samples) == 0) and source_dataset is not None:
            print(f"Gesture not found in {self.dataset_name}, trying {source_dataset}")
            gesture_samples, _, _ = get_gesture_embeddings_from_dataset(
                source_dataset, new_gesture_id, user_idx
            )
        
        if gesture_samples is None or len(gesture_samples) == 0:
            return False, {}, ["No samples found for this gesture"]
    
        
        # Average the samples to get a representative embedding
        new_gesture_embed = np.mean(gesture_samples, axis=0)
        
        # Calculate distinctiveness
        distinctiveness, closest_gestures = self.calculate_distinctiveness(new_gesture_embed)
        
        # Calculate coverage improvement
        coverage = self.calculate_coverage_improvement(new_gesture_embed, new_gesture_id)
        
        # Calculate user performance
        performance = self.calculate_user_performance(gesture_samples)
        
        # Calculate combined score
        combined_score = (
            weights[0] * distinctiveness + 
            weights[1] * coverage + 
            weights[2] * performance
        )
        
        # Make recommendation
        recommendation = combined_score >= threshold
        
        # Store all scores
        scores = {
            "distinctiveness": distinctiveness,
            "coverage": coverage,
            "performance": performance,
            "combined": combined_score
        }
        
        # Create explanation
        gesture_name = self.activity_labels[new_gesture_id] if new_gesture_id < len(self.activity_labels) else f"Gesture {new_gesture_id}"
        
        explanation = [
            f"Evaluation for '{gesture_name}' (Gesture ID {new_gesture_id}):",
            f"- Distinctiveness: {distinctiveness:.2f} - " + 
            ("High (distinct from existing gestures)" if distinctiveness >= 0.5 
             else "Low (similar to existing gestures)"),
            
            f"- Coverage: {coverage:.2f} - " + 
            ("High (fills gap in gesture space)" if coverage >= 0.5 
             else "Low (redundant in gesture space)"),
            
            f"- User Performance: {performance:.2f} - " + 
            ("High (consistent execution)" if performance >= 0.5 
             else "Low (inconsistent execution)"),
            
            f"- Combined Score: {combined_score:.2f} with threshold {threshold}",
            
            f"Recommendation: {'ADD' if recommendation else 'DO NOT ADD'} this gesture"
        ]
        
        # Add details about closest gestures
        if closest_gestures:
            explanation.append("Most similar gestures:")
            for label, similarity in closest_gestures:
                gesture_name = self.activity_labels[label] if label < len(self.activity_labels) else f"Gesture {label}"
                explanation.append(f"  - {gesture_name} (similarity: {similarity:.2f})")
        
        return recommendation, scores, explanation
    
    def visualize_recommendation(self, scores, recommendation, threshold=0.6):
        """
        Visualize the recommendation scores.
        
        Args:
            scores: Dictionary of component scores
            recommendation: Final recommendation (boolean)
            threshold: Recommendation threshold
        """
        # Extract scores
        distinctiveness = scores.get("distinctiveness", 0)
        coverage = scores.get("coverage", 0)
        performance = scores.get("performance", 0)
        combined_score = scores.get("combined", 0)
        
        # Create visualization
        labels = ['Distinctiveness', 'Coverage', 'User Performance', 'Combined Score', 'Threshold']
        values = [distinctiveness, coverage, performance, combined_score, threshold]
        colors = ['#3498db', '#2ecc71', '#f39c12', 
                 '#e74c3c' if not recommendation else '#2ecc71', 
                 '#95a5a6']
        
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(labels, values, color=colors)
        plt.ylim(0, 1.1)
        plt.title('Gesture Recommendation Scores', fontsize=16)
        plt.ylabel('Score (higher is better)', fontsize=12)
        
        # Add text labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', fontsize=12)
        
        # Add recommendation as text
        recommendation_text = "Recommendation: ADD GESTURE" if recommendation else "Recommendation: DON'T ADD"
        plt.text(0.5, 1.05, recommendation_text, 
                transform=plt.gca().transAxes, ha='center', 
                fontsize=14, fontweight='bold',
                color='green' if recommendation else 'red')
        
        plt.tight_layout()
        
        # Return figure for saving or displaying
        return plt.gcf()
    
    def analyze_multiple_gestures(self, gesture_ids, user_idx=None):
        """
        Analyze multiple gestures and recommend which ones to add.
        
        Args:
            gesture_ids: List of gesture IDs to analyze
            user_idx: Optional user index to filter by
            
        Returns:
            results: Dictionary of results for each gesture
        """
        results = {}
        
        for gesture_id in gesture_ids:
            print(f"Analyzing gesture {gesture_id}...")
            
            # Make recommendation
            recommendation, scores, explanation = self.make_recommendation(
                gesture_id, user_idx
            )
            
            # Store results
            gesture_name = self.activity_labels[gesture_id] if gesture_id < len(self.activity_labels) else f"Gesture {gesture_id}"
            results[gesture_id] = {
                "name": gesture_name,
                "recommendation": recommendation,
                "scores": scores,
                "explanation": explanation
            }
            
            # Print explanation
            for line in explanation:
                print(line)
            print()
        
        # Sort results by combined score
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]["scores"].get("combined", 0), 
            reverse=True
        )
        
        # Print summary
        print("\nSummary of recommendations:")
        for gesture_id, result in sorted_results:
            gesture_name = result["name"]
            combined_score = result["scores"].get("combined", 0)
            recommendation = result["recommendation"]
            
            print(f"- {gesture_name} (ID: {gesture_id}): Score = {combined_score:.2f}, " +
                 f"Recommendation: {'ADD' if recommendation else 'DONT ADD'}")
        
        return results