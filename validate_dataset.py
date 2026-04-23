#!/usr/bin/env python3
"""
Batch validation of gesture datasets.
Validates all existing gestures in a dataset to identify:
- Similar gesture pairs
- Low recognizability gestures

Usage:
    python validate_dataset.py --dataset Alexandra
    python validate_dataset.py --dataset Turiya
    python validate_dataset.py --dataset Julius
"""

import numpy as np
import argparse
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_embedding_label(model_file, dataset, dataset_version):
    """Load pre-computed embeddings and labels."""
    embed_name = 'embed_' + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    
    embed_path = os.path.join('embed', embed_name + '.npy')
    label_path = os.path.join('dataset', dataset, label_name + '.npy')
    
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    embed = np.load(embed_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)
    
    return embed, labels


def extract_gesture_info(labels):
    """Extract gesture IDs from labels."""
    if len(labels.shape) == 3:
        gesture_ids = labels[:, 0, 0].astype(int)
    else:
        gesture_ids = labels.astype(int)
    return gesture_ids


def validate_dataset(embeddings, labels, dataset_name,
                    similarity_threshold=0.75,
                    recognizability_threshold=0.70,
                    max_epochs=10):
    """
    Validate all gestures in a dataset.
    
    Returns: dict with validation results
    """
    gesture_ids = extract_gesture_info(labels)
    unique_gestures = np.unique(gesture_ids)
    
    # Reduce embeddings if needed
    if len(embeddings.shape) == 3:
        embeddings_reduced = np.mean(embeddings, axis=1)
    else:
        embeddings_reduced = embeddings
    
    print(f"\n{'='*80}")
    print(f"Validating Dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Gestures: {len(unique_gestures)}")
    print(f"Total samples: {len(gesture_ids)}")
    print(f"Embeddings shape: {embeddings_reduced.shape}")
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_norm = scaler.fit_transform(embeddings_reduced)
    
    # Compute similarity matrix
    print(f"\nComputing gesture similarities...")
    sim_matrix = cosine_similarity(embeddings_norm)
    
    # Get per-gesture mean embedding for similarity
    gesture_embeddings = {}
    for g_id in unique_gestures:
        mask = gesture_ids == g_id
        gesture_embeddings[int(g_id)] = np.mean(embeddings_norm[mask], axis=0)
    
    # Compute per-gesture similarity to most similar other gesture
    gesture_similarities = {}
    for g_id in unique_gestures:
        g_id_int = int(g_id)
        emb = gesture_embeddings[g_id_int].reshape(1, -1)
        sims = cosine_similarity(emb, embeddings_norm)[0]
        
        # Find max similarity to OTHER gesture class
        max_sim = 0
        most_similar = None
        for g_id_other in unique_gestures:
            if g_id_other == g_id:
                continue
            mask = gesture_ids == g_id_other
            if np.any(mask):
                sim_to_other = np.mean(sims[mask])
                if sim_to_other > max_sim:
                    max_sim = sim_to_other
                    most_similar = int(g_id_other)
        
        gesture_similarities[g_id_int] = {
            'most_similar': most_similar,
            'similarity': float(max_sim)
        }
    
    # Compute recognizability using cross-validation
    print(f"Computing recognizability (cross-validation with max 10 epochs)...")
    
    clf = LogisticRegression(max_iter=min(max_epochs, 10), random_state=42)
    cv_scores = cross_val_score(clf, embeddings_norm, gesture_ids, cv=5, 
                               scoring='accuracy')
    baseline_accuracy = np.mean(cv_scores)
    
    # Per-gesture recognizability
    clf.fit(embeddings_norm, gesture_ids)
    y_pred = clf.predict(embeddings_norm)
    
    gesture_recognizability = {}
    for g_id in unique_gestures:
        mask = gesture_ids == g_id
        g_id_int = int(g_id)
        correct = np.sum((gesture_ids == g_id) & (y_pred == g_id))
        total = np.sum(gesture_ids == g_id)
        accuracy = correct / total if total > 0 else 0
        gesture_recognizability[g_id_int] = float(accuracy)
    
    # Categorize gestures
    accepted = []
    borderline = []
    problematic = []
    
    for g_id in unique_gestures:
        g_id_int = int(g_id)
        sim = gesture_similarities[g_id_int]['similarity']
        rec = gesture_recognizability[g_id_int]
        
        issues = []
        if sim > similarity_threshold:
            issues.append(f"similar_to_{gesture_similarities[g_id_int]['most_similar']}")
        if rec < recognizability_threshold:
            issues.append("low_recognizability")
        
        if not issues:
            accepted.append(g_id_int)
        elif len(issues) == 1 and sim <= 0.85:
            borderline.append((g_id_int, issues))
        else:
            problematic.append((g_id_int, issues))
    
    # Print results
    print(f"\n{'─'*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*80}")
    print(f"✅ Accepted: {len(accepted)} gestures")
    print(f"⚠️  Borderline: {len(borderline)} gestures")
    print(f"❌ Problematic: {len(problematic)} gestures")
    print(f"\nBaseline Accuracy (5-fold CV): {baseline_accuracy:.2%}")
    
    # Show problem gestures
    if borderline:
        print(f"\n⚠️  BORDERLINE GESTURES:")
        for g_id, issues in sorted(borderline):
            sim = gesture_similarities[g_id]['similarity']
            rec = gesture_recognizability[g_id]
            similar_to = gesture_similarities[g_id]['most_similar']
            print(f"  Gesture {g_id:2d}: similarity={sim:.3f} (vs {similar_to}), "
                  f"recognizability={rec:.1%}")
    
    if problematic:
        print(f"\n❌ PROBLEMATIC GESTURES:")
        for g_id, issues in sorted(problematic):
            sim = gesture_similarities[g_id]['similarity']
            rec = gesture_recognizability[g_id]
            similar_to = gesture_similarities[g_id]['most_similar']
            print(f"  Gesture {g_id:2d}: similarity={sim:.3f} (vs {similar_to}), "
                  f"recognizability={rec:.1%}")
    
    # Find confused pairs
    print(f"\n{'─'*80}")
    print(f"MOST CONFUSED GESTURE PAIRS:")
    print(f"{'─'*80}")
    
    pairs = []
    for i, g1 in enumerate(unique_gestures):
        for j, g2 in enumerate(unique_gestures):
            if i < j:
                g1_int = int(g1)
                g2_int = int(g2)
                
                mask1 = gesture_ids == g1
                mask2 = gesture_ids == g2
                
                emb1 = np.mean(embeddings_norm[mask1], axis=0).reshape(1, -1)
                emb2 = np.mean(embeddings_norm[mask2], axis=0).reshape(1, -1)
                
                sim = cosine_similarity(emb1, emb2)[0, 0]
                pairs.append((g1_int, g2_int, float(sim)))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for rank, (g1, g2, sim) in enumerate(pairs[:5], 1):
        marker = "🔴" if sim > similarity_threshold else "🟡" if sim > 0.70 else "🟢"
        print(f"{rank}. Gesture {g1:2d} ↔ Gesture {g2:2d}: {sim:.3f} {marker}")
    
    # Save results
    results = {
        'dataset': dataset_name,
        'baseline_accuracy': float(baseline_accuracy),
        'accepted': len(accepted),
        'borderline': len(borderline),
        'problematic': len(problematic),
        'gesture_similarities': gesture_similarities,
        'gesture_recognizability': gesture_recognizability,
        'accepted_list': accepted,
        'borderline_list': [g for g, _ in borderline],
        'problematic_list': [g for g, _ in problematic]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch validate gesture dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Alexandra, Turiya, Julius)')
    parser.add_argument('--version', type=str, default='20_120',
                       help='Dataset version')
    parser.add_argument('--embedding_model', type=str, default='limu_v1',
                       help='Embedding model')
    parser.add_argument('--similarity_threshold', type=float, default=0.75,
                       help='Similarity threshold')
    parser.add_argument('--recognizability_threshold', type=float, default=0.70,
                       help='Recognizability threshold')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Max epochs for training')
    
    args = parser.parse_args()
    
    # Load data
    try:
        embeddings, labels = load_embedding_label(
            args.embedding_model, args.dataset, args.version
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Validate
    results = validate_dataset(
        embeddings, labels, args.dataset,
        similarity_threshold=args.similarity_threshold,
        recognizability_threshold=args.recognizability_threshold,
        max_epochs=args.max_epochs
    )
    
    # Save results
    output_file = f'dataset_validation_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
