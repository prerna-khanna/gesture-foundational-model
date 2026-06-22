#!/usr/bin/env python3
"""
Gesture Validation System

Validates if a user-provided gesture can be added to the dataset.

This system:
1. Analyzes candidate gesture samples
2. Compares with existing gesture classes
3. Checks if gesture is:
   - Unique (distinct from existing)
   - Recognizable (model can classify it)
   - Not too similar to any existing gesture
4. Provides accept/reject recommendation with detailed reasoning

Usage:
    python gesture_validator.py \
        --candidate_data data_20_120.npy \
        --candidate_labels label_20_120.npy \
        --dataset Alexandra \
        --version 20_120 \
        --embedding_model limu_v1 \
        --semantic_model limu_gru_v1
"""

import numpy as np
import argparse
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


def load_gesture_names(dataset):
    """Load gesture names from config."""
    try:
        from config import load_dataset_label_names
        label_names, _, _ = load_dataset_label_names(None, label_index=0, dataset=dataset)
        return {i: name for i, name in enumerate(label_names)}
    except:
        return {i: f"Gesture {i}" for i in range(15)}


def extract_gesture_info(labels):
    """Extract gesture IDs and user IDs from labels."""
    if len(labels.shape) == 3:
        gesture_ids = labels[:, 0, 0].astype(int)
    else:
        gesture_ids = labels.astype(int)
    
    return gesture_ids


def compute_gesture_embedding(embedding, reduce_fn='mean'):
    """
    Reduce temporal embedding to single vector.
    
    Args:
        embedding: (seq_len, embed_dim) single gesture embedding
        reduce_fn: 'mean', 'max', or 'first'
    
    Returns:
        (embed_dim,) gesture embedding vector
    """
    if len(embedding.shape) == 1:
        return embedding
    
    if reduce_fn == 'mean':
        return np.mean(embedding, axis=0)
    elif reduce_fn == 'max':
        return np.max(embedding, axis=0)
    elif reduce_fn == 'first':
        return embedding[0]
    else:
        raise ValueError(f"Unknown reduce function: {reduce_fn}")


def validate_gesture(candidate_embeddings, candidate_name, 
                    existing_embeddings, existing_gesture_ids, 
                    existing_gesture_names,
                    classifier_model, 
                    similarity_threshold=0.75,
                    recognizability_threshold=0.7):
    """
    Validate if a candidate gesture should be accepted.
    
    Args:
        candidate_embeddings: (N, seq_len, embed_dim) or (N, embed_dim) candidate gesture samples
        candidate_name: name of the candidate gesture
        existing_embeddings: (M, embed_dim) existing gesture embeddings (mean-pooled)
        existing_gesture_ids: (M,) IDs of existing gestures
        existing_gesture_names: dict mapping ID to name
        classifier_model: trained classifier
        similarity_threshold: max allowed similarity to existing gesture
        recognizability_threshold: min required accuracy for model to recognize
    
    Returns:
        decision: dict with accept/reject decision and reasoning
    """
    
    # Reduce candidate embeddings if needed
    if len(candidate_embeddings.shape) == 3:
        # Multiple samples: compute mean embedding
        # If shape is (N, seq_len, embed_dim), reduce across samples first, then time
        if candidate_embeddings.shape[-1] != existing_embeddings.shape[-1]:
            # Shape mismatch - average across all dimensions
            candidate_mean_emb = np.mean(candidate_embeddings, axis=(0, 1))
        else:
            candidate_mean_emb = np.mean(candidate_embeddings, axis=(0, 1))
        
        candidate_embs_list = [compute_gesture_embedding(e) for e in candidate_embeddings]
    else:
        candidate_mean_emb = np.mean(candidate_embeddings, axis=0) if len(candidate_embeddings) > 0 else candidate_embeddings
        candidate_embs_list = [candidate_embeddings[i] for i in range(len(candidate_embeddings))]
    
    decision = {
        'gesture_name': candidate_name,
        'recommendation': 'UNKNOWN',
        'confidence': 0.0,
        'reasoning': [],
        'issues': [],
        'similarities': [],
        'model_performance': None
    }
    
    # 1. Check similarity to existing gestures
    print(f"\n{'='*80}")
    print(f"Validating Gesture: {candidate_name}")
    print(f"{'='*80}")
    
    print(f"\n1. SIMILARITY ANALYSIS")
    print(f"{'-'*80}")
    
    # Normalize embeddings
    scaler = StandardScaler()
    existing_norm = scaler.fit_transform(existing_embeddings)
    candidate_norm = scaler.transform(candidate_mean_emb.reshape(1, -1))
    
    similarities = cosine_similarity(candidate_norm, existing_norm)[0]
    
    most_similar_idx = np.argmax(similarities)
    most_similar_score = similarities[most_similar_idx]
    most_similar_gesture = int(existing_gesture_ids[most_similar_idx])
    most_similar_name = existing_gesture_names.get(most_similar_gesture, f"Gesture {most_similar_gesture}")
    
    decision['similarities'] = [
        {
            'gesture_id': int(existing_gesture_ids[i]),
            'gesture_name': existing_gesture_names.get(int(existing_gesture_ids[i]), f"Gesture {int(existing_gesture_ids[i])}"),
            'similarity': float(similarities[i])
        }
        for i in np.argsort(similarities)[::-1][:3]  # Top 3
    ]
    
    print(f"Most similar to: {most_similar_name} (ID {most_similar_gesture})")
    print(f"Similarity score: {most_similar_score:.4f}")
    print(f"Threshold: {similarity_threshold:.4f}")
    
    if most_similar_score > similarity_threshold:
        decision['issues'].append(
            f"TOO SIMILAR to '{most_similar_name}' (similarity: {most_similar_score:.2%})"
        )
        print(f"⚠️  WARNING: Too similar!")
    else:
        decision['reasoning'].append(
            f"✓ Distinct from existing gestures (most similar: {most_similar_score:.2%})"
        )
        print(f"✓ Sufficiently distinct")
    
    # Show all similarities
    print(f"\nSimilarity to all existing gestures:")
    for i in np.argsort(similarities)[::-1]:
        g_id = int(existing_gesture_ids[i])
        g_name = existing_gesture_names.get(g_id, f"Gesture {g_id}")
        print(f"  • {g_name:30s} (ID {g_id:2d}): {similarities[i]:.4f}")
    
    # 2. Check model recognizability
    print(f"\n2. RECOGNIZABILITY ANALYSIS")
    print(f"{'-'*80}")
    
    # Train a binary classifier: candidate vs all others
    try:
        # Create training data: existing gestures
        X_train = existing_embeddings
        y_train = np.zeros(len(existing_embeddings))  # Label 0: existing gestures
        
        # Add candidate samples
        candidate_train = np.vstack(candidate_embs_list)
        X_train = np.vstack([X_train, candidate_train])
        y_train = np.concatenate([y_train, np.ones(len(candidate_train))])
        
        # Train binary classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Get prediction accuracy on candidate samples
        candidate_scores = clf.predict_proba(candidate_train)[:, 1]  # Prob of being candidate
        candidate_accuracy = np.mean(candidate_scores)
        
        decision['model_performance'] = {
            'recognizability_score': float(candidate_accuracy),
            'threshold': recognizability_threshold,
            'samples_tested': len(candidate_train)
        }
        
        print(f"Model recognizability score: {candidate_accuracy:.2%}")
        print(f"Threshold: {recognizability_threshold:.2%}")
        print(f"Samples tested: {len(candidate_train)}")
        
        if candidate_accuracy < recognizability_threshold:
            decision['issues'].append(
                f"HARD TO RECOGNIZE: Model only {candidate_accuracy:.1%} confident "
                f"(threshold: {recognizability_threshold:.1%})"
            )
            print(f"⚠️  WARNING: Model struggles to recognize this gesture!")
        else:
            decision['reasoning'].append(
                f"✓ Model can reliably recognize gesture ({candidate_accuracy:.1%} confidence)"
            )
            print(f"✓ Model can recognize this gesture")
        
    except Exception as e:
        print(f"Warning: Could not evaluate recognizability: {e}")
        decision['model_performance'] = None
    
    # 3. Make final recommendation
    print(f"\n3. FINAL RECOMMENDATION")
    print(f"{'-'*80}")
    
    if decision['issues']:
        decision['recommendation'] = 'REJECT'
        confidence = 0.95 if len(decision['issues']) > 1 else 0.80
        decision['confidence'] = confidence
        
        print(f"❌ RECOMMENDATION: REJECT")
        print(f"Confidence: {confidence:.0%}")
        print(f"\nReasons:")
        for issue in decision['issues']:
            print(f"  ❌ {issue}")
        
        print(f"\nSuggestions:")
        print(f"  • Modify this gesture to make it more distinct")
        print(f"  • Try a different hand position or movement pattern")
        print(f"  • Or replace this gesture with a new one")
        if most_similar_score > similarity_threshold:
            print(f"  • Current gesture is too similar to '{most_similar_name}'")
    
    else:
        decision['recommendation'] = 'ACCEPT'
        confidence = 0.95
        decision['confidence'] = confidence
        
        print(f"✅ RECOMMENDATION: ACCEPT")
        print(f"Confidence: {confidence:.0%}")
        print(f"\nReasons:")
        for reason in decision['reasoning']:
            print(f"  {reason}")
    
    return decision


def main():
    parser = argparse.ArgumentParser(
        description='Validate if a new gesture can be added to the dataset'
    )
    parser.add_argument(
        '--candidate_embedding',
        type=str,
        help='Path to candidate gesture embedding file (.npy) in embed folder'
    )
    parser.add_argument(
        '--candidate_data',
        type=str,
        help='Alias: raw candidate embedding/data file (will be used as candidate_embedding if provided)'
    )
    parser.add_argument(
        '--candidate_labels',
        type=str,
        help='Optional: candidate labels file (used to infer candidate_name if --candidate_name not given)'
    )
    parser.add_argument(
        '--candidate_name',
        type=str,
        required=False,
        help='Name of the candidate gesture (inferred from candidate_data if omitted)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Existing dataset name (e.g., Alexandra, Turiya)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='20_120',
        help='Dataset version (default: 20_120)'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='limu_v1',
        help='Embedding model name (default: limu_v1)'
    )
    parser.add_argument(
        '--semantic_model',
        type=str,
        default='limu_gru_v1',
        help='Semantic model name (default: limu_gru_v1)'
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.75,
        help='Max allowed similarity to existing gesture (default: 0.75)'
    )
    parser.add_argument(
        '--recognizability_threshold',
        type=float,
        default=0.7,
        help='Min required recognizability (default: 0.7)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='gesture_validation_result.json',
        help='Save validation result to JSON file'
    )
    
    args = parser.parse_args()
    # Accept candidate_data as fallback for candidate_embedding
    if not args.candidate_embedding and args.candidate_data:
        args.candidate_embedding = args.candidate_data
    
    # Infer candidate_name if not provided
    if not args.candidate_name:
        if args.candidate_labels and os.path.exists(args.candidate_labels):
            try:
                lbls = np.load(args.candidate_labels)
                uniq = np.unique(lbls)
                # use single unique label or fallback to filename
                if len(uniq) == 1:
                    args.candidate_name = f"NewGesture_{int(uniq[0])}"
                else:
                    args.candidate_name = os.path.splitext(os.path.basename(args.candidate_data or args.candidate_embedding))[0]
            except Exception:
                args.candidate_name = os.path.splitext(os.path.basename(args.candidate_data or args.candidate_embedding))[0]
        elif args.candidate_data:
            args.candidate_name = os.path.splitext(os.path.basename(args.candidate_data))[0]
    
    # Validate input file
    if not args.candidate_embedding:
        print(f"Error: --candidate_embedding (or --candidate_data) is required")
        print(f"Example:")
        print(f"  python gesture_validator.py \\")
        print(f"    --candidate_embedding embed/embed_limu_v1_NewGesture_20_120.npy \\")
        print(f"    --candidate_name 'my_gesture' \\")
        print(f"    --dataset Alexandra \\")
        print(f"    --version 20_120")
        return
    
    if not os.path.exists(args.candidate_embedding):
        print(f"Error: Embedding file not found: {args.candidate_embedding}")
        print(f"Make sure embeddings are computed and saved in the embed folder")
        return
    
    print("="*80)
    print("GESTURE VALIDATION SYSTEM")
    print("="*80)
    print(f"\nCandidate Gesture: {args.candidate_name}")
    print(f"Existing Dataset: {args.dataset}")
    print(f"Embedding Model: {args.embedding_model}")
    
    # Load existing dataset
    print(f"\nLoading existing dataset...")
    try:
        existing_emb, existing_labels = load_embedding_label(
            args.embedding_model, args.dataset, args.version
        )
        existing_gesture_ids = extract_gesture_info(existing_labels)
        
        # Reduce temporal dimension
        if len(existing_emb.shape) == 3:
            existing_emb = np.mean(existing_emb, axis=1)
        
        print(f"  ✓ Loaded {len(existing_emb)} existing gesture samples")
        print(f"  ✓ Embedding dimension: {existing_emb.shape[1]}")
    
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        return
    
    # Load candidate gesture
    print(f"\nLoading candidate gesture embeddings...")
    try:
        candidate_emb = np.load(args.candidate_embedding).astype(np.float32)
        print(f"  ✓ Loaded from: {args.candidate_embedding}")
        print(f"  ✓ Shape: {candidate_emb.shape}")
        
        # Verify embedding dimension matches
        if len(candidate_emb.shape) == 3:
            embed_dim = candidate_emb.shape[-1]
        else:
            embed_dim = candidate_emb.shape[-1]
        
        existing_embed_dim = existing_emb.shape[-1]
        
        if embed_dim != existing_embed_dim:
            print(f"  ⚠️  WARNING: Embedding dimension mismatch!")
            print(f"      Candidate: {embed_dim}D, Existing: {existing_embed_dim}D")
            print(f"      Make sure you used the same embedding model")
        else:
            print(f"  ✓ Embedding dimension matches: {embed_dim}D")
    
    except Exception as e:
        print(f"Error loading candidate embedding: {e}")
        return
    
    # Get gesture names
    gesture_names = load_gesture_names(args.dataset)
    
    # Train a base classifier on existing gestures
    print(f"\nTraining baseline classifier...")
    try:
        X = existing_emb
        y = existing_gesture_ids
        
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X, y)
        
        baseline_acc = classifier.score(X, y)
        print(f"  ✓ Baseline classifier accuracy: {baseline_acc:.2%}")
    except Exception as e:
        print(f"Warning: Could not train classifier: {e}")
        classifier = None
    
    # Validate gesture
    decision = validate_gesture(
        candidate_emb,
        args.candidate_name,
        existing_emb,
        existing_gesture_ids,
        gesture_names,
        classifier,
        similarity_threshold=args.similarity_threshold,
        recognizability_threshold=args.recognizability_threshold
    )
    
    # Save result
    print(f"\n" + "="*80)
    print(f"Saving validation result to {args.output_file}...")
    
    with open(args.output_file, 'w') as f:
        json.dump(decision, f, indent=2)
    
    print(f"✓ Result saved")
    print("="*80)


if __name__ == '__main__':
    main()
