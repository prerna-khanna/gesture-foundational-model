#!/usr/bin/env python3
"""
Batch Gesture Validation - Assess quality of all gestures in multiple datasets.

Validates all existing gestures in Alexandra, Turiya, and Julius datasets.
Generates a comprehensive quality report.

Usage:
    python batch_gesture_validation.py \
        --datasets Alexandra Turiya Julius \
        --max_epochs 10 \
        --output_dir validation_results
"""

import numpy as np
import json
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
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


def assess_gesture_quality(embeddings, labels, gesture_ids_to_assess=None):
    """
    Assess quality of all gestures in a dataset.
    
    Returns:
        per_gesture_stats: dict with quality metrics for each gesture
        confusion_pairs: list of high-confusion gesture pairs
    """
    gesture_ids = extract_gesture_info(labels)
    unique_gestures = np.unique(gesture_ids)
    
    # Reduce temporal dimension if needed
    if len(embeddings.shape) == 3:
        embeddings = np.mean(embeddings, axis=1)
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings_scaled)
    
    # For each gesture, find its mean embedding
    gesture_embeddings = {}
    for g_id in unique_gestures:
        g_id = int(g_id)
        mask = gesture_ids == g_id
        gesture_embeddings[g_id] = np.mean(embeddings_scaled[mask], axis=0)
    
    # Train classifier for recognizability
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings_scaled, gesture_ids)
    y_pred = clf.predict(embeddings_scaled)
    
    # Compute per-gesture metrics
    per_gesture_stats = {}
    for g_id in unique_gestures:
        g_id = int(g_id)
        mask = gesture_ids == g_id
        
        # Find most similar gesture
        g_embedding = gesture_embeddings[g_id]
        similarities = []
        for other_id in unique_gestures:
            if other_id == g_id:
                continue
            other_embedding = gesture_embeddings[int(other_id)]
            sim = cosine_similarity([g_embedding], [other_embedding])[0, 0]
            similarities.append((int(other_id), float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar_id, most_similar_sim = similarities[0] if similarities else (None, 0)
        
        # Recognizability
        correct = np.sum(y_pred[mask] == g_id)
        total = np.sum(mask)
        recognizability = correct / total if total > 0 else 0
        
        per_gesture_stats[g_id] = {
            'total_samples': int(total),
            'most_similar_gesture': most_similar_id,
            'max_similarity': float(most_similar_sim),
            'recognizability': float(recognizability),
            'status': 'ACCEPT' if (most_similar_sim < 0.75 and recognizability >= 0.70) else 'BORDERLINE'
        }
    
    # Find high-confusion pairs
    confusion_pairs = []
    for i, g1 in enumerate(unique_gestures):
        for j, g2 in enumerate(unique_gestures):
            if i < j:
                g1_emb = gesture_embeddings[int(g1)]
                g2_emb = gesture_embeddings[int(g2)]
                sim = cosine_similarity([g1_emb], [g2_emb])[0, 0]
                if sim > 0.75:
                    confusion_pairs.append({
                        'gesture1': int(g1),
                        'gesture2': int(g2),
                        'similarity': float(sim)
                    })
    
    confusion_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return per_gesture_stats, confusion_pairs


def generate_quality_report(dataset_name, per_gesture_stats, confusion_pairs, version='20_120'):
    """Generate quality report for dataset."""
    
    print(f"\n{'='*80}")
    print(f"GESTURE QUALITY ASSESSMENT - {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Summary
    total_gestures = len(per_gesture_stats)
    accepted = sum(1 for s in per_gesture_stats.values() if s['status'] == 'ACCEPT')
    borderline = total_gestures - accepted
    
    print(f"Summary:")
    print(f"  ✅ Accepted: {accepted}/{total_gestures}")
    print(f"  ⚠️  Borderline: {borderline}/{total_gestures}\n")
    
    # Per-gesture table
    print(f"{'ID':<4} {'Samples':<8} {'Max Sim':<10} {'Recogn':<10} {'Status':<12} {'Most Similar To':<20}")
    print(f"{'-'*80}")
    
    for g_id in sorted(per_gesture_stats.keys()):
        stat = per_gesture_stats[g_id]
        status = "✅ ACCEPT" if stat['status'] == 'ACCEPT' else "⚠️  BORDERLINE"
        print(f"{g_id:<4} {stat['total_samples']:<8} {stat['max_similarity']:<10.3f} "
              f"{stat['recognizability']:<10.1%} {status:<12} Gesture {stat['most_similar_gesture']:<15}")
    
    # Confusion pairs
    if confusion_pairs:
        print(f"\n{'High-Confusion Gesture Pairs (similarity > 0.75):'}")
        print(f"{'-'*80}")
        for i, pair in enumerate(confusion_pairs[:5], 1):
            print(f"{i}. Gesture {pair['gesture1']} ↔ Gesture {pair['gesture2']}: {pair['similarity']:.3f}")
    
    return {
        'dataset': dataset_name,
        'version': version,
        'total_gestures': total_gestures,
        'accepted': accepted,
        'borderline': borderline,
        'per_gesture_stats': per_gesture_stats,
        'confusion_pairs': confusion_pairs
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch validate gestures across multiple datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['Alexandra', 'Turiya', 'Julius'],
        help='Datasets to validate'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='20_120',
        help='Dataset version'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='limu_v1',
        help='Embedding model'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10,
        help='Max epochs (for reference)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='validation_results',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("BATCH GESTURE QUALITY VALIDATION")
    print("="*80)
    print(f"\nDatasets: {', '.join(args.datasets)}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Version: {args.version}")
    print(f"Max Epochs: {args.max_epochs}\n")
    
    all_results = {}
    
    for dataset in args.datasets:
        try:
            # Load data
            embeddings, labels = load_embedding_label(
                args.embedding_model, dataset, args.version
            )
            
            print(f"\nLoading {dataset}...")
            print(f"  ✓ Embeddings: {embeddings.shape}")
            print(f"  ✓ Labels: {labels.shape}")
            
            # Assess quality
            per_gesture_stats, confusion_pairs = assess_gesture_quality(embeddings, labels)
            
            # Generate report
            report = generate_quality_report(dataset, per_gesture_stats, confusion_pairs, args.version)
            all_results[dataset] = report
            
        except Exception as e:
            print(f"\n⚠️  Error processing {dataset}: {e}")
            continue
    
    # Save detailed results
    print(f"\n{'='*80}")
    print(f"Saving results to {args.output_dir}...")
    print(f"{'='*80}\n")
    
    for dataset, report in all_results.items():
        output_file = os.path.join(args.output_dir, f'{dataset}_quality_assessment.json')
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ {output_file}")
    
    # Create summary CSV
    summary_data = []
    for dataset, report in all_results.items():
        for g_id, stat in report['per_gesture_stats'].items():
            summary_data.append({
                'Dataset': dataset,
                'Gesture_ID': g_id,
                'Samples': stat['total_samples'],
                'Max_Similarity': f"{stat['max_similarity']:.3f}",
                'Recognizability': f"{stat['recognizability']:.1%}",
                'Most_Similar_To': stat['most_similar_gesture'],
                'Status': stat['status']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(args.output_dir, 'quality_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ {summary_csv}")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Dataset':<15} {'Total':<8} {'Accepted':<12} {'Borderline':<12} {'Accept %':<12}")
    print(f"{'-'*60}")
    
    for dataset, report in sorted(all_results.items()):
        accept_pct = report['accepted'] / report['total_gestures'] * 100
        print(f"{dataset:<15} {report['total_gestures']:<8} {report['accepted']:<12} "
              f"{report['borderline']:<12} {accept_pct:<12.1f}%")


if __name__ == '__main__':
    main()
