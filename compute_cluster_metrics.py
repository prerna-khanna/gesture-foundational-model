#!/usr/bin/env python
"""Compute clustering metrics for classifier feature space.

Outputs a short summary to stdout and saves a JSON with detailed numbers and a CSV for threshold sweep.
"""
import argparse
import json
import os
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score


def extract_features_from_checkpoint(checkpoint, data_file=None, embeddings_file=None, label_file=None, device='cpu', batch_size=256):
    import torch
    from contrastive.models import ContrastiveTransformerClassifier, ContrastiveGRUClassifier, ContrastiveLSTMClassifier

    # load data (prefer embeddings_file when provided)
    if embeddings_file is not None:
        data = np.load(embeddings_file).astype(np.float32)
    elif data_file is not None:
        data = np.load(data_file).astype(np.float32)
    else:
        raise RuntimeError('Either data_file or embeddings_file must be provided')
    labels = np.load(label_file)
    # Normalize labels to per-sample 1D array if needed
    if labels.ndim > 1:
        try:
            labels = labels.reshape(labels.shape[0], -1)[:, 0]
        except Exception:
            labels = labels.reshape(-1)

    # load checkpoint and infer arch/input_dim
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    state_keys = list(state.keys()) if isinstance(state, dict) else []

    arch = 'transformer'
    if any('gru' in k for k in state_keys) or any('bigru' in k for k in state_keys):
        arch = 'gru'
    if any('lstm' in k for k in state_keys):
        arch = 'lstm'
    if any(k.startswith('input_projection') or 'transformer_encoder' in k for k in state_keys):
        arch = 'transformer'

    inferred_input_dim = None
    try:
        if arch == 'transformer' and 'input_projection.weight' in state:
            inferred_input_dim = state['input_projection.weight'].shape[1]
        elif arch == 'gru' and any(k.endswith('weight_ih_l0') for k in state_keys):
            for k in state_keys:
                if k.endswith('weight_ih_l0'):
                    inferred_input_dim = state[k].shape[1]
                    break
        elif arch == 'lstm' and any(k.endswith('weight_ih_l0') for k in state_keys):
            for k in state_keys:
                if k.endswith('weight_ih_l0'):
                    inferred_input_dim = state[k].shape[1]
                    break
    except Exception:
        inferred_input_dim = None

    if data.ndim == 2:
        data_in = data.reshape(data.shape[0], 1, data.shape[1])
    elif data.ndim == 3 and inferred_input_dim is not None and data.shape[2] == inferred_input_dim:
        data_in = data
    else:
        # fallback: assume data is already (N, seq_len, feat)
        data_in = data

    # instantiate model
    # choose a reasonable hidden_dim and num_classes placeholder; load_state_dict will set weights
    # Prefer num_classes from the checkpoint classifier if available to avoid shape mismatch
    num_classes = int(np.unique(labels).size)
    try:
        if isinstance(state, dict):
            if 'classifier.weight' in state:
                num_classes = int(state['classifier.weight'].shape[0])
            elif 'classifier.bias' in state:
                num_classes = int(state['classifier.bias'].shape[0])
    except Exception:
        pass
    hidden_dim = 128
    if arch == 'gru':
        model = ContrastiveGRUClassifier(input_dim=inferred_input_dim or data_in.shape[2], hidden_dim=hidden_dim, num_classes=num_classes)
    elif arch == 'lstm':
        model = ContrastiveLSTMClassifier(input_dim=inferred_input_dim or data_in.shape[2], hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        model = ContrastiveTransformerClassifier(input_dim=inferred_input_dim or data_in.shape[2], hidden_dim=hidden_dim, num_classes=num_classes)

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    feats_list = []
    with torch.no_grad():
        for i in range(0, data_in.shape[0], batch_size):
            batch = torch.from_numpy(data_in[i:i+batch_size]).float().to(device)
            out = model(batch, True)
            if isinstance(out, tuple) and len(out) >= 2:
                feats = out[1].cpu().numpy()
            else:
                # fallback: if model returns only logits, use logits
                feats = out.cpu().numpy()
            feats_list.append(feats)
    features = np.vstack(feats_list)
    return features, labels


def compute_metrics(features, labels, thresholds=100):
    labels = np.array(labels).reshape(-1)
    classes = np.unique(labels)
    K = classes.size

    # centroids
    centroids = np.vstack([features[labels == k].mean(axis=0) for k in classes])

    # 1) Intra-class variance (per-class mean squared distance to centroid)
    intra_per_class = np.array([np.mean(np.sum((features[labels == k] - centroids[i])**2, axis=1)) if np.sum(labels==k)>0 else 0.0 for i,k in enumerate(classes)])
    intra_mean = intra_per_class.mean()
    intra_std = intra_per_class.std()

    # 2) Inter-class distances (pairwise centroid distances)
    pdist = pairwise_distances(centroids, metric='euclidean')
    # ignore diagonal
    inter_vals = pdist[np.triu_indices(K, k=1)] if K>1 else np.array([0.0])
    inter_min = float(inter_vals.min()) if inter_vals.size>0 else 0.0
    inter_mean = float(inter_vals.mean()) if inter_vals.size>0 else 0.0

    # 3) Separation ratio: min_inter / max_intra
    max_intra = float(intra_per_class.max()) if intra_per_class.size>0 else 0.0
    separation_ratio = (inter_min / max_intra) if max_intra>0 else float('inf')

    # 4) Overlap score (error rate where sample is closer to wrong centroid)
    d_to_centroids = pairwise_distances(features, centroids, metric='euclidean')
    closest = d_to_centroids.argmin(axis=1)
    # map class labels to index in classes array
    class_to_idx = {c:i for i,c in enumerate(classes)}
    true_idx = np.array([class_to_idx[l] for l in labels])
    overlap_errors = (closest != true_idx).sum()
    overlap_error_rate = float(overlap_errors) / float(features.shape[0])

    # 5) Silhouette score (use euclidean)
    sil_score = None
    try:
        if len(np.unique(labels)) > 1 and features.shape[0] > len(np.unique(labels)):
            sil_score = float(silhouette_score(features, labels, metric='euclidean'))
    except Exception:
        sil_score = None

    # 6) Threshold instability: sweep thresholds on distance to own centroid
    d_own = d_to_centroids[np.arange(d_to_centroids.shape[0]), true_idx]
    d_other_min = np.min(np.where(np.arange(K)[None,:] == true_idx[:,None], np.inf, d_to_centroids), axis=1)

    t_min = 0.0
    t_max = max(d_own.max(), d_other_min.max())
    thresholds = np.linspace(t_min, t_max, thresholds)
    sweep = []
    for t in thresholds:
        # accept if distance to own centroid < t
        accept_own = d_own < t
        # false reject = own sample rejected
        false_reject = (~accept_own).sum()
        # false accept = number of samples where some other centroid < t
        false_accept = (d_other_min < t).sum()
        sweep.append({'t': float(t), 'false_reject': int(false_reject), 'false_reject_rate': float(false_reject)/features.shape[0], 'false_accept': int(false_accept), 'false_accept_rate': float(false_accept)/features.shape[0]})

    results = {
        'intra_per_class': intra_per_class.tolist(),
        'intra_mean': float(intra_mean),
        'intra_std': float(intra_std),
        'inter_min': float(inter_min),
        'inter_mean': float(inter_mean),
        'separation_ratio': float(separation_ratio),
        'overlap_error_rate': float(overlap_error_rate),
        'silhouette_score': sil_score,
        'threshold_sweep': sweep,
        'num_classes': int(K),
        'num_samples': int(features.shape[0])
    }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_file', type=str, required=False, help='Raw dataset (N, seq, feat) or embeddings (N, seq, feat)')
    p.add_argument('--embeddings_file', type=str, required=False, help='Precomputed embeddings (.npy) to feed into the model')
    p.add_argument('--label_file', type=str, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--threshold_steps', type=int, default=200)
    p.add_argument('--out', type=str, default='results/cluster_metrics.json')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    features, labels = extract_features_from_checkpoint(args.checkpoint, data_file=args.data_file, embeddings_file=getattr(args, 'embeddings_file', None), label_file=args.label_file, device=args.device, batch_size=args.batch_size)
    results = compute_metrics(features, labels, thresholds=args.threshold_steps)

    # Save JSON
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    # Save threshold sweep CSV
    import csv
    csv_path = os.path.splitext(args.out)[0] + '_thresholds.csv'
    keys = ['t', 'false_reject', 'false_reject_rate', 'false_accept', 'false_accept_rate']
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for row in results['threshold_sweep']:
            writer.writerow({k: row[k] for k in keys})

    # Print concise summary
    print('Cluster metrics summary:')
    print(' num_classes:', results['num_classes'], ' num_samples:', results['num_samples'])
    print(' Intra mean:', results['intra_mean'], ' Intra std:', results['intra_std'])
    print(' Inter min:', results['inter_min'], ' Inter mean:', results['inter_mean'])
    print(' Separation ratio (min_inter / max_intra):', results['separation_ratio'])
    print(' Overlap error rate:', results['overlap_error_rate'])
    print(' Silhouette score:', results['silhouette_score'])
    print(' JSON saved to', args.out)
    print(' Thresholds CSV saved to', csv_path)


if __name__ == '__main__':
    main()
