#!/usr/bin/env python
"""Plot pseudo-sample clusters produced by gaussian_transfer_augment.py.

Loads pseudo feature files from `results/pseudo/*.npy`, maps them into a fixed t-SNE
computed on filtered-class features, and saves an overlay plot and CSV of mapped coords.
"""
import os
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt


def infer_label_array(labels):
    labs = np.array(labels)
    if labs.ndim > 1:
        labs = labs.reshape(labs.shape[0], -1)[:, 0]
    if labs.size > 0 and labs.min() >= 1:
        return (labs - 1).astype(int)
    return labs.astype(int)


def build_model_from_checkpoint(ckpt_path, arch_hint='gru', num_classes=None):
    import torch
    from contrastive.models import ContrastiveTransformerClassifier, ContrastiveGRUClassifier, ContrastiveLSTMClassifier

    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
        state = state.get('model_state_dict', state.get('state_dict'))
    state_keys = list(state.keys()) if isinstance(state, dict) else []

    arch = arch_hint
    if any(k.startswith('input_projection') or 'transformer_encoder' in k for k in state_keys):
        arch = 'transformer'
    elif any('lstm' in k for k in state_keys):
        arch = 'lstm'
    elif any('gru' in k for k in state_keys) or any('bigru' in k for k in state_keys):
        arch = 'gru'

    inferred_input_dim = None
    try:
        if arch == 'transformer' and 'input_projection.weight' in state:
            inferred_input_dim = int(state['input_projection.weight'].shape[1])
        else:
            for k in state_keys:
                if k.endswith('weight_ih_l0'):
                    inferred_input_dim = int(state[k].shape[1])
                    break
    except Exception:
        inferred_input_dim = None

    try:
        if isinstance(state, dict) and 'classifier.weight' in state:
            num_classes = int(state['classifier.weight'].shape[0])
    except Exception:
        pass

    hidden_dim = 128
    if arch == 'gru':
        model = ContrastiveGRUClassifier(input_dim=inferred_input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    elif arch == 'lstm':
        model = ContrastiveLSTMClassifier(input_dim=inferred_input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        model = ContrastiveTransformerClassifier(input_dim=inferred_input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    return model


def extract_features(model, embeddings, device='cpu', batch_size=256):
    import torch
    model = model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            out = model(batch, True)
            if isinstance(out, tuple) and len(out) >= 2:
                f = out[1].cpu().numpy()
            else:
                f = out.cpu().numpy()
            feats.append(f)
    return np.vstack(feats)


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def map_to_2d(points, ref_feats, ref_2d, k=10, eps=1e-8):
    mapped = []
    for p in points:
        d = np.linalg.norm(ref_feats - p[None, :], axis=1)
        nn = np.argsort(d)[:k]
        w = 1.0 / (d[nn] + eps)
        w = w / w.sum()
        coord = (w[:, None] * ref_2d[nn]).sum(axis=0)
        mapped.append(coord)
    return np.vstack(mapped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--emb_filtered', default='embed/embed_limu_v1_blind_user_filtered_20_120.npy')
    p.add_argument('--labels_filtered', default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--pseudo_dir', default='results/pseudo')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out_png', default='results/plots/pseudo_overlay.png')
    p.add_argument('--out_csv', default='results/pseudo_coords.csv')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)

    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_filt = l2_normalize(feats_filt)

    # fit t-SNE on filtered features
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    coords = tsne.fit_transform(feats_filt)

    # plot base filtered classes
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab20')
    for i, c in enumerate(sorted(np.unique(labs_filt))):
        idx = np.where(labs_filt == c)[0]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=12, color=cmap(i % 20), alpha=0.5, label=str(c))

    # load pseudo files and map
    pseudo_files = sorted(glob.glob(os.path.join(args.pseudo_dir, '*_pseudo.npy')))
    rows = []
    # fit a KNeighborsRegressor to predict 2D coords from feature space (better out-of-sample mapping)
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    knn.fit(feats_filt, coords)

    for i, pf in enumerate(pseudo_files):
        name = os.path.basename(pf).replace('_pseudo.npy', '').replace('_', ' ')
        samples = np.load(pf)
        # ensure normalized and scale to prototype norm if needed (samples likely already scaled)
        samples = l2_normalize(samples)
        mapped = knn.predict(samples)
        plt.scatter(mapped[:, 0], mapped[:, 1], s=18, alpha=0.7, marker='x', label=f'pseudo {name}')
        for rcoord in mapped:
            rows.append({'file': os.path.basename(pf), 'class_name': name, 'x': float(rcoord[0]), 'y': float(rcoord[1])})

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title('Filtered-class t-SNE with pseudo-sample clusters')
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print('Saved plot to', args.out_png)

    # save CSV
    import csv
    with open(args.out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['file', 'class_name', 'x', 'y'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Saved pseudo coords to', args.out_csv)


if __name__ == '__main__':
    main()
