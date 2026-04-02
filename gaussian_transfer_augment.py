#!/usr/bin/env python
"""Gaussian-transfer few-shot augmentation.

For each missing class, compute mu_new from K shots, borrow covariance from k nearest seen classes
and sample pseudo-features from N(mu_new, Sigma_new). Saves pseudo-features to disk.

Outputs: results/pseudo/{class_name}_pseudo.npy and results/pseudo/summary.csv
"""
import os
import argparse
import json
import numpy as np


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


def compute_prototypes(features, labels):
    classes = np.unique(labels)
    protos = {}
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            continue
        protos[int(c)] = features[idx].mean(axis=0)
    return protos


def compute_class_covariances(features, labels, protos, eps=1e-6):
    classes = list(protos.keys())
    covs = {}
    D = features.shape[1]
    # global var fallback
    global_var = np.var(features, axis=0).mean() + eps
    for c in classes:
        idx = labels == c
        if idx.sum() <= 1:
            # fallback to isotropic covariance using global var
            covs[c] = np.eye(D) * global_var
            continue
        X = features[idx]
        # rowvar=False to get D x D cov
        cov = np.cov(X, rowvar=False)
        # ensure shape D x D
        if cov.ndim == 0:
            cov = np.eye(D) * float(cov)
        # regularize
        cov += np.eye(D) * eps
        covs[c] = cov
    return covs


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
    p.add_argument('--emb_full', default='embed/embed_limu_v1_blind_user_20_120.npy')
    p.add_argument('--labels_full', default='dataset/blind_user/label_20_120.npy')
    p.add_argument('--emb_filtered', default='embed/embed_limu_v1_blind_user_filtered_20_120.npy')
    p.add_argument('--labels_filtered', default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--data_config', default='dataset/data_config.json')
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--k', type=int, default=3, help='number of seen classes to borrow covariance from')
    p.add_argument('--n_samples', type=int, default=200, help='pseudo-samples to draw per new class')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out_dir', default='results/pseudo')
    p.add_argument('--plot', action='store_true', help='also save overlay plot using t-SNE mapping from filtered classes')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load names
    with open(args.data_config, 'r') as f:
        cfg = json.load(f)
    full_names = cfg.get('blind_user_20_120', {}).get('activity_label', [])
    filt_names = cfg.get('blind_user_filtered_20_120', {}).get('activity_label', [])
    missing = [n for n in full_names if n not in filt_names]
    print('Missing classes:', missing)

    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_full = extract_features(model, emb_full, device=args.device)

    feats_filt = l2_normalize(feats_filt)
    feats_full = l2_normalize(feats_full)

    protos_exist = compute_prototypes(feats_filt, labs_filt)
    covs = compute_class_covariances(feats_filt, labs_filt, protos_exist)

    # if plotting, fit t-SNE on filtered features
    if args.plot:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=args.seed)
        coords = tsne.fit_transform(feats_filt)
    else:
        coords = None

    rng = np.random.default_rng(args.seed)
    summary = []
    D = feats_filt.shape[1]
    for cls_name in missing:
        cls_idx = full_names.index(cls_name)
        idxs = np.where(labs_full == cls_idx)[0]
        if idxs.size == 0:
            print('No samples for', cls_name)
            continue
        chosen = idxs if idxs.size <= args.shots else rng.choice(idxs, size=args.shots, replace=False)
        shot_feats = feats_full[chosen]
        mu_new = shot_feats.mean(axis=0)
        # find k nearest seen classes by prototype distance
        existing_ids = list(protos_exist.keys())
        existing_vecs = np.vstack([protos_exist[k] for k in existing_ids])
        dists = np.linalg.norm(existing_vecs - mu_new[None, :], axis=1)
        nn_idx = np.argsort(dists)[:args.k]
        borrowed_covs = [covs[existing_ids[i]] for i in nn_idx]
        Sigma_new = np.mean(np.stack(borrowed_covs, axis=0), axis=0)
        # regularize Sigma_new
        Sigma_new += np.eye(D) * 1e-6

        # sample pseudo-features
        try:
            samples = rng.multivariate_normal(mean=mu_new, cov=Sigma_new, size=args.n_samples)
        except Exception:
            # fallback to isotropic
            var = np.trace(Sigma_new) / D
            samples = rng.normal(loc=mu_new, scale=np.sqrt(var), size=(args.n_samples, D))

        # scale samples to match prototype norm (avoid projecting all to unit sphere)
        proto_norm = np.linalg.norm(mu_new) + 1e-12
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / (norms + 1e-12) * proto_norm

        out_path = os.path.join(args.out_dir, f"{cls_name.replace(' ', '_')}_pseudo.npy")
        np.save(out_path, samples)
        print('Saved', out_path)

        entry = {'class_name': cls_name, 'class_index': int(cls_idx), 'n_shots': int(chosen.size), 'n_pseudo': int(args.n_samples), 'borrowed_from': [existing_ids[i] for i in nn_idx]}
        summary.append(entry)

        if args.plot and coords is not None:
            proto_2d = map_to_2d(mu_new[None, :], feats_filt, coords, k=10)[0]
            sample_2d = map_to_2d(samples, feats_filt, coords, k=10)
            # save a small per-class plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.4)
            plt.scatter(sample_2d[:, 0], sample_2d[:, 1], s=12, c='r', alpha=0.6)
            plt.scatter(proto_2d[0], proto_2d[1], s=120, marker='*', c='k')
            plt.title(f'Pseudo-samples for {cls_name}')
            fn = os.path.join(args.out_dir, f'{cls_name.replace(" ", "_")}_pseudo.png')
            plt.tight_layout()
            plt.savefig(fn, dpi=150)
            plt.close()

    # write summary
    import csv
    sum_path = os.path.join(args.out_dir, 'summary.csv')
    with open(sum_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['class_name', 'class_index', 'n_shots', 'n_pseudo', 'borrowed_from'])
        writer.writeheader()
        for r in summary:
            writer.writerow(r)
    print('Wrote summary to', sum_path)


if __name__ == '__main__':
    main()
