#!/usr/bin/env python
"""Plot missing-class prototypes overlaid on fixed t-SNE of filtered classes.

Produces: results/plots/overlay_missing_prototypes.png and results/prototype_coords.csv
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


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


def infer_label_array(labels):
    labs = np.array(labels)
    if labs.ndim > 1:
        labs = labs.reshape(labs.shape[0], -1)[:, 0]
    if labs.size > 0 and labs.min() >= 1:
        return (labs - 1).astype(int)
    return labs.astype(int)


def compute_prototypes(features, labels):
    classes = np.unique(labels)
    protos = {}
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            continue
        protos[int(c)] = features[idx].mean(axis=0)
    return protos


def map_to_2d(points, ref_feats, ref_2d, k=10, eps=1e-8):
    # weighted k-NN in feature space to map new points into existing 2D layout
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
    p.add_argument('--device', default='cpu')
    p.add_argument('--out_png', default='results/plots/overlay_missing_prototypes.png')
    p.add_argument('--out_csv', default='results/prototype_coords.csv')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    # load names
    with open(args.data_config, 'r') as f:
        cfg = json.load(f)
    full_names = cfg.get('blind_user_20_120', {}).get('activity_label', [])
    filt_names = cfg.get('blind_user_filtered_20_120', {}).get('activity_label', [])
    missing = [n for n in full_names if n not in filt_names]

    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_full = extract_features(model, emb_full, device=args.device)

    feats_filt = l2_normalize(feats_filt)
    feats_full = l2_normalize(feats_full)

    # compute protos
    protos_exist = compute_prototypes(feats_filt, labs_filt)

    # fit t-SNE on filtered features
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    coords = tsne.fit_transform(feats_filt)

    # prepare plot
    plt.figure(figsize=(10, 8))
    num_classes = len(np.unique(labs_filt))
    cmap = plt.get_cmap('tab20')
    for i, c in enumerate(sorted(np.unique(labs_filt))):
        idx = np.where(labs_filt == c)[0]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=10, color=cmap(i % 20), alpha=0.6, label=filt_names[c] if c < len(filt_names) else str(c))

    # for each missing class, compute prototype and shots, map to 2D
    proto_rows = []
    for cls_name in missing:
        cls_idx = full_names.index(cls_name)
        idxs = np.where(labs_full == cls_idx)[0]
        if idxs.size == 0:
            continue
        chosen = idxs if idxs.size <= args.shots else np.random.choice(idxs, size=args.shots, replace=False)
        shot_feats = feats_full[chosen]
        new_proto = shot_feats.mean(axis=0)
        new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-12)

        # map prototype and shots into 2D using weighted knn
        proto_2d = map_to_2d(np.vstack([new_proto]), feats_filt, coords, k=10)[0]
        shots_2d = map_to_2d(shot_feats, feats_filt, coords, k=10)

        # plot prototype and shots
        plt.scatter(proto_2d[0], proto_2d[1], s=200, marker='*', color='k', edgecolors='w', linewidths=0.8)
        plt.scatter(shots_2d[:, 0], shots_2d[:, 1], s=60, marker='x', color='k')
        plt.text(proto_2d[0], proto_2d[1], f' {cls_name}', fontsize=9)

        proto_rows.append({'class_name': cls_name, 'class_index': int(cls_idx), 'proto_x': float(proto_2d[0]), 'proto_y': float(proto_2d[1])})

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title('Filtered-class t-SNE with missing-class prototypes (mapped)')
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print('Saved plot to', args.out_png)

    # save prototype coords
    import csv
    with open(args.out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['class_name', 'class_index', 'proto_x', 'proto_y'])
        writer.writeheader()
        for r in proto_rows:
            writer.writerow(r)
    print('Saved prototype coords to', args.out_csv)


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""Overlay new-class prototypes on the fixed 12-class t-SNE layout.

This script fits t-SNE on the filtered 12-class classifier features (so layout stays fixed),
then maps K-shot embeddings and prototypes (for missing classes) into that 2D space and
plots them. Existing classes are shown as colored points (no large + markers or text labels).
K-shot points are drawn with distinct markers and prototypes are drawn as colored stars
(green=accepted, red=rejected). A legend is shown.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def compute_prototypes(features, labels):
    classes = np.unique(labels)
    protos = {}
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            continue
        protos[int(c)] = features[idx].mean(axis=0)
    return protos


def map_to_2d(vec, base_feats, base_2d, k=10):
    # map a high-dim vector into 2D by weighted average of k nearest base points' 2D coords
    dists = np.linalg.norm(base_feats - vec[None, :], axis=1)
    kn = min(k, base_feats.shape[0])
    idx = np.argpartition(dists, kn-1)[:kn]
    sel = base_2d[idx]
    sel_d = dists[idx]
    # weights: inverse distance (add eps)
    w = 1.0 / (sel_d + 1e-8)
    w = w / w.sum()
    mapped = (sel * w[:, None]).sum(axis=0)
    return mapped


def evaluate_accept(new_proto, shot_feats, protos_exist, min_agreement=0.6):
    existing_ids = list(protos_exist.keys())
    existing_vecs = np.vstack([protos_exist[k] for k in existing_ids])
    # per-shot agreement
    d_shot_new = np.linalg.norm(shot_feats - new_proto[None, :], axis=1)
    d_shot_exist = np.linalg.norm(shot_feats[:, None, :] - existing_vecs[None, :, :], axis=2)
    d_shot_exist_min = d_shot_exist.min(axis=1)
    agreement = float((d_shot_new < d_shot_exist_min).mean())
    mean_gap = float((d_shot_exist_min - d_shot_new).mean())
    accept = (agreement >= min_agreement) and (mean_gap > 0)
    return accept, agreement, mean_gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--emb_full', default='embed/embed_limu_v1_blind_user_20_120.npy')
    p.add_argument('--labels_full', default='dataset/blind_user/label_20_120.npy')
    p.add_argument('--emb_filtered', default='embed/embed_limu_v1_blind_user_filtered_20_120.npy')
    p.add_argument('--labels_filtered', default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--data_config', default='dataset/data_config.json')
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--k_map', type=int, default=10, help='k neighbors for mapping into 2D')
    p.add_argument('--min_agreement', type=float, default=0.6)
    p.add_argument('--out', default='results/plots/overlay_missing_prototypes.png')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    cfg = json.load(open(args.data_config))
    full_names = cfg['blind_user_20_120']['activity_label']
    filt_names = cfg['blind_user_filtered_20_120']['activity_label']
    missing = [n for n in full_names if n not in filt_names]
    print('Missing classes:', missing)

    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint)
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_full = extract_features(model, emb_full, device=args.device)

    # t-SNE on filtered features only (fixed layout)
    tsne = TSNE(n_components=2, init='pca', perplexity=30, random_state=42)
    feats2 = tsne.fit_transform(feats_filt)

    protos_exist = compute_prototypes(feats_filt, labs_filt)

    # prepare plot: show filtered-class points only (no big + markers or text labels)
    plt.figure(figsize=(10,8))
    unique = np.unique(labs_filt)
    n = unique.size
    try:
        cmap = plt.cm.get_cmap('tab20' if n <= 20 else 'hsv', n)
        colors = [cmap(i) for i in range(n)]
    except Exception:
        cmap = plt.cm.get_cmap('hsv')
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    label_to_color = {int(l): colors[i % len(colors)] for i,l in enumerate(unique)}

    for lab in unique:
        idx = labs_filt == lab
        name = filt_names[int(lab)] if int(lab) < len(filt_names) else str(int(lab))
        plt.scatter(feats2[idx,0], feats2[idx,1], s=12, color=label_to_color[int(lab)], alpha=0.7, label=name)

    # for each missing class compute prototype from first-K shots, map and plot K-shot embeddings
    coords_out = []
    markers = ['^', '*', 'o', 's', 'D']
    for i, cls_name in enumerate(missing):
        cls_idx = full_names.index(cls_name)
        idxs = np.where(labs_full == cls_idx)[0]
        if idxs.size == 0:
            continue
        chosen = np.sort(idxs)[:args.shots]
        shot_feats = feats_full[chosen]
        new_proto = shot_feats.mean(axis=0)
        # map shots and proto
        shot2d = np.vstack([map_to_2d(s, feats_filt, feats2, k=args.k_map) for s in shot_feats])
        proto2d_new = map_to_2d(new_proto, feats_filt, feats2, k=args.k_map)

        accept, agreement, mean_gap = evaluate_accept(new_proto, shot_feats, protos_exist, min_agreement=args.min_agreement)

        mk = markers[i % len(markers)]
        # plot each shot with that marker (hollow, black edge)
        for s2 in shot2d:
            plt.scatter(s2[0], s2[1], marker=mk, s=90, facecolors='none', edgecolors='k', linewidths=1.2)

        # plot the prototype as a colored star (green=accept, red=reject)
        color = 'green' if accept else 'red'
        plt.scatter(proto2d_new[0], proto2d_new[1], marker='*', s=170, color=color, edgecolor='white', linewidths=0.8)
        coords_out.append((cls_name, accept, float(agreement), float(mean_gap), float(proto2d_new[0]), float(proto2d_new[1])))

    # legend: include class names and shot/proto proxies
    handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D
    shot_proxies = [Line2D([0],[0], marker=m, color='w', label=f'K-shot ({i+1})', markerfacecolor='none', markeredgecolor='k', markersize=8) for i,m in enumerate(markers)]
    proto_accept = Line2D([0],[0], marker='*', color='w', label='Accepted prototype', markerfacecolor='green', markersize=10)
    proto_reject = Line2D([0],[0], marker='*', color='w', label='Rejected prototype', markerfacecolor='red', markersize=10)
    handles.extend(shot_proxies[:len(markers)])
    labels.extend([p.get_label() for p in shot_proxies[:len(markers)]])
    handles.append(proto_accept)
    handles.append(proto_reject)
    labels.append('Accepted prototype')
    labels.append('Rejected prototype')
    plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.02,1), fontsize='small')

    plt.title('Filtered 12-class t-SNE (fixed) with overlaid K-shot points and prototypes')
    plt.tight_layout(rect=[0,0,0.85,1])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close()

    # save coords
    os.makedirs('results', exist_ok=True)
    import csv
    csv_path = os.path.join('results','prototype_coords.csv')
    with open(csv_path,'w',newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['class','accept','agreement','mean_gap','x','y'])
        for r in coords_out:
            w.writerow(r)
    print('Saved overlay plot to', args.out)
    print('Saved prototype coords to', csv_path)


if __name__ == '__main__':
    main()
