#!/usr/bin/env python
"""Evaluate prototype-based classification after adding a new class prototype.

Creates prototypes from filtered dataset, adds a new prototype for a selected missing class using K-shot samples
(deterministically chosen: first K samples), excludes those K samples from evaluation, and computes accuracy on the remaining data.
"""
import argparse
import json
import os
import numpy as np
from collections import defaultdict


def load_data_config(path='dataset/data_config.json'):
    import json
    with open(path, 'r') as f:
        return json.load(f)


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


def get_logits_and_features(model, embeddings, device='cpu', batch_size=256):
    import torch
    model = model.to(device)
    model.eval()
    logits_list = []
    feats = []
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            out = model(batch, True)
            # out expected: (logits, features, projected)
            if isinstance(out, tuple) and len(out) >= 2:
                logits = out[0].cpu().numpy()
                f = out[1].cpu().numpy()
            else:
                # fallback: model returns logits only
                logits = out.cpu().numpy()
                f = logits
            logits_list.append(logits)
            feats.append(f)
    return np.vstack(logits_list), np.vstack(feats)


def compute_prototypes(features, labels):
    classes = np.unique(labels)
    protos = {}
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            continue
        protos[int(c)] = features[idx].mean(axis=0)
    return protos


def nearest_proto_label(sample_feat, proto_map):
    # proto_map: dict of label_index -> vector
    labels = list(proto_map.keys())
    vecs = np.vstack([proto_map[l] for l in labels])
    dists = np.linalg.norm(vecs - sample_feat[None, :], axis=1)
    idx = int(np.argmin(dists))
    return labels[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--emb_full', default='embed/embed_limu_v1_blind_user_20_120.npy')
    p.add_argument('--labels_full', default='dataset/blind_user/label_20_120.npy')
    p.add_argument('--emb_filtered', default='embed/embed_limu_v1_blind_user_filtered_20_120.npy')
    p.add_argument('--labels_filtered', default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--data_config', default='dataset/data_config.json')
    p.add_argument('--new_class', required=True, help='name of the new class to add (must be in blind_user_20_120)')
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='results/eval_with_new_proto.json')
    p.add_argument('--gap_multiplier', type=float, default=1.0, help='accept new class if mean_gap > gap_multiplier * gap_std')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    cfg = load_data_config(args.data_config)
    full_key = 'blind_user_20_120'
    filt_key = 'blind_user_filtered_20_120'
    full_names = cfg.get(full_key, {}).get('activity_label', None)
    filt_names = cfg.get(filt_key, {}).get('activity_label', None)
    if full_names is None or filt_names is None:
        raise RuntimeError('failed to load label names')

    if args.new_class not in full_names:
        raise RuntimeError('new_class not in full_names')

    new_idx_full = full_names.index(args.new_class)

    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')
    # get logits and features so we can use classifier top-1
    logits_full, feats_full = get_logits_and_features(model, emb_full, device=args.device)
    logits_filt, feats_filt = get_logits_and_features(model, emb_filt, device=args.device)

    # protos from filtered dataset: keys are filtered label indices (0..K-1)
    protos_filt = compute_prototypes(feats_filt, labs_filt)

    # map filtered labels to full indices
    proto_map_full = {}
    for filt_i, name in enumerate(filt_names):
        if filt_i in protos_filt:
            full_i = full_names.index(name)
            proto_map_full[full_i] = protos_filt[filt_i]

    # build new prototype from first `shots` examples of new class in full set
    idxs_new = np.where(labs_full == new_idx_full)[0]
    if idxs_new.size == 0:
        raise RuntimeError('no samples for new class in full dataset')
    chosen = np.sort(idxs_new)[:args.shots]
    new_proto = feats_full[chosen].mean(axis=0)

    # add new prototype into mapping (for distance calculations)
    proto_map_full[new_idx_full] = new_proto

    # build evaluation set: exclude chosen indices
    eval_mask = np.ones(feats_full.shape[0], dtype=bool)
    eval_mask[chosen] = False
    eval_feats = feats_full[eval_mask]
    eval_logits = logits_full[eval_mask]
    eval_labels = labs_full[eval_mask]

    # precompute existing proto vectors for distance calcs
    existing_ids = sorted(proto_map_full.keys())
    existing_vecs = np.vstack([proto_map_full[k] for k in existing_ids])

    # compute shot-based gap std (used as threshold scale)
    # per-shot gaps to nearest existing proto
    shot_d_new = np.linalg.norm(feats_full[chosen] - new_proto[None, :], axis=1)
    shot_d_exist = np.linalg.norm(feats_full[chosen][:, None, :] - existing_vecs[None, :, :], axis=2)
    shot_d_exist_min = shot_d_exist.min(axis=1)
    shot_gaps = shot_d_exist_min - shot_d_new
    gap_std = float(np.std(shot_gaps)) if shot_gaps.size > 1 else 0.0

    preds = []
    for i, (feat, logit) in enumerate(zip(eval_feats, eval_logits)):
        # classifier top-1 maps logits to filtered label indices; map to full index via name
        clf_top = int(np.argmax(logit))
        clf_name = filt_names[clf_top] if clf_top < len(filt_names) else None
        clf_full_idx = full_names.index(clf_name) if clf_name in full_names else None

        # distance to new proto and to classifier-top proto (if available)
        d_new = float(np.linalg.norm(feat - new_proto))
        d_to_existing = np.linalg.norm(existing_vecs - feat[None, :], axis=1)
        d_to_nearest_existing = float(d_to_existing.min())

        # distance to classifier-top proto (if mapped), else large
        if clf_full_idx is not None and clf_full_idx in proto_map_full:
            d_clf_top = float(np.linalg.norm(feat - proto_map_full[clf_full_idx]))
        else:
            d_clf_top = float('inf')

        # decision: accept new class if closer to new proto than classifier top-1 OR mean_gap > multiplier*gap_std
        mean_gap_sample = d_to_nearest_existing - d_new
        accept_new = (d_new < d_clf_top) or (mean_gap_sample > args.gap_multiplier * gap_std)

        if accept_new:
            preds.append(new_idx_full)
        else:
            # map classifier top to full index if possible, otherwise nearest existing proto
            if clf_full_idx is not None:
                preds.append(clf_full_idx)
            else:
                # fallback to nearest existing proto id
                nearest_pos = int(np.argmin(d_to_existing))
                preds.append(existing_ids[nearest_pos])

    preds = np.array(preds)

    overall_acc = float((preds == eval_labels).mean())

    # per-class accuracy
    per_class = {}
    for cls in np.unique(eval_labels):
        idx = eval_labels == cls
        acc = float((preds[idx] == eval_labels[idx]).mean()) if idx.sum() > 0 else None
        per_class[int(cls)] = {
            'name': full_names[int(cls)] if int(cls) < len(full_names) else str(int(cls)),
            'count': int(idx.sum()),
            'accuracy': acc
        }

    # confusion matrix
    labels_sorted = np.unique(np.concatenate([list(proto_map_full.keys()), eval_labels]))
    label_to_pos = {l:i for i,l in enumerate(labels_sorted)}
    cm = np.zeros((labels_sorted.size, labels_sorted.size), dtype=int)
    for t,p in zip(eval_labels, preds):
        cm[label_to_pos[int(t)], label_to_pos[int(p)]] += 1

    result = {
        'new_class': args.new_class,
        'new_class_index_full': int(new_idx_full),
        'n_shots': int(chosen.size),
        'excluded_indices': chosen.tolist(),
        'overall_accuracy': overall_acc,
        'per_class': per_class,
        'labels_sorted': labels_sorted.tolist(),
        'confusion_matrix': cm.tolist()
    }

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    # also save confusion matrix CSV
    import csv
    csv_path = os.path.splitext(args.out)[0] + '_confusion.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        header = ['true\/pred'] + [full_names[int(l)] if int(l) < len(full_names) else str(int(l)) for l in labels_sorted]
        writer.writerow(header)
        for i,l in enumerate(labels_sorted):
            row_vals = cm[i].tolist() if hasattr(cm[i], 'tolist') else list(cm[i])
            row = [full_names[int(l)] if int(l) < len(full_names) else str(int(l))] + row_vals
            writer.writerow(row)

    print('Saved eval JSON to', args.out)
    print('Saved confusion CSV to', csv_path)
    print('Overall accuracy (excluding shots):', overall_acc)


if __name__ == '__main__':
    main()
