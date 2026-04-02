#!/usr/bin/env python
"""Few-shot prototype tester for adding new gesture classes without retraining.

Workflow:
- Load precomputed embeddings (or raw data) and labels for the full `blind_user` dataset.
- Load embeddings+labels for `blind_user_filtered` (the current 12 classes).
- Run both through the trained classifier checkpoint to extract classifier features.
- Compute prototypes (mean features) for existing classes.
- For each target (missing) class, pick K few-shot examples, compute prototype, and evaluate:
  - distance to nearest existing prototype
  - sample agreement: fraction of shots closer to the new prototype than to any existing prototype
  - gap statistics (mean margin to closest existing prototype)

Decision rule (configurable): accept if sample_agreement >= min_agreement (default 0.6) AND mean_gap > 0

Outputs a small CSV and prints decisions.
"""
import os
import json
import argparse
import numpy as np
import csv

def load_data_config(path='dataset/data_config.json'):
    with open(path, 'r') as f:
        return json.load(f)


def infer_label_array(labels):
    labs = np.array(labels)
    if labs.ndim > 1:
        labs = labs.reshape(labs.shape[0], -1)[:, 0]
    # Detect 1-based labelled datasets (common in this repo)
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

    # infer arch
    arch = arch_hint
    if any(k.startswith('input_projection') or 'transformer_encoder' in k for k in state_keys):
        arch = 'transformer'
    elif any('lstm' in k for k in state_keys):
        arch = 'lstm'
    elif any('gru' in k for k in state_keys) or any('bigru' in k for k in state_keys):
        arch = 'gru'

    # infer input dim
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

    # infer num_classes
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


def compute_class_variances(features, labels, protos, eps=1e-6):
    classes = list(protos.keys())
    vars = {}
    for c in classes:
        idx = labels == c
        if idx.sum() == 0:
            vars[c] = eps
            continue
        dif = features[idx] - protos[c][None, :]
        d2 = np.sum(dif ** 2, axis=1)
        vars[c] = float(np.mean(d2)) + eps
    return vars


def distance(a, b):
    return np.linalg.norm(a - b)


def evaluate_new_prototype(new_proto, existing_protos, shot_features, existing_vars=None, eps=1e-6):
    # distances to existing prototypes
    existing_ids = list(existing_protos.keys())
    existing_vecs = np.vstack([existing_protos[k] for k in existing_ids])
    dists = np.linalg.norm(existing_vecs - new_proto[None, :], axis=1)
    nearest_idx = int(np.argmin(dists))
    nearest_id = existing_ids[nearest_idx]
    nearest_dist = float(dists[nearest_idx])
    second_dist = float(np.partition(dists, 1)[1]) if dists.size > 1 else float('inf')

    # normalize if not already (assume caller normalized features)
    # per-shot agreement (distance-based) as before using Euclidean
    d_shot_new = np.linalg.norm(shot_features - new_proto[None, :], axis=1)
    d_shot_exist = np.linalg.norm(shot_features[:, None, :] - existing_vecs[None, :, :], axis=2)
    d_shot_exist_min = d_shot_exist.min(axis=1)
    agreement = float((d_shot_new < d_shot_exist_min).mean())
    mean_gap = float((d_shot_exist_min - d_shot_new).mean())

    # Mahalanobis-like scoring: divide squared distances by class variance
    if existing_vars is None:
        # fallback: use unit variance
        existing_vars = {k: 1.0 for k in existing_ids}
    # score_new and score_old per shot
    # compute squared distances to existing protos
    d2_exist = np.sum((shot_features[:, None, :] - existing_vecs[None, :, :]) ** 2, axis=2)  # shots x C
    # scale by per-class variance
    var_arr = np.array([existing_vars[k] for k in existing_ids])[None, :]
    scaled_exist = d2_exist / (var_arr + eps)
    # for new proto, use its empirical variance (mean squared distance of shots to new_proto) or small eps
    var_new = float(np.mean(np.sum((shot_features - new_proto[None, :]) ** 2, axis=1))) + eps
    # regularize new-prototype variance with mean existing class variance to avoid instability for small K
    try:
        mean_existing_var = float(np.mean(list(existing_vars.values()))) if existing_vars is not None else var_new
    except Exception:
        mean_existing_var = var_new
    var_new = max(var_new, mean_existing_var + eps)
    d2_new = np.sum((shot_features - new_proto[None, :]) ** 2, axis=1)
    scaled_new = d2_new / (var_new + eps)
    # convert to scores (higher is better): negative scaled distance
    score_new = -scaled_new
    score_old = -scaled_exist.min(axis=1)
    margins = score_new - score_old
    mean_margin = float(margins.mean())

    # Prototypical-net style probabilities: use squared Euclidean distances and softmax over -d^2
    protos_all = np.vstack([existing_vecs, new_proto[None, :]])
    # compute squared dists: shots x (C+1)
    diff = shot_features[:, None, :] - protos_all[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    # logits = -d2 (optionally / (2*sigma^2))
    logits = -d2
    # stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - logits_max)
    probs = ex / ex.sum(axis=1, keepdims=True)
    # proto-net agreement: fraction of shots where argmax is the new prototype (last column)
    proto_net_preds = probs.argmax(axis=1)
    proto_net_agreement = float((proto_net_preds == (protos_all.shape[0]-1)).mean())
    mean_prob_new = float(probs[:, -1].mean())

    return {
        'nearest_existing_id': int(nearest_id),
        'nearest_dist': nearest_dist,
        'second_dist': second_dist,
        'agreement': agreement,
        'mean_gap': mean_gap,
        'mean_margin': mean_margin,
        'proto_net_agreement': proto_net_agreement,
        'mean_prob_new': mean_prob_new
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--emb_full', type=str, default='embed/embed_limu_v1_blind_user_20_120.npy', help='embeddings for full blind_user (contains missing classes)')
    p.add_argument('--labels_full', type=str, default='dataset/blind_user/label_20_120.npy')
    p.add_argument('--emb_filtered', type=str, default='embed/embed_limu_v1_blind_user_filtered_20_120.npy', help='embeddings for filtered (current) dataset')
    p.add_argument('--labels_filtered', type=str, default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--data_config', type=str, default='dataset/data_config.json')
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--min_agreement', type=float, default=0.6, help='min fraction of shots closer to new proto than to any existing proto to accept')
    p.add_argument('--out_csv', type=str, default='results/few_shot_prototype_results.csv')
    p.add_argument('--acceptor', type=str, default='resources/acceptor.joblib', help='path to trained acceptor joblib')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    cfg = load_data_config(args.data_config)
    # names
    full_key = 'blind_user_20_120'
    filt_key = 'blind_user_filtered_20_120'
    full_names = cfg.get(full_key, {}).get('activity_label', None)
    filt_names = cfg.get(filt_key, {}).get('activity_label', None)
    if full_names is None or filt_names is None:
        raise RuntimeError('Failed to load label names from data_config.json')

    # missing classes
    missing = [n for n in full_names if n not in filt_names]
    print('Missing classes to evaluate:', missing)

    # load embeddings and labels
    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    # build model
    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')

    # extract features for filtered and full datasets (classifier features)
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_full = extract_features(model, emb_full, device=args.device)

    # prototypes for existing classes (from filtered dataset)
    protos_exist = compute_prototypes(feats_filt, labs_filt)
    # normalize features and prototypes
    def l2_normalize(x, eps=1e-12):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)

    feats_filt = l2_normalize(feats_filt)
    feats_full = l2_normalize(feats_full)
    # recompute normalized prototypes
    protos_exist = compute_prototypes(feats_filt, labs_filt)
    # compute per-class variances (in normalized space)
    class_vars = compute_class_variances(feats_filt, labs_filt, protos_exist)

    # Calibration: simulate fake-new classes using existing classes to get distribution of mean_margin and proto_net_agreement
    calib_margins = []
    calib_proto_agree = []
    rng = np.random.default_rng(42)
    for c in protos_exist.keys():
        idxs = np.where(labs_filt == c)[0]
        if idxs.size <= args.shots:
            continue
        chosen = rng.choice(idxs, size=args.shots, replace=False)
        shot_feats = feats_filt[chosen]
        fake_proto = shot_feats.mean(axis=0)
        stats_fake = evaluate_new_prototype(fake_proto, {k: v for k, v in protos_exist.items() if k != c}, shot_feats, existing_vars=class_vars)
        calib_margins.append(stats_fake.get('mean_margin', 0.0))
        calib_proto_agree.append(stats_fake.get('proto_net_agreement', 0.0))
    calib_margins = np.array(calib_margins) if len(calib_margins) else np.array([0.0])
    calib_proto_agree = np.array(calib_proto_agree) if len(calib_proto_agree) else np.array([0.0])
    calib_margin_mean = float(calib_margins.mean())
    calib_margin_std = float(calib_margins.std())

    rows = []
    for cls_name in missing:
        cls_idx = full_names.index(cls_name)
        # find samples in full dataset belonging to this class
        idxs = np.where(labs_full == cls_idx)[0]
        if idxs.size == 0:
            print(f'No samples found for {cls_name} (index {cls_idx})')
            continue
        # select shots
        if idxs.size <= args.shots:
            chosen = idxs
        else:
            chosen = np.random.choice(idxs, size=args.shots, replace=False)

        shot_feats = feats_full[chosen]
        new_proto = shot_feats.mean(axis=0)
        # normalize new prototype to match normalized existing prototypes
        new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-12)

        # pass per-class variances into evaluation so Mahalanobis-like scoring uses them
        stats = evaluate_new_prototype(new_proto, protos_exist, shot_feats, existing_vars=class_vars)

        # confusion check: if shots mostly map to a single existing class, treat as duplicate
        # compute nearest existing class votes
        existing_ids = list(protos_exist.keys())
        existing_vecs = np.vstack([protos_exist[k] for k in existing_ids])
        d_shot_exist = np.linalg.norm(shot_feats[:, None, :] - existing_vecs[None, :, :], axis=2)
        v = d_shot_exist.argmin(axis=1)
        votes = np.bincount(v, minlength=len(existing_ids)).astype(float)
        votes = votes / votes.sum()
        max_vote_frac = votes.max()

        # acceptance: prefer trained acceptor if available (uses same features as train_acceptor)
        mean_margin = stats.get('mean_margin', 0.0)
        proto_agree = stats.get('proto_net_agreement', 0.0)
        mean_prob_new = stats.get('mean_prob_new', 0.0)

        acceptor_prob = None
        accept = False
        if args.acceptor and os.path.exists(args.acceptor):
            try:
                import joblib
                acc = joblib.load(args.acceptor)
                scaler = acc.get('scaler')
                model = acc.get('model')
                # feature order must match train_acceptor.make_feature_vector
                fv = np.array([mean_margin, proto_agree, mean_prob_new, float(stats.get('agreement', 0.0)), float(stats.get('mean_gap', 0.0)), float(stats.get('nearest_dist', 0.0)), float(stats.get('second_dist', 0.0))])
                fv_s = scaler.transform(fv.reshape(1, -1)) if scaler is not None else fv.reshape(1, -1)
                acceptor_prob = float(model.predict_proba(fv_s)[:, 1][0])
                # require not duplicate
                accept = (acceptor_prob > 0.5) and (max_vote_frac < 0.8)
            except Exception:
                acceptor_prob = None
                accept = False
        else:
            # fallback: use z-score of mean_margin against calibration (data-driven, no magic constants)
            z = (mean_margin - calib_margin_mean) / (calib_margin_std + 1e-12)
            calib_proto_mean = float(np.mean(calib_proto_agree)) if calib_proto_agree.size else 0.0
            accept = (z > 0.0) and (proto_agree > calib_proto_mean) and (max_vote_frac < 0.8)

        row = {
            'class_name': cls_name,
            'class_index': int(cls_idx),
            'n_shots': int(chosen.size),
            'nearest_existing_id': int(stats['nearest_existing_id']),
            'nearest_existing_name': filt_names[stats['nearest_existing_id']] if stats['nearest_existing_id'] < len(filt_names) else str(stats['nearest_existing_id']),
            'nearest_dist': float(stats['nearest_dist']),
            'second_dist': float(stats['second_dist']),
            'agreement': float(stats['agreement']),
            'mean_gap': float(stats['mean_gap']),
            'proto_net_agreement': float(stats.get('proto_net_agreement', 0.0)),
            'mean_prob_new': float(stats.get('mean_prob_new', 0.0)),
            'mean_margin': float(stats.get('mean_margin', 0.0)),
            'max_vote_frac': float(max_vote_frac),
            'acceptor_prob': (float(acceptor_prob) if acceptor_prob is not None else ''),
            'accept': bool(accept)
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    # save CSV
    keys = ['class_name','class_index','n_shots','nearest_existing_id','nearest_existing_name','nearest_dist','second_dist','agreement','mean_gap','proto_net_agreement','mean_prob_new','mean_margin','max_vote_frac','accept']
    with open(args.out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})

    print('Results saved to', args.out_csv)


if __name__ == '__main__':
    main()
