#!/usr/bin/env python
"""Train a logistic acceptor using real missing prototypes, pseudo-generated prototypes, and fake-new negatives.

Saves: resources/acceptor.joblib (model + scaler) and results/acceptor_training.csv
"""
import os
import argparse
import glob
import numpy as np
import json
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

from few_shot_add_class import (
    infer_label_array, build_model_from_checkpoint, extract_features,
    compute_prototypes, compute_class_variances, evaluate_new_prototype
)


def make_feature_vector(stats):
    # assemble feature vector from stats dict
    return [
        stats.get('mean_margin', 0.0),
        stats.get('proto_net_agreement', 0.0),
        stats.get('mean_prob_new', 0.0),
        stats.get('agreement', 0.0),
        stats.get('mean_gap', 0.0),
        stats.get('nearest_dist', 0.0),
        stats.get('second_dist', 0.0)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--emb_full', default='embed/embed_limu_v1_blind_user_20_120.npy')
    p.add_argument('--labels_full', default='dataset/blind_user/label_20_120.npy')
    p.add_argument('--emb_filtered', default='embed/embed_limu_v1_blind_user_filtered_20_120.npy')
    p.add_argument('--labels_filtered', default='dataset/blind_user_filtered/label_20_120.npy')
    p.add_argument('--pseudo_dir', default='results/pseudo')
    p.add_argument('--shots', type=int, default=5)
    p.add_argument('--n_pseudo_protos', type=int, default=50)
    p.add_argument('--n_neg_per_class', type=int, default=30)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out_model', default='resources/acceptor.joblib')
    p.add_argument('--out_csv', default='results/acceptor_training.csv')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_model) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    with open('dataset/data_config.json', 'r') as f:
        cfg = json.load(f)
    full_names = cfg.get('blind_user_20_120', {}).get('activity_label', [])
    filt_names = cfg.get('blind_user_filtered_20_120', {}).get('activity_label', [])
    missing_names = [n for n in full_names if n not in filt_names]

    emb_full = np.load(args.emb_full)
    labs_full = infer_label_array(np.load(args.labels_full))
    emb_filt = np.load(args.emb_filtered)
    labs_filt = infer_label_array(np.load(args.labels_filtered))

    model = build_model_from_checkpoint(args.checkpoint, arch_hint='gru')
    feats_filt = extract_features(model, emb_filt, device=args.device)
    feats_full = extract_features(model, emb_full, device=args.device)

    # normalize
    def l2_normalize(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-12)
    feats_filt = l2_normalize(feats_filt)
    feats_full = l2_normalize(feats_full)

    protos_exist = compute_prototypes(feats_filt, labs_filt)
    class_vars = compute_class_variances(feats_filt, labs_filt, protos_exist)

    X = []
    y = []
    rows = []

    # Positive examples: real missing prototypes (from full dataset)
    for cls_name in missing_names:
        cls_idx = full_names.index(cls_name)
        idxs = np.where(labs_full == cls_idx)[0]
        if idxs.size == 0:
            continue
        chosen = idxs if idxs.size <= args.shots else np.random.choice(idxs, size=args.shots, replace=False)
        shot_feats = feats_full[chosen]
        new_proto = shot_feats.mean(axis=0)
        stats = evaluate_new_prototype(new_proto, protos_exist, shot_feats, existing_vars=class_vars)
        fv = make_feature_vector(stats)
        X.append(fv)
        y.append(1)
        rows.append({'role': 'real_missing', 'class_name': cls_name, 'features': fv, 'label': 1})

        # Also include pseudo-derived prototypes if available
        pseudo_path = os.path.join(args.pseudo_dir, f"{cls_name.replace(' ', '_')}_pseudo.npy")
        if os.path.exists(pseudo_path):
            samples = np.load(pseudo_path)
            # create many pseudo-prototypes by sampling K and taking mean
            for _ in range(args.n_pseudo_protos):
                sel = np.random.choice(samples.shape[0], size=args.shots, replace=False)
                s_shots = samples[sel]
                p_proto = s_shots.mean(axis=0)
                stats_p = evaluate_new_prototype(p_proto, protos_exist, s_shots, existing_vars=class_vars)
                fv_p = make_feature_vector(stats_p)
                X.append(fv_p)
                y.append(1)
                rows.append({'role': 'pseudo', 'class_name': cls_name, 'features': fv_p, 'label': 1})

    # Negative examples: fake-new from existing classes
    existing_ids = list(protos_exist.keys())
    for c in existing_ids:
        idxs = np.where(labs_filt == c)[0]
        if idxs.size <= args.shots:
            continue
        for _ in range(args.n_neg_per_class):
            chosen = np.random.choice(idxs, size=args.shots, replace=False)
            shot_feats = feats_filt[chosen]
            fake_proto = shot_feats.mean(axis=0)
            # evaluate against protos excluding c to simulate newness
            protos_except = {k: v for k, v in protos_exist.items() if k != c}
            stats_n = evaluate_new_prototype(fake_proto, protos_except, shot_feats, existing_vars=class_vars)
            fv_n = make_feature_vector(stats_n)
            X.append(fv_n)
            y.append(0)
            rows.append({'role': 'fake_existing', 'class_name': c, 'features': fv_n, 'label': 0})

    X = np.array(X)
    y = np.array(y)

    # save training examples CSV
    with open(args.out_csv, 'w', newline='') as cf:
        fieldnames = ['role', 'class_name', 'label'] + [f'f{i}' for i in range(X.shape[1])]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r, xv, lab in zip(rows, X, y):
            row = {'role': r['role'], 'class_name': r['class_name'], 'label': int(lab)}
            for i in range(X.shape[1]):
                row[f'f{i}'] = float(xv[i])
            writer.writerow(row)

    # scale and train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')

    # cross-validate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for train_idx, test_idx in skf.split(Xs, y):
        clf_cv = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf_cv.fit(Xs[train_idx], y[train_idx])
        probs = clf_cv.predict_proba(Xs[test_idx])[:, 1]
        preds = clf_cv.predict(Xs[test_idx])
        aucs.append(roc_auc_score(y[test_idx], probs))
        accs.append(accuracy_score(y[test_idx], preds))

    print('CV AUC: %.4f +- %.4f' % (np.mean(aucs), np.std(aucs)))
    print('CV Acc: %.4f +- %.4f' % (np.mean(accs), np.std(accs)))

    # fit final model
    clf.fit(Xs, y)
    joblib.dump({'scaler': scaler, 'model': clf}, args.out_model)
    print('Saved acceptor to', args.out_model)


if __name__ == '__main__':
    main()
