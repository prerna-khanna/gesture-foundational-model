#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BioPM baseline row for Table 3: their full downstream pipeline, our label rate.

BioPM's own protocol (biopm.logreg_nested_cv) is subject-aware nested CV with an
inner regularisation search -- stricter than this paper's protocol and therefore
not comparable to Table 3. So we keep BioPM's *classifier* (standardise ->
multinomial logistic regression with C chosen on a held-out split, their
DEFAULT_C_GRID) and substitute *our* data protocol: the same
prepare_classifier_dataset split every other row of Table 3 uses, i.e. 10% of the
80% train partition as labels (= 8% of the data) and the 10% test partition.

    python run_biopm_baseline.py
    python run_biopm_baseline.py --datasets HGAG_DATA --native_cv   # also their CV

`--native_cv` additionally reports their subject-aware nested CV for reference.
That number is not comparable to Table 3, but it is what BioPM's authors would
quote, and the gap between the two is informative about how much the random
split inflates every row of Table 3.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from datasets_common import DATASETS, activity_label_index
from utils import prepare_classifier_dataset

DEFAULT_TABLE3 = ("sony_watch", "umahand_filtered", "UTD_MHAD_filtered",
                  "HGAG_DATA", "blind_user_filtered")
# BioPM's own regularisation grid (biopm/evaluation.py DEFAULT_C_GRID).
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def flat(x):
    return x.reshape(len(x), -1)


def fit_eval(tr, ltr, va, lva, te, lte):
    """Standardise, pick C on validation, refit, score on test."""
    scaler = StandardScaler().fit(flat(tr))
    Xtr, Xva, Xte = (scaler.transform(flat(a)) for a in (tr, va, te))

    best_c, best_va = None, -1.0
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=5000).fit(Xtr, ltr)
        s = m.score(Xva, lva)
        if s > best_va:
            best_va, best_c = s, c

    model = LogisticRegression(C=best_c, max_iter=5000).fit(Xtr, ltr)
    pred = model.predict(Xte)
    return (float((pred == lte).mean()),
            float(f1_score(lte, pred, average="macro")),
            best_c, len(ltr), len(lte))


def native_cv(repo, feats, labels, subjects):
    """BioPM's own subject-aware nested CV, for reference only."""
    repo = os.path.expanduser(repo)
    collisions = ("utils", "models", "config", "data", "features",
                  "preprocessing", "evaluation", "inference")
    stashed = {n: sys.modules.pop(n) for n in collisions if n in sys.modules}
    sys.path.insert(0, repo)
    try:
        from biopm.evaluation import logreg_nested_cv
        return logreg_nested_cv(feats, labels, subjects, verbose=False)
    finally:
        sys.path.remove(repo)
        for n in [n for n in collisions if n in sys.modules]:
            del sys.modules[n]
        sys.modules.update(stashed)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_TABLE3), choices=list(DATASETS))
    p.add_argument("--embed_dir", default="new_embed")
    p.add_argument("--tag", default="biopm")
    p.add_argument("--dataset_version", default="20_120")
    p.add_argument("--label_rate", type=float, default=0.1,
                   help="Fraction of the 80%% train partition used as labels, as in the paper")
    p.add_argument("--training_rate", type=float, default=0.8)
    p.add_argument("--seeds", nargs="+", type=int, default=[3431, 1234, 2024])
    p.add_argument("--native_cv", action="store_true")
    p.add_argument("--repo", default="~/Desktop/biopm")
    p.add_argument("--out", default=os.path.join("results", "biopm_baseline.csv"))
    cli = p.parse_args()

    cfg_all = json.load(open(os.path.join("dataset", "data_config.json")))
    rows = []

    for ds in cli.datasets:
        path = os.path.join(cli.embed_dir, f"embed_{cli.tag}_{ds}_{cli.dataset_version}.npy")
        if not os.path.exists(path):
            print(f"[{ds}] no {path}; run extract_biopm_embeddings.py --datasets {ds}")
            continue
        emb = np.load(path).astype(np.float32)
        labels_all = np.load(os.path.join("dataset", ds,
                                          f"label_{cli.dataset_version}.npy")).astype(np.float32)
        idx = activity_label_index(ds, cli.dataset_version)
        seq = emb.shape[1]
        labels = labels_all[:, :seq, :] if labels_all.shape[1] != seq else labels_all
        n_cls = len(np.unique(labels[:, 0, idx]))

        accs, f1s, cs = [], [], []
        for seed in cli.seeds:
            tr, ltr, va, lva, te, lte = prepare_classifier_dataset(
                emb, labels, label_index=idx, training_rate=cli.training_rate,
                label_rate=cli.label_rate, merge=min(20, seq), seed=seed, balance=True)
            a, f, c, ntr, nte = fit_eval(tr, ltr, va, lva, te, lte)
            accs.append(a); f1s.append(f); cs.append(c)

        row = {"dataset": ds, "n_classes": n_cls, "n_train": ntr, "n_test": nte,
               "acc_mean": np.mean(accs), "acc_std": np.std(accs),
               "f1_mean": np.mean(f1s), "f1_std": np.std(f1s), "C": cs}
        rows.append(row)
        print(f"[{ds:20s}] {n_cls:2d} cls  train={ntr:6d} test={nte:6d}  "
              f"acc={row['acc_mean']:.3f}±{row['acc_std']:.3f}  "
              f"F1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  C={cs}")

        if cli.native_cv:
            uidx = cfg_all[f"{ds}_{cli.dataset_version}"].get("user_label_index")
            if uidx is None:
                print("      (no user_label_index; skipping native CV)")
            else:
                res = native_cv(cli.repo, flat(emb), labels[:, 0, idx].astype(int),
                                labels[:, 0, uidx].astype(int))
                print(f"      BioPM native {res['cv_strategy']}: "
                      f"acc={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f} "
                      f"F1={res['macro_f1_mean']:.3f}±{res['macro_f1_std']:.3f}")
                row["native_acc"] = res["accuracy_mean"]
                row["native_f1"] = res["macro_f1_mean"]
                row["native_cv"] = res["cv_strategy"]

    if rows:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(cli.out, index=False)
        print(f"\nwritten to {cli.out}")
        print("\n% LaTeX row for Table 3 (Acc / F1):")
        cells = []
        for ds in cli.datasets:
            r = next((r for r in rows if r["dataset"] == ds), None)
            cells.append("--" if r is None else f"{r['acc_mean']:.2f} / {r['f1_mean']:.2f}")
        print(r"BioPM~\citep{biopm2026} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
