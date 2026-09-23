#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qualitative companion to the Tier 1 text ablation.

Gives the mechanistic argument for why semantics would help, rather than only an
accuracy delta: it dumps the class-similarity matrix each arm actually feeds to
the semantic loss, lists each class's nearest semantic neighbours, and -- if you
point it at a confusion matrix -- reports how well semantic similarity predicts
which classes the model confuses.

This is the fallback framing if the ablation arms come out close together: even
where accuracy is flat, a real correlation between semantic proximity and
confusion structure is evidence the descriptions encode something meaningful.

Usage
-----
    python analyze_semantic_structure.py
    python analyze_semantic_structure.py --datasets blind_user_filtered \
        --confusion results/text_ablation/<stamp>/details.json
"""

import argparse
import json
import os

import numpy as np
import torch

from config import load_dataset_label_names, load_dataset_stats
from contrastive import text_variants
from utils import get_device

from datasets_common import DATASETS, DEFAULT_DATASETS
ARMS = ("real", "shuffled", "mismatched", "onehot", "random")


def build_similarity(arm, descriptions, pooling, device, seed, tokenizer, model):
    anchors, info = text_variants.build_class_anchors(
        arm, descriptions, pooling, device, seed=seed, tokenizer=tokenizer, model=model)
    return text_variants.anchors_to_similarity(anchors).cpu().numpy(), info


def nearest_neighbours(sim, label_names, k=3):
    """Each class's k most semantically similar other classes."""
    out = {}
    for i, name in enumerate(label_names):
        order = np.argsort(-sim[i])
        neighbours = [(label_names[j], float(sim[i, j])) for j in order if j != i][:k]
        out[name] = neighbours
    return out


def confusion_correlation(sim, confusion):
    """Spearman correlation between off-diagonal semantic similarity and
    off-diagonal confusion rate. A positive value means the model confuses
    semantically similar classes -- the ordering the method assumes."""
    confusion = np.asarray(confusion, dtype=float)
    if confusion.shape != sim.shape:
        return None

    # Row-normalise so classes with more test samples do not dominate.
    row_sums = confusion.sum(axis=1, keepdims=True)
    rates = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)

    off = ~np.eye(sim.shape[0], dtype=bool)
    x, y = sim[off], rates[off]
    if np.std(x) == 0 or np.std(y) == 0:
        return None

    # Spearman = Pearson on ranks; avoids a scipy dependency this repo does not use.
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    xr, yr = rank(x), rank(y)
    return float(np.corrcoef(xr, yr)[0, 1])


def load_confusion_matrices(path):
    """Pull {tag: confusion_matrix} out of a run's details.json."""
    if not path:
        return {}
    with open(path) as f:
        return {d["tag"]: d["confusion_matrix"] for d in json.load(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DATASETS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(text_variants.TEXT_MODES))
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--seed", type=int, default=3431)
    parser.add_argument("--confusion", default=None, help="details.json from run_text_ablation.py")
    parser.add_argument("--out_dir", default=os.path.join("results", "semantic_structure"))
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--plot", action="store_true", help="Also save heatmaps")
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    device = get_device(cli.gpu)
    confusions = load_confusion_matrices(cli.confusion)

    # Load BERT once and share it across datasets and arms.
    tokenizer = model = None
    if any(text_variants.anchor_mode(a) in text_variants.BERT_MODES for a in cli.arms):
        from transformers import AutoModel, AutoTokenizer
        bert_name = text_variants.bert_model_name()
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        model = AutoModel.from_pretrained(bert_name).to(device)

    report = {}
    for dataset in cli.datasets:
        dataset_cfg = load_dataset_stats(dataset, cli.dataset_version)
        label_names, label_num, descriptions = load_dataset_label_names(dataset_cfg, 0)
        if descriptions is None:
            print(f"[{dataset}] no descriptions in data_config.json, skipping")
            continue

        report[dataset] = {}
        for arm in cli.arms:
            sim, info = build_similarity(arm, descriptions, cli.pooling, device,
                                         cli.seed, tokenizer, model)

            np.savetxt(os.path.join(cli.out_dir, f"sim_{dataset}_{arm}.csv"),
                       sim, delimiter=",", header=",".join(label_names), comments="")

            off = ~np.eye(label_num, dtype=bool)
            entry = {
                "descriptions_used": info["descriptions_used"],
                "offdiag_mean": float(sim[off].mean()),
                "offdiag_std": float(sim[off].std()),
                "offdiag_max": float(sim[off].max()),
                # The loss only distinguishes pairs across these thresholds, so
                # a matrix with no mass above them is semantically inert to it.
                "frac_pairs_margin_2.0": float((sim[off] > 0.8).mean()),
                "frac_pairs_margin_1.5": float(((sim[off] > 0.5) & (sim[off] <= 0.8)).mean()),
                "nearest_neighbours": nearest_neighbours(sim, label_names),
            }

            for tag, matrix in confusions.items():
                if tag.startswith(f"{dataset}_"):
                    rho = confusion_correlation(sim, matrix)
                    if rho is not None:
                        entry.setdefault("confusion_spearman", {})[tag] = rho

            report[dataset][arm] = entry

            print(f"\n[{dataset} / {arm}] off-diagonal similarity "
                  f"mean={entry['offdiag_mean']:.3f} std={entry['offdiag_std']:.3f} "
                  f"max={entry['offdiag_max']:.3f}")
            print(f"  pairs reaching margin 2.0: {entry['frac_pairs_margin_2.0']:.1%}, "
                  f"margin 1.5: {entry['frac_pairs_margin_1.5']:.1%}")
            if arm == "real":
                for name, neighbours in entry["nearest_neighbours"].items():
                    pretty = ", ".join(f"{n} ({s:.3f})" for n, s in neighbours)
                    print(f"    {name:<28} -> {pretty}")

            if cli.plot:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(1 + label_num * 0.7, 1 + label_num * 0.7))
                im = ax.matshow(sim, cmap="Blues", vmin=0, vmax=1)
                fig.colorbar(im)
                ax.set_xticks(range(label_num))
                ax.set_yticks(range(label_num))
                ax.set_xticklabels(label_names, rotation=90, fontsize=7)
                ax.set_yticklabels(label_names, fontsize=7)
                ax.set_title(f"{dataset} / {arm}", pad=40)
                fig.tight_layout()
                fig.savefig(os.path.join(cli.out_dir, f"sim_{dataset}_{arm}.png"), dpi=150)
                plt.close(fig)

    with open(os.path.join(cli.out_dir, "semantic_structure.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten to {cli.out_dir}")


if __name__ == "__main__":
    main()
