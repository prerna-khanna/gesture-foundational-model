#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier 0 encoder-vs-classifier disentanglement grid (ICLR revision).

Answers the question "is the gain the representation or the text-guided
classifier?" by crossing every encoder with two classifiers:

    simple  -- plain cross-entropy, both losses off. This is row 4 of Table 4
               (`nucleus + sig-axis + no contrastive + no semantic`), so the
               GestureLens/simple cell should land near 0.79 (Hand SU) and
               0.73 (Hand BU). Treat that as a harness check, not as a number
               to paste -- Table 4 was produced under a different protocol.
    full    -- the text-guided classifier (classification + semantic + contrastive)

Reading the grid:
  * `simple` column varies only the encoder, so a GestureLens win there is
    evidence about the *representation*.
  * `full - simple` per encoder is how much the classifier adds. If that delta
    is similar for every encoder, the classifier is not what makes GestureLens
    special -- which is the claim Tier 0 needs to support.

Protocol: every encoder is FROZEN. Note this differs from Table 3, where the
baselines were fine-tuned end-to-end from pretrained weights (see
contrasense_imp/run_full.py and unihar_impl/create_splits_uni.py), so these
numbers are not expected to reproduce that table.

Embeddings must follow the shared contract: new_embed/embed_<encoder>_<dataset>_
<version>.npy, float32, [N, 120, D], N matching the dataset's label file.

    python run_encoder_grid.py --encoders limu_v1 unihar
    python run_encoder_grid.py --encoders unihar --classifiers simple --n_epochs 5
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from run_text_ablation import build_args, load_embedding_label, run_one

# Tier 0 datasets: Hand (SU) and Hand (BU) from Table 4. Earbud is deliberately
# out of scope for this grid.
from datasets_common import DATASETS, DEFAULT_DATASETS, activity_label_index

# Encoder tag -> the <encoder> field of the embedding filename.
ENCODERS = {
    "limu_v1": "GestureLens (nucleus masking)",
    "unihar": "UniHAR (augmentation-based SSL)",
    "contrastsense": "ContrastSense (contrastive SSL)",
    "yuan": "Yuan et al. (large-scale foundation)",
    "biopm": "BioPM (movement-element transformer)",
    "limubert": "LIMU-BERT (random masking)",
}

# Classifier mode -> (text_mode, use_contrastive)
CLASSIFIERS = {
    "simple": ("none", False),
    "full": ("real", True),
}

DEFAULT_SEEDS = (3431, 1234, 2024)

# Per-encoder hyperparameter search. The classifier's defaults were tuned on
# 72-d GestureLens embeddings; applying them unchanged to a 32-d ContrastSense or
# 1024-d harnet representation would handicap those encoders and make the grid
# unfair in GestureLens's favour. The paper already claims baselines were tuned
# "to ensure a fair comparison", so the grid has to do the same.
#
# Selection is on VALIDATION accuracy, never test.
SEARCH_GRID = [
    {"lr": lr, "hidden_dim": hidden}
    for lr in (1e-4, 3e-4, 1e-3)
    for hidden in (64, 128)
]


def to_markdown(summary, encoders, classifiers, datasets):
    header = "| Encoder | Classifier | " + " | ".join(f"{d} acc | {d} F1" for d in datasets) + " |"
    lines = [header, "|---|---|" + "---|" * (2 * len(datasets))]
    for encoder in encoders:
        for classifier in classifiers:
            cells = []
            for dataset in datasets:
                row = summary[(summary["dataset"] == dataset) &
                              (summary["encoder"] == encoder) &
                              (summary["classifier"] == classifier)]
                if row.empty:
                    cells += ["--", "--"]
                    continue
                row = row.iloc[0]
                acc_std = 0.0 if pd.isna(row["acc_std"]) else row["acc_std"]
                f1_std = 0.0 if pd.isna(row["f1_std"]) else row["f1_std"]
                cells.append(f"{row['acc_mean']:.3f} ± {acc_std:.3f}")
                cells.append(f"{row['f1_mean']:.3f} ± {f1_std:.3f}")
            lines.append(f"| {encoder} | {classifier} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(summary, encoders, datasets):
    """How much the text-guided classifier adds on top of each encoder."""
    lines = ["| Encoder | " + " | ".join(f"{d} Δacc" for d in datasets) + " |",
             "|---|" + "---|" * len(datasets)]
    for encoder in encoders:
        cells = []
        for dataset in datasets:
            def cell(mode):
                r = summary[(summary["dataset"] == dataset) & (summary["encoder"] == encoder) &
                            (summary["classifier"] == mode)]
                return None if r.empty else r.iloc[0]["acc_mean"]
            simple, full = cell("simple"), cell("full")
            cells.append("--" if simple is None or full is None else f"{full - simple:+.3f}")
        lines.append(f"| {encoder} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoders", nargs="+", default=["limu_v1", "unihar"], choices=list(ENCODERS))
    parser.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--model_version", default="v2")
    parser.add_argument("--embed_dir", default="new_embed")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--no_search", action="store_true",
                        help="Skip the per-encoder hyperparameter search and use config defaults")
    parser.add_argument("--search_epochs", type=int, default=None,
                        help="Shorter budget for search runs; final runs always use the full budget")
    parser.add_argument("--label_rate", type=float, default=0.1)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--out_dir", default=None)
    cli = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
    out_dir = cli.out_dir or os.path.join("results", "encoder_grid", stamp)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "runs.csv")
    overrides = {} if cli.n_epochs is None else {"n_epochs": cli.n_epochs}

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({**vars(cli), "encoder_descriptions": ENCODERS}, f, indent=2)

    rows, search_rows = [], []
    total = len(cli.encoders) * len(cli.classifiers) * len(cli.datasets) * len(cli.seeds)
    done = 0

    for encoder in cli.encoders:
        for dataset in cli.datasets:
            try:
                embedding, labels = load_embedding_label(
                    cli.embed_dir, encoder, dataset, cli.dataset_version)
            except FileNotFoundError as exc:
                print(f"\n[skip] {encoder} / {dataset}: {exc}")
                done += len(cli.classifiers) * len(cli.seeds)
                continue
            print(f"\n[{encoder} / {dataset}] embeddings {embedding.shape}")

            for classifier in cli.classifiers:
                text_mode, use_contrastive = CLASSIFIERS[classifier]

                # --- hyperparameter search, selected on validation accuracy ---
                best_hp = {}
                if not cli.no_search:
                    search_overrides = dict(overrides)
                    if cli.search_epochs is not None:
                        search_overrides["n_epochs"] = cli.search_epochs
                    best_vali = -1.0
                    for hp in SEARCH_GRID:
                        done += 0  # search runs are not counted against the final total
                        sargs = build_args(dataset, cli.dataset_version, cli.model_version,
                                           encoder, cli.gpu,
                                           save_model=f"search_{encoder}_{dataset}_{classifier}")
                        sargs.label_index = activity_label_index(dataset)
                        try:
                            r = run_one(sargs, embedding, labels, text_mode, cli.seeds[0],
                                        {**search_overrides, "lr": hp["lr"]},
                                        label_rate=cli.label_rate,
                                        use_contrastive=use_contrastive,
                                        hidden_dim=hp["hidden_dim"])
                        except Exception as exc:
                            print(f"  search {hp} failed: {exc}")
                            continue
                        print(f"  search {hp} -> vali_acc={r['vali_accuracy']:.4f}")
                        search_rows.append({"encoder": encoder, "dataset": dataset,
                                            "classifier": classifier, **hp,
                                            "vali_accuracy": r["vali_accuracy"]})
                        if r["vali_accuracy"] > best_vali:
                            best_vali, best_hp = r["vali_accuracy"], hp
                    if not best_hp:
                        print(f"  every search run failed for {encoder}/{dataset}/{classifier}")
                    else:
                        print(f"  chosen {best_hp} (vali_acc={best_vali:.4f})")
                    pd.DataFrame(search_rows).to_csv(
                        os.path.join(out_dir, "search.csv"), index=False)

                final_overrides = {**overrides}
                if "lr" in best_hp:
                    final_overrides["lr"] = best_hp["lr"]

                for seed in cli.seeds:
                    done += 1
                    tag = f"{encoder}_{dataset}_{classifier}_seed{seed}"
                    print(f"\n{'=' * 70}\n[{done}/{total}] {tag}\n{'=' * 70}")

                    run_args = build_args(dataset, cli.dataset_version, cli.model_version,
                                          encoder, cli.gpu, save_model=f"grid_{tag}")
                    run_args.label_index = activity_label_index(dataset)

                    try:
                        result = run_one(run_args, embedding, labels, text_mode, seed,
                                         final_overrides, label_rate=cli.label_rate,
                                         use_contrastive=use_contrastive,
                                         hidden_dim=best_hp.get("hidden_dim"))
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        rows.append({"encoder": encoder, "dataset": dataset, "classifier": classifier,
                                     "seed": seed, "accuracy": np.nan, "f1": np.nan, "error": str(exc)})
                        pd.DataFrame(rows).to_csv(rows_path, index=False)
                        continue

                    rows.append({"encoder": encoder, "dataset": dataset, "classifier": classifier,
                                 "seed": seed, "accuracy": result["accuracy"], "f1": result["f1"],
                                 "vali_accuracy": result["vali_accuracy"],
                                 "embed_dim": int(embedding.shape[-1]),
                                 "lr": final_overrides.get("lr"),
                                 "hidden_dim": result["hidden_dim"], "error": ""})
                    pd.DataFrame(rows).to_csv(rows_path, index=False)
                    print(f"\n>>> {tag}: acc={result['accuracy']:.4f} f1={result['f1']:.4f}")

    df = pd.DataFrame(rows)
    finished = df[df["accuracy"].notna()] if len(df) else df
    if not len(finished):
        print("\nNo cell completed; see runs.csv.")
        return

    summary = finished.groupby(["encoder", "dataset", "classifier"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        n_seeds=("seed", "count")).reset_index()
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    grid = to_markdown(summary, cli.encoders, cli.classifiers, cli.datasets)
    deltas = delta_table(summary, cli.encoders, cli.datasets)
    table = (grid + "\n\nGain from the text-guided classifier (full - simple):\n\n" + deltas + "\n")
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(table)

    print("\n" + table)
    failed = len(df) - len(finished)
    if failed:
        print(f"\n{failed} of {len(df)} cells failed; see {rows_path}.")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
