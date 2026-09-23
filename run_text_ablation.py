#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier 1 text-description ablation (ICLR revision).

Answers R3's question of whether the *semantic content* of the class
descriptions matters, or whether the gain comes from merely having a
distinguishable per-class anchor. See contrastive/text_variants.py for what each
arm changes and why.

The training loop here mirrors classify_embeddings() in
classifier_with_contrastive.py exactly -- same model, same optimizer, same
augmentation, same trainer -- and adds only (a) the text arm, (b) a seed
override so each cell can be reported with a spread rather than a single number,
and (c) a configurable embedding folder.

The `real` arm is implemented but excluded from the default sweep because those
numbers already exist from the submitted paper. Note that the paper's cells were
produced with per-dataset model-selection rules (see getting_main_res.txt),
whereas every arm here uses the single rule currently in train.py:377. Arms are
therefore comparable to each other; comparing them against the published `real`
column carries that caveat. Run with `--arms real ...` to get a like-for-like
`real` cell under the unified rule.

Usage
-----
    python run_text_ablation.py                       # full default sweep
    python run_text_ablation.py --datasets sighted_user --seeds 3431
    python run_text_ablation.py --arms onehot none --n_epochs 50   # smoke test
"""

import argparse
import copy
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import train as train_module
from config import create_io_config, load_dataset_label_names, load_dataset_stats, load_model_config
from contrastive.augmenter import GestureAugmenter
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.models import ContrastiveTransformerClassifier
from contrastive.text_variants import DEFAULT_ARMS, TEXT_MODES
from statistic import stat_acc_f1, stat_results
from utils import IMUDataset, TrainConfig, get_device, prepare_classifier_dataset, set_seeds

# Datasets this sweep can run over. Kept in sync with run_encoder_grid.DATASETS
# so Tier 0 and Tier 1 can report over the same data; the default pair matches
# the Tier 0 grid (Hand SU, Hand BU).
from datasets_common import DATASETS, DEFAULT_DATASETS, activity_label_index

DEFAULT_SEEDS = (3431, 1234, 2024)

TARGET = "classifier_text_ablation"


def load_embedding_label(embed_dir, model_file, dataset, dataset_version):
    """Same as embedding.load_embedding_label but with a configurable folder,
    since our frozen embeddings live in new_embed/ rather than embed/."""
    embed_name = f"embed_{model_file}_{dataset}_{dataset_version}.npy"
    embed_path = os.path.join(embed_dir, embed_name)
    if not os.path.exists(embed_path):
        raise FileNotFoundError(
            f"No embeddings at {embed_path}. Generate them with:\n"
            f"    python embedding.py v1 {dataset} {dataset_version} -f {model_file}")
    embed = np.load(embed_path).astype(np.float32)
    labels = np.load(os.path.join("dataset", dataset, f"label_{dataset_version}.npy")).astype(np.float32)
    return embed, labels


def build_args(dataset, dataset_version, model_version, model_file, gpu, save_model):
    """Assemble the same args namespace handle_argv() would build, without going
    through argparse, so one process can sweep many datasets."""
    args = argparse.Namespace(
        dataset=dataset, dataset_version=dataset_version, model_version=model_version,
        model_file=model_file, gpu=gpu, save_model=save_model,
        train_cfg="./config/train.json", label_index=0)

    model_cfg = load_model_config(TARGET, "gru", model_version)
    if model_cfg is None:
        sys.exit(f"Unable to find model config gru_{model_version} in config/classifier.json")
    args.model_cfg = model_cfg

    dataset_cfg = load_dataset_stats(dataset, dataset_version)
    if dataset_cfg is None:
        sys.exit(f"Unable to find dataset config for {dataset}_{dataset_version}")
    args.dataset_cfg = dataset_cfg

    return create_io_config(args, dataset, dataset_version, pretrain_model=None, target=TARGET)


def run_one(args, embedding, labels, text_mode, seed, train_cfg_overrides,
            training_rate=0.8, label_rate=0.1, balance=True, use_contrastive=True,
            hidden_dim=None):
    """Train one cell and return its test accuracy and F1.

    `text_mode='none'` with `use_contrastive=False` is the plain cross-entropy
    classifier -- the "simple classifier" column of the Tier 0 encoder grid, and
    row 4 of Table 4 in the paper.
    """
    train_cfg = TrainConfig.from_json(args.train_cfg)
    train_cfg = train_cfg._replace(seed=seed, **train_cfg_overrides)
    model_cfg = args.model_cfg
    dataset_cfg = args.dataset_cfg

    set_seeds(seed)

    label_names, label_num, descriptions = load_dataset_label_names(dataset_cfg, args.label_index)
    device = get_device(args.gpu)
    hidden_dim = hidden_dim or getattr(model_cfg, "hidden_dim", 128)

    if descriptions is None:
        raise ValueError(
            f"{args.dataset} has no descriptions in dataset/data_config.json; the "
            f"text ablation is meaningless without them.")

    # Encoders differ in how much time survives: most emit [N, 120, D] and get
    # chunked into 6 windows of model_cfg.seq_len, but harnet/Yuan collapses the
    # time axis entirely and emits [N, 1, D]. Chunk by whichever is smaller, and
    # trim the label array to match -- every timestep of a window carries the
    # same gesture label, so slicing is lossless.
    seq_len = embedding.shape[1]
    merge = min(model_cfg.seq_len, seq_len)
    if labels.shape[1] != seq_len:
        labels = labels[:, :seq_len, :]

    data_train, label_train, data_vali, label_vali, data_test, label_test = \
        prepare_classifier_dataset(embedding, labels, label_index=args.label_index,
                                   training_rate=training_rate, label_rate=label_rate,
                                   merge=merge, seed=seed, balance=balance)

    augmenter = GestureAugmenter()
    data_loader_train = DataLoader(IMUDataset(data_train, label_train, pipeline=[augmenter.augment]),
                                   shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(IMUDataset(data_vali, label_vali),
                                  shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(IMUDataset(data_test, label_test),
                                  shuffle=False, batch_size=train_cfg.batch_size)

    model = ContrastiveTransformerClassifier(
        input_dim=data_train.shape[-1], hidden_dim=hidden_dim, num_classes=label_num).to(device)

    # Use a precomputed [C, C] similarity matrix when one has been cached, so the
    # job does not need bert-base-uncased on disk. Falls back to BERT silently.
    pooling = getattr(train_cfg, "pooling", "cls")
    sims_cache = os.path.join("assets", "semantic_sims",
                              f"sims_{args.dataset}_{text_mode}_{pooling}_seed{seed}.npy")

    criterion = ContrastiveCombinedLoss(
        label_names=label_names, descriptions=descriptions,
        pooling=pooling, device=device,
        hidden_dim=hidden_dim, text_mode=text_mode, text_seed=seed,
        use_contrastive=use_contrastive, sims_cache=sims_cache)

    # model.parameters() only, matching classifier_with_contrastive.py -- the
    # criterion's semantic_projection stays a frozen random projection in every
    # arm, exactly as in the submitted results.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train_module.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch, current_epoch=0):
        inputs, label = batch
        logits, features, projected = model(inputs, True)
        return criterion(logits=logits, features=features, projected=projected,
                         labels=label, epoch=current_epoch)

    def func_forward(model, batch):
        inputs, label = batch
        return model(inputs, False), label

    def func_evaluate(label, predicts):
        return stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())

    best_stat = trainer.train(func_loss, func_forward, func_evaluate,
                              data_loader_train, data_loader_test, data_loader_vali)

    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)

    anchor_info = criterion.semantic_criterion.anchor_info if criterion.semantic_criterion else None
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        # Selection metric. Never pick hyperparameters on the test columns above.
        "vali_accuracy": float(best_stat[1]) if best_stat else float("nan"),
        "vali_f1": float(best_stat[4]) if best_stat else float("nan"),
        "hidden_dim": int(hidden_dim),
        "n_train": int(label_train.shape[0]),
        "n_test": int(label_test.shape[0]),
        "confusion_matrix": matrix.tolist(),
        "anchor_info": anchor_info,
    }


def summarise(df):
    """Mean +/- std over seeds, in the accuracy/F1/variance format of the paper."""
    grouped = df.groupby(["dataset", "arm"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), n_seeds=("seed", "count"))
    return grouped.reset_index()


def to_markdown(summary, arms, datasets):
    """One row per arm, one column pair per dataset -- the shape the paper table
    needs, so it can be pasted next to the existing five-template table."""
    lines = ["| Arm | " + " | ".join(f"{d} acc | {d} F1" for d in datasets) + " |",
             "|---|" + "---|" * (2 * len(datasets))]
    for arm in arms:
        cells = []
        for dataset in datasets:
            row = summary[(summary["dataset"] == dataset) & (summary["arm"] == arm)]
            if row.empty:
                cells += ["--", "--"]
                continue
            row = row.iloc[0]
            std_acc = 0.0 if pd.isna(row["acc_std"]) else row["acc_std"]
            std_f1 = 0.0 if pd.isna(row["f1_std"]) else row["f1_std"]
            cells.append(f"{row['acc_mean']:.3f} ± {std_acc:.3f}")
            cells.append(f"{row['f1_mean']:.3f} ± {std_f1:.3f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DATASETS))
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(TEXT_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--model_version", default="v2", help="Classifier config version (gru_<v>)")
    parser.add_argument("--model_file", default="limu_v1", help="Pretrained encoder used for the embeddings")
    parser.add_argument("--embed_dir", default="new_embed")
    parser.add_argument("--n_epochs", type=int, default=None, help="Override config/train.json")
    parser.add_argument("--label_rate", type=float, default=0.1)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--out_dir", default=None)
    args_cli = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
    out_dir = args_cli.out_dir or os.path.join("results", "text_ablation", stamp)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "runs.csv")

    overrides = {} if args_cli.n_epochs is None else {"n_epochs": args_cli.n_epochs}

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2)

    rows, details = [], []
    total = len(args_cli.datasets) * len(args_cli.arms) * len(args_cli.seeds)
    done = 0

    for dataset in args_cli.datasets:
        embedding, labels = load_embedding_label(
            args_cli.embed_dir, args_cli.model_file, dataset, args_cli.dataset_version)
        print(f"\n[{dataset}] embeddings {embedding.shape}, labels {labels.shape}")

        for arm in args_cli.arms:
            for seed in args_cli.seeds:
                done += 1
                tag = f"{dataset}_{arm}_seed{seed}"
                print(f"\n{'=' * 70}\n[{done}/{total}] {tag}\n{'=' * 70}")

                run_args = build_args(dataset, args_cli.dataset_version, args_cli.model_version,
                                      args_cli.model_file, args_cli.gpu,
                                      save_model=f"text_ablation_{tag}")
                run_args.label_index = activity_label_index(dataset)

                try:
                    result = run_one(run_args, embedding, labels, arm, seed, overrides,
                                     label_rate=args_cli.label_rate)
                except Exception as exc:  # keep the sweep alive; record the gap
                    import traceback
                    traceback.print_exc()
                    rows.append({"dataset": dataset, "arm": arm, "seed": seed,
                                 "accuracy": np.nan, "f1": np.nan, "error": str(exc)})
                    pd.DataFrame(rows).to_csv(rows_path, index=False)
                    continue

                rows.append({"dataset": dataset, "arm": arm, "seed": seed,
                             "accuracy": result["accuracy"], "f1": result["f1"],
                             "n_train": result["n_train"], "n_test": result["n_test"],
                             "error": ""})
                details.append({"tag": tag, **{k: result[k] for k in
                                               ("confusion_matrix", "anchor_info")}})

                # Written after every cell so a crashed sweep is still usable.
                pd.DataFrame(rows).to_csv(rows_path, index=False)
                print(f"\n>>> {tag}: acc={result['accuracy']:.4f} f1={result['f1']:.4f}")

    df = pd.DataFrame(rows)
    finished = df[df["accuracy"].notna()]
    if finished.empty:
        print("\nNo cell completed; see runs.csv for the errors.")
        return

    summary = summarise(finished)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "details.json"), "w") as f:
        json.dump(details, f, indent=2)

    table = to_markdown(summary, args_cli.arms, args_cli.datasets)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(table + "\n")

    print("\n" + table)
    failed = len(df) - len(finished)
    if failed:
        print(f"\n{failed} of {len(df)} cells failed; see {rows_path}.")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
