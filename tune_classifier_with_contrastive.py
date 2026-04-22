#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Small in-code hyperparameter tuning setup for classifier_with_contrastive.

Edit the constants below and run the file directly.
"""

from types import SimpleNamespace

from config import create_io_config, load_model_config, load_dataset_stats
from embedding import load_embedding_label
from classifier_with_contrastive import run_grid_search


MODEL_VERSION = "v2"
DATASET = "UTD_MHAD"
DATASET_VERSION = "20_120"
MODEL_FILE = "limu_v1"
TARGET = "classifier_contrastive_gru"
PREFIX = "gru"
LABEL_INDEX = 1

TRAINING_RATE = 0.8
LABEL_RATE = 0.2
BALANCE = True

# Keep this grid small enough to run locally, but focused on the knobs that
# matter most for the overfitting pattern we saw.
PARAM_GRID = {
    "lr": [1e-4, 3e-4],
    "weight_decay": [1e-4, 5e-4],
    "lambda2": [0.001, 0.005],
    "grad_clip_norm": [0.5, 1.0],
}


def main():
    model_cfg = load_model_config(TARGET, PREFIX, MODEL_VERSION)
    if model_cfg is None:
        raise SystemExit("Unable to find corresponding model config!")

    dataset_cfg = load_dataset_stats(DATASET, DATASET_VERSION)
    if dataset_cfg is None:
        raise SystemExit("Unable to find corresponding dataset config!")

    args = SimpleNamespace(
        model_version=MODEL_VERSION,
        dataset=DATASET,
        dataset_version=DATASET_VERSION,
        gpu=None,
        model_file=MODEL_FILE,
        train_cfg="./config/train.json",
        mask_cfg="./config/mask.json",
        label_index=LABEL_INDEX,
        save_model=f"tune_{DATASET.lower()}_{DATASET_VERSION}",
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
    )
    args = create_io_config(args, DATASET, DATASET_VERSION, pretrain_model=args.model_file, target=TARGET)

    embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
    best_params, best_acc, best_f1, _ = run_grid_search(
        args,
        embedding,
        labels,
        LABEL_INDEX,
        TRAINING_RATE,
        LABEL_RATE,
        balance=BALANCE,
        method=PREFIX,
        param_grid=PARAM_GRID,
    )

    print("\nBest tuning result")
    print(f"Best params: {best_params}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()