#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute the class-similarity matrices the semantic loss consumes.

BERT contributes exactly one thing to GestureLens's semantic loss: a [C, C]
class-similarity matrix, fixed for the whole run. Caching that matrix lets a
training job run without the 439MB bert-base-uncased checkpoint -- which matters
for Bolt, where the model would otherwise dominate the upload.

Run once on a machine that can load BERT, then stage assets/semantic_sims/:

    python precompute_semantic_sims.py
    ls assets/semantic_sims/

The consumer is SemanticLoss(sims_cache=...), wired up in run_text_ablation.run_one.
"""

import argparse
import os

import numpy as np
import torch

from config import load_dataset_label_names, load_dataset_stats
from contrastive import text_variants
from utils import get_device

from datasets_common import DATASETS, DEFAULT_DATASETS


def cache_path(out_dir, dataset, text_mode, pooling, seed):
    return os.path.join(out_dir, f"sims_{dataset}_{text_mode}_{pooling}_seed{seed}.npy")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["sony_watch", "blind_user_filtered"],
                        choices=list(DATASETS))
    parser.add_argument("--arms", nargs="+", default=["real"], choices=list(text_variants.TEXT_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[3431, 1234, 2024],
                        help="Arms whose text assignment is seeded (shuffled/mismatched) need one "
                             "matrix per seed; deterministic arms reuse the first.")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--out_dir", default=os.path.join("assets", "semantic_sims"))
    parser.add_argument("--gpu", default=None)
    cli = parser.parse_args()

    os.makedirs(cli.out_dir, exist_ok=True)
    device = get_device(cli.gpu)

    tokenizer = model = None
    if any(text_variants.anchor_mode(a) in text_variants.BERT_MODES for a in cli.arms):
        from transformers import AutoModel, AutoTokenizer
        name = text_variants.bert_model_name()
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device)

    for dataset in cli.datasets:
        cfg = load_dataset_stats(dataset, cli.dataset_version)
        _, _, descriptions = load_dataset_label_names(cfg, 0)
        if descriptions is None:
            print(f"[{dataset}] no descriptions in data_config.json, skipping")
            continue

        for arm in cli.arms:
            if arm == "none":
                continue
            # Only the seeded arms vary with seed; the rest would write identical
            # copies, so emit one matrix under every seed name for a simple
            # lookup at training time.
            seeded = arm in ("shuffled", "mismatched", "random")
            for seed in cli.seeds:
                anchors, _ = text_variants.build_class_anchors(
                    arm, descriptions, cli.pooling, device,
                    seed=seed if seeded else cli.seeds[0], tokenizer=tokenizer, model=model)
                sims = text_variants.anchors_to_similarity(anchors).cpu().numpy().astype(np.float32)
                path = cache_path(cli.out_dir, dataset, arm, cli.pooling, seed)
                np.save(path, sims)
                off = ~np.eye(sims.shape[0], dtype=bool)
                print(f"[{dataset}/{arm}/seed{seed}] {sims.shape} "
                      f"offdiag mean={sims[off].mean():.3f} max={sims[off].max():.3f} -> {path}")

    total = sum(os.path.getsize(os.path.join(cli.out_dir, f)) for f in os.listdir(cli.out_dir))
    print(f"\n{len(os.listdir(cli.out_dir))} matrices, {total/1024:.1f}KB total "
          f"(vs ~439MB for bert-base-uncased)")


if __name__ == "__main__":
    main()
