#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vendor bert-base-uncased into assets/ so the Bolt job needs no internet.

Compute nodes frequently reach an internal pip mirror but not huggingface.co, so
the three BERT-backed arms (real, shuffled, mismatched) would fail at runtime
after the sweep has already started. Running this once on a machine with
internet, before bolt/stage.sh, removes that failure mode.

    python bolt/prefetch_bert.py
    bash bolt/stage.sh

At runtime the job sets GESTURELENS_BERT_PATH to the vendored directory; see
contrastive/text_variants.bert_model_name().
"""

import os
import sys

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "bert-base-uncased")


def main():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("transformers is not installed here. `pip install transformers` first.")

    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading bert-base-uncased -> {DEST}")

    AutoTokenizer.from_pretrained("bert-base-uncased").save_pretrained(DEST)
    AutoModel.from_pretrained("bert-base-uncased").save_pretrained(DEST)

    # Prove it loads from disk with no network, which is what the job will do.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    AutoTokenizer.from_pretrained(DEST)
    AutoModel.from_pretrained(DEST)

    total = sum(os.path.getsize(os.path.join(DEST, f)) for f in os.listdir(DEST))
    print(f"Verified offline load. {len(os.listdir(DEST))} files, {total / 1e6:.0f}MB")
    print("Now run: bash bolt/stage.sh")


if __name__ == "__main__":
    main()
