#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract frozen UniHAR embeddings in the GestureLens embedding format (Tier 0).

Tier 0 asks whether GestureLens's gains come from the representation or from the
text-guided classifier. Answering that needs every encoder's embeddings in one
common format so a single classifier can be run on top of all of them.

UniHAR is the cheapest and most load-bearing encoder to start with: it is built
on LIMU-BERT, so its checkpoint is architecturally identical to ours (it loads
straight into models.Transformer) and it differs from GestureLens essentially
only in the masking/augmentation strategy. That makes it the cleanest available
control for the nucleus-masking claim.

Output contract (shared by every Tier 0 extractor):
    embed_<tag>_<dataset>_<version>.npy, float32, shape [N, 120, D]
    N must match dataset/<dataset>/label_<version>.npy so labels stay aligned.
    T must be 120 because prepare_classifier_dataset() chunks 120 -> 6 x 20.

The UniHAR checkpoint's positional embedding is (20, 36), so it only accepts
20-step sequences. Each 120-step window is therefore split into 6 sub-windows of
20, encoded independently, and restacked to [N, 120, 36] -- which preserves the
time axis and lands exactly on the contract above.

    python extract_unihar_embeddings.py --datasets sony_watch blind_user_filtered

IMPORTANT: n_layers / n_heads / emb_norm are NOT recoverable from the checkpoint,
because LIMU-BERT-style Transformers share one block across all layers. The
defaults below mirror the LIMU-BERT base config; confirm them against UniHAR's
own repo config before treating the numbers as final. Use --n_layers/--n_heads
to override.
"""

import argparse
import os

import numpy as np
import torch

from config import PretrainModelConfig
from models import Transformer
from utils import Preprocess4Normalization, get_device

DEFAULT_CKPT = "unihar_impl/unihar_bert_fed_d1.pt"
from datasets_common import DATASETS, DEFAULT_DATASETS


def infer_shapes(state_dict):
    """Read the dimensions that the checkpoint *does* determine."""
    seq_len, hidden = state_dict["encoder.embed.pos_embed.weight"].shape
    feature_num = state_dict["encoder.embed.lin.weight"].shape[1]
    hidden_ff = state_dict["encoder.pwff.fc1.weight"].shape[0]
    return dict(seq_len=seq_len, hidden=hidden, feature_num=feature_num, hidden_ff=hidden_ff)


def load_encoder(ckpt_path, n_layers, n_heads, emb_norm, device):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    shapes = infer_shapes(state_dict)
    if shapes["hidden"] % n_heads:
        raise ValueError(f"hidden={shapes['hidden']} is not divisible by n_heads={n_heads}")

    cfg = PretrainModelConfig(n_layers=n_layers, n_heads=n_heads, emb_norm=emb_norm, **shapes)
    encoder = Transformer(cfg)

    # The checkpoint stores the transformer under an `encoder.` prefix; the
    # `decoder.*` tensors are the reconstruction head and are not needed.
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    # strict=False on purpose: nucleus_embed / sig_axis_embed are GestureLens-only
    # additions that UniHAR has no weights for. They are only read when a mask is
    # passed to forward(), and we never pass one, so they stay unused.
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    gesturelens_only = {"embed.nucleus_embed.weight", "embed.sig_axis_embed.weight"}
    unexplained = set(missing) - gesturelens_only
    if unexplained:
        raise RuntimeError(f"Checkpoint is missing weights this architecture needs: {sorted(unexplained)}")
    if unexpected:
        raise RuntimeError(f"Checkpoint has weights the architecture cannot place: {sorted(unexpected)}")

    print(f"Loaded {ckpt_path}")
    print(f"  inferred: {shapes}")
    print(f"  assumed : n_layers={n_layers} n_heads={n_heads} emb_norm={emb_norm}  <- verify against UniHAR's repo")
    print(f"  unused GestureLens-only embeddings left at init: {sorted(gesturelens_only & set(missing))}")

    return encoder.to(device).eval(), cfg


@torch.no_grad()
def extract(encoder, cfg, data, device, batch_size=256):
    """[N, 120, F] raw IMU -> [N, 120, hidden] frozen embeddings."""
    n, total_len, _ = data.shape
    if total_len % cfg.seq_len:
        raise ValueError(f"window length {total_len} is not divisible by the encoder's seq_len {cfg.seq_len}")
    chunks = total_len // cfg.seq_len

    # Same normalisation the GestureLens embedding pipeline applies, so the
    # comparison is between encoders rather than between preprocessing choices.
    normalize = Preprocess4Normalization(cfg.feature_num)
    normalized = np.stack([normalize(window) for window in data]).astype(np.float32)

    out = np.empty((n, total_len, cfg.hidden), dtype=np.float32)
    for start in range(0, n, batch_size):
        batch = normalized[start:start + batch_size]
        b = batch.shape[0]
        # [b, 120, F] -> [b * 6, 20, F] so each sub-window gets its own positions
        sub = torch.from_numpy(batch.reshape(b * chunks, cfg.seq_len, cfg.feature_num)).to(device)
        embedded = encoder(sub)
        out[start:start + b] = embedded.reshape(b, total_len, cfg.hidden).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["sony_watch", "blind_user_filtered"],
                        choices=list(DATASETS))
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--tag", default="unihar", help="Encoder tag used in the output filename")
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--out_dir", default="new_embed")
    parser.add_argument("--n_layers", type=int, default=4, help="Not stored in the checkpoint; see module docstring")
    parser.add_argument("--n_heads", type=int, default=4, help="Not stored in the checkpoint; see module docstring")
    parser.add_argument("--no_emb_norm", action="store_true")
    parser.add_argument("--gpu", default=None)
    cli = parser.parse_args()

    device = get_device(cli.gpu)
    encoder, cfg = load_encoder(cli.ckpt, cli.n_layers, cli.n_heads, not cli.no_emb_norm, device)
    os.makedirs(cli.out_dir, exist_ok=True)

    for dataset in cli.datasets:
        data_path = os.path.join("dataset", dataset, f"data_{cli.dataset_version}.npy")
        label_path = os.path.join("dataset", dataset, f"label_{cli.dataset_version}.npy")
        if not os.path.exists(data_path):
            print(f"[{dataset}] no {data_path}, skipping")
            continue

        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path)
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"[{dataset}] {data.shape[0]} windows but {labels.shape[0]} labels")

        embeddings = extract(encoder, cfg, data, device)

        if not np.isfinite(embeddings).all():
            raise RuntimeError(f"[{dataset}] embeddings contain NaN/Inf")
        if float(embeddings.std()) < 1e-6:
            raise RuntimeError(f"[{dataset}] embeddings are ~constant (std={embeddings.std():.2e}); "
                               f"the encoder almost certainly did not load correctly")

        out_path = os.path.join(cli.out_dir, f"embed_{cli.tag}_{dataset}_{cli.dataset_version}.npy")
        np.save(out_path, embeddings)
        print(f"[{dataset}] {data.shape} -> {embeddings.shape}  "
              f"mean={embeddings.mean():.4f} std={embeddings.std():.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
