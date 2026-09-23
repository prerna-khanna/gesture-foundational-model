#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract frozen ContrastSense embeddings in the GestureLens format (Tier 0).

Uses ContrastSense's OWN encoder class, imported from a clone of
https://github.com/MaginaDai/ContrastSense-Public, rather than a reconstruction
from the checkpoint's tensor shapes. That matters: the layer shapes alone do not
determine padding, activation order, or the residual wiring, and guessing wrong
produces embeddings that look plausible but carry no signal.

    git clone --depth 1 https://github.com/MaginaDai/ContrastSense-Public.git
    python extract_contrastsense_embeddings.py --repo ~/Desktop/ContrastSense-Public

The checkpoint is a MoCo pair; we take `encoder_q` (the query encoder) and drop
the queues, `encoder_k`, and the projection head -- the projector maps a
flattened 6400-d feature, which is specific to their 200-step input, and we want
per-timestep features anyway.

ContrastSense_encoder returns [B, T, 32] for a [B, 1, T, 6] input, so the output
lands directly on the shared Tier 0 contract: [N, 120, 32], N matching the
dataset's label file.

NOTE on the repo's own loader: contrasense_imp/run_full.py renames `encoder_q.*`
and loads it into a BiLSTM with strict=False, where zero keys match -- so that
script trains a randomly-initialised model. This extractor asserts a non-trivial
key overlap so the same failure cannot happen silently here.
"""

import argparse
import os
import sys
import tempfile
import zipfile

import numpy as np
import torch

from utils import Preprocess4Normalization, get_device

from datasets_common import DATASETS, DEFAULT_DATASETS
DEFAULT_CKPT = "contrasense_imp/HHAR/model_best.pth"


def load_state_dict(ckpt_path):
    """The checkpoint in this repo was committed as an *unzipped* torch archive
    (a directory of byteorder/data/data.pkl/version), so torch.load rejects it.
    Re-zip in a temp file when that is what we are handed."""
    if os.path.isdir(ckpt_path):
        tmp = tempfile.mktemp(suffix=".pt")
        with zipfile.ZipFile(tmp, "w") as z:
            for root, _, files in os.walk(ckpt_path):
                for f in files:
                    full = os.path.join(root, f)
                    z.write(full, os.path.join("model_best", os.path.relpath(full, ckpt_path)))
        ckpt_path = tmp

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw


def load_encoder(repo, ckpt_path, dims, device):
    repo = os.path.expanduser(repo)
    if not os.path.isdir(repo):
        sys.exit(f"ContrastSense repo not found at {repo}. Clone it first (see module docstring).")

    # Their ContrastSense.py does `import utils`, and so do we. Ours is already in
    # sys.modules, so a naive import silently binds their code to our utils and
    # fails. Shadow the colliding top-level modules for the duration of the
    # import, then put ours back.
    collisions = ("utils", "models", "config", "data_loader")
    stashed = {name: sys.modules.pop(name) for name in collisions if name in sys.modules}
    sys.path.insert(0, repo)

    # Their data_loader/imu_transforms.py has a stray, never-used
    # `from os import SCHED_RESET_ON_FORK` at module scope. That constant is
    # Linux-only, so the import chain to ContrastSense_encoder dies on macOS.
    # Defining it is safe precisely because nothing reads it.
    shimmed_sched = not hasattr(os, "SCHED_RESET_ON_FORK")
    if shimmed_sched:
        os.SCHED_RESET_ON_FORK = 0

    try:
        from ContrastSense import ContrastSense_encoder
    except ImportError as exc:
        sys.exit(f"Could not import ContrastSense_encoder from {repo}: {exc}")
    finally:
        sys.path.remove(repo)
        if shimmed_sched:
            del os.SCHED_RESET_ON_FORK
        for name in [n for n in collisions if n in sys.modules]:
            del sys.modules[name]
        sys.modules.update(stashed)

    # Dropout is active in their forward(); eval() disables it, which is what we
    # want for deterministic frozen features.
    encoder = ContrastSense_encoder(dims=dims)

    state_dict = load_state_dict(ckpt_path)
    prefix = "encoder_q.encoder."
    encoder_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not encoder_state:
        sys.exit(f"No '{prefix}*' tensors in {ckpt_path}; is this a ContrastSense checkpoint?")

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    matched = len(encoder_state) - len(unexpected)
    if matched == 0:
        sys.exit("ZERO pretrained tensors matched the encoder -- this is the exact silent "
                 "failure in contrasense_imp/run_full.py. Refusing to emit random embeddings.")
    if unexpected:
        raise RuntimeError(f"Checkpoint tensors the encoder cannot place: {sorted(unexpected)}")
    if missing:
        raise RuntimeError(f"Encoder weights absent from the checkpoint: {sorted(missing)}")

    print(f"Loaded {ckpt_path}")
    print(f"  matched {matched}/{len(encoder_state)} pretrained encoder tensors (dims={dims})")
    return encoder.to(device).eval()


@torch.no_grad()
def extract(encoder, data, device, feature_num=6, batch_size=128):
    """[N, 120, 6] raw IMU -> [N, 120, dims] frozen features."""
    normalize = Preprocess4Normalization(feature_num)
    normalized = np.stack([normalize(w) for w in data]).astype(np.float32)

    out = []
    for start in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
        # ContrastSense_encoder expects [B, 1, T, 6]
        feats = encoder(batch.unsqueeze(1))
        out.append(feats.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="~/Desktop/ContrastSense-Public")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--datasets", nargs="+", default=["sony_watch", "blind_user_filtered"],
                        choices=list(DATASETS))
    parser.add_argument("--dims", type=int, default=32, help="Must match the checkpoint's channel width")
    parser.add_argument("--tag", default="contrastsense")
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--out_dir", default="new_embed")
    parser.add_argument("--gpu", default=None)
    cli = parser.parse_args()

    device = get_device(cli.gpu)
    encoder = load_encoder(cli.repo, cli.ckpt, cli.dims, device)
    os.makedirs(cli.out_dir, exist_ok=True)

    for dataset in cli.datasets:
        data_path = os.path.join("dataset", dataset, f"data_{cli.dataset_version}.npy")
        if not os.path.exists(data_path):
            print(f"[{dataset}] no {data_path}, skipping")
            continue
        data = np.load(data_path).astype(np.float32)
        labels = np.load(os.path.join("dataset", dataset, f"label_{cli.dataset_version}.npy"))
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"[{dataset}] {data.shape[0]} windows but {labels.shape[0]} labels")

        embeddings = extract(encoder, data, device, feature_num=data.shape[-1])

        if embeddings.shape[:2] != data.shape[:2]:
            raise RuntimeError(f"[{dataset}] expected [N, {data.shape[1]}, D], got {embeddings.shape}; "
                               f"the contract requires the time axis to stay at {data.shape[1]}")
        if not np.isfinite(embeddings).all():
            raise RuntimeError(f"[{dataset}] embeddings contain NaN/Inf")
        if float(embeddings.std()) < 1e-6:
            raise RuntimeError(f"[{dataset}] embeddings are ~constant (std={embeddings.std():.2e})")

        out_path = os.path.join(cli.out_dir, f"embed_{cli.tag}_{dataset}_{cli.dataset_version}.npy")
        np.save(out_path, embeddings)
        print(f"[{dataset}] {data.shape} -> {embeddings.shape}  "
              f"mean={embeddings.mean():.4f} std={embeddings.std():.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
