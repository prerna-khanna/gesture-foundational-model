#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract frozen Yuan et al. (harnet) embeddings for the Tier 0 grid.

Yuan et al., "Self-supervised learning for human activity recognition using
700,000 person-days of wearable data" -- the large-scale foundation-model arm of
the grid. Uses the official Resnet from a clone of
https://github.com/OxWearables/ssl-wearables plus weights vendored to
assets/harnet10.pt.

    git clone --depth 1 https://github.com/OxWearables/ssl-wearables.git
    python -c "import torch; m=torch.hub.load('OxWearables/ssl-wearables','harnet10',
               class_num=5, pretrained=True); torch.save(m.state_dict(),'assets/harnet10.pt')"
    python extract_yuan_embeddings.py --repo ~/Desktop/ssl-wearables

THREE CAVEATS THAT BELONG IN THE PAPER
--------------------------------------
1. Accelerometer only. harnet takes N x 3 x T; the gyroscope is discarded, while
   every other encoder in the grid sees all 6 axes.
2. Resampled 20 Hz -> 30 Hz. harnet assumes 30 Hz.
3. The time axis collapses. harnet10's downsampling factors multiply to
   2*2*5*5*3 = 300 against a 300-sample input, so the feature map is
   [N, 1024, 1] -- one vector per window, not a sequence. Every other encoder
   emits [N, 120, D]. We therefore emit [N, 1, 1024] and let the classifier
   consume a length-1 sequence (option 3). This is a real asymmetry: Yuan's arm
   gets no temporal structure for the classifier's attention to work with.

Window-length handling: our windows are 120 samples at 20 Hz (6 s), harnet10
wants 300 at 30 Hz (10 s). `--resample_mode pad` (default) resamples to 30 Hz
(180 samples) and centre-pads with edge values to 300, preserving true gesture
speed. `--resample_mode stretch` interpolates 120 -> 300 directly, preserving
shape but slowing the gesture by 1.67x. Padding is the safer default because
harnet was trained on real-time-scale motion.
"""

import argparse
import os
import sys

import numpy as np
import torch

from utils import get_device

from datasets_common import DATASETS, DEFAULT_DATASETS
# (model name, samples expected at 30 Hz)
HARNETS = {"harnet5": 150, "harnet10": 300, "harnet30": 900}


def load_encoder(repo, weights, variant, device):
    repo = os.path.expanduser(repo)
    if not os.path.isdir(repo):
        sys.exit(f"ssl-wearables repo not found at {repo}. Clone it first (see module docstring).")
    weights = os.path.expanduser(weights)
    if not os.path.isfile(weights):
        sys.exit(f"No vendored weights at {weights}. Download them on a machine with "
                 f"internet access (see module docstring).")

    collisions = ("utils", "models", "config", "data")
    stashed = {name: sys.modules.pop(name) for name in collisions if name in sys.modules}
    sys.path.insert(0, repo)
    try:
        from sslearning.models.accNet import Resnet
    except ImportError as exc:
        sys.exit(f"Could not import Resnet from {repo}: {exc}")
    finally:
        sys.path.remove(repo)
        for name in [n for n in collisions if n in sys.modules]:
            del sys.modules[name]
        sys.modules.update(stashed)

    epoch_len = {"harnet5": 5, "harnet10": 10, "harnet30": 30}[variant]
    model = Resnet(output_size=5, n_channels=3, is_eva=True, epoch_len=epoch_len)

    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    # Drop the downstream classifier head; we only want the frozen feature extractor.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    matched = len(state_dict) - len(unexpected)
    if matched == 0:
        sys.exit("ZERO pretrained tensors matched the harnet feature extractor; refusing "
                 "to emit random embeddings.")
    feature_missing = [k for k in missing if not k.startswith("classifier")]
    if feature_missing:
        raise RuntimeError(f"Feature-extractor weights absent from the checkpoint: {feature_missing}")

    print(f"Loaded {weights} into {variant}")
    print(f"  matched {matched}/{len(state_dict)} pretrained tensors "
          f"(classifier head intentionally dropped)")
    return model.feature_extractor.to(device).eval()


def resample(data, target_len, mode, src_hz=20, dst_hz=30):
    """[N, 120, 3] at src_hz -> [N, target_len, 3] at dst_hz."""
    n, src_len, ch = data.shape
    if mode == "stretch":
        resampled_len = target_len
    else:
        resampled_len = int(round(src_len * dst_hz / src_hz))
        if resampled_len > target_len:
            raise ValueError(f"window is {resampled_len} samples at {dst_hz}Hz but the model "
                             f"only accepts {target_len}; use --resample_mode stretch")

    src_grid = np.linspace(0.0, 1.0, src_len)
    dst_grid = np.linspace(0.0, 1.0, resampled_len)
    out = np.empty((n, resampled_len, ch), dtype=np.float32)
    for i in range(n):
        for c in range(ch):
            out[i, :, c] = np.interp(dst_grid, src_grid, data[i, :, c])

    if resampled_len == target_len:
        return out
    # Centre-pad with edge values so the gesture keeps its true speed and the
    # padding reads as "holding still", not as a discontinuity to zero.
    total = target_len - resampled_len
    before, after = total // 2, total - total // 2
    return np.pad(out, ((0, 0), (before, after), (0, 0)), mode="edge")


@torch.no_grad()
def extract(feature_extractor, data, device, target_len, mode, batch_size=64):
    """[N, 120, 6] raw IMU -> [N, 1, 1024] frozen features."""
    # harnet is accelerometer-only; this repo stores acc in channels 0:3.
    acc = data[:, :, :3]
    # Match the training convention: harnet expects acceleration in g, and this
    # repo's raw arrays are in m/s^2 (see Preprocess4Normalization's 9.8).
    acc = resample(acc, target_len, mode) / 9.8

    out = []
    for start in range(0, len(acc), batch_size):
        batch = torch.from_numpy(acc[start:start + batch_size]).to(device)
        feats = feature_extractor(batch.permute(0, 2, 1))       # [B, 3, T] -> [B, C, T']
        out.append(feats.flatten(start_dim=1).unsqueeze(1).cpu().numpy())   # -> [B, 1, C*T']
    return np.concatenate(out).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="~/Desktop/ssl-wearables")
    parser.add_argument("--weights", default="assets/harnet10.pt")
    parser.add_argument("--variant", default="harnet10", choices=list(HARNETS))
    parser.add_argument("--datasets", nargs="+", default=["sony_watch", "blind_user_filtered"],
                        choices=list(DATASETS))
    parser.add_argument("--resample_mode", default="pad", choices=["pad", "stretch"])
    parser.add_argument("--tag", default="yuan")
    parser.add_argument("--dataset_version", default="20_120")
    parser.add_argument("--out_dir", default="new_embed")
    parser.add_argument("--gpu", default=None)
    cli = parser.parse_args()

    device = get_device(cli.gpu)
    feature_extractor = load_encoder(cli.repo, cli.weights, cli.variant, device)
    target_len = HARNETS[cli.variant]
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

        embeddings = extract(feature_extractor, data, device, target_len, cli.resample_mode)

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
