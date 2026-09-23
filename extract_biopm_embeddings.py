#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract frozen BioPM embeddings for the Tier 0 grid.

BioPM (ICML 2026) is a movement-element transformer for 3-axis accelerometer
data: https://github.com/Prithvitarale/biopm. It ships three pretrained
checkpoints (25/50/75% masking rate), so no pretraining is needed.

    git clone --depth 1 https://github.com/Prithvitarale/biopm.git
    python extract_biopm_embeddings.py --repo ~/Desktop/biopm

Pipeline, mirroring their scripts/preprocess_mhealth.py but per-window rather
than over a continuous stream (our data is already cut into 120-sample windows,
so their sliding-window loop in windowize_and_extract does not apply):

    1. acc[:, :, :3]                      -- BioPM is accelerometer-only
    2. resample 20 Hz -> target_fs (30)
    3. 0.5-12 Hz Butterworth bandpass     -- body acceleration, drives ME detection
    4. 0.5 Hz Butterworth lowpass         -- gravity signal
    5. detect_zero_crossings + pack_window per window -> (pad_size, 37)
    6. encode_window -> (B, 1023) per-axis pooled features

THREE THINGS THAT BELONG IN THE PAPER
-------------------------------------
1. Accelerometer only; the gyroscope is discarded, as with Yuan et al.

2. OUR WINDOWS ARE MOSTLY ZERO PADDING. The dataset arrays follow the paper's
   "if the signal length is <120, append zeros at the start and end", and the
   result is severe: 83.6% of every sony_watch window and 42.3% of every
   blind_user_filtered window is exactly zero on all three axes. Only ~1 s of
   each 6 s sony_watch window carries signal.

   That breaks BioPM specifically, because its movement elements come from
   spline zero-crossings of the band-passed signal -- a mostly-flat-zero window
   produces degenerate crossings. So by default we CROP each window to its
   non-zero extent before resampling (`--no_crop_padding` to disable). BioPM
   NaN-pads movement elements to pad_size anyway, so variable input length is
   fine; this feeds it a real continuous signal instead of padding.

3. GRAVITY IS EXCLUDED BY DEFAULT. fuse_window_feature concatenates
   [acc_feat (384), gravity_feat (639)]. Measured over non-padding samples both
   datasets are in m/s^2 (median |acc| 10.33 and 9.81, i.e. ~1 g), so the unit
   convention is fine and we divide by 9.8. Gravity is still excluded because a
   6th-order 0.5 Hz low-pass needs far more than ~1 s of signal to estimate a DC
   component, so the gravity half -- 62% of the feature width -- would be
   dominated by filter transients. `--gravity include` restores all 1023 dims.

4. The time axis collapses. BioPM pools over movement elements, so there is one
   vector per window, not a sequence. Like harnet/Yuan we emit [N, 1, D] and let
   the classifier consume a length-1 sequence.

Also note our windows are 6 s where BioPM was trained on 10 s: a distribution
shift, not a shape error.
"""

import argparse
import os
import sys

import numpy as np
import torch

from datasets_common import DATASETS, DEFAULT_DATASETS
from utils import get_device

NORM = 32  # samples per movement element; the column split in load_preprocessed_h5


def import_biopm(repo):
    """Import the biopm package from a clone, shielding our same-named modules."""
    repo = os.path.expanduser(repo)
    if not os.path.isdir(repo):
        sys.exit(f"biopm repo not found at {repo}. Clone it first (see module docstring).")

    collisions = ("utils", "models", "config", "data", "features", "preprocessing",
                  "evaluation", "inference")
    stashed = {n: sys.modules.pop(n) for n in collisions if n in sys.modules}
    sys.path.insert(0, repo)
    try:
        from biopm.inference import load_pretrained, encode_window
        from biopm.preprocessing import (PreprocessConfig, bandpass_filter,
                                         lowpass_filter, detect_zero_crossings,
                                         pack_window)
    except ImportError as exc:
        sys.exit(f"Could not import biopm from {repo}: {exc}\n"
                 f"Its requirements are torch, numpy, scipy, pandas, h5py, "
                 f"scikit-learn, tqdm.")
    finally:
        sys.path.remove(repo)
        for n in [n for n in collisions if n in sys.modules]:
            del sys.modules[n]
        sys.modules.update(stashed)

    return dict(load_pretrained=load_pretrained, encode_window=encode_window,
                PreprocessConfig=PreprocessConfig, bandpass_filter=bandpass_filter,
                lowpass_filter=lowpass_filter,
                detect_zero_crossings=detect_zero_crossings, pack_window=pack_window)


def resample(acc, src_len, dst_len):
    """[N, src_len, 3] -> [N, dst_len, 3] by linear interpolation on a unit grid."""
    if src_len == dst_len:
        return acc
    src = np.linspace(0.0, 1.0, src_len)
    dst = np.linspace(0.0, 1.0, dst_len)
    out = np.empty((acc.shape[0], dst_len, acc.shape[2]), dtype=np.float32)
    for i in range(acc.shape[0]):
        for c in range(acc.shape[2]):
            out[i, :, c] = np.interp(dst, src, acc[i, :, c])
    return out


GRAVITY_MS2 = 9.80665   # both datasets measure ~1 g over non-padding samples


def crop_to_signal(window, min_samples=16):
    """Trim leading/trailing all-zero padding from one [T, 3] window.

    Returns None when too little real signal survives for zero-crossing
    detection to mean anything.
    """
    nonzero = np.abs(window).sum(axis=1) > 0
    if not nonzero.any():
        return None
    first, last = np.argmax(nonzero), len(nonzero) - 1 - np.argmax(nonzero[::-1])
    cropped = window[first:last + 1]
    return cropped if len(cropped) >= min_samples else None


def build_window_features(api, raw_windows, cfg, crop_padding, verbose=True):
    """Per-window movement-element extraction.

    Each window is cropped to its non-zero extent (optional), resampled to
    cfg.target_fs at its true duration, band-passed for ME detection and
    low-passed for gravity. Window lengths therefore differ, which is why this
    loops rather than filtering the whole array at once -- BioPM pads MEs to
    cfg.pad_size, so the encoder sees a fixed shape regardless.

    Returns (patches, pos, extra, gravity, n_empty, kept_lengths).
    """
    n, t_src, _ = raw_windows.shape
    packed = np.empty((n, cfg.pad_size, cfg.feature_columns), dtype=np.float32)
    # Gravity is kept at the full resampled window length so the tensor is
    # rectangular; it is excluded from the features by default anyway.
    t_dst = int(round(t_src / 20.0 * cfg.target_fs))
    gravity = np.zeros((n, t_dst, 3), dtype=np.float32)
    n_empty = 0
    kept_lengths = []

    for i in range(n):
        win = raw_windows[i]
        if crop_padding:
            cropped = crop_to_signal(win)
            if cropped is None:
                packed[i] = np.full((cfg.pad_size, cfg.feature_columns), np.nan, np.float32)
                n_empty += 1
                continue
            win = cropped
        kept_lengths.append(len(win))

        # Resample this window's true duration to target_fs.
        dur_s = len(win) / 20.0
        dst_len = max(cfg.normalize_size, int(round(dur_s * cfg.target_fs)))
        w = resample(win[None, ...], len(win), dst_len)[0]

        try:
            bp = api["bandpass_filter"](w, cfg.low_f1, cfg.high_f1, cfg.target_fs, cfg.order)
            lp = api["lowpass_filter"](w, cfg.low_f1, cfg.target_fs, cfg.order)
            time_index = np.arange(dst_len, dtype=np.float64) / cfg.target_fs
            me_norm, me_info, pos, _, _ = api["detect_zero_crossings"](bp, time_index, cfg)
        except Exception:
            me_norm, me_info, pos, lp = None, None, None, None

        if me_norm is None or len(me_norm) == 0:
            # pack_window emits an all-NaN block, which BioPM's valid_mask reads
            # as "no movement elements". Counted and reported rather than dropped,
            # so window/label alignment is preserved.
            packed[i] = np.full((cfg.pad_size, cfg.feature_columns), np.nan, np.float32)
            n_empty += 1
        else:
            packed[i] = api["pack_window"](me_norm, me_info, pos, cfg)
            m = min(t_dst, len(lp))
            gravity[i, :m] = lp[:m]

        if verbose and (i + 1) % 500 == 0:
            print(f"    segmented {i + 1}/{n} windows", flush=True)

    return (packed[:, :, :NORM], packed[:, :, NORM], packed[:, :, NORM + 1:],
            gravity, n_empty, np.asarray(kept_lengths))


@torch.no_grad()
def encode(api, model, patches, pos, extra, grav, per_axis, batch_size, device):
    out = []
    for s in range(0, len(patches), batch_size):
        sl = slice(s, s + batch_size)
        feats = api["encode_window"](
            model,
            torch.from_numpy(patches[sl]), torch.from_numpy(pos[sl]),
            torch.from_numpy(extra[sl]), torch.from_numpy(grav[sl]),
            per_axis=per_axis, device=device)
        out.append(feats.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="~/Desktop/biopm")
    p.add_argument("--masking_rate", type=float, default=0.5, choices=[0.25, 0.5, 0.75])
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DATASETS))
    p.add_argument("--per_axis", action="store_true", default=True,
                   help="Per-axis mean+std pooling (BioPM's recommended default)")
    p.add_argument("--gravity", default="exclude", choices=["exclude", "include"],
                   help="Gravity features are unreliable on this data; see module docstring")
    # Cropping to the non-zero extent sounds right but is empirically much worse:
    # on sony_watch it leaves ~20 samples, where a 6th-order 0.5-12 Hz bandpass is
    # degenerate, so 91% of windows yield no movement elements and a linear probe
    # collapses to 0.119. Keeping the full padded window gives the filter enough
    # samples to work (probe 0.942 on sony_watch, 0.637 on blind_user_filtered vs
    # 0.551 cropped), and matches what every other encoder in the grid is fed.
    p.add_argument("--crop_padding", action="store_true", default=False,
                   help="Trim each window to its non-zero extent (worse; see comment)")
    p.add_argument("--no_crop_padding", dest="crop_padding", action="store_false")
    p.add_argument("--target_fs", type=int, default=30)
    p.add_argument("--tag", default="biopm")
    p.add_argument("--dataset_version", default="20_120")
    p.add_argument("--out_dir", default="new_embed")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", default=None)
    cli = p.parse_args()

    api = import_biopm(cli.repo)
    device = get_device(cli.gpu)
    device_str = str(device)
    model = api["load_pretrained"](masking_rate=cli.masking_rate, device=device_str)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"Loaded BioPM (masking_rate={cli.masking_rate}), {n_params:,} parameters")

    os.makedirs(cli.out_dir, exist_ok=True)

    for dataset in cli.datasets:
        dpath = os.path.join("dataset", dataset, f"data_{cli.dataset_version}.npy")
        if not os.path.exists(dpath):
            print(f"[{dataset}] no {dpath}, skipping")
            continue
        data = np.load(dpath).astype(np.float32)
        labels = np.load(os.path.join("dataset", dataset, f"label_{cli.dataset_version}.npy"))
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"[{dataset}] {data.shape[0]} windows but {labels.shape[0]} labels")

        src_len = data.shape[1]
        window_sec = src_len / 20.0                     # our arrays are 20 Hz
        cfg = api["PreprocessConfig"](ori_fs=20, target_fs=cli.target_fs,
                                      window_sec=int(round(window_sec)))

        # Both datasets measure ~1 g over non-padding samples, so the unit is
        # m/s^2 and the conversion is a known constant -- NOT a statistic
        # estimated from the data, which the zero padding would corrupt.
        acc = data[:, :, :3] / GRAVITY_MS2
        pad_frac = float((np.abs(acc).sum(axis=2) == 0).mean())
        print(f"[{dataset}] {data.shape}; zero padding = {pad_frac:.1%}; "
              f"acc scaled by 1/{GRAVITY_MS2:.2f} (m/s^2 -> g)")

        patches, pos, extra, grav, n_empty, kept = build_window_features(
            api, acc, cfg, cli.crop_padding)
        if len(kept):
            print(f"           cropped window length: median {int(np.median(kept))}/"
                  f"{src_len} samples (min {kept.min()}, max {kept.max()})")
        if n_empty:
            print(f"           {n_empty}/{len(acc)} windows yielded no movement elements "
                  f"({n_empty / len(acc):.1%}); their features will be all-NaN")

        feats = encode(api, model, patches, pos, extra, grav,
                       cli.per_axis, cli.batch_size, device_str)

        acc_dim = (2 * 3 if cli.per_axis else 2) * 64
        if cli.gravity == "exclude":
            feats = feats[:, :acc_dim]

        # NaN windows come from windows with no detected movement elements. Zero
        # them so the classifier sees a finite, uninformative vector rather than
        # poisoning the whole batch.
        bad = ~np.isfinite(feats).all(axis=1)
        if bad.any():
            print(f"           zeroing {bad.sum()} non-finite feature vectors")
            feats[bad] = 0.0

        if float(feats.std()) < 1e-6:
            raise RuntimeError(f"[{dataset}] features are ~constant (std={feats.std():.2e})")

        embeddings = feats[:, None, :]      # [N, 1, D] -- pooled, no time axis
        out = os.path.join(cli.out_dir,
                           f"embed_{cli.tag}_{dataset}_{cli.dataset_version}.npy")
        np.save(out, embeddings)
        print(f"[{dataset}] -> {embeddings.shape} (gravity={cli.gravity})  "
              f"mean={embeddings.mean():.4f} std={embeddings.std():.4f}  -> {out}")


if __name__ == "__main__":
    main()
