#!/usr/bin/env python3
"""Plot IMU datasets stored as (N, 120, 6) numpy arrays.

Edit the USER CONFIG section below, then run:
    python dataste_plots/plot_dataset.py
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# USER CONFIG
DATA_PATH = "dataset/UTD_MHAD/data_20_120.npy"
LABELS_PATH = "dataset/UTD_MHAD/label_20_120.npy"  # Set to None if labels are not available
OUTDIR = "dataste_plots/output_UTD_MHAD"
LABEL_INDEX = 1
# Only these activities will be considered for plotting.
# Example: [0, 1, 2] for first three classes.
SELECTED_ACTIVITIES = [10, 11, 12]
# Number of figures to generate. Each figure has 6 axes (one per channel).
NUM_ACTIVITY_PLOTS = 3
SEED = 42


def _load_data(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3 or data.shape[1:] != (120, 6):
        raise ValueError(f"Expected data shape (N,120,6), got {data.shape}")
    return data.astype(np.float32)


def _load_labels(path: Optional[str], n_samples: int, label_index: int) -> Optional[np.ndarray]:
    if not path:
        return None
    labels = np.load(path)

    if labels.shape[0] != n_samples:
        raise ValueError(f"Label sample count {labels.shape[0]} does not match data sample count {n_samples}")

    if labels.ndim == 3:
        # Common case: (N,120,K) repeated per timestep
        if label_index >= labels.shape[2]:
            raise ValueError(f"label_index={label_index} out of range for labels shape {labels.shape}")
        label_vec = labels[:, 0, label_index]
    elif labels.ndim == 2:
        # Alternative case: (N,K)
        if label_index >= labels.shape[1]:
            raise ValueError(f"label_index={label_index} out of range for labels shape {labels.shape}")
        label_vec = labels[:, label_index]
    else:
        raise ValueError(f"Unsupported labels shape {labels.shape}")

    return label_vec.astype(np.int64)


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_random_sequences(data: np.ndarray, outdir: str, max_samples: int, seed: int, labels: Optional[np.ndarray]) -> None:
    n = data.shape[0]
    k = min(max_samples, n)
    rng = np.random.default_rng(seed)
    picks = rng.choice(n, size=k, replace=False)

    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    t = np.arange(data.shape[1])

    for idx in picks:
        for ch in range(6):
            axes[ch].plot(t, data[idx, :, ch], alpha=0.75, linewidth=1)

    for ch in range(6):
        axes[ch].set_ylabel(CHANNEL_NAMES[ch])
        axes[ch].grid(alpha=0.25)
    axes[-1].set_xlabel("time step")

    if labels is not None:
        shown_labels = labels[picks].tolist()
        fig.suptitle(f"Random sample traces (indices={picks.tolist()}, labels={shown_labels})")
    else:
        fig.suptitle(f"Random sample traces (indices={picks.tolist()})")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "random_sequences.png"), dpi=180)
    plt.close(fig)


def _plot_selected_activity_samples(
    data: np.ndarray,
    labels: np.ndarray,
    outdir: str,
    selected_activities: list[int],
    num_plots: int,
    seed: int,
) -> None:
    if labels is None:
        raise ValueError("LABELS_PATH must be provided to filter by activity.")
    if not selected_activities:
        raise ValueError("SELECTED_ACTIVITIES cannot be empty.")

    rng = np.random.default_rng(seed)
    t = np.arange(data.shape[1])

    class_to_indices: dict[int, np.ndarray] = {}
    for act in selected_activities:
        idx = np.where(labels == act)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples found for activity {act}. Check SELECTED_ACTIVITIES.")
        class_to_indices[act] = idx

    for i in range(num_plots):
        act = selected_activities[i % len(selected_activities)]
        sample_idx = int(rng.choice(class_to_indices[act]))

        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
        for ch in range(6):
            axes[ch].plot(t, data[sample_idx, :, ch], linewidth=1.5)
            axes[ch].set_ylabel(CHANNEL_NAMES[ch])
            axes[ch].grid(alpha=0.25)
        axes[-1].set_xlabel("time step")

        fig.suptitle(f"Activity {act} | sample index {sample_idx}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"activity_plot_{i + 1}.png"), dpi=180)
        plt.close(fig)


def _plot_channel_mean_std(data: np.ndarray, outdir: str) -> None:
    mean = data.mean(axis=0)  # (120,6)
    std = data.std(axis=0)    # (120,6)
    t = np.arange(data.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for ch in range(6):
        ax = axes[ch]
        ax.plot(t, mean[:, ch], color="tab:blue", linewidth=2, label="mean")
        ax.fill_between(
            t,
            mean[:, ch] - std[:, ch],
            mean[:, ch] + std[:, ch],
            alpha=0.25,
            color="tab:blue",
            label="+/- 1 std",
        )
        ax.set_title(CHANNEL_NAMES[ch])
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle("Channel-wise mean and standard deviation across N samples")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "channel_mean_std.png"), dpi=180)
    plt.close(fig)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    # x shape: (N, D)
    x_centered = x - x.mean(axis=0, keepdims=True)
    # SVD-based PCA for numerical stability
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    # Project to first two principal components
    return u[:, :2] * s[:2]


def _plot_2d_projection(data: np.ndarray, outdir: str, labels: Optional[np.ndarray]) -> None:
    flat = data.reshape(data.shape[0], -1)  # (N, 120*6)
    proj = _pca_2d(flat)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is None:
        ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.6)
    else:
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            m = labels == lab
            ax.scatter(proj[m, 0], proj[m, 1], s=12, alpha=0.65, label=f"class {int(lab)}")
        # Avoid massive legends for many classes
        if len(unique_labels) <= 20:
            ax.legend(loc="best", fontsize=8)

    ax.set_title("2D PCA projection of flattened sequences")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_2d.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    _ensure_outdir(OUTDIR)

    data = _load_data(DATA_PATH)
    labels = _load_labels(LABELS_PATH, data.shape[0], LABEL_INDEX)

    _plot_selected_activity_samples(
        data=data,
        labels=labels,
        outdir=OUTDIR,
        selected_activities=SELECTED_ACTIVITIES,
        num_plots=NUM_ACTIVITY_PLOTS,
        seed=SEED,
    )

    print("Saved plots:")
    for i in range(NUM_ACTIVITY_PLOTS):
        print(f"  - {os.path.join(OUTDIR, f'activity_plot_{i + 1}.png')}")


if __name__ == "__main__":
    main()
