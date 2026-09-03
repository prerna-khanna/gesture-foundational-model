"""
wg_pretrain.py -- WatchGuardian Stage 1, faithfully reproduced.

WatchGuardian adopts Yuan et al.'s OxWearables model: a 5-layer ResNet feature
extractor pre-trained with multi-task self-supervised learning using three
pretext tasks -- Arrow of Time (AoT), Permutation, and Time Warping (TW).
Their data is UK Biobank (accelerometer, not obtainable). We instead pretrain
the SAME method on HHAR, an unlabeled human-activity dataset already present in
our pipeline at the same 6-axis / 120-frame format as our gesture data. This
keeps the modality and data format identical to our own encoder, so the
WatchGuardian baseline differs from our method in the SSL architecture and
pretext tasks (ResNet + AoT/Perm/TW), not in the data or sensor modality.

    python -m customization.wg_pretrain            # pretrain on HHAR, save encoder
    -> saved/watchguardian/wg_resnet_ssl.pt

The three pretext tasks (Saeed et al. / Yuan et al.):
  - Arrow of Time: predict whether the sequence is played forward or reversed.
  - Permutation:  split into segments, predict whether they were shuffled.
  - Time Warping: predict whether the sequence was time-warped.
Each is a binary head on the shared ResNet features; total loss is their sum.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# 5-layer ResNet feature extractor (their Stage 1 architecture)
# --------------------------------------------------------------------------- #

class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, k=5, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = (nn.Sequential() if (cin == cout and stride == 1)
                      else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                         nn.BatchNorm1d(cout)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.short(x))


class WGResNetEncoder(nn.Module):
    """5-layer ResNet feature extractor + 2 FC layers -> embedding (their Fig 2a)."""
    def __init__(self, in_channels=6, emb_dim=128):
        super().__init__()
        chs = [in_channels, 32, 64, 128, 128, 128]      # 5 residual stages
        self.blocks = nn.ModuleList([
            ResBlock1d(chs[i], chs[i + 1], stride=(2 if i < 3 else 1))
            for i in range(5)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(chs[-1], emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        h = self.pool(h).flatten(1)
        return self.fc2(F.relu(self.fc1(h)))            # (B, emb_dim)


# --------------------------------------------------------------------------- #
# the three SSL pretext transforms + labels
# --------------------------------------------------------------------------- #

def _aot(x):
    return x[::-1].copy()                                # reverse time


def _permute(x, n_seg=4):
    T = len(x); seg = T // n_seg
    order = np.random.permutation(n_seg)
    return np.concatenate([x[o * seg:(o + 1) * seg] for o in order] + [x[n_seg * seg:]])[:T]


def _timewarp(x, sigma=0.2):
    T = len(x)
    knots = np.linspace(0, T - 1, 4)
    w = np.clip(np.random.normal(1.0, sigma, 4), 0.3, 2)
    warp = np.interp(np.arange(T), knots, np.sort(knots * w))
    return np.stack([np.interp(np.arange(T), warp, x[:, c]) for c in range(x.shape[1])], axis=1)


def make_ssl_batch(raw, rng):
    """
    For a batch of raw windows, create the three binary pretext labels.
    Returns x (possibly transformed) and 3 binary targets (aot, perm, tw).
    Each sample independently may or may not receive each transform.
    """
    B = len(raw)
    x = raw.copy()
    y_aot = rng.integers(0, 2, B)
    y_perm = rng.integers(0, 2, B)
    y_tw = rng.integers(0, 2, B)
    for i in range(B):
        if y_aot[i]:
            x[i] = _aot(x[i])
        if y_perm[i]:
            x[i] = _permute(x[i])
        if y_tw[i]:
            x[i] = _timewarp(x[i])
    return x.astype(np.float32), y_aot, y_perm, y_tw


class WGPretrainModel(nn.Module):
    """Shared ResNet + three binary SSL heads."""
    def __init__(self, in_channels=6, emb_dim=128):
        super().__init__()
        self.encoder = WGResNetEncoder(in_channels, emb_dim)
        self.h_aot = nn.Linear(emb_dim, 2)
        self.h_perm = nn.Linear(emb_dim, 2)
        self.h_tw = nn.Linear(emb_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        return self.h_aot(z), self.h_perm(z), self.h_tw(z)


# --------------------------------------------------------------------------- #
# pretraining loop on HHAR
# --------------------------------------------------------------------------- #

def pretrain(hhar_raw, in_channels=6, emb_dim=128, epochs=200, bs=256,
             lr=1e-3, device='cpu', seed=0, verbose=True,
             patience=15, min_delta=1e-3, val_frac=0.1):
    """
    Train the SSL encoder to CONVERGENCE rather than for a fixed number of
    epochs. We hold out a validation split, evaluate the three pretext tasks'
    combined loss each epoch, and stop when it stops improving (patience
    epochs without a min_delta gain). This removes the arbitrary epoch count
    and guards against the pretext tasks overfitting a small corpus like HHAR
    (~9k windows). `epochs` is now an upper bound.
    """
    rng = np.random.default_rng(seed)
    N = len(hhar_raw)
    idx_all = rng.permutation(N)
    n_val = max(bs, int(N * val_frac))
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]
    val_raw = hhar_raw[val_idx]

    model = WGPretrainModel(in_channels, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    def eval_val():
        model.eval()
        with torch.no_grad():
            xb, ya, yp, yt = make_ssl_batch(val_raw, np.random.default_rng(0))
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            la, lp, lt = model(xb)
            v = (lossf(la, torch.tensor(ya, device=device))
                 + lossf(lp, torch.tensor(yp, device=device))
                 + lossf(lt, torch.tensor(yt, device=device))).item()
        model.train()
        return v

    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        perm = rng.permutation(len(tr_idx))
        model.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[perm[i:i + bs]]
            if len(b) < 2:
                continue
            xb, ya, yp, yt = make_ssl_batch(hhar_raw[b], rng)
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            la, lp, lt = model(xb)
            loss = (lossf(la, torch.tensor(ya, device=device))
                    + lossf(lp, torch.tensor(yp, device=device))
                    + lossf(lt, torch.tensor(yt, device=device)))
            opt.zero_grad(); loss.backward(); opt.step()
        vloss = eval_val()
        if verbose:
            print(f"  [wg-ssl] epoch {ep+1:3d}  val-loss {vloss:.4f}"
                  + ("  *" if vloss < best - min_delta else ""))
        if vloss < best - min_delta:
            best, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  [wg-ssl] converged at epoch {ep+1} "
                          f"(no improvement for {patience} epochs); best val-loss {best:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model.encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='20_120')
    ap.add_argument('--epochs', type=int, default=200,
                    help='UPPER BOUND; training stops early on val-loss plateau')
    ap.add_argument('--emb_dim', type=int, default=128)
    ap.add_argument('--save', default='saved/watchguardian/wg_resnet_ssl.pt')
    args = ap.parse_args()

    hhar_path = os.path.join(ROOT, 'dataset', 'hhar', f'data_{args.version}.npy')
    if not os.path.exists(hhar_path):
        raise FileNotFoundError(
            f"HHAR not found at {hhar_path}. It should be in your dataset/ "
            f"folder (you use it as a background set). Preprocess it with "
            f"dataset/hhar.py if missing.")
    hhar = np.load(hhar_path).astype(np.float32)
    print(f"[wg-ssl] HHAR: {hhar.shape} (pretraining WatchGuardian's ResNet "
          f"with AoT/Permutation/TimeWarp)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = pretrain(hhar, in_channels=hhar.shape[-1], emb_dim=args.emb_dim,
                   epochs=args.epochs, device=device)

    save = os.path.join(ROOT, args.save)
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({'state_dict': enc.state_dict(),
                'in_channels': hhar.shape[-1], 'emb_dim': args.emb_dim}, save)
    print(f"[wg-ssl] saved encoder -> {save}")


if __name__ == '__main__':
    main()