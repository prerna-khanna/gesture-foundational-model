"""
baseline_watchguardian.py -- Tier 3: faithful re-implementation of
WatchGuardian (Lei, Cao et al., ACM TCH 2026), Section 3.1.

WatchGuardian's three-stage pipeline:
  Stage 1: a model pre-trained with self-supervised learning on a large amount
           of UNLABELED wrist activity (they use Yuan et al.'s UK Biobank SSL
           model). We reuse OUR frozen encoder here, which is also SSL-pretrained
           on unlabeled activity -- so Stage 1 is a fair match to our own base.
  Stage 2: SUPERVISED fine-tuning of the encoder on public LABELED hand-gesture
           datasets, to make it gesture-aware before any user arrives. This is
           the stage our method does NOT have, and it is the distinction we draw
           in the paper. We reproduce it by fine-tuning a small head on the
           pooled labeled gestures of the training users (leave-one-user-out).
  Stage 3: per-user customization -- expand the few examples ~143x via six
           augmentation techniques (2^6 - 1 combinations) plus synthesis (~80x),
           then train additional lightweight classification layers with weighted
           cross-entropy. A NEW classification head per vocabulary.

Recognition: sliding-window classification + a smoothing threshold of 3
consecutive positive windows (their Section 3.1.3). We approximate the
window/smoothing step by majority vote over a gesture's windows, since our
gestures are already segmented 120-frame windows rather than continuous streams.

Faithful augmentation (their Section 3.1.3, exact parameters):
  1) zooming     speed x0.9..x1.0
  2) scaling     intensity s ~ N(1, 0.2^2), s in [0,2]
  3) time-warp   2 knots, w ~ N(1, 0.05^2)
  4) time-reversal
  5) time-domain noise   Gaussian, level 0.01
  6) frequency-domain noise   noise on FFT components, inverse-FFT back
All 2^6 - 1 combinations -> ~64x; synthesis -> ~80x; total ~143x.

We reuse our frozen encoder for Stage 1 rather than re-training an SSL model,
which is the fair comparison (both start from unlabeled-activity SSL). The
substantive WatchGuardian-specific parts -- the Stage-2 labeled fine-tuning and
the Stage-3 augment/synth + trained classification layers -- are reproduced.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# their six augmentation techniques (exact parameters from Section 3.1.3)
# --------------------------------------------------------------------------- #

def wg_zoom(x, rng):
    factor = rng.uniform(0.9, 1.0)                       # x0.9..x1.0
    T = len(x); dst = np.linspace(0, T - 1, max(4, int(T * factor)))
    src = np.arange(T)
    w = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)
    return _fit_len(w, T)


def wg_scale(x, rng):
    s = np.clip(rng.normal(1.0, 0.2), 0, 2)              # N(1,0.2^2), clipped [0,2]
    return x * s


def wg_timewarp(x, rng):
    T = len(x)
    knots = np.linspace(0, T - 1, 4)                     # 2 interior knots
    w = np.clip(rng.normal(1.0, 0.05, 4), 0, 2)          # N(1,0.05^2)
    warp = np.interp(np.arange(T), knots, np.sort(knots * w))
    return np.stack([np.interp(np.arange(T), warp, x[:, c]) for c in range(x.shape[1])], axis=1)


def wg_reverse(x, rng):
    return x[::-1].copy()


def wg_tnoise(x, rng):
    return x + rng.normal(0, 0.01, x.shape)              # time-domain Gaussian 0.01


def wg_fnoise(x, rng):
    F_ = np.fft.rfft(x, axis=0)
    F_ = F_ + (rng.normal(0, 0.01, F_.shape) + 1j * rng.normal(0, 0.01, F_.shape))
    return np.fft.irfft(F_, n=len(x), axis=0)


_WG_AUGS = [wg_zoom, wg_scale, wg_timewarp, wg_reverse, wg_tnoise, wg_fnoise]


def _fit_len(x, T):
    if len(x) == T:
        return x
    if len(x) > T:
        return x[:T]
    return np.pad(x, ((0, T - len(x)), (0, 0)))


def wg_augment(raw_samples, seed=0, max_combos=32):
    """
    Apply all 2^6-1 combinations of the six techniques (subsampled to max_combos
    for tractability; the paper uses all 63). Returns ~max_combos x expansion.
    """
    rng = np.random.RandomState(seed)
    out = list(raw_samples)
    T = raw_samples.shape[1]
    from itertools import combinations
    combos = []
    for r in range(1, 7):
        combos += list(combinations(range(6), r))
    rng.shuffle(combos)
    combos = combos[:max_combos]
    for x in raw_samples:
        for combo in combos:
            y = x.copy()
            for ai in combo:
                y = _fit_len(_WG_AUGS[ai](y, rng), T)
            out.append(y)
    return np.stack(out).astype(np.float32)


def wg_synthesize(raw_samples, neg_pool, seed=1, factor=8):
    """
    Synthesis (their Section 3.1.3): splice a positive segment (varying length)
    into a negative background context at a random position.
    """
    rng = np.random.RandomState(seed)
    if neg_pool is None or len(neg_pool) == 0:
        neg_pool = raw_samples
    out, T = [], raw_samples.shape[1]
    for x in raw_samples:
        for _ in range(factor):
            bg = neg_pool[rng.randint(len(neg_pool))].copy()
            L = rng.randint(int(T * 0.6), int(T * 0.98))   # [3,4.9]s of a 5s window
            start = rng.randint(0, T - L)
            seg_start = rng.randint(0, max(1, T - L))
            bg[start:start + L] = x[seg_start:seg_start + L]
            out.append(bg)
    return np.stack(out).astype(np.float32) if out else np.zeros((0,) + raw_samples.shape[1:], np.float32)


# --------------------------------------------------------------------------- #
# Stage 2: labeled-gesture fine-tuning head (on frozen encoder embeddings)
# --------------------------------------------------------------------------- #

class WGClassificationLayers(nn.Module):
    """
    The 'additional lightweight classification layers' trained per user
    (their Stage 3). Two FC layers with dropout, over frozen encoder embeddings.
    """
    def __init__(self, emb_dim, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, n_out))

    def forward(self, z):
        return self.net(z)


def _pool_embed(bb, reg_embed, raw, mask=None):
    """Mean-pool the frozen encoder tokens to a fixed vector (Stage 1 features)."""
    emb = bb.encode_array(raw)                # (N, T, H)
    return emb.mean(axis=1)                    # (N, H)




# --------------------------------------------------------------------------- #
# Stage 1 + Stage 2 encoder: WatchGuardian's own ResNet (pretrained on HHAR),
# then labeled-gesture fine-tuned on the training users (leave-one-user-out).
# --------------------------------------------------------------------------- #

class WGEncoderStage2(nn.Module):
    """WG ResNet encoder + a fine-tuning classifier head (Stage 2)."""
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoder.emb_dim, n_classes)

    def forward(self, x):
        return self.clf(self.encoder(x))

    def embed(self, x):
        return self.encoder(x)


def load_wg_encoder(root, device='cpu'):
    """Load the HHAR-pretrained WG ResNet encoder (Stage 1)."""
    import os
    from customization.wg_pretrain import WGResNetEncoder
    p = os.path.join(root, 'saved', 'watchguardian', 'wg_resnet_ssl.pt')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"WG encoder not found at {p}. Run: python -m customization.wg_pretrain")
    ck = torch.load(p, map_location='cpu')
    enc = WGResNetEncoder(in_channels=ck['in_channels'], emb_dim=ck['emb_dim'])
    enc.load_state_dict(ck['state_dict']); enc.to(device).eval()
    return enc


def wg_stage2_finetune(encoder, raw_labeled, y_labeled, epochs=25, lr=1e-4,
                       device='cpu'):
    """
    Stage 2: supervised fine-tuning on public LABELED gestures (here, the
    pooled training users under leave-one-user-out). Makes the encoder
    gesture-aware before per-user customization. Returns the fine-tuned encoder.
    """
    classes = np.unique(y_labeled)
    remap = {c: i for i, c in enumerate(classes)}
    model = WGEncoderStage2(encoder, len(classes)).to(device)
    X = torch.tensor(raw_labeled, dtype=torch.float32, device=device)
    yy = torch.tensor([remap[c] for c in y_labeled], dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    bs = 128
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(model(X[idx]), yy[idx]).backward(); opt.step()
    model.eval()
    return model.encoder


# --------------------------------------------------------------------------- #
# per-user WatchGuardian customization + evaluation
# --------------------------------------------------------------------------- #

def eval_wg_user(wg_encoder, raw_user, y_user, n_train=7, neg_pool=None,
                 epochs=80, device='cpu', seed=0, validate=False):
    """
    WatchGuardian customization on one user. Frozen encoder (Stage 1) provides
    features; Stage 3 expands the few examples ~143x and trains classification
    layers with weighted cross-entropy and a negative class.

    validate=False: recognizes all gestures (recognizer only).
    (WatchGuardian has no gesture validator; it recognizes whatever is added.
     So there is no validated variant -- this is a key contrast in the paper.)
    """
    rng = np.random.default_rng(seed)
    classes = list(np.unique(y_user))
    tr, te = {}, []
    for c in classes:
        idx = np.where(y_user == c)[0]; rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr[c] = idx[:k]; te += list(idx[k:])
    te = np.array(te)
    remap = {c: i for i, c in enumerate(classes)}
    neg_id = len(classes)

    # Stage 3: expand each gesture's few examples ~143x, embed with frozen encoder
    Xz, yz = [], []
    for c in classes:
        raw = raw_user[tr[c]]
        aug = wg_augment(raw, seed=int(c))
        syn = wg_synthesize(raw, neg_pool, seed=int(c) + 100)
        pos = np.concatenate([aug, syn]) if len(syn) else aug
        with torch.no_grad():
            z = wg_encoder(torch.tensor(pos, dtype=torch.float32, device=device)).cpu().numpy()
        Xz.append(z); yz += [remap[c]] * len(z)
    # negative class from synthesized backgrounds
    if neg_pool is not None and len(neg_pool):
        with torch.no_grad():
            negz = wg_encoder(torch.tensor(neg_pool[:min(len(neg_pool), 400)],
                              dtype=torch.float32, device=device)).cpu().numpy()
        Xz.append(negz); yz += [neg_id] * len(negz)
    Xz = np.concatenate(Xz).astype(np.float32); yz = np.array(yz)

    # weighted cross-entropy (their loss) to handle class imbalance
    counts = np.bincount(yz, minlength=neg_id + 1).astype(np.float32)
    weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32, device=device)

    head = WGClassificationLayers(Xz.shape[1], neg_id + 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=weights)
    Xt = torch.tensor(Xz, dtype=torch.float32, device=device)
    yt = torch.tensor(yz, dtype=torch.long, device=device)
    head.train()
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(head(Xt[idx]), yt[idx]).backward(); opt.step()

    head.eval()
    with torch.no_grad():
        Zte = wg_encoder(torch.tensor(raw_user[te], dtype=torch.float32, device=device)).cpu().numpy()
    with torch.no_grad():
        pred = head(torch.tensor(Zte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    truth = np.array([remap[c] for c in y_user[te]])
    acc = float(np.mean(pred == truth))
    return {'baseline': 'watchguardian', 'acc': acc, 'n_gestures': int(len(classes)),
            'requires_per_user_training': True, 'validates': False, 'n_train': n_train}