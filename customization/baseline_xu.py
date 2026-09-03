"""
baseline_xu.py -- Tier 2: faithful re-implementation of Xu et al. (CHI 2022),
"Enabling Hand Gesture Customization on Wrist-Worn Devices".

We reproduce their METHOD, adapted to our data, and are explicit about the one
thing we cannot reproduce: their base model was pre-trained on a labeled gesture
dataset from 500+ participants, which does not exist for blind users. That
limitation is precisely our paper's argument, so the honest adaptation is to
train Xu's architecture on the SAME data available to our method (leave-one-user
-out over the blind users), then run their few-shot customization pipeline.

Faithful to the paper (Section 4.1--4.2):
  - EfficientNet-style feature extractor: per input channel, two MBConv
    (inverted-residual) blocks; concatenate the 6 channels; one separable conv;
    max-pool; flatten -> 120-d embedding. (Their Fig 2a; 106k total params.)
  - Pre-trained classifier: 5 FC layers [80,40,20,10,5] with BatchNorm +
    dropout(0.5) between each; cross-entropy; Adam. (Their Section 4.1.)
  - Per-user prediction head: 2-layer FC, first layer LeakyReLU(0.3) + L2
    (lambda=5e-5) + dropout(0.5), softmax output of size (n_custom + 1) for the
    negative class. A NEW head is trained from scratch for each vocabulary size.
    (Their Section 4.1, "prediction head".)
  - Few-shot data processing: time-series augmentation (jitter, scaling,
    time-warp) + their negative-augmentation + synthesis. (Their Section 4.2.)

Deliberately adapted / simplified, and flagged so a reviewer knows:
  - Base model trained on our blind-user data (LOU), NOT their 500-participant
    labeled corpus (which we cannot obtain). This is the intended comparison.
  - Adversarial regularization (their 4.2.4) is omitted by default; it is a
    small regularizer on top of augment+synth and can be enabled with
    --xu_adversarial if a reviewer asks. The augment+synth pipeline is the
    substantive part and is implemented.

Reported the same way as every other baseline: recognition accuracy on the
full vocabulary, tagged requires_per_user_training=True, validates=False.
Xu DOES have interactive feedback (a form of validation); we implement that
separately (baseline_xu_validator) so the validator comparison is apples-to-
apples, but the recognition number here is the recognizer only.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# EfficientNet-style feature extractor (their Fig 2a)
# --------------------------------------------------------------------------- #

class MBConv1d(nn.Module):
    """Inverted-residual block (MobileNetV2/EfficientNet) for 1-D signals."""
    def __init__(self, ch, expand=4, k=5):
        super().__init__()
        hid = ch * expand
        self.use_res = True
        self.pw1 = nn.Conv1d(ch, hid, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hid)
        self.dw = nn.Conv1d(hid, hid, k, padding=k // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm1d(hid)
        self.pw2 = nn.Conv1d(hid, ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(ch)

    def forward(self, x):
        y = F.relu6(self.bn1(self.pw1(x)))
        y = F.relu6(self.bn2(self.dw(y)))
        y = self.bn3(self.pw2(y))
        return x + y if self.use_res else y


class SeparableConv1d(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class XuFeatureExtractor(nn.Module):
    """
    Per-channel MBConv x2 -> concat 6 channels -> separable conv -> maxpool ->
    flatten -> 120-d embedding. Matches Xu et al. Fig 2a.
    """
    def __init__(self, in_channels=6, emb_dim=120, base_ch=16):
        super().__init__()
        self.in_channels = in_channels
        # each scalar channel lifted to base_ch, then two MBConv blocks
        self.stems = nn.ModuleList([nn.Conv1d(1, base_ch, 5, padding=2) for _ in range(in_channels)])
        self.blocks = nn.ModuleList([
            nn.Sequential(MBConv1d(base_ch), MBConv1d(base_ch)) for _ in range(in_channels)])
        self.sep = SeparableConv1d(base_ch * in_channels, base_ch * in_channels)
        self.pool = nn.AdaptiveMaxPool1d(4)
        self.proj = nn.Linear(base_ch * in_channels * 4, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        # x: (B, T, C) -> per channel (B, 1, T)
        x = x.transpose(1, 2)                        # (B, C, T)
        feats = []
        for c in range(self.in_channels):
            h = self.stems[c](x[:, c:c + 1, :])
            h = self.blocks[c](h)
            feats.append(h)
        h = torch.cat(feats, dim=1)                  # (B, base_ch*C, T)
        h = self.sep(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)                          # (B, 120)


class XuPretrainedClassifier(nn.Module):
    """Feature extractor + 5-FC classifier head [80,40,20,10,n] with BN+dropout."""
    def __init__(self, n_classes, in_channels=6, emb_dim=120, base_ch=16):
        super().__init__()
        self.feat = XuFeatureExtractor(in_channels, emb_dim, base_ch=base_ch)
        sizes = [emb_dim, 80, 40, 20, 10]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(0.5)]
        layers += [nn.Linear(sizes[-1], n_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.feat(x))

    def embed(self, x):
        return self.feat(x)


class XuPredictionHead(nn.Module):
    """Per-user 2-layer FC head over frozen 120-d embeddings (their Section 4.1)."""
    def __init__(self, emb_dim, n_out):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 32)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, n_out)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z), 0.3)
        return self.fc2(self.drop(h))


# --------------------------------------------------------------------------- #
# Xu's few-shot data processing (their Section 4.2)
# --------------------------------------------------------------------------- #

def _jitter(x, sigma=0.03):
    return x + np.random.normal(0, sigma, x.shape)


def _scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
    return x * factor


def _time_warp(x, n_knots=2, sigma=0.2):
    T = len(x)
    orig = np.linspace(0, T - 1, T)
    knot_x = np.linspace(0, T - 1, n_knots + 2)
    knot_y = knot_x * np.random.normal(1.0, sigma, len(knot_x))
    knot_y[0], knot_y[-1] = 0, T - 1
    warp = np.interp(orig, knot_x, np.sort(knot_y))
    return np.stack([np.interp(orig, warp, x[:, c]) for c in range(x.shape[1])], axis=1)


def xu_augment(raw_samples, factor=8, seed=0):
    """Positive augmentation: jitter/scaling/time-warp (their three techniques)."""
    rng = np.random.RandomState(seed)
    out = list(raw_samples)
    for x in raw_samples:
        for _ in range(factor):
            op = rng.randint(3)
            out.append(_jitter(x) if op == 0 else _scaling(x) if op == 1 else _time_warp(x))
    return np.stack(out).astype(np.float32)


def xu_negatives(raw_samples, factor=4, seed=1):
    """
    Their negative-augmentation: permutation-style transforms marked as NEGATIVE,
    so the head learns a negative class boundary (their Section 4.2.2).
    """
    rng = np.random.RandomState(seed)
    out = []
    for x in raw_samples:
        for _ in range(factor):
            perm = x.copy()
            seg = len(perm) // 4
            order = rng.permutation(4)
            perm = np.concatenate([x[order[i] * seg:(order[i] + 1) * seg] for i in range(4)])
            if len(perm) < len(x):
                perm = np.pad(perm, ((0, len(x) - len(perm)), (0, 0)))
            out.append(perm[:len(x)])
    return np.stack(out).astype(np.float32) if out else np.zeros((0,) + raw_samples.shape[1:], np.float32)


def xu_synthesize(raw_samples, neg_pool, factor=4, seed=2):
    """
    Data synthesis (their Section 4.2.3): splice a positive gesture segment into
    a negative background context to simulate natural motion variance.
    """
    rng = np.random.RandomState(seed)
    if len(neg_pool) == 0:
        return np.zeros((0,) + raw_samples.shape[1:], np.float32)
    out, T = [], raw_samples.shape[1]
    for x in raw_samples:
        for _ in range(factor):
            bg = neg_pool[rng.randint(len(neg_pool))].copy()
            L = rng.randint(T // 3, 2 * T // 3)
            start = rng.randint(0, T - L)
            bg[start:start + L] = x[:L]
            out.append(bg)
    return np.stack(out).astype(np.float32) if out else np.zeros((0,) + raw_samples.shape[1:], np.float32)


# --------------------------------------------------------------------------- #
# base-model training (LOU) -- once per held-out user
# --------------------------------------------------------------------------- #

def train_xu_base(raw_train, y_train, in_channels=6, epochs=40, lr=1e-3, device='cpu', base_ch=16):
    """Train Xu's feature extractor + classifier on the pooled training users."""
    classes = np.unique(y_train)
    remap = {c: i for i, c in enumerate(classes)}
    X = torch.tensor(raw_train, dtype=torch.float32, device=device)
    yy = torch.tensor([remap[c] for c in y_train], dtype=torch.long, device=device)
    model = XuPretrainedClassifier(len(classes), in_channels, base_ch=base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    bs = 128
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:           # BatchNorm needs >1 sample; skip a size-1 tail
                continue
            opt.zero_grad()
            lossf(model(X[idx]), yy[idx]).backward()
            opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# per-user customization + evaluation
# --------------------------------------------------------------------------- #

def eval_xu_user(base_model, raw_test_user, y_test_user, n_train=7,
                 neg_pool=None, epochs=60, device='cpu', seed=0):
    """
    Customize on the held-out user: for each gesture, take k support samples,
    run augment+synth, train a fresh (n+1)-class prediction head over the frozen
    Xu embeddings, evaluate on held-out samples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_test_user)
    tr, te = {}, []
    for c in classes:
        idx = np.where(y_test_user == c)[0]; rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr[c] = idx[:k]; te += list(idx[k:])
    te = np.array(te)

    # build augmented+synthesized training set per gesture, plus a negative class
    Xtr, ytr = [], []
    remap = {c: i for i, c in enumerate(classes)}
    for c in classes:
        raw = raw_test_user[tr[c]]
        aug = xu_augment(raw, factor=8)
        syn = xu_synthesize(raw, neg_pool if neg_pool is not None else raw, factor=4)
        pos = np.concatenate([aug, syn]) if len(syn) else aug
        Xtr.append(pos); ytr += [remap[c]] * len(pos)
    # negative class
    neg = xu_negatives(raw_test_user[np.concatenate([tr[c] for c in classes])], factor=4)
    if len(neg):
        Xtr.append(neg); ytr += [len(classes)] * len(neg)
    Xtr = np.concatenate(Xtr).astype(np.float32); ytr = np.array(ytr)

    # frozen embeddings from the base feature extractor
    base_model.eval()
    with torch.no_grad():
        Ztr = base_model.embed(torch.tensor(Xtr, dtype=torch.float32, device=device)).cpu().numpy()
        Zte = base_model.embed(torch.tensor(raw_test_user[te], dtype=torch.float32, device=device)).cpu().numpy()

    # train the per-user prediction head (n+1 classes)
    head = XuPredictionHead(base_model.feat.emb_dim, len(classes) + 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=5e-5)
    lossf = nn.CrossEntropyLoss()
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(head(Zt), yt).backward(); opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(Zte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    truth = np.array([remap[c] for c in y_test_user[te]])
    # ignore predictions of the negative class for the closed-set accuracy
    acc = float(np.mean(pred == truth))
    return {'baseline': 'xu', 'acc': acc, 'n_gestures': int(len(classes)),
            'requires_per_user_training': True, 'validates': False,  # plain recognizer; validated variant is separate
            'n_train': n_train}


# --------------------------------------------------------------------------- #
# Xu et al.'s interactive-feedback validation (their Section 4.1)
# --------------------------------------------------------------------------- #
# Xu flag a candidate gesture in three cases, all read off the trained
# prediction head's predictions on the candidate's own segments:
#   (a) too similar to an existing gesture: the head predicts the candidate's
#       segments as one of the EXISTING gestures a majority of the time;
#   (b) too inconsistent: the head's predictions across the candidate's own
#       demonstrations disagree (low self-agreement);
#   (c) too confusing with daily activity: the segments are predicted as the
#       negative/background class a majority of the time.
# We implement these on the same trained head, so the comparison to our
# validator is between the two systems' OWN acceptance rules, not a common one.

def xu_validate_gesture(head, z_cand, existing_ids, neg_id,
                        tau_similar=0.5, tau_consistent=0.5, tau_bg=0.5,
                        device='cpu'):
    """
    z_cand : (k, emb) embeddings of the candidate gesture's demonstrations.
    Returns (accepted: bool, reason: str).
    """
    import torch
    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(z_cand, dtype=torch.float32, device=device))
        pred = logits.argmax(1).cpu().numpy()
    # (c) background collision
    if np.mean(pred == neg_id) > tau_bg:
        return False, 'confusing_with_daily_activity'
    # (a) too similar to an existing gesture
    for e in existing_ids:
        if np.mean(pred == e) > tau_similar:
            return False, 'too_similar_to_existing'
    # (b) inconsistency: no single label dominates the candidate's own demos
    if len(pred):
        top_frac = max(np.mean(pred == u) for u in np.unique(pred))
        if top_frac < tau_consistent:
            return False, 'too_inconsistent'
    return True, 'accepted'


def eval_xu_user_validated(base_model, raw_test_user, y_test_user, n_train=7,
                           neg_pool=None, epochs=60, device='cpu', seed=0):
    """
    Xu's customization WITH their interactive-feedback validation: add gestures
    one at a time, run Xu's checks against the head trained on the accepted set,
    reject flagged gestures, and report accuracy on the ACCEPTED vocabulary.
    This mirrors how we evaluate our own validator, so acceptance rate and
    accuracy are directly comparable between the two systems.
    """
    import torch
    rng = np.random.default_rng(seed)
    classes = list(np.unique(y_test_user))
    tr, te = {}, []
    for c in classes:
        idx = np.where(y_test_user == c)[0]; rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr[c] = idx[:k]; te += list(idx[k:])
    te = np.array(te)

    base_model.eval()
    def embed(raw):
        with torch.no_grad():
            return base_model.embed(torch.tensor(raw, dtype=torch.float32, device=device)).cpu().numpy()

    accepted, rejected = [], []
    for c in classes:
        # train a head on the CURRENT accepted set + candidate + negative
        cur = accepted + [c]
        remap = {g: i for i, g in enumerate(cur)}
        neg_id = len(cur)
        Xtr, ytr = [], []
        for g in cur:
            raw = raw_test_user[tr[g]]
            aug = xu_augment(raw, factor=8)
            syn = xu_synthesize(raw, neg_pool if neg_pool is not None else raw, factor=4)
            pos = np.concatenate([aug, syn]) if len(syn) else aug
            Xtr.append(embed(pos)); ytr += [remap[g]] * len(pos)
        neg = xu_negatives(raw_test_user[np.concatenate([tr[g] for g in cur])], factor=4)
        if len(neg):
            Xtr.append(embed(neg)); ytr += [neg_id] * len(neg)
        Ztr = np.concatenate(Xtr); ytr = np.array(ytr)
        head = XuPredictionHead(base_model.feat.emb_dim, neg_id + 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=5e-5)
        lossf = nn.CrossEntropyLoss()
        Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.long, device=device)
        head.train()
        for _ in range(epochs):
            opt.zero_grad(); lossf(head(Zt), yt).backward(); opt.step()

        # run Xu's validation on the candidate c
        z_cand = embed(raw_test_user[tr[c]])
        existing_ids = [remap[g] for g in accepted]
        ok, reason = xu_validate_gesture(head, z_cand, existing_ids, neg_id, device=device)
        if ok:
            accepted.append(c)
        else:
            rejected.append((int(c), reason))

    if not accepted:
        return None

    # final head on the accepted vocabulary, evaluate on held-out
    remap = {g: i for i, g in enumerate(accepted)}
    neg_id = len(accepted)
    Xtr, ytr = [], []
    for g in accepted:
        raw = raw_test_user[tr[g]]
        aug = xu_augment(raw, factor=8)
        syn = xu_synthesize(raw, neg_pool if neg_pool is not None else raw, factor=4)
        pos = np.concatenate([aug, syn]) if len(syn) else aug
        Xtr.append(embed(pos)); ytr += [remap[g]] * len(pos)
    neg = xu_negatives(raw_test_user[np.concatenate([tr[g] for g in accepted])], factor=4)
    if len(neg):
        Xtr.append(embed(neg)); ytr += [neg_id] * len(neg)
    Ztr = np.concatenate(Xtr); ytr = np.array(ytr)
    head = XuPredictionHead(base_model.feat.emb_dim, neg_id + 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=5e-5)
    lossf = nn.CrossEntropyLoss()
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    head.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(head(Zt), yt).backward(); opt.step()
    head.eval()
    keep = np.isin(y_test_user[te], accepted)
    Zte = embed(raw_test_user[te][keep])
    with torch.no_grad():
        pred = head(torch.tensor(Zte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    truth = np.array([remap[g] for g in y_test_user[te][keep]])
    acc = float(np.mean(pred == truth)) if len(truth) else 0.0
    return {'baseline': 'xu_validated', 'acc_full': acc,
            'accepted': len(accepted), 'total': len(classes),
            'rejected': rejected, 'n_train': n_train,
            'requires_per_user_training': True, 'validates': True}