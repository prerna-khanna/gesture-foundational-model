"""
baselines.py -- Tier 1 baselines (DTW, SVM, RandomForest, fine-tune).

All baselines share the leave-one-user-out protocol and the SAME evaluation
metric as our method: register k gestures from the test user, recognize the
held-out samples, report accuracy on the vocabulary. This makes every row in
the results table directly comparable.

Design choices, matched to how Xu et al. and WatchGuardian report baselines:
  - SVM and RandomForest operate on the FROZEN encoder embeddings (mean-pooled),
    the same features our nearest-class-mean ablation uses. This is Xu's exact
    setup ("input for these traditional models are the feature embeddings from
    the pre-trained model").
  - DTW is a template-matcher on the raw signal (uWave-style), one template per
    gesture. This is the classic training-free baseline and also mirrors the
    template-matching lineage of AccessWear.
  - fine-tune replaces the classifier head and fine-tunes on the k support
    samples, the standard transfer-learning baseline.

None of these baselines has a validator; they recognize whatever they are
given. They therefore report accuracy on the FULL vocabulary (no acceptance
step), which is the honest comparison: "what does recognition look like without
validation?" Our method's headline is that validation raises this number while
keeping the gestures users chose.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# DTW template matcher (uWave-style)
# --------------------------------------------------------------------------- #

def _dtw_distance(a, b, radius=10):
    """DTW distance between two (T, C) sequences using fastdtw (linear-time)."""
    from fastdtw import fastdtw
    dist, _ = fastdtw(a, b, radius=radius,
                      dist=lambda p, q: np.linalg.norm(p - q))
    return dist


class DTWBaseline:
    """One template per gesture (the first support sample); 1-NN by DTW."""

    def __init__(self, radius=10):
        self.radius = radius
        self.templates = {}   # name -> (T, C)

    def fit(self, raw_support, y_support, names):
        for c in np.unique(y_support):
            idx = np.where(y_support == c)[0][0]     # first sample as template
            self.templates[names[c]] = raw_support[idx]

    def predict(self, raw_query):
        out = []
        for x in raw_query:
            best, bestd = None, np.inf
            for name, tmpl in self.templates.items():
                d = _dtw_distance(x, tmpl, radius=self.radius)
                if d < bestd:
                    bestd, best = d, name
            out.append(best)
        return out


# --------------------------------------------------------------------------- #
# embedding-space classical ML (SVM, RandomForest)
# --------------------------------------------------------------------------- #

def _pool(emb):
    """(N, T, H) -> (N, H) mean-pool, matching the NCM ablation's features."""
    return emb.mean(axis=1) if emb.ndim == 3 else emb


class SklearnBaseline:
    """SVM or RandomForest on frozen mean-pooled embeddings."""

    def __init__(self, kind='svm'):
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        self.kind = kind
        self.scaler = StandardScaler()
        if kind == 'svm':
            self.clf = SVC(kernel='rbf', C=10, gamma='scale')
        elif kind == 'rf':
            self.clf = RandomForestClassifier(n_estimators=200, random_state=0)
        else:
            raise ValueError(kind)

    def fit(self, emb_support, y_support):
        X = self.scaler.fit_transform(_pool(emb_support))
        self.clf.fit(X, y_support)

    def predict(self, emb_query):
        X = self.scaler.transform(_pool(emb_query))
        return self.clf.predict(X)


# --------------------------------------------------------------------------- #
# fine-tune a linear head on the frozen embeddings
# --------------------------------------------------------------------------- #

class FineTuneBaseline:
    """
    Transfer-learning baseline: a fresh linear classifier over the frozen
    encoder embeddings, trained on the k support samples. This is the
    lightweight version of Xu's "fine-tuning on the pre-trained model"
    (their §5.2.4) -- we fine-tune the head, not the encoder, because the
    encoder is the shared frozen backbone in every condition.
    """

    def __init__(self, epochs=100, lr=1e-2, wd=1e-3):
        self.epochs, self.lr, self.wd = epochs, lr, wd

    def fit(self, emb_support, y_support):
        import torch
        import torch.nn as nn
        X = torch.tensor(_pool(emb_support), dtype=torch.float32)
        classes = np.unique(y_support)
        self.classes = classes
        remap = {c: i for i, c in enumerate(classes)}
        yy = torch.tensor([remap[c] for c in y_support], dtype=torch.long)
        self.head = nn.Linear(X.shape[1], len(classes))
        opt = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.CrossEntropyLoss()
        self.head.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            lossf(self.head(X), yy).backward()
            opt.step()

    def predict(self, emb_query):
        import torch
        self.head.eval()
        with torch.no_grad():
            X = torch.tensor(_pool(emb_query), dtype=torch.float32)
            idx = self.head(X).argmax(1).numpy()
        return self.classes[idx]


# --------------------------------------------------------------------------- #
# unified runner: evaluate any baseline on one user, LOU-style
# --------------------------------------------------------------------------- #

def eval_baseline_user(kind, bb, user, n_train=7, seed=0, raw_loader=None):
    """
    kind in {'dtw', 'svm', 'rf', 'finetune'}.
    Returns per-user accuracy on the FULL vocabulary (no validation step),
    which is the honest 'recognition without a validator' number.
    """
    from customization.backbone import gesture_ids

    emb, mask, lab = bb.encode_dataset(user, return_masks=True)
    y = gesture_ids(lab)

    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr_idx += list(idx[:k]); te_idx += list(idx[k:])
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    names = {c: f'g{int(c)}' for c in np.unique(y)}

    if kind == 'dtw':
        raw = raw_loader(user)                 # (N, T, C) raw signal
        m = DTWBaseline()
        m.fit(raw[tr_idx], y[tr_idx], names)
        pred = m.predict(raw[te_idx])
        truth = [names[c] for c in y[te_idx]]
    elif kind in ('svm', 'rf'):
        m = SklearnBaseline(kind)
        m.fit(emb[tr_idx], y[tr_idx])
        pred = [names[c] for c in m.predict(emb[te_idx])]
        truth = [names[c] for c in y[te_idx]]
    elif kind == 'finetune':
        m = FineTuneBaseline()
        m.fit(emb[tr_idx], y[tr_idx])
        pred = [names[c] for c in m.predict(emb[te_idx])]
        truth = [names[c] for c in y[te_idx]]
    else:
        raise ValueError(kind)

    acc = float(np.mean([p == t for p, t in zip(pred, truth)]))
    return {'user': user, 'baseline': kind, 'n_train': n_train,
            'acc': acc, 'n_gestures': int(len(np.unique(y))),
            'requires_per_user_training': kind in ('svm', 'rf', 'finetune', 'dtw'),
            'validates': False}