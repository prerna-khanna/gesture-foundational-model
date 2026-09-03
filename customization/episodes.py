"""
episodes.py -- meta-training task construction.

Two jobs:

1. EpisodeSampler -- draws n-way k-shot episodes. Episodes are WITHIN-USER
   (support and query from the same person), because that is the deployment
   condition: a user registers their own gestures and then uses them. What
   must generalize is across *users* and across *class identities*, and that
   is exactly what leave-one-user-out episodic training buys.

2. MetaAugmentor -- synthesizes novel gesture CLASSES from existing raw data.

   Why this matters more than ordinary augmentation: with ~60 real gesture
   classes across all pooled datasets, the head will meta-overfit to the task
   distribution long before it learns a general separation rule. Apple hit the
   same wall with only six two-handed classes and solved it by splicing two
   one-handed gestures into a synthetic two-handed class.

   Their splice does not transfer (single wrist IMU, one hand). The transfer
   is the *principle*: synthesize the class families that customization
   actually fails on. Ch 7.4.3 says rejections are almost entirely
     (a) repetition-count variants (tap / double tap / triple tap), and
     (b) same short motion, different direction (wrist left / right / down).
   So we synthesize exactly those families. If the head is meta-trained on
   thousands of "differs only by repeat count" and "differs only by axis sign"
   class pairs, it may learn a space that separates them -- turning gestures
   the current system rejects into gestures the user gets to keep.

   Augmentation happens on RAW IMU and is then re-encoded by the frozen
   backbone. Doing it in embedding space would be wrong: the backbone is
   nonlinear, and a time-warp in embedding space is not the embedding of a
   time-warped signal.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# raw-level class synthesis
# --------------------------------------------------------------------------- #

class MetaAugmentor:
    """Produces synthetic gesture classes from a base class's raw samples."""

    def __init__(self, seq_len=120, rng=None):
        self.seq_len = seq_len
        self.rng = rng or np.random.default_rng(0)

    # -- helpers ------------------------------------------------------- #

    @staticmethod
    def _energy(x):
        return np.sqrt((x ** 2).sum(axis=-1))

    def _nucleus_span(self, x, thresh_frac=0.35):
        """Crude energy-based nucleus span; mirrors features.detect_nucleus intent."""
        e = self._energy(x)
        e = (e - e.min()) / (np.ptp(e) + 1e-8)
        above = np.where(e > thresh_frac)[0]
        if len(above) < 3:
            return 0, min(self.seq_len, 30)
        return int(above[0]), int(above[-1]) + 1

    def _fit_length(self, x):
        T = self.seq_len
        if len(x) == T:
            return x
        if len(x) > T:
            return x[:T]
        pad = np.zeros((T - len(x), x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    # -- class-level transforms ---------------------------------------- #

    def repeat_nucleus(self, x, n_repeat=2, gap=6):
        """'single X' -> 'double/triple X'. The dominant rejection family."""
        s, e = self._nucleus_span(x)
        nuc = x[s:e]
        if len(nuc) < 2:
            return x.copy()
        gap_block = np.zeros((gap, x.shape[1]), dtype=x.dtype)
        gap_block[:, :3] = x[:s].mean(axis=0)[:3] if s > 0 else 0.0  # hold gravity
        body = [x[:s]]
        for i in range(n_repeat):
            body.append(nuc)
            if i < n_repeat - 1:
                body.append(gap_block)
        body.append(x[e:])
        return self._fit_length(np.concatenate(body, axis=0))

    def flip_axis(self, x, axis=None):
        """'wrist left' -> 'wrist right'. Negates one gyro axis + its accel pair."""
        y = x.copy()
        if axis is None:
            axis = int(np.abs(y[:, 3:6]).mean(axis=0).argmax())
        y[:, 3 + axis] *= -1.0
        y[:, axis] *= -1.0
        return y

    def time_warp(self, x, factor=1.5):
        """'X' -> 'slow X' / 'fast X'. Same trajectory, different tempo."""
        T = len(x)
        src = np.linspace(0, T - 1, T)
        dst = np.linspace(0, T - 1, max(4, int(T / factor)))
        warped = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)
        return self._fit_length(warped.astype(x.dtype))

    def scale_amplitude(self, x, factor=0.5):
        """'big X' -> 'small X'. Gyro scaled; gravity component of accel preserved."""
        y = x.copy()
        g = y[:, :3].mean(axis=0, keepdims=True)
        y[:, :3] = g + (y[:, :3] - g) * factor
        y[:, 3:6] *= factor
        return y

    # -- bank builder --------------------------------------------------- #

    def synthesize(self, raw, labels_g, per_class_variants=3, max_new_classes=None):
        """
        raw      : (N, T, C) raw IMU
        labels_g : (N,) integer gesture ids

        Returns (syn_raw, syn_labels) where syn_labels are NEW class ids offset
        past the real ones. Each synthetic class is a coherent transform of one
        base class applied to ALL of that class's samples, so within-class
        variation is preserved -- these are genuine classes, not noise.
        """
        ops = [
            ('rep2', lambda s: self.repeat_nucleus(s, 2)),
            ('rep3', lambda s: self.repeat_nucleus(s, 3)),
            ('flip', lambda s: self.flip_axis(s)),
            ('slow', lambda s: self.time_warp(s, 1.6)),
            ('fast', lambda s: self.time_warp(s, 0.65)),
            ('small', lambda s: self.scale_amplitude(s, 0.45)),
        ]
        base_classes = np.unique(labels_g)
        next_id = int(labels_g.max()) + 1
        out_raw, out_lab = [], []

        for c in base_classes:
            idx = np.where(labels_g == c)[0]
            if len(idx) < 2:
                continue
            chosen = self.rng.choice(len(ops), size=min(per_class_variants, len(ops)),
                                     replace=False)
            for oi in chosen:
                _, fn = ops[oi]
                for i in idx:
                    out_raw.append(fn(raw[i]))
                    out_lab.append(next_id)
                next_id += 1
                if max_new_classes and (next_id - int(labels_g.max()) - 1) >= max_new_classes:
                    break
            if max_new_classes and (next_id - int(labels_g.max()) - 1) >= max_new_classes:
                break

        if not out_raw:
            return np.empty((0,) + raw.shape[1:], dtype=raw.dtype), np.empty((0,), dtype=int)
        return np.stack(out_raw).astype(raw.dtype), np.asarray(out_lab, dtype=int)


# --------------------------------------------------------------------------- #
# episode sampling
# --------------------------------------------------------------------------- #

class TaskPool:
    """
    Holds encoded gesture data grouped by (source, user, class).

    A 'task group' is one (source, user) pair -- the set of classes one person
    performed on one device. Episodes never mix groups.
    """

    def __init__(self):
        self.groups = {}       # key -> {'emb':(N,T,H), 'mask':(N,T), 'y':(N,), 'classes':[...]}

    def add(self, key, emb, y, mask=None, min_per_class=3):
        y = np.asarray(y).astype(int)
        keep_classes = [c for c in np.unique(y) if (y == c).sum() >= min_per_class]
        if len(keep_classes) < 2:
            return
        sel = np.isin(y, keep_classes)
        self.groups[key] = {
            'emb': emb[sel],
            'mask': mask[sel] if mask is not None else None,
            'y': y[sel],
            'classes': list(keep_classes),
        }

    def summary(self):
        return {k: (len(v['classes']), len(v['y'])) for k, v in self.groups.items()}

    @property
    def n_classes_total(self):
        return sum(len(v['classes']) for v in self.groups.values())


class EpisodeSampler:
    """
    Draws (support, query, null) episodes.

    n_way is sampled per episode so the head sees vocabularies of the size real
    users ask for. Ch 7.2.2: participants wanted 5-12 gestures, clustering at 10.
    """

    def __init__(self, pool, null_emb=None, null_mask=None,
                 n_way=(3, 10), k_shot=(3, 7), n_query=3, n_null=8, seed=0):
        self.pool = pool
        self.null_emb = null_emb
        self.null_mask = null_mask
        self.n_way = n_way if isinstance(n_way, tuple) else (n_way, n_way)
        self.k_shot = k_shot if isinstance(k_shot, tuple) else (k_shot, k_shot)
        self.n_query = n_query
        self.n_null = n_null
        self.rng = np.random.default_rng(seed)
        self.keys = [k for k, v in pool.groups.items() if len(v['classes']) >= self.n_way[0]]
        if not self.keys:
            raise ValueError(
                f"No task group has >= {self.n_way[0]} usable classes. "
                f"Pool summary: {pool.summary()}")

    def sample(self):
        key = self.keys[self.rng.integers(len(self.keys))]
        g = self.pool.groups[key]

        n_way = int(self.rng.integers(self.n_way[0], min(self.n_way[1], len(g['classes'])) + 1))
        k = int(self.rng.integers(self.k_shot[0], self.k_shot[1] + 1))
        classes = self.rng.choice(g['classes'], size=n_way, replace=False)

        s_idx, s_y, q_idx, q_y = [], [], [], []
        for local, c in enumerate(classes):
            idx = np.where(g['y'] == c)[0]
            self.rng.shuffle(idx)
            k_eff = min(k, max(1, len(idx) - 1))
            q_eff = min(self.n_query, len(idx) - k_eff)
            s_idx += list(idx[:k_eff]);            s_y += [local] * k_eff
            q_idx += list(idx[k_eff:k_eff + q_eff]); q_y += [local] * q_eff

        ep = {
            'support_emb': g['emb'][s_idx],
            'support_mask': g['mask'][s_idx] if g['mask'] is not None else None,
            'support_y': np.asarray(s_y),
            'query_emb': g['emb'][q_idx],
            'query_mask': g['mask'][q_idx] if g['mask'] is not None else None,
            'query_y': np.asarray(q_y),
            'n_classes': n_way,
            'group': key,
        }

        if self.null_emb is not None and self.n_null > 0:
            ni = self.rng.choice(len(self.null_emb), size=min(self.n_null, len(self.null_emb)),
                                 replace=False)
            ep['null_emb'] = self.null_emb[ni]
            ep['null_mask'] = self.null_mask[ni] if self.null_mask is not None else None
        else:
            ep['null_emb'] = None
            ep['null_mask'] = None
        return ep
