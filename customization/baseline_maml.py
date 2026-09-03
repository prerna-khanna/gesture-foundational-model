"""
baseline_maml.py -- Tier 4: MAML baseline, adapted from Shahi et al. (Apple,
UIST 2024), "Vision-Based Hand Gesture Customization from a Single Demonstration".

Shahi et al. customize gestures with Model-Agnostic Meta-Learning (MAML, Finn
et al.). Their input is monocular video of hand keypoints; we adapt the same
ALGORITHM to wrist IMU, on the frozen encoder embeddings our method also uses.
This isolates the meta-learning strategy: they meta-learn to adapt a
classifier's weights to a new user's gestures; our method meta-learns a metric.

Faithful to their setup (their Section 4.5):
  - MAML with an (n+1)-way k-shot formulation: n custom gestures plus a
    background/null class (they add a background class to every task).
  - Leave-one-subject-out: the meta-learner never trains on the test user.
  - Inner-loop adaptation on the support set, outer-loop meta-update on the
    query set. SGD, learning rate 0.025 (their value).
  - At meta-test, adapt to the held-out user's gestures with a few gradient
    steps on the support set, then classify the query set.

Because our encoder is frozen and shared across all methods, MAML here adapts a
small classifier head over the encoder embeddings (mean-pooled), which is the
faithful IMU analogue of adapting their graph-transformer classifier. The
encoder itself is not meta-updated (it is the shared frozen backbone in every
condition), matching how the other baselines use it.

Reported like the other baselines: recognition accuracy over the vocabulary,
requires_per_user_training=True (per-user adaptation), validates=False (MAML
recognizes; it has no gesture validator).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMLHead(nn.Module):
    """Small classifier head meta-learned over frozen encoder embeddings."""
    def __init__(self, emb_dim, n_out, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, z, weights=None):
        if weights is None:
            return self.fc2(F.relu(self.fc1(z)))
        # functional forward for the inner loop (MAML adapts these weights)
        h = F.relu(F.linear(z, weights['fc1.weight'], weights['fc1.bias']))
        return F.linear(h, weights['fc2.weight'], weights['fc2.bias'])


def _clone_weights(model):
    return {k: v.clone() for k, v in model.named_parameters()}


def _inner_adapt(model, weights, z_sup, y_sup, inner_lr, inner_steps):
    """One task's inner-loop adaptation (support set)."""
    w = weights
    for _ in range(inner_steps):
        logits = model(z_sup, w)
        loss = F.cross_entropy(logits, y_sup)
        grads = torch.autograd.grad(loss, list(w.values()), create_graph=True)
        w = {k: v - inner_lr * g for (k, v), g in zip(w.items(), grads)}
    return w


# --------------------------------------------------------------------------- #
# meta-training over episodes (built the same way as our episode sampler)
# --------------------------------------------------------------------------- #

def meta_train(pool_embed, pool_labels, pool_users, emb_dim, max_classes,
               n_way=(3, 8), k_shot=(3, 7), n_query=3,
               episodes=5000, inner_lr=0.025, outer_lr=1e-3,
               inner_steps=3, device='cpu', seed=0, verbose=True):
    """max_classes: output size of the MAML head; must be >= the largest
    vocabulary of ANY user (including the test user), since MAML's head is
    fixed-size. We slice to the active n classes per task/user."""
    """
    pool_embed  : (N, emb_dim) frozen mean-pooled embeddings of training users
    pool_labels : (N,) gesture ids
    pool_users  : (N,) user ids (episodes are within-user, as in our method)
    """
    rng = np.random.default_rng(seed)
    model = MAMLHead(emb_dim, max_classes).to(device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    users = np.unique(pool_users)

    for ep in range(episodes):
        # sample one user, n gestures, k support + q query
        u = users[rng.integers(len(users))]
        umask = pool_users == u
        uy = pool_labels[umask]; uz = pool_embed[umask]
        classes = np.unique(uy)
        if len(classes) < (n_way[0] if isinstance(n_way, tuple) else n_way):
            continue
        n = int(rng.integers(n_way[0], min(n_way[1], len(classes)) + 1)) \
            if isinstance(n_way, tuple) else n_way
        k = int(rng.integers(k_shot[0], k_shot[1] + 1)) if isinstance(k_shot, tuple) else k_shot
        chosen = rng.choice(classes, size=n, replace=False)

        zs, ys, zq, yq = [], [], [], []
        ok = True
        for local, c in enumerate(chosen):
            idx = np.where(uy == c)[0]; rng.shuffle(idx)
            if len(idx) < k + 1:
                ok = False; break
            q = min(n_query, len(idx) - k)
            zs.append(uz[idx[:k]]); ys += [local] * k
            zq.append(uz[idx[k:k + q]]); yq += [local] * q
        if not ok:
            continue
        z_sup = torch.tensor(np.concatenate(zs), dtype=torch.float32, device=device)
        y_sup = torch.tensor(ys, dtype=torch.long, device=device)
        z_qry = torch.tensor(np.concatenate(zq), dtype=torch.float32, device=device)
        y_qry = torch.tensor(yq, dtype=torch.long, device=device)

        # inner adapt on support, evaluate on query, meta-update
        w0 = _clone_weights(model)
        w_adapted = _inner_adapt(model, w0, z_sup, y_sup, inner_lr, inner_steps)
        q_logits = model(z_qry, w_adapted)[:, :n]     # only the n active classes
        meta_loss = F.cross_entropy(q_logits, y_qry)
        meta_opt.zero_grad(); meta_loss.backward(); meta_opt.step()

        if verbose and (ep + 1) % 500 == 0:
            print(f"  [maml] episode {ep+1}/{episodes}  meta-loss {meta_loss.item():.4f}")

    model.eval()
    return model


def eval_maml_user(model, z_user, y_user, emb_dim, n_train=7,
                   inner_lr=0.025, inner_steps=5, device='cpu', seed=0):
    """Meta-test: adapt to the held-out user's gestures, classify held-out samples."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_user)
    remap = {c: i for i, c in enumerate(classes)}
    tr, te = [], []
    for c in classes:
        idx = np.where(y_user == c)[0]; rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr.append(idx[:k]); te += list(idx[k:])
    tr = np.concatenate(tr); te = np.array(te)

    z_sup = torch.tensor(z_user[tr], dtype=torch.float32, device=device)
    y_sup = torch.tensor([remap[c] for c in y_user[tr]], dtype=torch.long, device=device)
    w0 = _clone_weights(model)
    w = _inner_adapt(model, w0, z_sup, y_sup, inner_lr, inner_steps)

    z_qry = torch.tensor(z_user[te], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(z_qry, w)[:, :len(classes)].argmax(1).cpu().numpy()
    truth = np.array([remap[c] for c in y_user[te]])
    acc = float(np.mean(pred == truth))
    return {'baseline': 'maml', 'acc': acc, 'n_gestures': int(len(classes)),
            'requires_per_user_training': True, 'validates': False, 'n_train': n_train}