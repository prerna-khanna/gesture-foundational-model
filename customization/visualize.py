"""
visualize.py -- figures that show what the head does to the embedding space.

    python -m customization.visualize --user Nihal
    python -m customization.visualize --all          # cohort-level summary

Produces, per user:
  fig_tsne_<user>.png         raw vs head embeddings, side by side, same gestures
  fig_separation_<user>.png   quantitative: intra/inter distance, silhouette
And cohort-level:
  fig_cohort_separation.png   the "head helps" bar chart across all users

The story every figure tells: the frozen backbone already knows a lot, but its
space mixes gesture-defining directions with nuisance ones (posture, tempo,
placement). The head is a learned metric that collapses the nuisance and keeps
the gesture-defining directions -- so SAME-gesture samples tighten and
DIFFERENT-gesture clusters separate, for gestures the head never trained on.

Key methodological point for the paper: t-SNE is fit SEPARATELY on raw and head
embeddings (you cannot share a projection across two different spaces), so read
t-SNE panels qualitatively -- cluster *structure*, not absolute positions. The
quantitative claim lives in fig_separation, which is computed in the real
embedding space, not in the 2-D projection.
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customization.backbone import FrozenBackbone, gesture_ids   # noqa: E402
from customization.head import CustomizationHead                 # noqa: E402
from customization.registry import GestureRegistry               # noqa: E402
from customization.evaluate import IdentityHead                  # noqa: E402
from customization.meta_train import BLIND_USERS                 # noqa: E402


# --------------------------------------------------------------------------- #
# metrics computed in the REAL embedding space (not the 2-D projection)
# --------------------------------------------------------------------------- #

def separation_metrics(z, y):
    """
    z: (N, m) unit-norm embeddings, y: (N,) labels.

    Returns a dict of space-quality numbers that do NOT depend on any 2-D
    projection, so they are the defensible version of what t-SNE shows.

      intra   mean distance of a sample to its own class mean (want small)
      inter   mean distance between class means            (want large)
      ratio   inter / intra                                (want large)
      sil     silhouette score                             (want high, [-1,1])
      knn     leave-one-out 1-NN accuracy                  (want high)
    """
    from sklearn.metrics import silhouette_score

    classes = np.unique(y)
    protos = {c: z[y == c].mean(0) for c in classes}
    protos = {c: p / (np.linalg.norm(p) + 1e-8) for c, p in protos.items()}

    intra = np.mean([np.linalg.norm(z[i] - protos[y[i]]) for i in range(len(z))])
    pm = np.stack([protos[c] for c in classes])
    dd = np.linalg.norm(pm[:, None] - pm[None, :], axis=2)
    inter = dd[~np.eye(len(classes), dtype=bool)].mean()

    # LOO 1-NN
    correct = 0
    for i in range(len(z)):
        d = np.linalg.norm(z - z[i], axis=1); d[i] = np.inf
        correct += int(y[np.argmin(d)] == y[i])
    knn = correct / len(z)

    try:
        sil = silhouette_score(z, y)
    except Exception:
        sil = float('nan')

    return {'intra': intra, 'inter': inter, 'ratio': inter / (intra + 1e-8),
            'sil': sil, 'knn': knn}


# --------------------------------------------------------------------------- #
# per-user t-SNE: raw vs head, same gestures, side by side
# --------------------------------------------------------------------------- #

def _tsne(z, seed=0):
    from sklearn.manifold import TSNE
    n = len(z)
    perp = max(5, min(30, (n - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init='pca',
                random_state=seed, max_iter=1000).fit_transform(z.astype(np.float64))


def plot_user_tsne(user, z_raw, z_head, y, out_dir):
    classes = np.unique(y)
    cmap = plt.get_cmap('tab20' if len(classes) > 10 else 'tab10')
    color = {c: cmap(i % cmap.N) for i, c in enumerate(classes)}

    p_raw, p_head = _tsne(z_raw), _tsne(z_head)
    m_raw = separation_metrics(z_raw, y)
    m_head = separation_metrics(z_head, y)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, p, m, title in [
        (ax[0], p_raw, m_raw, 'Frozen backbone (mean-pooled)'),
        (ax[1], p_head, m_head, 'After customization head $g_\\phi$')]:
        for c in classes:
            s = y == c
            a.scatter(p[s, 0], p[s, 1], color=color[c], s=42,
                      edgecolors='white', linewidths=0.5, label=f'g{int(c)}')
            mx, my = p[s, 0].mean(), p[s, 1].mean()
            a.text(mx, my, f'g{int(c)}', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        a.set_title(f"{title}\nsep ratio {m['ratio']:.2f} | "
                    f"silhouette {m['sil']:.2f} | 1-NN {m['knn']:.0%}",
                    fontsize=11)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor('#cccccc')

    fig.suptitle(f"{user}: embedding space, same {len(classes)} gestures "
                 f"(t-SNE fit separately per panel -- read structure, not position)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig_tsne_{user}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path, m_raw, m_head


def plot_user_separation(user, m_raw, m_head, out_dir):
    keys = ['ratio', 'sil', 'knn']
    labels = ['inter/intra\ndist ratio', 'silhouette', 'LOO 1-NN\naccuracy']
    raw = [m_raw[k] for k in keys]; head = [m_head[k] for k in keys]
    x = np.arange(len(keys)); w = 0.36

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, raw, w, label='frozen backbone', color='#b0b7c3')
    b2 = ax.bar(x + w/2, head, w, label='+ head $g_\\phi$', color='#3b6fb0')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"{user}: space quality, raw vs head")
    ax.legend(frameon=False)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'{b.get_height():.2f}', ha='center', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig_separation_{user}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# per-gesture attention: which frames the head pools over
# --------------------------------------------------------------------------- #

def plot_attention(user, bb, head, out_dir, max_gestures=6):
    """
    Shows the nucleus-attention weights across the 120 frames for one example
    of each gesture -- evidence that the head focuses on the gesture nucleus
    rather than the whole window (the tap/double-tap discrimination story).
    """
    import torch
    emb, mask, lab = bb.encode_dataset(user, return_masks=True)
    y = gesture_ids(lab)
    classes = np.unique(y)[:max_gestures]

    fig, axes = plt.subplots(len(classes), 1, figsize=(9, 1.3 * len(classes)),
                             sharex=True)
    if len(classes) == 1:
        axes = [axes]
    with torch.no_grad():
        for a, c in zip(axes, classes):
            i = np.where(y == c)[0][0]
            h = torch.as_tensor(emb[i:i+1], dtype=torch.float32)
            nm = torch.as_tensor(mask[i:i+1], dtype=torch.float32)
            raw_score = head.pool.score(h).squeeze(-1)
            att = torch.softmax(raw_score + head.pool.nucleus_bias * nm, dim=1).squeeze(0).numpy()
            # learned-only attention (no nucleus prior) -- shows what g_phi adds
            att_learned = torch.softmax(raw_score, dim=1).squeeze(0).numpy()
            a.fill_between(range(len(att)), att, color='#3b6fb0', alpha=0.6)
            a.plot(att_learned, color='#2a2a2a', lw=1.1, alpha=0.9)
            a.plot(mask[i] * att.max(), color='#d0743c', lw=1.2, alpha=0.85)
            a.set_ylabel(f'g{int(c)}', rotation=0, ha='right', va='center', fontsize=9)
            a.set_yticks([])
            a.spines[['top', 'right', 'left']].set_visible(False)
    axes[-1].set_xlabel('frame (0-119)')
    legend = [Line2D([0], [0], color='#3b6fb0', lw=6, alpha=0.6, label='final attention'),
              Line2D([0], [0], color='#2a2a2a', lw=1.5, label='learned score (no nucleus prior)'),
              Line2D([0], [0], color='#d0743c', lw=1.5, label='detected nucleus')]
    fig.legend(handles=legend, loc='upper right', frameon=False, fontsize=9)
    fig.suptitle(f"{user}: where the head attends across the window", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig_attention_{user}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# cohort-level: the headline "head helps" figure
# --------------------------------------------------------------------------- #

def plot_cohort(rows, out_dir):
    users = [r['user'] for r in rows]
    x = np.arange(len(users)); w = 0.38

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))

    raw_ratio = [r['raw']['ratio'] for r in rows]
    head_ratio = [r['head']['ratio'] for r in rows]
    ax[0].bar(x - w/2, raw_ratio, w, label='frozen backbone', color='#b0b7c3')
    ax[0].bar(x + w/2, head_ratio, w, label='+ head', color='#3b6fb0')
    ax[0].set_ylabel('inter/intra distance ratio')
    ax[0].set_title('Class separation improves with the head, per user '
                    '(held-out gestures)')
    ax[0].set_xticks(x); ax[0].set_xticklabels(users, rotation=30, ha='right')
    ax[0].legend(frameon=False); ax[0].spines[['top', 'right']].set_visible(False)

    raw_knn = [r['raw']['knn'] for r in rows]
    head_knn = [r['head']['knn'] for r in rows]
    ax[1].bar(x - w/2, raw_knn, w, label='frozen backbone', color='#b0b7c3')
    ax[1].bar(x + w/2, head_knn, w, label='+ head', color='#3b6fb0')
    ax[1].set_ylabel('LOO 1-NN accuracy'); ax[1].set_ylim(0, 1)
    ax[1].set_title('Nearest-neighbour recoverability, per user')
    ax[1].set_xticks(x); ax[1].set_xticklabels(users, rotation=30, ha='right')
    ax[1].legend(frameon=False); ax[1].spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_cohort_separation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #

def load_head(user, bb, heads_dir, root):
    import torch
    ck = os.path.join(root, heads_dir, f'head_loo_{user}.pt')
    if not os.path.exists(ck):
        return None
    head = CustomizationHead(backbone_dim=bb.hidden)
    head.load_state_dict(torch.load(ck, map_location='cpu')); head.eval()
    return head


def embed_user(bb, user, head):
    emb, mask, lab = bb.encode_dataset(user, return_masks=True)
    y = gesture_ids(lab)
    reg = GestureRegistry(head)
    z_head = reg.embed(emb, mask)
    reg_raw = GestureRegistry(IdentityHead(bb.hidden))
    z_raw = reg_raw.embed(emb, mask)
    return z_raw, z_head, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--heads_dir', default='saved/customization')
    ap.add_argument('--pretrain_dataset', default='blind_user')
    ap.add_argument('--version', default='20_120')
    ap.add_argument('--out_dir', default='figures')
    ap.add_argument('--attention', action='store_true',
                    help='also render per-gesture attention strips')
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bb = FrozenBackbone(pretrain_dataset=args.pretrain_dataset, version=args.version)

    targets = BLIND_USERS if args.all else [args.user or 'Nihal']
    cohort = []
    for user in targets:
        head = load_head(user, bb, args.heads_dir, root)
        if head is None:
            print(f"  {user}: no head, skipping")
            continue
        z_raw, z_head, y = embed_user(bb, user, head)

        p, m_raw, m_head = plot_user_tsne(user, z_raw, z_head, y, out_dir)
        ps = plot_user_separation(user, m_raw, m_head, out_dir)
        print(f"  {user}: sep ratio {m_raw['ratio']:.2f} -> {m_head['ratio']:.2f} | "
              f"1-NN {m_raw['knn']:.0%} -> {m_head['knn']:.0%}  [{os.path.basename(p)}]")
        if args.attention:
            plot_attention(user, bb, head, out_dir)
        cohort.append({'user': user, 'raw': m_raw, 'head': m_head})

    if len(cohort) > 1:
        p = plot_cohort(cohort, out_dir)
        r = np.mean([c['raw']['ratio'] for c in cohort])
        h = np.mean([c['head']['ratio'] for c in cohort])
        rk = np.mean([c['raw']['knn'] for c in cohort])
        hk = np.mean([c['head']['knn'] for c in cohort])
        print(f"\nCOHORT: sep ratio {r:.2f} -> {h:.2f} | "
              f"1-NN {rk:.0%} -> {hk:.0%}  [{os.path.basename(p)}]")
    print(f"\nfigures in {out_dir}/")


if __name__ == '__main__':
    main()