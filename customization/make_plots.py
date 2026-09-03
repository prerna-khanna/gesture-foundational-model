"""
make_plots.py -- generate the paper figures from results/*.json.

Matches the figure style of the UniMotion paper (grouped accuracy bars,
class-count degradation curves, ablation, per-user breakdown). Every figure
reads only from saved JSON; nothing is recomputed.

    python -m customization.make_plots          # writes figures/*.pdf

Figures produced (only those whose JSON exists):
  fig_baselines.pdf     our method vs baselines (bar), with the two axes that
                        matter: accuracy AND what each method requires.
  fig_validators.pdf    our validator vs Xu's validator: acceptance vs accuracy.
  fig_ngestures.pdf     accuracy as vocabulary grows (our method).
  fig_ablation.pdf      loss + validator-component ablation (grouped bars).
  fig_thresholds.pdf    threshold sensitivity: acceptance vs accuracy.
  fig_peruser.pdf       per-user accepted/accuracy (transparency figure).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

# UniMotion-style palette: muted, print-friendly
C_OURS = '#2c5f8a'      # deep blue for our method
C_BASE = '#b0b7c3'      # grey for baselines
C_ACC = '#3e6b4f'       # green (accuracy)
C_F1 = '#c98a3c'        # amber (F1 / secondary)
C_REJ = '#b24a3c'       # brick (rejection / acceptance)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})


def _load(name):
    p = os.path.join(RES, f'{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------- #
# Figure 1: baselines -- accuracy bar, our method highlighted
# --------------------------------------------------------------------------- #

def fig_baselines():
    main = _load('main'); bl = _load('baselines')
    xu = _load('xu'); wg = _load('watchguardian'); mm = _load('maml')
    if not (main and bl):
        return
    # our method: report acc_unfiltered (full-vocab, fair to baselines) AND acc_full (validated)
    ours_unf = main['none']['acc_unfiltered']
    ours_val = main['none']['acc_full']

    labels, accs, colors = [], [], []
    for kind in ['svm', 'rf', 'finetune', 'ncm']:
        if kind in bl:
            labels.append(kind.upper() if kind in ('svm', 'rf') else kind.capitalize())
            accs.append(bl[kind]['acc']); colors.append(C_BASE)
    if xu:
        labels.append('Xu et al.'); accs.append(xu['xu']['acc']); colors.append(C_BASE)
    if mm:
        labels.append('MAML'); accs.append(mm['maml']['acc']); colors.append(C_BASE)
    if wg:
        labels.append('WatchGuardian'); accs.append(wg['watchguardian']['acc']); colors.append(C_BASE)
    labels.append('Ours\n(unfiltered)'); accs.append(ours_unf); colors.append('#7fa8c9')
    labels.append('Ours\n(validated)'); accs.append(ours_val); colors.append(C_OURS)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, edgecolor='white', width=0.7)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.012, f'{a:.2f}',
                ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Recognition accuracy'); ax.set_ylim(0, 1.05)
    ax.set_title('Recognition accuracy on blind-user gestures')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_baselines.pdf')); plt.close(fig)
    print('wrote fig_baselines.pdf')


# --------------------------------------------------------------------------- #
# Figure 2: our validator vs Xu's validator -- acceptance vs accuracy
# --------------------------------------------------------------------------- #

def fig_validators():
    main = _load('main'); xv = _load('xu_validated')
    if not (main and xv):
        return
    methods = ['Our validator', "Xu's validator"]
    accepted = [main['none']['accepted'], xv['xu_validated']['accepted']]
    total = [main['none']['total'], xv['xu_validated']['total']]
    acc = [main['none']['acc_full'], xv['xu_validated']['acc_full']]

    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(methods)); w = 0.35
    b1 = ax1.bar(x - w / 2, [a / t for a, t in zip(accepted, total)], w,
                 color=C_REJ, label='Acceptance rate', edgecolor='white')
    ax1.set_ylabel('Acceptance rate', color=C_REJ); ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=C_REJ)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, acc, w, color=C_OURS, label='Accuracy (accepted)', edgecolor='white')
    ax2.set_ylabel('Accuracy on accepted', color=C_OURS); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=C_OURS); ax2.grid(False)
    for b, v in zip(b1, [a / t for a, t in zip(accepted, total)]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.0%}', ha='center', fontsize=9, color=C_REJ)
    for b, v in zip(b2, acc):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, color=C_OURS)
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_title("Validators: what they accept, and how well it recognizes")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_validators.pdf')); plt.close(fig)
    print('wrote fig_validators.pdf')


# --------------------------------------------------------------------------- #
# Figure 3: accuracy vs number of gestures (UniMotion Fig 11 style)
# --------------------------------------------------------------------------- #

def fig_ngestures():
    ng = _load('ngestures')
    if not ng:
        return
    ns = sorted([int(k) for k in ng if k.isdigit()])
    accs = [ng[str(n)]['acc'] for n in ns]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, accs, '-o', color=C_OURS, lw=2, markersize=6)
    for n, a in zip(ns, accs):
        ax.text(n, a + 0.012, f'{a:.2f}', ha='center', fontsize=8)
    ax.set_xlabel('Number of gestures'); ax.set_ylabel('Accuracy')
    ax.set_ylim(min(accs) - 0.08, 1.02)
    ax.set_title('Accuracy as the vocabulary grows')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_ngestures.pdf')); plt.close(fig)
    print('wrote fig_ngestures.pdf')


# --------------------------------------------------------------------------- #
# Figure 4: ablation (loss + validator-component)
# --------------------------------------------------------------------------- #

def fig_ablation():
    main = _load('main')
    if not main:
        return
    # validator-component ablation: acceptance + accuracy
    conds = ['val_full', 'val_no_rep', 'val_no_dist', 'val_no_null']
    names = ['Full', 'no rep.', 'no distinct.', 'no null']
    conds = [c for c in conds if c in main]
    names = names[:len(conds)]
    accept = [main[c]['accepted'] / main[c]['total'] for c in conds]
    acc = [main[c]['acc_full'] for c in conds]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(conds)); w = 0.38
    ax1.bar(x - w / 2, accept, w, color=C_REJ, label='Acceptance', edgecolor='white')
    ax1.set_ylabel('Acceptance rate', color=C_REJ); ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=C_REJ)
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, acc, w, color=C_OURS, label='Accuracy', edgecolor='white')
    ax2.set_ylabel('Accuracy', color=C_OURS); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=C_OURS); ax2.grid(False)
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_title('Validator-component ablation')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_ablation.pdf')); plt.close(fig)
    print('wrote fig_ablation.pdf')


# --------------------------------------------------------------------------- #
# Figure 5: threshold sensitivity
# --------------------------------------------------------------------------- #

def fig_thresholds():
    th = _load('thresholds_repstat') or _load('thresholds')
    if not th:
        return
    pcts = sorted([int(k) for k in th if k.isdigit()])
    accept = [th[str(p)]['accept_rate'] for p in pcts]
    acc = [th[str(p)]['acc_full'] for p in pcts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pcts, accept, '-o', color=C_REJ, lw=2, label='Acceptance rate')
    ax.plot(pcts, acc, '-s', color=C_OURS, lw=2, label='Accuracy')
    ax.set_xlabel('Repeatability percentile'); ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    ax.set_title('Threshold sensitivity')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_thresholds.pdf')); plt.close(fig)
    print('wrote fig_thresholds.pdf')


# --------------------------------------------------------------------------- #
# Figure 6: per-user transparency (accepted count + accuracy)
# --------------------------------------------------------------------------- #

def fig_peruser():
    main = _load('main')
    if not main:
        return
    rows = main['none']['rows']
    users = [r['user'] for r in rows]
    accepted = [r['accepted'] for r in rows]
    total = [r['total'] for r in rows]
    acc = [r['acc_full'] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(users)); w = 0.38
    ax1.bar(x - w / 2, [a / t for a, t in zip(accepted, total)], w,
            color=C_REJ, edgecolor='white', label='Accept rate')
    ax1.set_ylabel('Acceptance rate', color=C_REJ); ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor=C_REJ)
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, acc, w, color=C_OURS, edgecolor='white', label='Accuracy')
    ax2.set_ylabel('Accuracy (accepted)', color=C_OURS); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelcolor=C_OURS); ax2.grid(False)
    ax1.set_xticks(x); ax1.set_xticklabels(users, rotation=30, ha='right', fontsize=9)
    ax1.set_title('Per-user acceptance and accuracy')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_peruser.pdf')); plt.close(fig)
    print('wrote fig_peruser.pdf')


def main():
    fig_baselines()
    fig_validators()
    fig_ngestures()
    fig_ablation()
    fig_thresholds()
    fig_peruser()
    print(f'\nfigures in {FIG}/')


if __name__ == '__main__':
    main()