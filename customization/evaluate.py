"""
evaluate.py -- end-to-end customization evaluation, leave-one-user-out.

Reproduces the Ch 7 protocol (7 demos train / 3 held out per gesture, 10 blind
users, 10 self-designed gestures each) and adds the things Ch 7 does not
measure.

    python -m customization.evaluate --heads_dir saved/customization

Reported per user:
  accepted       gestures surviving the validator  (Ch 7 Table 7.4)
  acc_full       accuracy on the ACCEPTED vocabulary, closed-set
  acc_unfiltered accuracy on all 10 gestures with no validator (Ch 7 Table 7.5
                 "No validator", mean 0.8057 -- this is the number to beat,
                 because accepting more gestures is the actual goal)
  acc_open       accuracy with the reject rule active, on gesture data
  bg_fpr         fraction of background windows falsely accepted as a gesture
  add_ms         wall-clock to add one gesture to an existing vocabulary

A validator that rejects everything trivially scores 1.00 on acc_full, so
never read acc_full without `accepted` next to it.

ABLATIONS (--ablation)
  none          full proposed system
  no_null       drop L_null and the reject rule
  no_margin     drop L_margin (ball-overlap check degrades to noise)
  no_rep        drop L_rep
  no_head       nearest-class-mean directly on mean-pooled backbone embeddings,
                i.e. no learned reorganization at all. This is the row
                reviewers will demand: it tests whether g_phi does anything
                the frozen backbone doesn't already do.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customization.backbone import FrozenBackbone, gesture_ids   # noqa: E402
from customization.head import CustomizationHead                 # noqa: E402
from customization.registry import GestureRegistry               # noqa: E402
from customization.meta_train import BLIND_USERS                 # noqa: E402


class IdentityHead(torch.nn.Module):
    """--ablation no_head: mean-pool + L2 normalize, zero learned parameters."""
    def __init__(self, backbone_dim=72, out_dim=None):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.out_dim = backbone_dim
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, h, nucleus_mask=None):
        z = h.mean(dim=1)
        return torch.nn.functional.normalize(z, dim=-1)


def split_user(emb, mask, y, n_train=7, seed=0):
    """Per-gesture split: first n_train demos register, the rest test."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = min(n_train, max(1, len(idx) - 1))
        tr.append(idx[:k]); te.append(idx[k:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    return (emb[tr], mask[tr], y[tr]), (emb[te], mask[te], y[te])


def eval_user(bb, user, head, null_emb, null_mask, corpus_support,
              corpus_groups=None, n_train=7, target_fpr=0.05,
              device='cpu', seed=0, verbose=True):
    emb, mask, lab = bb.encode_dataset(user, return_masks=True)
    y = gesture_ids(lab)
    (etr, mtr, ytr), (ete, mte, yte) = split_user(emb, mask, y, n_train, seed)

    reg = GestureRegistry(head, device=device)
    z_tr = reg.embed(etr, mtr)
    z_te = reg.embed(ete, mte)
    z_null = reg.embed(null_emb, null_mask) if null_emb is not None and len(null_emb) else None

    # All thresholds come from users the head neither trained on nor is being
    # tested on. Every one of tau_rep / tau_null_ratio / min_sep_ratio is a
    # percentile of real gesture behaviour, so each carries an explicit
    # false-rejection budget instead of being a hand-picked constant.
    def enc(zc):
        # saved corpora round-trip through np.save(dtype=object); coerce back
        zc = np.asarray(zc, dtype=np.float32)
        return reg.embed(zc, None) if zc.ndim == 3 else zc
    corpus_z = [enc(zc) for zc in corpus_support]
    groups_z = ([[enc(zc) for zc in g] for g in corpus_groups]
                if corpus_groups else None)
    reg.calibrate(corpus_z, null_z=z_null, corpus_groups=groups_z, verbose=verbose)

    # ---- register gestures one at a time, exactly as a user would ---- #
    decisions, add_times = [], []
    for c in sorted(np.unique(ytr)):
        zc = z_tr[ytr == c]
        t0 = time.perf_counter()
        d = reg.add_gesture(f'g{int(c)}', zc, validate=True, null_z=z_null)
        add_times.append((time.perf_counter() - t0) * 1000)
        decisions.append(d)

    accepted = [d['name'] for d in decisions if d['added']]
    rejected = [(d['name'], d['message']) for d in decisions if not d['added']]
    if not accepted:
        return None

    if z_null is not None:
        reg.calibrate_reject(z_null, target_fpr=target_fpr, verbose=False)

    # ---- closed-set accuracy on the accepted vocabulary --------------- #
    keep_ids = {int(n[1:]) for n in accepted}
    sel = np.isin(yte, list(keep_ids))
    pred, _ = reg.predict(z_te[sel], allow_reject=False)
    truth = [f'g{int(c)}' for c in yte[sel]]
    acc_full = float(np.mean([p == t for p, t in zip(pred, truth)])) if sel.any() else 0.0

    # ---- open-set accuracy (reject rule live) ------------------------- #
    pred_o, _ = reg.predict(z_te[sel], allow_reject=True)
    acc_open = float(np.mean([p == t for p, t in zip(pred_o, truth)])) if sel.any() else 0.0

    # ---- background false-accept rate --------------------------------- #
    bg_fpr = None
    if z_null is not None:
        lab_n, _ = reg.predict(z_null, allow_reject=True)
        bg_fpr = float(np.mean([l is not None for l in lab_n]))

    # ---- unfiltered baseline (Ch 7 Table 7.5 column 1) ---------------- #
    reg_u = GestureRegistry(head, device=device)
    for c in sorted(np.unique(ytr)):
        reg_u.add_gesture(f'g{int(c)}', z_tr[ytr == c], validate=False)
    pred_u, _ = reg_u.predict(z_te, allow_reject=False)
    truth_u = [f'g{int(c)}' for c in yte]
    acc_unfiltered = float(np.mean([p == t for p, t in zip(pred_u, truth_u)]))

    if verbose:
        print(f"  {user:<10} accepted {len(accepted)}/{len(decisions)} "
              f"| acc_full {acc_full:.4f} | acc_open {acc_open:.4f} "
              f"| unfiltered {acc_unfiltered:.4f} "
              f"| bg_fpr {('%.3f' % bg_fpr) if bg_fpr is not None else 'n/a'} "
              f"| add {np.mean(add_times):.1f} ms")
        for n, msg in rejected:
            print(f"      rejected {n}: {msg}")

    return {'user': user, 'accepted': len(accepted), 'total': len(decisions),
            'rejected': rejected, 'acc_full': acc_full, 'acc_open': acc_open,
            'acc_unfiltered': acc_unfiltered, 'bg_fpr': bg_fpr,
            'add_ms_mean': float(np.mean(add_times)),
            'add_ms_last': float(add_times[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads_dir', default='saved/customization')
    ap.add_argument('--pretrain_dataset', default='blind_user')
    ap.add_argument('--version', default='20_120')
    ap.add_argument('--n_train', type=int, default=7)
    ap.add_argument('--target_fpr', type=float, default=0.05)
    ap.add_argument('--ablation', default='none',
                    choices=['none', 'no_null', 'no_margin', 'no_rep', 'no_head'])
    ap.add_argument('--out', default='customization_results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bb = FrozenBackbone(pretrain_dataset=args.pretrain_dataset,
                        version=args.version, device=device)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rows = []
    for user in BLIND_USERS:
        ckpt = os.path.join(root, args.heads_dir, f'head_loo_{user}.pt')
        if args.ablation == 'no_head':
            head = IdentityHead(bb.hidden).to(device)
            calib = os.path.join(root, args.heads_dir, f'head_loo_{user}_calib.npz')
        elif not os.path.exists(ckpt):
            print(f"  {user:<10} SKIP -- no head at {ckpt} "
                  f"(run meta_train.py --held_out {user})")
            continue
        else:
            head = CustomizationHead(backbone_dim=bb.hidden)
            head.load_state_dict(torch.load(ckpt, map_location='cpu'))
            head.to(device).eval()
            calib = ckpt.replace('.pt', '_calib.npz')

        null_emb = null_mask = None
        if os.path.exists(calib):
            z = np.load(calib)
            if len(z['null']):
                null_emb, null_mask = z['null'], z['null_mask']

        corpus_p = ckpt.replace('.pt', '_corpus.npy')
        corpus = ([np.asarray(z, dtype=np.float32)
                   for z in np.load(corpus_p, allow_pickle=True)]
                  if os.path.exists(corpus_p) else [])
        groups_p = ckpt.replace('.pt', '_corpus_groups.npy')
        groups = ([[np.asarray(z, dtype=np.float32) for z in g]
                   for g in np.load(groups_p, allow_pickle=True)]
                  if os.path.exists(groups_p) else None)

        r = eval_user(bb, user, head, null_emb, null_mask, corpus,
                      corpus_groups=groups, n_train=args.n_train,
                      target_fpr=args.target_fpr, device=device)
        if r:
            r['ablation'] = args.ablation
            rows.append(r)

    if not rows:
        print("No users evaluated.")
        return

    def m(k):
        v = [r[k] for r in rows if r[k] is not None]
        return float(np.mean(v)) if v else float('nan')

    print("\n" + "=" * 78)
    print(f"ABLATION: {args.ablation}   (n = {len(rows)} users)")
    print(f"  accepted        {m('accepted'):.1f} / 10")
    print(f"  acc_full        {m('acc_full'):.4f}   <- Ch 7 Table 7.4 full validator = 0.9614")
    print(f"  acc_open        {m('acc_open'):.4f}   (reject rule live)")
    print(f"  acc_unfiltered  {m('acc_unfiltered'):.4f}   <- Ch 7 Table 7.5 no-validator = 0.8057")
    print(f"  bg_fpr          {m('bg_fpr'):.4f}")
    print(f"  add gesture     {m('add_ms_mean'):.1f} ms")
    print("=" * 78)

    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
