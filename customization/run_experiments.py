"""
run_experiments.py -- Tier 1 experiment driver.

Produces the tables and sweeps every reviewer of a gesture-customization paper
will look for (matched to Xu et al., WatchGuardian, and Shahi et al.):

  --exp main            our method + loss ablations + validator-component
                        ablation, one row per condition (the headline table)
  --exp baselines       DTW / SVM / RF / fine-tune (Tier 1)
  --exp kshot           accuracy vs number of demonstrations (k = 1..7)
  --exp ngestures       accuracy vs vocabulary size (n = 2..max)
  --exp thresholds      threshold sensitivity: acceptance vs accuracy as the
                        rep/sep/null percentiles vary

All conditions use the same leave-one-user-out heads in saved/customization/
and write JSON to results/.

Run:
    python -m customization.run_experiments --exp main
    python -m customization.run_experiments --exp baselines
    python -m customization.run_experiments --exp kshot
    python -m customization.run_experiments --exp ngestures
    python -m customization.run_experiments --exp thresholds
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customization.backbone import FrozenBackbone, gesture_ids            # noqa: E402
from customization.head import CustomizationHead                          # noqa: E402
from customization.registry import GestureRegistry                        # noqa: E402
from customization.evaluate import IdentityHead, eval_user                # noqa: E402
from customization.meta_train import BLIND_USERS                          # noqa: E402
from customization import baselines as BL                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS = os.path.join(ROOT, 'saved', 'customization')
OUTDIR = os.path.join(ROOT, 'results')
os.makedirs(OUTDIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# loaders shared across experiments
# --------------------------------------------------------------------------- #

def load_head(user, backbone_dim, ablation='none'):
    """Return the head for a user's LOU condition, or an IdentityHead for no_head."""
    if ablation == 'no_head':
        return IdentityHead(backbone_dim)
    ck = os.path.join(HEADS, f'head_loo_{user}.pt')
    if not os.path.exists(ck):
        return None
    head = CustomizationHead(backbone_dim=backbone_dim)
    head.load_state_dict(torch.load(ck, map_location='cpu'))
    head.eval()
    return head


def load_calib(user, suffix=''):
    """Return (null_emb, null_mask, corpus, corpus_groups) saved during meta_train.
    suffix selects a loss-ablation head's own calibration, e.g. '_no_rep'."""
    ck = os.path.join(HEADS, f'head_loo_{user}{suffix}.pt')
    null_emb = null_mask = None
    calib = ck.replace('.pt', '_calib.npz')
    if os.path.exists(calib):
        z = np.load(calib)
        if len(z['null']):
            null_emb, null_mask = z['null'], z['null_mask']
    corpus_p = ck.replace('.pt', '_corpus.npy')
    corpus = ([np.asarray(z, dtype=np.float32) for z in np.load(corpus_p, allow_pickle=True)]
              if os.path.exists(corpus_p) else [])
    groups_p = ck.replace('.pt', '_corpus_groups.npy')
    groups = ([[np.asarray(z, dtype=np.float32) for z in g]
               for g in np.load(groups_p, allow_pickle=True)]
              if os.path.exists(groups_p) else None)
    return null_emb, null_mask, corpus, groups


def raw_loader(user, version='20_120'):
    return np.load(os.path.join(ROOT, 'dataset', user, f'data_{version}.npy')).astype(np.float32)


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else float('nan')
    out['n_users'] = len(rows)
    return out


# --------------------------------------------------------------------------- #
# experiment 1: main table (method + loss ablations + validator ablation)
# --------------------------------------------------------------------------- #

def exp_main(bb, args):
    # (a) our method + loss ablations: these use pre-trained heads named by ablation.
    #     The loss ablations require heads trained with that loss removed; if those
    #     heads are not present we note it and skip, rather than silently using the
    #     full head.
    loss_conditions = ['none', 'no_rep', 'no_sep', 'no_null', 'no_head']
    # (b) validator-component ablations: SAME full head, but a check turned off.
    validator_conditions = ['val_full', 'val_no_rep', 'val_no_dist', 'val_no_null']

    all_results = {}

    # ---- loss / architecture ablations ---- #
    for cond in loss_conditions:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden, ablation=('no_head' if cond == 'no_head' else 'none'
                             if cond == 'none' else cond))
            # loss-ablation heads: head_loo_<user>_<cond>.pt if present, else skip
            if cond not in ('none', 'no_head'):
                ck = os.path.join(HEADS, f'head_loo_{user}_{cond}.pt')
                if not os.path.exists(ck):
                    head = None
                else:
                    head = CustomizationHead(backbone_dim=bb.hidden)
                    head.load_state_dict(torch.load(ck, map_location='cpu')); head.eval()
            if head is None:
                continue
            calib_suffix = f'_{cond}' if cond not in ('none', 'no_head') else ''
            ne, nm, corpus, groups = load_calib(user, suffix=calib_suffix)
            r = eval_user(bb, user, head, ne, nm, corpus, corpus_groups=groups,
                          n_train=args.n_train, target_fpr=args.target_fpr,
                          device='cpu', verbose=False)
            if r:
                rows.append(r)
        if rows:
            all_results[cond] = summarize(
                rows, ['accepted', 'total', 'acc_full', 'acc_open',
                       'acc_unfiltered', 'bg_fpr', 'add_ms_mean'])
            all_results[cond]['rows'] = rows
        print(f"[main] {cond:12s}: "
              + (f"acc_full {all_results[cond]['acc_full']:.3f} "
                 f"accepted {all_results[cond]['accepted']:.1f}/{all_results[cond]['total']:.1f} "
                 f"(n={all_results[cond]['n_users']})" if cond in all_results
                 else "no heads found -- train with this loss ablation first"))

    # ---- validator-component ablation (full head, toggle one check) ---- #
    for cond in validator_conditions:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden)
            if head is None:
                continue
            ne, nm, corpus, groups = load_calib(user)
            r = eval_user_with_toggles(bb, user, head, ne, nm, corpus, groups,
                                       cond, args.n_train, args.target_fpr)
            if r:
                rows.append(r)
        if rows:
            all_results[cond] = summarize(
                rows, ['accepted', 'total', 'acc_full', 'acc_open',
                       'acc_unfiltered', 'bg_fpr'])
            all_results[cond]['rows'] = rows
            print(f"[main] {cond:12s}: acc_full {all_results[cond]['acc_full']:.3f} "
                  f"accepted {all_results[cond]['accepted']:.1f}/{all_results[cond]['total']:.1f} "
                  f"bg_fpr {all_results[cond]['bg_fpr']:.3f}")

    _dump('main', all_results)


def eval_user_with_toggles(bb, user, head, null_emb, null_mask, corpus, groups,
                           cond, n_train, target_fpr, seed=0):
    """eval_user, but with one validator check disabled (validator-component ablation)."""
    from customization.evaluate import split_user
    emb, mask, lab = bb.encode_dataset(user, return_masks=True)
    y = gesture_ids(lab)
    (etr, mtr, ytr), (ete, mte, yte) = split_user(emb, mask, y, n_train, seed)

    reg = GestureRegistry(head)
    reg.enabled_checks = {
        'repeatability': cond != 'val_no_rep',
        'distinguishability': cond != 'val_no_dist',
        'null': cond != 'val_no_null',
    }
    z_tr, z_te = reg.embed(etr, mtr), reg.embed(ete, mte)
    z_null = reg.embed(null_emb, null_mask) if null_emb is not None and len(null_emb) else None
    enc = lambda zc: reg.embed(np.asarray(zc, np.float32), None) if np.ndim(zc) == 3 else np.asarray(zc, np.float32)
    corpus_z = [enc(c) for c in corpus]
    groups_z = [[enc(c) for c in g] for g in groups] if groups else None
    if not corpus_z:
        return None
    reg.calibrate(corpus_z, null_z=z_null, corpus_groups=groups_z, verbose=False)

    decisions = []
    for c in sorted(np.unique(ytr)):
        decisions.append(reg.add_gesture(f'g{int(c)}', z_tr[ytr == c],
                                         validate=True, null_z=z_null))
    accepted = [d['name'] for d in decisions if d['added']]
    if not accepted:
        return None
    if z_null is not None:
        reg.calibrate_reject(z_null, target_fpr=target_fpr, verbose=False)

    keep = {int(n[1:]) for n in accepted}
    sel = np.isin(yte, list(keep))
    truth = [f'g{int(c)}' for c in yte[sel]]
    acc_full = float(np.mean([p == t for p, t in zip(reg.predict(z_te[sel], False)[0], truth)])) if sel.any() else 0.0
    acc_open = float(np.mean([p == t for p, t in zip(reg.predict(z_te[sel], True)[0], truth)])) if sel.any() else 0.0
    bg_fpr = None
    if z_null is not None:
        bg_fpr = float(np.mean([l is not None for l in reg.predict(z_null, True)[0]]))
    return {'user': user, 'condition': cond, 'accepted': len(accepted),
            'total': len(decisions), 'acc_full': acc_full, 'acc_open': acc_open,
            'acc_unfiltered': None, 'bg_fpr': bg_fpr}


# --------------------------------------------------------------------------- #
# experiment 2: cheap baselines
# --------------------------------------------------------------------------- #

def exp_baselines(bb, args):
    out = {}
    for kind in ['ncm', 'svm', 'rf', 'finetune']:
        rows = []
        for user in BLIND_USERS:
            if kind == 'ncm':
                head = IdentityHead(bb.hidden)
                ne, nm, corpus, groups = load_calib(user)
                r = eval_user(bb, user, head, ne, nm, corpus, corpus_groups=groups,
                              n_train=args.n_train, device='cpu', verbose=False)
                if r:
                    rows.append({'user': user, 'baseline': 'ncm', 'acc': r['acc_unfiltered'],
                                 'requires_per_user_training': False, 'validates': False})
            else:
                try:
                    r = BL.eval_baseline_user(kind, bb, user, n_train=args.n_train,
                                              raw_loader=raw_loader)
                    rows.append(r)
                except Exception as e:
                    print(f"  {kind} {user}: {e}")
        if rows:
            out[kind] = {'acc': float(np.mean([r['acc'] for r in rows])),
                         'n_users': len(rows),
                         'requires_per_user_training': rows[0].get('requires_per_user_training'),
                         'validates': False, 'rows': rows}
            print(f"[baselines] {kind:10s}: acc {out[kind]['acc']:.3f} "
                  f"(n={out[kind]['n_users']}, "
                  f"per-user-train={out[kind]['requires_per_user_training']})")
    _dump('baselines', out)


# --------------------------------------------------------------------------- #
# experiment 3: k-shot sweep
# --------------------------------------------------------------------------- #

def exp_kshot(bb, args):
    out = {}
    for k in [1, 2, 3, 4, 5, 7]:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden)
            if head is None:
                continue
            ne, nm, corpus, groups = load_calib(user)
            r = eval_user(bb, user, head, ne, nm, corpus, corpus_groups=groups,
                          n_train=k, device='cpu', verbose=False)
            if r:
                rows.append(r)
        if rows:
            out[str(k)] = summarize(rows, ['accepted', 'acc_full', 'acc_open', 'acc_unfiltered'])
            out[str(k)]['rows'] = [{'user': r['user'], 'accepted': r['accepted'],
                                    'acc_full': r['acc_full'], 'acc_open': r['acc_open'],
                                    'acc_unfiltered': r['acc_unfiltered']} for r in rows]
            print(f"[kshot] k={k}: acc_full {out[str(k)]['acc_full']:.3f} "
                  f"accepted {out[str(k)]['accepted']:.1f}")
    _dump('kshot', out)


# --------------------------------------------------------------------------- #
# experiment 4: n-gestures sweep  (accuracy vs vocabulary size)
# --------------------------------------------------------------------------- #

def exp_ngestures(bb, args):
    from customization.evaluate import split_user
    out = {}
    for n in [2, 3, 4, 5, 6, 8, 10]:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden)
            if head is None:
                continue
            emb, mask, lab = bb.encode_dataset(user, return_masks=True)
            y = gesture_ids(lab)
            classes = np.unique(y)
            if len(classes) < n:
                continue
            sub = classes[:n]
            selm = np.isin(y, sub)
            (etr, mtr, ytr), (ete, mte, yte) = split_user(
                emb[selm], mask[selm], y[selm], args.n_train, seed=0)
            reg = GestureRegistry(head)
            z_tr, z_te = reg.embed(etr, mtr), reg.embed(ete, mte)
            for c in np.unique(ytr):
                reg.add_gesture(f'g{int(c)}', z_tr[ytr == c], validate=False)
            truth = [f'g{int(c)}' for c in yte]
            acc = float(np.mean([p == t for p, t in zip(reg.predict(z_te, False)[0], truth)]))
            rows.append({'user': user, 'acc': acc})
        if rows:
            out[str(n)] = {'acc': float(np.mean([r['acc'] for r in rows])),
                           'n_users': len(rows), 'rows': rows}
            print(f"[ngestures] n={n}: acc {out[str(n)]['acc']:.3f} (n_users={len(rows)})")
    _dump('ngestures', out)


# --------------------------------------------------------------------------- #
# experiment 5: threshold sensitivity  (acceptance vs accuracy)
# --------------------------------------------------------------------------- #

def exp_thresholds(bb, args):
    """Vary the calibration percentiles; trace acceptance rate vs accuracy."""
    from customization.evaluate import split_user
    out = {}
    # sweep the repeatability percentile (stricter -> lower acceptance)
    for pct in [80, 85, 90, 95, 99]:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden)
            if head is None:
                continue
            ne, nm, corpus, groups = load_calib(user)
            emb, mask, lab = bb.encode_dataset(user, return_masks=True)
            y = gesture_ids(lab)
            (etr, mtr, ytr), (ete, mte, yte) = split_user(emb, mask, y, args.n_train, 0)
            reg = GestureRegistry(head)
            z_tr, z_te = reg.embed(etr, mtr), reg.embed(ete, mte)
            z_null = reg.embed(ne, nm) if ne is not None and len(ne) else None
            enc = lambda zc: reg.embed(np.asarray(zc, np.float32), None) if np.ndim(zc) == 3 else np.asarray(zc, np.float32)
            cz = [enc(c) for c in corpus]
            gz = [[enc(c) for c in g] for g in groups] if groups else None
            if not cz:
                continue
            reg.calibrate(cz, null_z=z_null, corpus_groups=gz,
                          rep_percentile=pct, verbose=False)
            dec = [reg.add_gesture(f'g{int(c)}', z_tr[ytr == c], validate=True, null_z=z_null)
                   for c in sorted(np.unique(ytr))]
            acc = [d['name'] for d in dec if d['added']]
            if not acc:
                rows.append((0, 0.0)); continue
            keep = {int(n[1:]) for n in acc}; sel = np.isin(yte, list(keep))
            truth = [f'g{int(c)}' for c in yte[sel]]
            a = float(np.mean([p == t for p, t in zip(reg.predict(z_te[sel], False)[0], truth)])) if sel.any() else 0.0
            rows.append({'user': user, 'accept_rate': len(acc) / len(dec), 'acc_full': a})
        if rows:
            out[str(pct)] = {'accept_rate': float(np.mean([r['accept_rate'] for r in rows])),
                             'acc_full': float(np.mean([r['acc_full'] for r in rows])),
                             'n_users': len(rows), 'rows': rows}
            print(f"[thresholds] rep_pct={pct}: accept {out[str(pct)]['accept_rate']:.2f} "
                  f"acc {out[str(pct)]['acc_full']:.3f}")
    # --- second sweep: the repeatability STATISTIC (worst-shot -> lenient) ---
    # rep_stat_pct=100 is worst-shot (max, the default); lower values relax it,
    # e.g. p90 ignores a single bad demonstration. This directly tests whether
    # the strict worst-shot rule over-rejects users like John.
    out_stat = {}
    for stat in [100, 95, 90, 85]:
        rows = []
        for user in BLIND_USERS:
            head = load_head(user, bb.hidden)
            if head is None:
                continue
            ne, nm, corpus, groups = load_calib(user)
            emb, mask, lab = bb.encode_dataset(user, return_masks=True)
            y = gesture_ids(lab)
            (etr, mtr, ytr), (ete, mte, yte) = split_user(emb, mask, y, args.n_train, 0)
            reg = GestureRegistry(head)
            reg.rep_stat_pct = stat
            z_tr, z_te = reg.embed(etr, mtr), reg.embed(ete, mte)
            z_null = reg.embed(ne, nm) if ne is not None and len(ne) else None
            enc = lambda zc: reg.embed(np.asarray(zc, np.float32), None) if np.ndim(zc) == 3 else np.asarray(zc, np.float32)
            cz = [enc(c) for c in corpus]
            gz = [[enc(c) for c in g] for g in groups] if groups else None
            if not cz:
                continue
            reg.calibrate(cz, null_z=z_null, corpus_groups=gz, verbose=False)
            dec = [reg.add_gesture(f'g{int(c)}', z_tr[ytr == c], validate=True, null_z=z_null)
                   for c in sorted(np.unique(ytr))]
            acc = [d['name'] for d in dec if d['added']]
            if not acc:
                rows.append({'user': user, 'accept_rate': 0.0, 'acc_full': 0.0}); continue
            keep = {int(n[1:]) for n in acc}; sel = np.isin(yte, list(keep))
            truth = [f'g{int(c)}' for c in yte[sel]]
            a = float(np.mean([p == t for p, t in zip(reg.predict(z_te[sel], False)[0], truth)])) if sel.any() else 0.0
            rows.append({'user': user, 'accept_rate': len(acc) / len(dec), 'acc_full': a})
        if rows:
            out_stat[str(stat)] = {'accept_rate': float(np.mean([r['accept_rate'] for r in rows])),
                                   'acc_full': float(np.mean([r['acc_full'] for r in rows])),
                                   'n_users': len(rows), 'rows': rows}
            print(f"[thresholds] rep_stat_pct={stat}: accept {out_stat[str(stat)]['accept_rate']:.2f} "
                  f"acc {out_stat[str(stat)]['acc_full']:.3f}")
    _dump('thresholds', out)
    _dump('thresholds_repstat', out_stat)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# experiment: Xu et al. baseline (Tier 2)
# --------------------------------------------------------------------------- #

def exp_xu(bb, args):
    """Faithful Xu et al. baseline: LOU base model + per-user prediction head."""
    from customization import baseline_xu as XU
    rows = []
    # pool raw training data per held-out user, train Xu base, customize on user
    for user in BLIND_USERS:
        # training pool = all OTHER users' raw gestures
        raw_tr, y_tr = [], []
        for u in BLIND_USERS:
            if u == user:
                continue
            r = raw_loader(u); l = gesture_ids(np.load(
                os.path.join(ROOT, 'dataset', u, f'label_{args.version}.npy')))
            raw_tr.append(r); y_tr.append(l + 1000 * BLIND_USERS.index(u))  # unique per user
        raw_tr = np.concatenate(raw_tr); y_tr = np.concatenate(y_tr)
        in_ch = raw_tr.shape[-1]
        base = XU.train_xu_base(raw_tr, y_tr, in_channels=in_ch,
                                epochs=args.xu_base_epochs, device='cpu')
        # background pool for synthesis/negatives (optional)
        neg_pool = None
        raw_user = raw_loader(user)
        y_user = gesture_ids(np.load(os.path.join(ROOT, 'dataset', user,
                                                  f'label_{args.version}.npy')))
        r = XU.eval_xu_user(base, raw_user, y_user, n_train=args.n_train,
                            neg_pool=neg_pool, device='cpu')
        r['user'] = user
        rows.append(r)
        print(f"[xu] {user:10s}: acc {r['acc']:.3f} (n_gestures={r['n_gestures']})")
    out = {'xu': {'acc': float(np.mean([r['acc'] for r in rows])),
                  'n_users': len(rows),
                  'requires_per_user_training': True, 'validates': True,
                  'rows': rows}}
    _dump('xu', out)



def exp_xu_validated(bb, args):
    """Xu et al. WITH their own interactive-feedback validation (rejects gestures)."""
    from customization import baseline_xu as XU
    rows = []
    for user in BLIND_USERS:
        raw_tr, y_tr = [], []
        for u in BLIND_USERS:
            if u == user:
                continue
            r = raw_loader(u); l = gesture_ids(np.load(
                os.path.join(ROOT, 'dataset', u, f'label_{args.version}.npy')))
            raw_tr.append(r); y_tr.append(l + 1000 * BLIND_USERS.index(u))
        raw_tr = np.concatenate(raw_tr); y_tr = np.concatenate(y_tr)
        base = XU.train_xu_base(raw_tr, y_tr, in_channels=raw_tr.shape[-1],
                                epochs=args.xu_base_epochs, device='cpu')
        raw_user = raw_loader(user)
        y_user = gesture_ids(np.load(os.path.join(ROOT, 'dataset', user,
                                                  f'label_{args.version}.npy')))
        r = XU.eval_xu_user_validated(base, raw_user, y_user, n_train=args.n_train,
                                      device='cpu')
        if r:
            r['user'] = user; rows.append(r)
            print(f"[xu_val] {user:10s}: acc {r['acc_full']:.3f} "
                  f"accepted {r['accepted']}/{r['total']}  rejected {r['rejected']}")
    out = {'xu_validated': {'acc_full': float(np.mean([r['acc_full'] for r in rows])),
                            'accepted': float(np.mean([r['accepted'] for r in rows])),
                            'total': float(np.mean([r['total'] for r in rows])),
                            'n_users': len(rows), 'validates': True,
                            'requires_per_user_training': True, 'rows': rows}}
    _dump('xu_validated', out)



def exp_watchguardian(bb, args):
    """
    WatchGuardian baseline (Tier 3), faithful three-stage pipeline:
      Stage 1: their ResNet, SSL-pretrained on HHAR (run wg_pretrain first)
      Stage 2: labeled-gesture fine-tune on the OTHER users (leave-one-user-out)
      Stage 3: per-user augment/synth + trained classification layers
    """
    from customization import baseline_watchguardian as WG
    base_enc = WG.load_wg_encoder(ROOT, device='cpu')
    rows = []
    for user in BLIND_USERS:
        # Stage 2: fine-tune on pooled labeled gestures of all OTHER users
        raw_lab, y_lab = [], []
        for u in BLIND_USERS:
            if u == user:
                continue
            raw_lab.append(raw_loader(u))
            y_lab.append(gesture_ids(np.load(os.path.join(ROOT, 'dataset', u,
                                     f'label_{args.version}.npy'))) + 1000 * BLIND_USERS.index(u))
        raw_lab = np.concatenate(raw_lab); y_lab = np.concatenate(y_lab)
        import copy
        enc_ft = WG.wg_stage2_finetune(copy.deepcopy(base_enc), raw_lab, y_lab, device='cpu')
        # Stage 3: customize on the held-out user
        raw_user = raw_loader(user)
        y_user = gesture_ids(np.load(os.path.join(ROOT, 'dataset', user,
                                                  f'label_{args.version}.npy')))
        r = WG.eval_wg_user(enc_ft, raw_user, y_user, n_train=args.n_train, device='cpu')
        r['user'] = user; rows.append(r)
        print(f"[wg] {user:10s}: acc {r['acc']:.3f} (n_gestures={r['n_gestures']})")
    out = {'watchguardian': {'acc': float(np.mean([r['acc'] for r in rows])),
                             'n_users': len(rows),
                             'requires_per_user_training': True, 'validates': False,
                             'rows': rows}}
    _dump('watchguardian', out)



def exp_maml(bb, args):
    """MAML baseline (Tier 4), adapted from Shahi et al. (Apple vision paper)."""
    from customization import baseline_maml as ML
    rows = []
    for user in BLIND_USERS:
        # pool OTHER users' frozen mean-pooled embeddings for meta-training
        pz, py, pu = [], [], []
        for ui, u in enumerate(BLIND_USERS):
            if u == user:
                continue
            emb, mask, lab = bb.encode_dataset(u, return_masks=True)
            z = emb.mean(axis=1)                       # frozen mean-pooled features
            pz.append(z); py.append(gesture_ids(lab)); pu.append(np.full(len(z), ui))
        pz = np.concatenate(pz); py = np.concatenate(py); pu = np.concatenate(pu)
        # head must be sized to the LARGEST vocabulary across all users (incl. test),
        # since MAML's output layer is fixed-size.
        max_classes = 0
        for u in BLIND_USERS:
            lab_u = gesture_ids(np.load(os.path.join(ROOT, 'dataset', u,
                                        f'label_{args.version}.npy')))
            max_classes = max(max_classes, int(len(np.unique(lab_u))))
        model = ML.meta_train(pz, py, pu, emb_dim=pz.shape[1], max_classes=max_classes,
                              episodes=args.maml_episodes, device='cpu', verbose=False)
        # meta-test on held-out user
        emb, mask, lab = bb.encode_dataset(user, return_masks=True)
        zu = emb.mean(axis=1); yu = gesture_ids(lab)
        r = ML.eval_maml_user(model, zu, yu, emb_dim=zu.shape[1], n_train=args.n_train)
        r['user'] = user; rows.append(r)
        print(f"[maml] {user:10s}: acc {r['acc']:.3f} (n_gestures={r['n_gestures']})")
    out = {'maml': {'acc': float(np.mean([r['acc'] for r in rows])),
                    'n_users': len(rows),
                    'requires_per_user_training': True, 'validates': False,
                    'rows': rows}}
    _dump('maml', out)


def _dump(name, obj):
    p = os.path.join(OUTDIR, f'{name}.json')
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    choices=['all', 'main', 'baselines', 'kshot', 'ngestures', 'thresholds', 'xu', 'xu_validated', 'watchguardian', 'maml'])
    ap.add_argument('--pretrain_dataset', default='blind_user')
    ap.add_argument('--version', default='20_120')
    ap.add_argument('--n_train', type=int, default=7)
    ap.add_argument('--target_fpr', type=float, default=0.05)
    ap.add_argument('--xu_base_epochs', type=int, default=40)
    ap.add_argument('--maml_episodes', type=int, default=2000)
    args = ap.parse_args()

    bb = FrozenBackbone(pretrain_dataset=args.pretrain_dataset, version=args.version)
    table = {'main': exp_main, 'baselines': exp_baselines, 'kshot': exp_kshot,
             'ngestures': exp_ngestures, 'thresholds': exp_thresholds, 'xu': exp_xu, 'xu_validated': exp_xu_validated, 'watchguardian': exp_watchguardian, 'maml': exp_maml}
    if args.exp == 'all':
        for name, fn in table.items():
            print(f"\n{'='*60}\nEXPERIMENT: {name}\n{'='*60}")
            fn(bb, args)
        print(f"\nAll results in {OUTDIR}/  --  run  python -m customization.report  for a summary")
    else:
        table[args.exp](bb, args)


if __name__ == '__main__':
    main()