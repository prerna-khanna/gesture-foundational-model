"""
meta_train.py -- trains the customization head episodically, leave-one-user-out.

    python -m customization.meta_train \
        --held_out Alexandra \
        --pretrain_dataset blind_user \
        --episodes 4000 \
        --save saved/customization/head_loo_Alexandra.pt

What gets pooled into the task distribution:
  * every per-user blind dataset EXCEPT the held-out user
  * the multi-user corpora (blind_user, sighted_user, earbud, sony_watch, ...),
    split into within-user task groups
  * synthetic classes from MetaAugmentor -- the repetition-count / direction /
    tempo families that Ch 7.4.3 identifies as the actual failure modes
  * background windows from HAR datasets, used as the null class in every
    episode and later for reject calibration

The held-out user's data NEVER enters meta-training, in any form -- not as
gesture classes, not as augmentation source. That is what makes the Ch 7
evaluation honest: the head has never seen this person move.
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customization.backbone import FrozenBackbone, gesture_ids, user_ids   # noqa: E402
from customization.head import CustomizationHead, EpisodicCustomizationLoss  # noqa: E402
from customization.episodes import TaskPool, EpisodeSampler, MetaAugmentor   # noqa: E402

BLIND_USERS = ['Alexandra', 'Rahkiya', 'Nihal', 'Angel', 'Julius',
               'Turiya', 'John', 'Kerry', 'Roafel', 'Edery']
MULTIUSER_CORPORA = ['blind_user', 'sighted_user', 'earbud', 'sony_watch', 'smart_watch']
BACKGROUND_SETS = ['hhar', 'motion', 'uci', 'shoaib']


def _to_t(x, device):
    return None if x is None else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def build_pool(bb, held_out, use_augmentation=True, aug_variants=3,
               max_background=3000, n_calib_users=2, seed=0, verbose=True):
    """
    n_calib_users : users withheld from EPISODE SAMPLING and used only to
        calibrate tau_rep. This matters. If the repeatability threshold is
        estimated on users the head trained on, it is measured on classes the
        head has memorized -- radii there collapse toward zero, and shipping
        that threshold rejects almost every gesture a NEW user demonstrates.
        The calibration split must be unseen, exactly like the test user.
    """
    rng = np.random.default_rng(seed)
    pool = TaskPool()
    corpus_support_sets = []
    corpus_groups = []

    trainable = [u for u in BLIND_USERS if u != held_out]
    # Calibration users are chosen DETERMINISTICALLY, not randomly, so the
    # thresholds a test user is judged by do not depend on a random seed. We
    # take the n_calib_users users that immediately follow the held-out user in
    # a fixed order (wrapping around), which spreads calibration duty evenly
    # across users over the leave-one-out sweep. They are still held out of
    # episode training, so their radii have not collapsed.
    if n_calib_users:
        order = [u for u in BLIND_USERS if u != held_out]
        start = BLIND_USERS.index(held_out) % len(order)
        rotated = order[start:] + order[:start]
        calib_users = rotated[:min(n_calib_users, len(order) - 1)]
    else:
        calib_users = []

    # ---- per-user blind datasets ------------------------------------- #
    for u in BLIND_USERS:
        if u == held_out:
            continue
        try:
            emb, mask, lab = bb.encode_dataset(u, return_masks=True)
        except FileNotFoundError:
            continue
        y = gesture_ids(lab)

        if u in calib_users:
            # calibration only -- never sampled into an episode
            grp = [emb[y == c] for c in np.unique(y) if (y == c).sum() >= 3]
            corpus_support_sets += grp
            if len(grp) >= 2:
                corpus_groups.append(grp)
            continue

        pool.add(('user', u), emb, y, mask)

        if use_augmentation:
            raw = np.load(os.path.join(bb.cache_dir, '..', 'dataset', u,
                                       f'data_{bb.version}.npy')).astype(np.float32)
            aug = MetaAugmentor(seq_len=bb.seq_len, rng=rng)
            syn_raw, syn_y = aug.synthesize(raw, y, per_class_variants=aug_variants)
            if len(syn_raw):
                syn_emb = bb.encode_array(syn_raw)
                syn_mask = bb.nucleus_masks(syn_raw)
                pool.add(('aug', u), syn_emb, syn_y, syn_mask)

    # ---- multi-user corpora, split per user -------------------------- #
    for ds in MULTIUSER_CORPORA:
        try:
            emb, mask, lab = bb.encode_dataset(ds, return_masks=True)
        except FileNotFoundError:
            continue
        y, uid = gesture_ids(lab), user_ids(lab)
        uniq = np.unique(uid)
        holdout_u = set(uniq[:max(1, len(uniq) // 5)])   # 20% of users -> calibration
        for u in uniq:
            sel = uid == u
            if int(u) in holdout_u:
                ys = y[sel]
                grp = [emb[sel][ys == c] for c in np.unique(ys) if (ys == c).sum() >= 3]
                corpus_support_sets += grp
                if len(grp) >= 2:
                    corpus_groups.append(grp)
                continue
            pool.add((ds, int(u)), emb[sel], y[sel], mask[sel])

    # ---- background ---------------------------------------------------- #
    null_emb, null_mask = [], []
    for ds in BACKGROUND_SETS:
        try:
            e, m, _ = bb.encode_dataset(ds, return_masks=True)
        except FileNotFoundError:
            continue
        take = min(len(e), max_background // max(1, len(BACKGROUND_SETS)))
        idx = rng.choice(len(e), size=take, replace=False)
        null_emb.append(e[idx]); null_mask.append(m[idx])
    null_emb = np.concatenate(null_emb) if null_emb else None
    null_mask = np.concatenate(null_mask) if null_mask else None

    if verbose:
        print(f"[pool] {len(pool.groups)} task groups, "
              f"{pool.n_classes_total} classes total, held_out={held_out}")
        print(f"[pool] calibration users (excluded from episodes): {calib_users}")
        print(f"[pool] calibration classes: {len(corpus_support_sets)} "
              f"in {len(corpus_groups)} within-user groups")
        print(f"[pool] background windows: "
              f"{0 if null_emb is None else len(null_emb)}")
        if null_emb is None:
            print("[pool] WARNING: no background found -- L_null disabled and the "
                  "reject rule cannot be calibrated. Fetch an ADL/HAR dataset.")
    return pool, null_emb, null_mask, corpus_support_sets, corpus_groups


def meta_train(head, sampler, device, episodes=4000, lr=1e-3, weight_decay=1e-4,
               log_every=200, loss_kwargs=None, grad_clip=1.0):
    head.to(device).train()
    crit = EpisodicCustomizationLoss(**(loss_kwargs or {}))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=episodes)

    hist = []
    for step in range(1, episodes + 1):
        ep = sampler.sample()

        zs = head(_to_t(ep['support_emb'], device), _to_t(ep['support_mask'], device))
        zq = head(_to_t(ep['query_emb'], device), _to_t(ep['query_mask'], device))
        zn = (head(_to_t(ep['null_emb'], device), _to_t(ep['null_mask'], device))
              if ep['null_emb'] is not None else None)

        loss, parts = crit(
            zs, torch.as_tensor(ep['support_y'], device=device),
            zq, torch.as_tensor(ep['query_y'], device=device),
            ep['n_classes'], z_null=zn, step=step, temperature=head.temperature)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step(); sched.step()
        hist.append(parts)

        if step % log_every == 0:
            w = hist[-log_every:]
            print(f"ep {step:5d} | loss {np.mean([h['total'] for h in w]):.4f} "
                  f"| proto {np.mean([h['proto'] for h in w]):.4f} "
                  f"| rep {np.mean([h['rep'] for h in w]):.4f} "
                  f"| margin {np.mean([h['margin'] for h in w]):.4f} "
                  f"| null {np.mean([h['null'] for h in w]):.4f} "
                  f"| q-acc {np.mean([h['acc'] for h in w]):.3f} "
                  f"| radius {np.mean([h['mean_radius'] for h in w]):.3f}")
    head.eval()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held_out', default='Alexandra')
    ap.add_argument('--pretrain_dataset', default='blind_user',
                    help='which saved/pretrain_base_<X>_<ver> checkpoint to load')
    ap.add_argument('--version', default='20_120')
    ap.add_argument('--episodes', type=int, default=4000)
    ap.add_argument('--out_dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_way_min', type=int, default=3)
    ap.add_argument('--n_way_max', type=int, default=10)
    ap.add_argument('--k_min', type=int, default=3)
    ap.add_argument('--k_max', type=int, default=7)
    ap.add_argument('--no_augmentation', action='store_true')
    ap.add_argument('--ablate_loss', default=None,
                    choices=['no_rep', 'no_sep', 'no_null'],
                    help='zero out one loss term, for the loss ablation; also '
                         'renames the output head to head_loo_<user>_<ablate>.pt')
    ap.add_argument('--n_calib_users', type=int, default=2,
                    help='users withheld from episodes, used only for tau_rep calibration')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bb = FrozenBackbone(pretrain_dataset=args.pretrain_dataset, version=args.version,
                        device=device)
    print(f"[backbone] hidden={bb.hidden} seq_len={bb.seq_len} ckpt={bb.ckpt_path}")

    pool, null_emb, null_mask, corpus, corpus_groups = build_pool(
        bb, args.held_out, use_augmentation=not args.no_augmentation,
        n_calib_users=args.n_calib_users, seed=args.seed)

    sampler = EpisodeSampler(pool, null_emb, null_mask,
                             n_way=(args.n_way_min, args.n_way_max),
                             k_shot=(args.k_min, args.k_max), seed=args.seed)

    head = CustomizationHead(backbone_dim=bb.hidden, hidden=args.hidden,
                             out_dim=args.out_dim)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head] {n_params:,} trainable params "
          f"(backbone stays frozen: {sum(p.numel() for p in bb.model.parameters()):,})")

    loss_kwargs = {}
    if args.ablate_loss == 'no_rep':
        loss_kwargs['w_rep'] = 0.0
    elif args.ablate_loss == 'no_sep':
        loss_kwargs['w_margin'] = 0.0
    elif args.ablate_loss == 'no_null':
        loss_kwargs['w_null'] = 0.0
    meta_train(head, sampler, device, episodes=args.episodes, lr=args.lr,
               loss_kwargs=loss_kwargs)

    suffix = f'_{args.ablate_loss}' if args.ablate_loss else ''
    save = args.save or f'saved/customization/head_loo_{args.held_out}{suffix}.pt'
    save = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), save)
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save(head.state_dict(), save)

    # stash calibration material next to the head so evaluate.py can reuse it
    np.savez(save.replace('.pt', '_calib.npz'),
             null=null_emb if null_emb is not None else np.zeros((0, bb.seq_len, bb.hidden), np.float32),
             null_mask=null_mask if null_mask is not None else np.zeros((0, bb.seq_len), np.float32),
             n_corpus=len(corpus))
    np.save(save.replace('.pt', '_corpus.npy'),
            np.array(corpus, dtype=object), allow_pickle=True)
    np.save(save.replace('.pt', '_corpus_groups.npy'),
            np.array([np.array(g, dtype=object) for g in corpus_groups], dtype=object),
            allow_pickle=True)
    print(f"[save] {save}")


if __name__ == '__main__':
    main()