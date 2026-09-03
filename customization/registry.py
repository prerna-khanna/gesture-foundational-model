"""
registry.py -- the deployed personalized recognizer + the validator.

This is the object that lives on the phone. It holds one prototype and one
radius per accepted gesture. Adding a gesture is: encode k demos, compute a
mean, run three closed-form checks, append. No retraining, no head resize,
cost independent of vocabulary size.

The important structural change vs Ch 7: the validator and the recognizer
operate in the SAME space and use the SAME decision rule. In the current
branch, gesture_validator.py measures cosine similarity in the frozen
backbone's mean-pooled space while recognition happens in the classifier's
learned feature space -- so a rejection is not actually a prediction about
what the deployed recognizer will do. Here, "will these two gestures get
confused?" is answered by literally evaluating the deployed decision rule.

THREE CHECKS
------------
1. Repeatability  -- worst-shot spread of the k demos around their own mean,
   against a threshold calibrated as a percentile of spreads observed across
   the whole meta-training corpus. Not a magic 0.75.

2. Distinguishability -- two parts:
     (a) leave-one-out nearest-prototype recall over the support sets of ALL
         gestures in the vocabulary (the new one can break an old one, which
         is why we re-check everything -- same as Ch 7);
     (b) ball separation: ||p_i - p_j|| / (r_i + r_j) >= min_sep_ratio.
         Scale-free, so it transfers across users whose embeddings have
         different absolute spread. This is the part L_margin was trained
         to make meaningful.
   Deterministic and O(|V|) *arithmetic*, versus O(|V|) *retraining* today.

3. Null collision -- NEW. Is the candidate gesture sitting INSIDE the
   background manifold, i.e. is it something the user does accidentally all
   day? Measured as a local-density ratio: the candidate prototype's mean
   distance to its k nearest background windows, divided by the background's
   own median k-NN distance. Below ~1.0 the candidate is indistinguishable
   from ordinary movement.

   Note this cannot be done by asking "how much background falls inside the
   candidate's ball" -- a tightly-performed gesture has a tiny ball, so that
   fraction is ~0 even when the gesture sits exactly on the background
   manifold. The ratio form is scale-free and does not have that blind spot.

   On a continuously streaming wrist IMU this is the real deployment killer,
   and neither Ch 7 nor the Apple paper vetoes it at registration time.

REJECTION AT INFERENCE
----------------------
Distance is normalized by class radius: u_c = ||z - p_c|| / r_c. Predict
argmin_c u_c, but emit REJECT if min_c u_c > tau_null. tau_null is calibrated
on held-out background so that a target false-accept rate is met, which makes
the operating point explicit rather than implicit in a softmax.
"""

import json
import os
import numpy as np
import torch

from .head import CustomizationHead


class GestureRegistry:

    def __init__(self, head, device=None,
                 tau_rep=0.35,          # max worst-shot spread (calibrate!)
                 tau_recall=0.70,       # per-class LOO recall floor (matches Ch 7)
                 min_sep_ratio=1.25,    # required ||p_i-p_j|| / (r_i+r_j)
                 tau_null=2.0,          # INFERENCE reject threshold on u = d/r (calibrate!)
                 tau_null_ratio=1.0,    # candidate must sit >= this x background's own kNN scale
                 null_k=10,             # k for the local-density estimate
                 radius_q=0.8,
                 min_radius=0.05,
                 min_tau_rep=0.10):   # floor: a degenerate calibration must not reject everything
        self.head = head.eval()
        self.device = device or next(head.parameters()).device
        self.tau_rep = tau_rep
        self.tau_recall = tau_recall
        self.min_sep_ratio = min_sep_ratio
        self.tau_null = tau_null
        self.tau_null_ratio = tau_null_ratio
        self.null_k = null_k
        self._bg_scale = None
        self.radius_q = radius_q
        self.min_radius = min_radius
        self.min_tau_rep = min_tau_rep
        # percentile of support-to-prototype distance used as the repeatability
        # statistic. 100 = worst-shot (max, the default): the check uses the
        # farthest performance, so a single inconsistent demonstration can
        # reject a gesture. Lower values (e.g. 95) relax this.
        self.rep_stat_pct = 100
        # which validator checks are active; used for the validator-component ablation
        self.enabled_checks = {'repeatability': True, 'distinguishability': True, 'null': True}

        self.names = []                 # list[str]
        self.protos = None              # (C, m)
        self.radii = None               # (C,)
        self._support = {}              # name -> (k, m) kept for LOO re-checks
        self._null_z = None             # (Nb, m) background embeddings, for checks

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def embed(self, backbone_tokens, nucleus_mask=None, batch_size=256):
        """(N, T, 72) -> (N, m) unit-norm."""
        out = []
        for i in range(0, len(backbone_tokens), batch_size):
            h = torch.as_tensor(backbone_tokens[i:i + batch_size],
                                dtype=torch.float32, device=self.device)
            m = None
            if nucleus_mask is not None:
                m = torch.as_tensor(nucleus_mask[i:i + batch_size],
                                    dtype=torch.float32, device=self.device)
            out.append(self.head(h, m).cpu().numpy())
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------ #
    # calibration
    # ------------------------------------------------------------------ #

    def calibrate(self, corpus_support_sets, null_z=None, corpus_groups=None,
                  rep_percentile=95, null_percentile=5, sep_percentile=5,
                  target_fpr=0.05, verbose=True):
        """
        corpus_support_sets : list of (k, m) arrays -- support sets of many
            real gesture classes from users the head did NOT train on. These
            define, empirically, what a genuine gesture looks like.

        Sets two thresholds from that reference distribution:
          tau_rep        worst-shot spread at `rep_percentile` -- a candidate
                         less consistent than 95% of real gestures fails.
          tau_null_ratio background-density ratio at `null_percentile` -- a
                         candidate that sits deeper in the background manifold
                         than 95% of real gestures fails.
          min_sep_ratio  pairwise separation at `sep_percentile`, measured on
                         WITHIN-USER pairs of real gestures (needs
                         `corpus_groups`). A hand-picked value here is the
                         worst of the three: set it at the median of real
                         gesture separation and you reject half of every
                         vocabulary.

        Both are percentiles of real behaviour rather than hand-picked
        constants, so they carry a stated false-rejection budget: at these
        settings roughly 5% of genuine gestures trip each check.

        null_z        : (Nb, m) background embeddings.
        corpus_groups : list of per-user lists of (k, m) support sets. Pairs
                        are only formed within a user, because that is the
                        comparison the validator actually makes.
        """
        spreads = []
        for zc in corpus_support_sets:
            if len(zc) < 2:
                continue
            p = zc.mean(0); p = p / (np.linalg.norm(p) + 1e-8)
            spreads.append(self._rep_spread(zc, p))
        if spreads:
            raw_tau = float(np.percentile(spreads, rep_percentile))
            self.tau_rep = max(raw_tau, self.min_tau_rep)
            if raw_tau < self.min_tau_rep:
                print(f"[calibrate] WARNING: raw tau_rep={raw_tau:.4f} below floor "
                      f"{self.min_tau_rep}; using the floor. "
                      f"This almost always means the calibration classes came from "
                      f"A degenerate value almost always means the calibration "
                      f"classes came from users the head TRAINED on -- their radii "
                      f"have collapsed. Calibrate on a held-out split "
                      f"(meta_train.build_pool --n_calib_users).")

        if null_z is not None and len(null_z) > 0:
            self._null_z = np.asarray(null_z, dtype=np.float32)
            self._bg_scale = None
            ratios = []
            for zc in corpus_support_sets:
                if len(zc) < 2:
                    continue
                p = zc.mean(0); p = p / (np.linalg.norm(p) + 1e-8)
                ratios.append(self._background_ratio(p, self._null_z))
            if ratios:
                self.tau_null_ratio = float(np.percentile(ratios, null_percentile))

        if corpus_groups:
            seps = []
            for grp in corpus_groups:
                pr = []
                for zc in grp:
                    if len(zc) < 2:
                        continue
                    p, r, _ = self._proto_radius(zc, self.radius_q, self.min_radius)
                    pr.append((p, r))
                for i in range(len(pr)):
                    for j in range(i + 1, len(pr)):
                        (pi, ri), (pj, rj) = pr[i], pr[j]
                        seps.append(float(np.linalg.norm(pi - pj)) / (ri + rj + 1e-8))
            if seps:
                self.min_sep_ratio = float(np.percentile(seps, sep_percentile))

        if verbose:
            print(f"[calibrate] tau_rep = {self.tau_rep:.4f} "
                  f"({rep_percentile}th pct of {len(spreads)} reference classes)")
            print(f"[calibrate] tau_null_ratio = {self.tau_null_ratio:.4f} "
                  f"({null_percentile}th pct of reference-class background distance)")
            print(f"[calibrate] min_sep_ratio = {self.min_sep_ratio:.4f} "
                  f"({sep_percentile}th pct of within-user real gesture pairs)")
            print(f"[calibrate] tau_null pending -- call calibrate_reject() "
                  f"once a vocabulary is registered (target FPR {target_fpr:.0%})")
        self._target_fpr = target_fpr
        return self

    def calibrate_reject(self, null_z=None, target_fpr=None, verbose=True):
        """
        Set tau_null from background so that only `target_fpr` of background
        windows get accepted as some gesture. Must run AFTER gestures are
        registered, since u depends on the current prototypes.
        """
        z = null_z if null_z is not None else self._null_z
        if z is None or self.protos is None:
            raise ValueError("need background embeddings and a non-empty vocabulary")
        fpr = target_fpr if target_fpr is not None else getattr(self, '_target_fpr', 0.05)
        _, _, u = self._score(z)
        self.tau_null = float(np.quantile(u, fpr))
        if verbose:
            print(f"[calibrate_reject] tau_null = {self.tau_null:.4f} "
                  f"-> background false-accept {fpr:.1%}")
        return self.tau_null

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def _rep_spread(self, z, p):
        """Repeatability statistic: percentile of sample-to-prototype distance."""
        d = np.linalg.norm(z - p, axis=1)
        if self.rep_stat_pct >= 100:
            return float(d.max())
        return float(np.percentile(d, self.rep_stat_pct))

    @staticmethod
    def _proto_radius(z, radius_q, min_radius):
        p = z.mean(0); p = p / (np.linalg.norm(p) + 1e-8)
        d = np.linalg.norm(z - p, axis=1)
        r = float(np.quantile(d, radius_q)) if len(d) > 1 else float(d.mean())
        return p.astype(np.float32), max(r, min_radius), float(d.max())

    def _bg_knn_scale(self, z_null, max_ref=500):
        """Median k-NN distance WITHIN the background set -- its own local scale."""
        if self._bg_scale is not None:
            return self._bg_scale
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z_null), size=min(max_ref, len(z_null)), replace=False)
        ref = z_null[idx]
        d = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        k = min(self.null_k, len(ref) - 1)
        self._bg_scale = float(np.median(np.sort(d, axis=1)[:, :k].mean(axis=1)))
        return self._bg_scale

    def _background_ratio(self, p, z_null):
        """(candidate's kNN distance into background) / (background's own kNN scale)."""
        k = min(self.null_k, len(z_null))
        d = np.sort(np.linalg.norm(z_null - p[None, :], axis=1))[:k].mean()
        return float(d / (self._bg_knn_scale(z_null) + 1e-8))

    def _score(self, z, protos=None, radii=None):
        """
        (N, m) -> (idx, d_raw, u) where
          idx   = argmin_c ||z - p_c||        <- class decision, RAW distance
          d_raw = the winning raw distance
          u     = d_raw / r_idx               <- reject statistic only

        The class decision must NOT be radius-normalized. Dividing by r_c
        before the argmin makes a high-variance class a black hole: every
        query is "close" to it in normalized units, so it swallows the whole
        vocabulary. Radius normalization is the right scale for "is this
        inside any class ball?" (reject), and the wrong scale for "which
        class is this?" (assignment).
        """
        protos = self.protos if protos is None else protos
        radii = self.radii if radii is None else radii
        d = np.linalg.norm(z[:, None, :] - protos[None, :, :], axis=2)
        idx = d.argmin(axis=1)
        d_raw = d[np.arange(len(z)), idx]
        return idx, d_raw, d_raw / radii[idx]

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #

    def _loo_recall(self, support_by_name):
        """
        Leave-one-out nearest-prototype recall per class. For each held-out
        demo, prototypes are recomputed WITHOUT it, then it is classified.
        This is the deployed rule, so the number predicts deployed behaviour.
        """
        names = list(support_by_name.keys())
        recalls, confusions = {}, {}
        for ni, name in enumerate(names):
            z = support_by_name[name]
            correct, conf = 0, {}
            for i in range(len(z)):
                loo = np.delete(z, i, axis=0)
                if len(loo) == 0:
                    continue
                protos, radii = [], []
                for m in names:
                    zm = loo if m == name else support_by_name[m]
                    p, r, _ = self._proto_radius(zm, self.radius_q, self.min_radius)
                    protos.append(p); radii.append(r)
                protos = np.stack(protos); radii = np.array(radii)
                d = np.linalg.norm(z[i][None, :] - protos, axis=1)   # raw, see _score
                pred = names[int(d.argmin())]
                if pred == name:
                    correct += 1
                else:
                    conf[pred] = conf.get(pred, 0) + 1
            recalls[name] = correct / max(1, len(z))
            confusions[name] = max(conf, key=conf.get) if conf else None
        return recalls, confusions

    def validate(self, name, z_cand, null_z=None):
        """
        Run all three checks for a candidate gesture without committing it.

        z_cand : (k, m) head embeddings of the user's k demonstrations.

        Returns a decision dict with a machine-readable `accept` plus a
        user-facing `message` -- Ch 7.2.2 found users would trust a system
        that says "not consistent enough" (M = 6.30), so the reason matters
        as much as the verdict.
        """
        z_cand = np.asarray(z_cand, dtype=np.float32)
        p_new, r_new, _ = self._proto_radius(z_cand, self.radius_q, self.min_radius)
        spread_new = self._rep_spread(z_cand, p_new)

        d = {'name': name, 'accept': False, 'checks': {}, 'message': '', 'conflict': None}

        # --- 1. repeatability ------------------------------------------ #
        ok_rep = (spread_new <= self.tau_rep) or (not self.enabled_checks['repeatability'])
        d['checks']['repeatability'] = {
            'spread': spread_new, 'threshold': self.tau_rep, 'pass': bool(ok_rep)}
        if not ok_rep:
            d['message'] = ("Your demonstrations of this gesture differ too much from each "
                            "other. Try performing it more consistently, or pick a simpler "
                            "motion.")
            return d

        # --- 2. distinguishability ------------------------------------- #
        support = dict(self._support)
        support[name] = z_cand

        if len(support) > 1:
            recalls, confusions = self._loo_recall(support)
            worst = min(recalls, key=recalls.get)
            ok_recall = (recalls[worst] >= self.tau_recall) or (not self.enabled_checks['distinguishability'])

            # ball separation against every existing prototype
            overlaps = []
            for other in self.names:
                p_o, r_o, _ = self._proto_radius(self._support[other],
                                                 self.radius_q, self.min_radius)
                ratio = float(np.linalg.norm(p_new - p_o)) / (r_new + r_o + 1e-8)
                if ratio < self.min_sep_ratio:
                    overlaps.append((other, ratio))
            ok_overlap = (len(overlaps) == 0) or (not self.enabled_checks['distinguishability'])

            d['checks']['distinguishability'] = {
                'min_recall': recalls[worst], 'worst_class': worst,
                'recall_threshold': self.tau_recall,
                'overlaps': [{'with': o, 'sep_ratio': g} for o, g in overlaps],
                'required_sep_ratio': self.min_sep_ratio,
                'pass': bool(ok_recall and ok_overlap)}

            if not ok_recall:
                d['conflict'] = confusions.get(worst)
                d['message'] = (f"'{worst}' would no longer be recognized reliably "
                                f"({recalls[worst]:.0%}) after adding this gesture"
                                + (f"; it gets confused with '{d['conflict']}'."
                                   if d['conflict'] else "."))
                return d
            if not ok_overlap:
                worst_o = min(overlaps, key=lambda t: t[1])
                d['conflict'] = worst_o[0]
                d['message'] = (f"This gesture is too close to '{worst_o[0]}'. "
                                f"Try changing the direction or the shape of the motion.")
                return d
        else:
            d['checks']['distinguishability'] = {'pass': True, 'note': 'first gesture'}

        # --- 3. null collision ----------------------------------------- #
        z_null = null_z if null_z is not None else self._null_z
        if z_null is not None and len(z_null) > self.null_k:
            ratio = self._background_ratio(p_new, z_null)
            ok_null = (ratio >= self.tau_null_ratio) or (not self.enabled_checks['null'])
            d['checks']['null_collision'] = {
                'density_ratio': ratio,
                'threshold': self.tau_null_ratio, 'pass': bool(ok_null)}
            if not ok_null:
                d['message'] = ("This gesture looks like ordinary hand movement, so it "
                                "would trigger by accident during everyday activity. "
                                "Try a more deliberate motion.")
                return d
        else:
            d['checks']['null_collision'] = {'pass': True, 'note': 'no background set'}

        d['accept'] = True
        d['message'] = "Gesture accepted."
        return d

    # ------------------------------------------------------------------ #
    # mutation
    # ------------------------------------------------------------------ #

    def add_gesture(self, name, z_cand, validate=True, null_z=None, force=False):
        """Validate then append. Returns the decision dict."""
        z_cand = np.asarray(z_cand, dtype=np.float32)
        decision = (self.validate(name, z_cand, null_z=null_z) if validate
                    else {'name': name, 'accept': True, 'checks': {},
                          'message': 'validation skipped'})
        if decision['accept'] or force:
            self._support[name] = z_cand
            self._rebuild()
            decision['added'] = True
        else:
            decision['added'] = False
        return decision

    def remove_gesture(self, name):
        self._support.pop(name, None)
        self._rebuild()

    def _rebuild(self):
        if not self._support:
            self.names, self.protos, self.radii = [], None, None
            return
        self.names = list(self._support.keys())
        protos, radii = [], []
        for n in self.names:
            p, r, _ = self._proto_radius(self._support[n], self.radius_q, self.min_radius)
            protos.append(p); radii.append(r)
        self.protos = np.stack(protos).astype(np.float32)
        self.radii = np.asarray(radii, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def predict(self, z, allow_reject=True):
        """
        z : (N, m) head embeddings.
        Returns (labels, scores) where labels[i] is a gesture name or None
        (rejected as background), and scores[i] is the normalized distance
        (lower = more confident).
        """
        if self.protos is None:
            raise ValueError("empty vocabulary")
        z = np.asarray(z, dtype=np.float32)
        idx, _, u = self._score(z)
        labels = []
        for i, j in enumerate(idx):
            labels.append(None if (allow_reject and u[i] > self.tau_null)
                          else self.names[j])
        return labels, u

    def predict_index(self, z, allow_reject=True):
        """Same, as integer class indices with -1 for reject (for sklearn metrics)."""
        labels, scores = self.predict(z, allow_reject=allow_reject)
        name_to_i = {n: i for i, n in enumerate(self.names)}
        return np.array([-1 if l is None else name_to_i[l] for l in labels]), scores

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez(path + '.npz',
                 protos=self.protos, radii=self.radii,
                 **{f'support::{n}': z for n, z in self._support.items()})
        with open(path + '.json', 'w') as f:
            json.dump({'names': self.names,
                       'tau_rep': self.tau_rep, 'tau_recall': self.tau_recall,
                       'min_sep_ratio': self.min_sep_ratio, 'tau_null': self.tau_null,
                       'tau_null_ratio': self.tau_null_ratio, 'null_k': self.null_k,
                       'radius_q': self.radius_q, 'min_radius': self.min_radius,
                       'min_tau_rep': self.min_tau_rep,
                       'out_dim': int(self.head.out_dim)}, f, indent=2)

    @classmethod
    def load(cls, path, head_ckpt, backbone_dim=72, device=None):
        with open(path + '.json') as f:
            meta = json.load(f)
        head = CustomizationHead(backbone_dim=backbone_dim, out_dim=meta['out_dim'])
        head.load_state_dict(torch.load(head_ckpt, map_location='cpu'))
        head.eval()
        reg = cls(head, device=device,
                  tau_rep=meta['tau_rep'], tau_recall=meta['tau_recall'],
                  min_sep_ratio=meta.get('min_sep_ratio', 1.25),
                  tau_null=meta['tau_null'],
                  tau_null_ratio=meta.get('tau_null_ratio', 1.0),
                  null_k=meta.get('null_k', 10),
                  radius_q=meta['radius_q'], min_radius=meta['min_radius'],
                  min_tau_rep=meta.get('min_tau_rep', 0.10))
        z = np.load(path + '.npz')
        for k in z.files:
            if k.startswith('support::'):
                reg._support[k.split('::', 1)[1]] = z[k]
        reg._rebuild()
        return reg