"""
head.py -- the customization head g_phi and its loss.

This REPLACES ContrastiveTransformerClassifier for the customization task.

Key difference: g_phi has no output layer sized to the vocabulary. It maps
frozen backbone tokens (B, T, 72) to a unit-norm metric embedding (B, m).
Classification is non-parametric (nearest prototype), so adding a gesture is
appending a mean vector -- no head resize, no retraining, O(1) in |V|.

g_phi is trained episodically. It never sees a global class index, only
"class 0..n-1 within this episode", which is what lets it generalize to
gesture classes invented by a user it has never seen.

Loss = L_proto + w_rep*L_rep + w_margin*L_margin + w_null*L_null

  L_proto   prototypical cross-entropy over episode queries. The main
            "reorganize the space" objective.
  L_rep     worst-shot compactness hinge. Penalizes the FARTHEST support
            sample from its prototype, not the average -- the real failure
            mode at registration is one sloppy demonstration, not uniformly
            high variance. This is what makes the repeatability check a
            calibrated statistic instead of an arbitrary cosine threshold.
  L_margin  prototype separation as a SCALE-FREE RATIO,
            ||p_i - p_j|| / (r_i + r_j) >= min_sep_ratio.
            An absolute gap (d >= r_i + r_j + gamma) is geometrically
            infeasible here: embeddings live on the unit sphere, so d <= 2,
            and with ~10 classes the best achievable pairwise distance is
            about sqrt(2). The ratio form has no such ceiling and makes
            "these two balls do not overlap" a transferable statement --
            which is what makes the distinguishability check closed-form
            at registration time.
  L_null    pushes background/ADL motion outside every gesture radius.
            Gives an open-set reject rule that a softmax head cannot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NucleusAttentionPool(nn.Module):
    """
    Attention pooling over the 120 tokens, biased toward the detected nucleus.

    Mean pooling (what ContrastiveTransformerClassifier does) throws away the
    temporal localization that Stage-1 pre-training worked to create. A gesture
    that differs from another only in the nucleus -- which is exactly the
    single/double/triple-tap family that dominates Ch 7.4.3 rejections -- gets
    washed out by averaging over 120 timesteps of mostly preparation and
    retraction.
    """

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.nucleus_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, nucleus_mask=None):
        # h: (B, T, D); nucleus_mask: (B, T) in {0,1} or None
        s = self.score(h).squeeze(-1)                       # (B, T)
        if nucleus_mask is not None:
            s = s + self.nucleus_bias * nucleus_mask
        a = torch.softmax(s, dim=1).unsqueeze(-1)           # (B, T, 1)
        return (h * a).sum(dim=1)                           # (B, D)


class CustomizationHead(nn.Module):
    """
    g_phi: (B, T, backbone_dim) -> (B, out_dim), L2-normalized.

    Deliberately small. It is not meant to learn gestures; it is meant to
    rotate/rescale a space the backbone already built so that few-shot classes
    separate. Keeping it small is also what keeps registration cheap on-device.
    """

    def __init__(self, backbone_dim=72, hidden=128, out_dim=64, dropout=0.1,
                 learn_temperature=True, init_temperature=0.1):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.out_dim = out_dim
        self.pool = NucleusAttentionPool(backbone_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # log-temperature so it stays positive under unconstrained optimization
        self.log_temp = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(init_temperature)))),
            requires_grad=learn_temperature)

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=1e-3, max=10.0)

    def forward(self, h, nucleus_mask=None):
        if h.dim() != 3:
            raise ValueError(f"expected (B, T, D) backbone tokens, got {tuple(h.shape)}")
        if h.shape[-1] != self.backbone_dim:
            raise ValueError(
                f"backbone dim mismatch: head built for {self.backbone_dim}, got {h.shape[-1]}. "
                f"The pretrained LIMU encoder is hidden=72 (config/limu_bert.json base_v1).")
        z = self.mlp(self.pool(h, nucleus_mask))
        return F.normalize(z, dim=-1)          # unit sphere -> d in [0, 2]


# --------------------------------------------------------------------------- #
# geometry helpers -- shared by training AND by the registry at deploy time,
# so the validator measures the same thing the recognizer uses.
# --------------------------------------------------------------------------- #

def prototypes_and_radii(z_support, y_support, n_classes, radius_q=0.8, eps=1e-8):
    """
    z_support: (S, m) unit-norm; y_support: (S,) in [0, n_classes)

    Returns
      protos : (n_classes, m) -- L2-renormalized class means
      radii  : (n_classes,)   -- soft quantile of support-to-prototype distance.
                                 This is the class "ball". Using a high
                                 quantile rather than the max makes it robust
                                 to a single outlier demo while still
                                 reflecting spread.
      spread : (n_classes,)   -- max support-to-prototype distance (the
                                 worst-shot statistic used by L_rep and by
                                 the repeatability check).
    """
    m = z_support.shape[1]
    protos = torch.zeros(n_classes, m, device=z_support.device, dtype=z_support.dtype)
    radii = torch.zeros(n_classes, device=z_support.device, dtype=z_support.dtype)
    spread = torch.zeros(n_classes, device=z_support.device, dtype=z_support.dtype)

    for c in range(n_classes):
        mask = (y_support == c)
        if mask.sum() == 0:
            continue
        zc = z_support[mask]
        p = F.normalize(zc.mean(dim=0), dim=-1)
        protos[c] = p
        d = torch.norm(zc - p.unsqueeze(0), dim=-1)
        spread[c] = d.max()
        if d.numel() == 1:
            radii[c] = d.mean()
        else:
            radii[c] = torch.quantile(d, radius_q)
    return protos, radii.clamp(min=eps), spread


def sq_dists(z, protos):
    """(B, m) x (C, m) -> (B, C) squared euclidean."""
    return torch.cdist(z, protos, p=2).pow(2)


class EpisodicCustomizationLoss(nn.Module):
    """
    Composite loss. All terms operate on within-episode geometry only.

    w_rep / w_margin / w_null ramp in over `warmup_episodes` so the space first
    becomes roughly discriminative (L_proto) before we start constraining its
    geometry -- the same staged-weighting logic as ContrastiveCombinedLoss,
    but with terms that mean something for open-vocabulary customization.
    """

    def __init__(self,
                 w_rep=0.3, w_margin=0.5, w_null=0.5,
                 target_radius=0.35,     # desired class ball radius on the unit sphere
                 min_sep_ratio=1.25,     # required ||p_i-p_j|| / (r_i+r_j)
                 null_margin=0.30,       # how far outside every ball background must sit
                 warmup_episodes=500,
                 radius_q=0.8):
        super().__init__()
        self.w_rep, self.w_margin, self.w_null = w_rep, w_margin, w_null
        self.target_radius = target_radius
        self.min_sep_ratio = min_sep_ratio
        self.null_margin = null_margin
        self.warmup = max(1, warmup_episodes)
        self.radius_q = radius_q

    def _ramp(self, step):
        return min(1.0, step / self.warmup)

    def forward(self, z_support, y_support, z_query, y_query, n_classes,
                z_null=None, step=0, temperature=0.1):
        """
        z_*         : (·, m) unit-norm head outputs
        y_*         : (·,) episode-local class indices in [0, n_classes)
        z_null      : (Nn, m) background samples for this episode, or None
        temperature : pass head.temperature so the learned tau stays in the graph
        """
        protos, radii, spread = prototypes_and_radii(
            z_support, y_support, n_classes, radius_q=self.radius_q)

        parts = {}

        # ---- 1. prototypical CE over queries ---------------------------- #
        d2 = sq_dists(z_query, protos)
        logits = -d2 / temperature
        l_proto = F.cross_entropy(logits, y_query)
        parts['proto'] = l_proto.item()

        r = self._ramp(step)

        # ---- 2. worst-shot repeatability -------------------------------- #
        l_rep = F.relu(spread - self.target_radius).mean()
        parts['rep'] = l_rep.item()

        # ---- 3. radius-normalized prototype separation ------------------ #
        # want: ||p_i - p_j|| / (r_i + r_j) >= min_sep_ratio  for all i != j
        pd = torch.cdist(protos, protos, p=2)
        denom = radii.unsqueeze(0) + radii.unsqueeze(1) + 1e-8
        ratio = pd / denom
        off = ~torch.eye(n_classes, dtype=torch.bool, device=protos.device)
        l_margin = (F.relu(self.min_sep_ratio - ratio)[off].mean()
                    if off.any() else pd.sum() * 0)
        parts['margin'] = l_margin.detach().item()

        # ---- 4. open-set / background ----------------------------------- #
        if z_null is not None and len(z_null) > 0:
            dn = torch.cdist(z_null, protos, p=2)              # (Nn, C)
            dmin, idx = dn.min(dim=1)
            need_null = radii[idx] + self.null_margin
            l_null = F.relu(need_null - dmin).mean()
        else:
            l_null = torch.zeros((), device=protos.device)
        parts['null'] = l_null.detach().item()

        total = (l_proto
                 + r * self.w_rep * l_rep
                 + r * self.w_margin * l_margin
                 + r * self.w_null * l_null)
        parts['total'] = total.item()

        with torch.no_grad():
            parts['acc'] = (logits.argmax(1) == y_query).float().mean().item()
            parts['mean_radius'] = radii.mean().item()
        return total, parts
