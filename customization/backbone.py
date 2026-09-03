"""
backbone.py -- frozen GestureLens feature extractor.

Single source of truth for turning raw IMU (N, 120, 6) into token embeddings
(N, 120, H) with H = model_cfg.hidden (72 for base_v1).

This mirrors embedding.py's forward path exactly -- same normalization, same
nucleus detection, same significant-axis mask -- so that embeddings produced
here are interchangeable with the ones the Stage-2 classifier already consumes.
The difference from embedding.py is that this module is importable (no argparse
side effects) and caches to disk.

The backbone is ALWAYS frozen. Nothing in the customization pipeline ever
writes gradients into it.

Usage
-----
    from customization.backbone import FrozenBackbone

    bb = FrozenBackbone(pretrain_dataset='blind_user', version='20_120')
    emb = bb.encode_dataset('Alexandra')          # (98, 120, 72), cached
    emb = bb.encode_array(raw_np_array)           # ad-hoc, e.g. live samples
"""

import os
import sys
import numpy as np
import torch

# Repo root on path so `models`, `features`, `utils`, `config` resolve.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models import LIMUBertModel4Pretrain                      # noqa: E402
from features import detect_nucleus, compute_energy, calculate_significant_axis  # noqa: E402
from config import load_model_config                            # noqa: E402
from utils import Preprocess4Normalization                      # noqa: E402


def _nucleus_mask_from_points(seq_len, batch_points):
    """Binary (B, T) mask marking the detected nucleus span of each sequence."""
    mask = torch.zeros((len(batch_points), seq_len), dtype=torch.long)
    for i, pts in enumerate(batch_points):
        if len(pts) == 2:
            start, end = int(pts[0]), int(pts[1])
            start = max(0, min(start, seq_len))
            end = max(start, min(end, seq_len))
            mask[i, start:end] = 1
    return mask


class FrozenBackbone:
    """Wraps the pretrained LIMU encoder in eval/no-grad mode."""

    def __init__(self,
                 pretrain_dataset='blind_user',
                 version='20_120',
                 model_file='limu_v1',
                 model_prefix='base',
                 model_version='v1',
                 device=None,
                 cache_dir='embed'):
        self.version = version
        self.cache_dir = os.path.join(_REPO, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        cfg = load_model_config('pretrain', model_prefix, model_version,
                                path_bert=os.path.join(_REPO, 'config/limu_bert.json'))
        if cfg is None:
            raise ValueError(
                f"No pretrain model config for prefix={model_prefix} version={model_version}. "
                f"Check config/limu_bert.json.")
        self.cfg = cfg
        self.hidden = cfg.hidden            # 72 for base_v1
        self.seq_len = cfg.seq_len          # 120
        self.feature_num = cfg.feature_num  # 6

        ckpt = os.path.join(_REPO, 'saved',
                            f'pretrain_{model_prefix}_{pretrain_dataset}_{version}',
                            f'{model_file}.pt')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Pretrained backbone not found at {ckpt}.\n"
                f"Run pretrain.py first, or point --pretrain_dataset at a dataset "
                f"whose saved/pretrain_{model_prefix}_*_{version}/{model_file}.pt exists.")
        self.ckpt_path = ckpt

        self.model = LIMUBertModel4Pretrain(cfg, output_embed=True)
        state = torch.load(ckpt, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            # decoder/linear heads are unused when output_embed=True; only warn on encoder gaps.
            enc_missing = [k for k in missing if k.startswith('transformer')]
            if enc_missing:
                raise RuntimeError(f"Encoder weights missing from checkpoint: {enc_missing[:5]}")

        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.normalizer = Preprocess4Normalization(cfg.feature_num)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_array(self, raw, batch_size=128):
        """raw: (N, T, 6) float ndarray -> (N, T, hidden) float32 ndarray."""
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 3:
            raise ValueError(f"expected (N, T, C), got {raw.shape}")
        if raw.shape[1] != self.seq_len:
            raise ValueError(
                f"sequence length {raw.shape[1]} != backbone seq_len {self.seq_len}. "
                f"Resample/pad upstream -- the positional encodings are length-specific.")
        if raw.shape[2] < self.feature_num:
            raise ValueError(f"need >= {self.feature_num} channels, got {raw.shape[2]}")

        normed = np.stack([self.normalizer(x) for x in raw]).astype(np.float32)

        out = []
        for i in range(0, len(normed), batch_size):
            seqs = torch.from_numpy(normed[i:i + batch_size]).to(self.device)

            energy = compute_energy(seqs)
            nuc_pts = detect_nucleus(energy)
            nuc_mask = _nucleus_mask_from_points(seqs.size(1), nuc_pts).to(self.device)

            sig_axis = calculate_significant_axis(seqs)
            sig_mask = (seqs.argmax(dim=-1) == sig_axis[:, None]).float()

            h = self.model(seqs, nucleus_mask=nuc_mask, sig_axis_mask=sig_mask)
            out.append(h.cpu().numpy().astype(np.float32))

        emb = np.concatenate(out, axis=0)
        assert emb.shape[-1] == self.hidden, (emb.shape, self.hidden)
        return emb

    @torch.no_grad()
    def nucleus_masks(self, raw, batch_size=128):
        """(N, T) binary nucleus masks -- reused by the head's attention pooling."""
        raw = np.asarray(raw, dtype=np.float32)
        normed = np.stack([self.normalizer(x) for x in raw]).astype(np.float32)
        out = []
        for i in range(0, len(normed), batch_size):
            seqs = torch.from_numpy(normed[i:i + batch_size]).to(self.device)
            pts = detect_nucleus(compute_energy(seqs))
            out.append(_nucleus_mask_from_points(seqs.size(1), pts).numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #

    def _cache_path(self, dataset, tag):
        stem = f'{tag}_{os.path.basename(self.ckpt_path)[:-3]}_{dataset}_{self.version}'
        return os.path.join(self.cache_dir, stem + '.npy')

    def encode_dataset(self, dataset, refresh=False, return_masks=False):
        """
        Load dataset/<dataset>/data_<version>.npy, encode, cache.
        Returns (embeddings, labels) or (embeddings, masks, labels).
        """
        data_p = os.path.join(_REPO, 'dataset', dataset, f'data_{self.version}.npy')
        label_p = os.path.join(_REPO, 'dataset', dataset, f'label_{self.version}.npy')
        if not os.path.exists(data_p):
            raise FileNotFoundError(data_p)

        raw = np.load(data_p).astype(np.float32)
        labels = np.load(label_p).astype(np.float32) if os.path.exists(label_p) else None

        emb_p, mask_p = self._cache_path(dataset, 'embed'), self._cache_path(dataset, 'nucmask')
        if os.path.exists(emb_p) and not refresh:
            emb = np.load(emb_p)
        else:
            emb = self.encode_array(raw)
            np.save(emb_p, emb)

        if not return_masks:
            return emb, labels

        if os.path.exists(mask_p) and not refresh:
            masks = np.load(mask_p)
        else:
            masks = self.nucleus_masks(raw)
            np.save(mask_p, masks)
        return emb, masks, labels


def gesture_ids(labels, label_index=0):
    """Repo labels are (N, T, 2) = [gesture_id, user_id] per timestep. -> (N,) int."""
    labels = np.asarray(labels)
    if labels.ndim == 3:
        return labels[:, 0, label_index].astype(int)
    if labels.ndim == 2:
        return labels[:, label_index].astype(int)
    return labels.astype(int)


def user_ids(labels):
    return gesture_ids(labels, label_index=1)
