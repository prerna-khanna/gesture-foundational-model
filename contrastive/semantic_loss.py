import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive import text_variants


class SemanticLoss(nn.Module):
    """Semantic-margin loss driven by a C x C class-similarity matrix.

    `text_mode` selects where that matrix comes from; see contrastive/text_variants.py
    for the full set of arms. The default ('real') reproduces the original
    behaviour exactly -- BERT-encoded descriptions from the dataset config.
    """

    def __init__(self, label_names, descriptions, pooling, device, temperature=0.07, hidden_dim=128,
                 text_mode='real', text_seed=0, vectorized=True, sims_cache=None):
        super().__init__()
        self.pooling = pooling
        self.device = device
        self.temperature = temperature
        self.pooling = pooling
        self.text_mode = text_mode
        self.text_seed = text_seed
        self.vectorized = vectorized
        self.skip_same_class = text_variants.skips_same_class(text_mode)
        self.anchor_info = None
        # Optional path to a precomputed [C, C] class-similarity matrix. BERT's
        # only contribution to this loss is that matrix, so caching it lets a
        # job run without the 439MB model (see precompute_semantic_sims.py).
        self.sims_cache = sims_cache
        self.loaded_from_cache = bool(sims_cache and os.path.exists(sims_cache))

        # Only load BERT for the arms that actually need a text encoder. The
        # semantics-free controls (onehot / random / none) build their anchors
        # directly, so they can run without transformers installed.
        if self.loaded_from_cache:
            self.tokenizer = None
            self.model = None
        elif text_variants.anchor_mode(text_mode) in text_variants.BERT_MODES:
            from transformers import AutoTokenizer, AutoModel
            bert_name = text_variants.bert_model_name()
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
            self.model = AutoModel.from_pretrained(bert_name)
            self.model.to(device)
        else:
            self.tokenizer = None
            self.model = None

        # Single projection layer with correct dimensions.
        # NOTE: this layer is not registered with the optimizer in
        # classifier_with_contrastive.py (which only passes model.parameters()),
        # so in the published setup it acts as a frozen random projection. Left
        # as-is here so every ablation arm is comparable to those numbers.
        self.semantic_projection = nn.Linear(hidden_dim, 32).to(device)

        # Compute semantic similarities between gesture labels
        self.semantic_sims = self.compute_semantic_similarities(label_names, descriptions)

    def compute_semantic_similarities(self, label_names, descriptions):

        """descriptions = [
            f"a {name} gesture with properties: " + 
            f"primary type: {'directional' if name in ['up', 'down', 'left', 'right'] else 'rotational' if 'rotate' in name or name in ['circle'] else 'shape' if name in ['square', 'triangle', 'infinity'] else 'complex'}, " +
            f"direction: {name.split()[0]}, " +
            f"complexity: {'simple' if name in ['up', 'down', 'left', 'right'] else 'complex'}"
            for name in label_names
        ]"""

        if self.loaded_from_cache:
            sims = np.load(self.sims_cache)
            if sims.shape != (len(descriptions), len(descriptions)):
                raise ValueError(f"{self.sims_cache} holds a {sims.shape} matrix but this "
                                 f"dataset has {len(descriptions)} classes")
            self.anchor_info = {"text_mode": self.text_mode, "descriptions_used": None,
                                "source": self.sims_cache}
            print(f"Loaded cached class-similarity matrix from {self.sims_cache} "
                  f"(arm={self.text_mode}); BERT not required")
            return torch.from_numpy(sims).float().to(self.device)

        print("description is ", descriptions)
        print(f"Text ablation arm: {self.text_mode} (seed {self.text_seed})")
        print("Pooling criterion used: " + self.pooling)

        anchors, info = text_variants.build_class_anchors(
            self.text_mode, descriptions, self.pooling, self.device,
            seed=self.text_seed, tokenizer=self.tokenizer, model=self.model)
        self.anchor_info = info

        if info["descriptions_used"] is not None and self.text_mode != "real":
            print("Descriptions actually used by this arm:")
            for name, text in zip(label_names, info["descriptions_used"]):
                print(f"  {name!r} -> {text}")

        # Compute similarity matrix with increased contrast
        return text_variants.anchors_to_similarity(anchors)

    def forward(self, features, labels, epoch=0):
        """
        Compute semantic loss for the features
        """
        # Project features to semantic space
        semantic_features = self.semantic_projection(features)
        semantic_features = F.normalize(semantic_features + 1e-8, p=2, dim=1)

        # Compute distances and loss
        dists = torch.cdist(semantic_features, semantic_features, p=2)
        batch_size = semantic_features.size(0)
        if batch_size < 2:
            # Nothing to compare against. Seeded from the features (times zero)
            # rather than torch.zeros so the result stays graph-connected.
            return semantic_features.sum() * 0.0

        if self.vectorized:
            return self._forward_vectorized(dists, labels, batch_size)
        return self._forward_loop(dists, labels, batch_size, semantic_features)

    def _forward_vectorized(self, dists, labels, batch_size):
        """Batched form of _forward_loop. Same arithmetic, ~20x faster.

        The nested Python loop costs ~B^2/2 interpreter steps per batch and
        dominated total training time (measured on blind_user_filtered: 19s per
        epoch with the loop, 1.3s per epoch with the semantic term removed
        entirely), which made the full ablation sweep infeasible.
        `self.semantic_sims` carries no gradient, so only `dists` is
        differentiable and the two forms agree in the backward pass too.
        """
        labels = labels.long()
        sem = self.semantic_sims[labels][:, labels]

        # Dynamic margin based on similarity
        margin = torch.where(sem > 0.8,
                             torch.full_like(sem, 2.0),
                             torch.where(sem > 0.5,
                                         torch.full_like(sem, 1.5),
                                         torch.full_like(sem, 1.0)))
        # Weight loss based on similarity
        weight = torch.pow(sem, 2) + 0.1
        pair_loss = weight * torch.clamp(margin - dists, min=0.0)

        # Upper triangle only, matching the loop's `for j in range(i + 1, ...)`.
        mask = torch.triu(torch.ones_like(pair_loss, dtype=torch.bool), diagonal=1)
        if self.skip_same_class:
            mask = mask & (labels.view(-1, 1) != labels.view(1, -1))

        return (pair_loss * mask).sum() / (batch_size * (batch_size - 1))

    def _forward_loop(self, dists, labels, batch_size, semantic_features):
        """Original nested-loop implementation. Kept as the reference that
        _forward_vectorized is checked against by test_semantic_loss_vectorized.py."""
        loss = semantic_features.sum() * 0.0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # The `real_nodiag` arm drops same-class pairs. Those always draw
                # sem_sim == 1.0 (the matrix diagonal), so they contribute the
                # largest margin and highest weight in every arm regardless of
                # what the text says -- excluding them isolates the part of the
                # loss that can actually see semantic content.
                if self.skip_same_class and labels[i] == labels[j]:
                    continue
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                # Dynamic margin based on similarity
                margin = 1.0 * (2.0 if sem_sim > 0.8 else
                             1.5 if sem_sim > 0.5 else 1.0)

                # Weight loss based on similarity
                weight = torch.pow(sem_sim, 2) + 0.1
                loss += weight * torch.max(torch.tensor(0.0).to(self.device),
                                       margin - dists[i, j])

        final_loss = loss / (batch_size * (batch_size - 1))
        return final_loss
