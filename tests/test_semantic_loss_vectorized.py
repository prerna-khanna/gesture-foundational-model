#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checks that the vectorised semantic loss matches the original nested loop.

The loop dominated training time (~95% of it), so the ablation sweep runs the
vectorised path. That is only safe if the two agree in both value and gradient,
which is what this asserts -- across every arm, including the single-class edge
case and the same-class-skipping `real_nodiag` arm.

    python tests/test_semantic_loss_vectorized.py
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub transformers UNCONDITIONALLY. The anchors are stubbed below, so the
# tokenizer/model objects are constructed but never used -- and this test must
# run where bert-base-uncased is absent and there is no network (the Bolt job
# ships 7KB of precomputed similarity matrices instead of the 439MB model).
# Deferring to a real `transformers` here would make the test try to reach
# huggingface.co and fail the whole job before any training starts.
_stub = types.ModuleType("transformers")


class _StubBert:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return _StubBert()

    def to(self, *a):
        return self


_stub.AutoTokenizer = _StubBert
_stub.AutoModel = _StubBert
sys.modules["transformers"] = _stub

import contrastive.text_variants as tv  # noqa: E402
from contrastive.semantic_loss import SemanticLoss  # noqa: E402

_real_build = tv.build_class_anchors


def _stub_build(mode, descriptions, pooling, device, seed=0, tokenizer=None, model=None, dim=768):
    """Deterministic stand-in for BERT with deliberately varied similarity, so
    the test exercises all three margin bands (>0.8, >0.5, else)."""
    if tv.anchor_mode(mode) in tv.BERT_MODES:
        g = torch.Generator().manual_seed(11)
        a = torch.randn(len(descriptions), 24, generator=g)
        a[1] = a[0] + 0.03 * a[1]   # near-duplicate -> lands in the >0.8 band
        a[2] = a[0] + 0.90 * a[2]   # moderately similar -> the >0.5 band
        return torch.nn.functional.normalize(a, dim=1).to(device), {
            "text_mode": mode, "descriptions_used": list(descriptions)}
    return _real_build(mode, descriptions, pooling, device, seed, tokenizer, model, dim)


def build(mode, num_classes, hidden_dim, vectorized):
    names = [f"class_{i}" for i in range(num_classes)]
    descriptions = [f"a {n} gesture" for n in names]
    torch.manual_seed(0)   # identical semantic_projection init in both variants
    return SemanticLoss(names, descriptions, "cls", "cpu",
                        hidden_dim=hidden_dim, text_mode=mode, vectorized=vectorized)


def check(mode, num_classes=6, batch_size=32, hidden_dim=64):
    torch.manual_seed(1)
    features = torch.randn(batch_size, hidden_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    results = {}
    for vectorized in (False, True):
        loss_fn = build(mode, num_classes, hidden_dim, vectorized)
        x = features.clone().requires_grad_(True)
        value = loss_fn(x, labels)
        value.backward()
        results[vectorized] = (value.detach(), x.grad.clone())

    (loop_val, loop_grad), (vec_val, vec_grad) = results[False], results[True]
    value_gap = (loop_val - vec_val).abs().item()
    grad_gap = (loop_grad - vec_grad).abs().max().item()

    assert value_gap < 1e-5, f"{mode}: value mismatch {value_gap}"
    assert grad_gap < 1e-5, f"{mode}: gradient mismatch {grad_gap}"
    print(f"  {mode:<12} loss={loop_val.item():.6f}  "
          f"|dvalue|={value_gap:.2e}  |dgrad|max={grad_gap:.2e}  ok")


def check_single_class(mode="real_nodiag", num_classes=6, batch_size=8, hidden_dim=64):
    """real_nodiag skips every pair when a batch holds one class; both paths must
    return a finite, backward-able zero rather than raising."""
    torch.manual_seed(2)
    features = torch.randn(batch_size, hidden_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for vectorized in (False, True):
        loss_fn = build(mode, num_classes, hidden_dim, vectorized)
        x = features.clone().requires_grad_(True)
        value = loss_fn(x, labels)
        value.backward()
        assert value.item() == 0.0, f"expected 0, got {value.item()}"
    print(f"  {'single-class':<12} both paths return 0 and backward cleanly  ok")


if __name__ == "__main__":
    tv.build_class_anchors = _stub_build
    import contrastive.semantic_loss as sl
    sl.text_variants = tv

    print("Vectorised vs nested-loop semantic loss:")
    for arm in ("real", "real_nodiag", "shuffled", "mismatched", "onehot", "random"):
        check(arm)
    check_single_class()

    # Batch sizes that stress the triangle mask and the odd/even boundary.
    for batch_size in (2, 3, 17, 64):
        check("real", batch_size=batch_size)
    print("\nAll equivalence checks passed.")
