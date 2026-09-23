"""Text-description controls for the semantic loss (ICLR revision, Tier 1).

The semantic loss consumes the class descriptions in exactly one way: it encodes
them with BERT, builds a C x C class-similarity matrix, and uses that matrix to
set the margin and the weight of a pairwise repulsion term. The text never
touches the forward pass. Every control below therefore differs only in how that
matrix is produced, which is what isolates the contribution of the *content* of
the text from the contribution of merely having a per-class anchor.

Concretely, for a pair of samples (i, j) the loss applies
    margin = 2.0 if sem_sim > 0.8 else 1.5 if sem_sim > 0.5 else 1.0
    weight = sem_sim ** 2 + 0.1
    loss  += weight * relu(margin - dist(i, j))
so the arms differ as follows:

    arm           same-class pair        cross-class pair
    real          margin 2.0, w 1.1      graded 1.0-2.0, w 0.1-1.1
    shuffled      margin 2.0, w 1.1      graded but misaligned
    mismatched    margin 2.0, w 1.1      graded by off-domain similarity
    onehot        margin 2.0, w 1.1      flat 1.0, w 0.1
    random        margin 2.0, w 1.1      flat-ish 1.0, w ~0.1
    real_nodiag   excluded               graded 1.0-2.0, w 0.1-1.1
    none          semantic term disabled entirely

`real` vs `onehot` differ in exactly one thing -- whether cross-class margins are
semantically graded -- which is the tightest available test of whether semantics
matter. `none` is the floor: it removes the term altogether, so a gap there only
shows the extra margin term helps, not that its semantic structure does.

`real_nodiag` exists because the diagonal of the similarity matrix is 1.0 by
construction (normalised anchors), so same-class pairs always draw the largest
margin and the highest weight in every arm. That component is identical across
arms and ignores semantics, so if it dominates the loss magnitude the other arms
will look flat for reasons unrelated to semantic content. Dropping same-class
pairs separates "semantics do not matter" from "the loss is dominated by a term
that cannot see semantics".
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

BERT_DIM = 768


def bert_model_name():
    """Where to load BERT from.

    Defaults to the hub id, but compute clusters frequently cannot reach
    huggingface.co even when pip works against an internal mirror. Set
    GESTURELENS_BERT_PATH to a local directory (see bolt/prefetch_bert.py) to
    load from disk instead.
    """
    return os.environ.get("GESTURELENS_BERT_PATH", "bert-base-uncased")


TEXT_MODES = ("real", "real_nodiag", "shuffled", "mismatched", "onehot",
              "random", "none")

# Arms whose anchors require a text encoder to be loaded at all.
BERT_MODES = ("real", "real_nodiag", "shuffled", "mismatched")

# Arms in the default sweep. `real` is omitted because those numbers already
# exist from the submitted paper; it stays implemented so it can be re-run if a
# reviewer asks for it under the unified model-selection rule.
DEFAULT_ARMS = ("shuffled", "mismatched", "onehot", "random", "real_nodiag", "none")

# Off-domain fillers kept in the *same* template as the real descriptions, so the
# control varies content while holding surface form and length roughly fixed.
# Bare nouns would confound meaning with sentence structure.
OFF_DOMAIN_DESCRIPTIONS = [
    "a Ripe banana item with properties: primary type: fruit, direction: yellow, complexity: simple",
    "a Wooden dining chair item with properties: primary type: furniture, direction: oak, complexity: simple",
    "a Crimson paint swatch item with properties: primary type: colour, direction: warm, complexity: simple",
    "a Sourdough loaf item with properties: primary type: bread, direction: baked, complexity: complex",
    "a Granite countertop item with properties: primary type: stone, direction: speckled, complexity: complex",
    "a Ceramic coffee mug item with properties: primary type: tableware, direction: glazed, complexity: simple",
    "a Woollen winter scarf item with properties: primary type: clothing, direction: knitted, complexity: simple",
    "a Pocket calculator item with properties: primary type: appliance, direction: battery, complexity: complex",
    "a Freshwater trout item with properties: primary type: fish, direction: river, complexity: complex",
    "a Cast iron skillet item with properties: primary type: cookware, direction: seasoned, complexity: simple",
    "a Paperback novel item with properties: primary type: book, direction: fiction, complexity: complex",
    "a Cobalt glass vase item with properties: primary type: decor, direction: blown, complexity: complex",
    "a Leather work boot item with properties: primary type: footwear, direction: laced, complexity: simple",
    "a Mountain spring water item with properties: primary type: drink, direction: bottled, complexity: simple",
    "a Brass door hinge item with properties: primary type: hardware, direction: mounted, complexity: simple",
    "a Cheddar cheese wheel item with properties: primary type: dairy, direction: aged, complexity: complex",
    "a Bamboo cutting board item with properties: primary type: kitchenware, direction: grained, complexity: simple",
    "a Emerald gemstone item with properties: primary type: mineral, direction: faceted, complexity: complex",
    "a Cotton bedsheet item with properties: primary type: linen, direction: woven, complexity: simple",
    "a Rye whiskey bottle item with properties: primary type: spirit, direction: barrelled, complexity: complex",
]


def skips_same_class(mode):
    """Whether this arm excludes same-class pairs from the semantic term."""
    return mode == "real_nodiag"


def anchor_mode(mode):
    """The arm that determines how anchors are built. `real_nodiag` reuses the
    real anchors and differs only in which pairs the loss visits."""
    return "real" if mode == "real_nodiag" else mode


def pool_hidden_states(hidden_states, attention_mask, pooling):
    """Pool BERT token states. Kept identical to the original SemanticLoss
    implementation so that `real` reproduces the published behaviour."""
    if pooling == "cls":
        return hidden_states[:, 0, :]
    if pooling == "mean":
        return torch.stack([h[m == 1].mean(dim=0)
                            for h, m in zip(hidden_states, attention_mask)])
    if pooling == "max":
        return torch.stack([h[m == 1].max(dim=0)[0]
                            for h, m in zip(hidden_states, attention_mask)])
    raise ValueError(f"Unknown pooling strategy: {pooling}")


def encode_descriptions(descriptions, tokenizer, model, pooling, device):
    """BERT-encode a list of descriptions into L2-normalised class anchors."""
    with torch.no_grad():
        inputs = tokenizer(descriptions, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = pool_hidden_states(outputs.last_hidden_state,
                                        inputs["attention_mask"], pooling)
        return F.normalize(embeddings, p=2, dim=1)


def derangement(n, seed):
    """A seeded permutation with no fixed point, so no class keeps its own
    description. Returns the identity for n < 2, where no derangement exists."""
    if n < 2:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def build_class_anchors(mode, descriptions, pooling, device, seed=0,
                        tokenizer=None, model=None, dim=BERT_DIM):
    """Build the C x D class anchor matrix for one arm.

    Returns (anchors, info) where `anchors` is an L2-normalised [C, D] tensor and
    `info` records exactly which text (if any) each class received, so the run
    manifest can be audited later.
    """
    mode = anchor_mode(mode)
    num_classes = len(descriptions)

    if mode == "real":
        used = list(descriptions)
        anchors = encode_descriptions(used, tokenizer, model, pooling, device)

    elif mode == "shuffled":
        perm = derangement(num_classes, seed)
        used = [descriptions[i] for i in perm]
        anchors = encode_descriptions(used, tokenizer, model, pooling, device)

    elif mode == "mismatched":
        if num_classes > len(OFF_DOMAIN_DESCRIPTIONS):
            raise ValueError(
                f"Only {len(OFF_DOMAIN_DESCRIPTIONS)} off-domain descriptions are "
                f"defined but the dataset has {num_classes} classes; add more to "
                f"OFF_DOMAIN_DESCRIPTIONS.")
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(OFF_DOMAIN_DESCRIPTIONS), size=num_classes, replace=False)
        used = [OFF_DOMAIN_DESCRIPTIONS[i] for i in pick]
        anchors = encode_descriptions(used, tokenizer, model, pooling, device)

    elif mode == "onehot":
        used = None
        if num_classes > dim:
            raise ValueError(f"Cannot build {num_classes} one-hot anchors in {dim} dims")
        anchors = torch.eye(num_classes, dim, device=device)

    elif mode == "random":
        used = None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        anchors = torch.randn(num_classes, dim, generator=generator).to(device)
        anchors = F.normalize(anchors, p=2, dim=1)

    else:
        raise ValueError(f"build_class_anchors does not handle mode '{mode}'")

    return anchors, {"text_mode": mode, "descriptions_used": used}


def anchors_to_similarity(anchors, contrast_power=3):
    """Cosine similarity between class anchors, with the same cubic contrast
    enhancement the original SemanticLoss applies."""
    sim = torch.matmul(anchors, anchors.t())
    return torch.pow(sim, contrast_power)
