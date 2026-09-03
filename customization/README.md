# Customization head for GestureLens

Replaces the Stage-2 text-guided classifier **for the customization task only**.
Stage-1 (the token-based pre-trained encoder) is unchanged and frozen.

Drop the `customization/` folder into the repo root, alongside `models.py`,
`features.py`, `config.py`.

```bash
for u in Alexandra Rahkiya Nihal Angel Julius Turiya John Kerry Roafel Edery; do
    python -m customization.meta_train --held_out $u --episodes 4000
done
python -m customization.evaluate --ablation none
python -m customization.evaluate --ablation no_head
```

## Why not the Stage-2 classifier

`ContrastiveTransformerClassifier` ends in `nn.Linear(hidden, num_classes)`.
Adding a gesture means resizing that head and retraining, so "incremental
fine-tuning" is not architecturally available. It also needs a hand-authored
BERT description per class, which a user inventing a gesture at setup does not
have; it has no reject option, which a continuously streaming wrist IMU needs
more than anything; and using it as the validator makes accept/reject a
property of one stochastic 30-50 epoch training run on 7 samples per class.

`CustomizationHead` (g_phi) has no vocabulary-sized layer. It maps frozen
backbone tokens to a unit-norm metric embedding; classification is nearest
prototype. Adding a gesture is appending a mean vector: no retraining, cost
independent of |V|, ~5 ms measured.

## Files

| file | role |
|---|---|
| `backbone.py` | frozen LIMU encoder, `(N,120,6) -> (N,120,72)`. Mirrors `embedding.py`'s forward path exactly (normalization, nucleus detection, significant-axis mask). Caches to `embed/`. |
| `head.py` | `CustomizationHead` (~20k params) + `EpisodicCustomizationLoss`. |
| `episodes.py` | within-user episode sampler + `MetaAugmentor` (synthesizes novel *classes* at raw-IMU level). |
| `registry.py` | deployed recognizer: prototype store, 3-check validator, open-set reject, save/load. |
| `meta_train.py` | leave-one-user-out episodic training driver. |
| `evaluate.py` | Ch 7 protocol + ablations. |

## The loss

```
L = L_proto + w_rep*L_rep + w_margin*L_margin + w_null*L_null
```

- **L_proto** — prototypical CE over episode queries. The "reorganize the space" term.
- **L_rep** — *worst-shot* compactness hinge. The registration failure mode is one
  sloppy demonstration, not uniformly high variance, so this penalizes the farthest
  support sample rather than the mean.
- **L_margin** — separation as a scale-free ratio `||p_i-p_j|| / (r_i+r_j)`. An
  absolute gap is geometrically infeasible: embeddings are on the unit sphere, so
  `d <= 2`, and ~10 near-orthogonal classes only reach `sqrt(2)`.
- **L_null** — pushes background/ADL motion outside every class radius.

`w_*` ramp in over `warmup_episodes` so the space becomes discriminative before
its geometry is constrained.

## The three registration checks

All closed-form in the same space the recognizer uses, so a rejection is a
prediction about deployed behaviour. Today's `gesture_validator.py` measures
cosine similarity in the frozen backbone's mean-pooled space while recognition
happens in the classifier's learned space — those are different spaces.

1. **Repeatability** — worst-shot spread of the k demos vs `tau_rep`.
2. **Distinguishability** — leave-one-out nearest-prototype recall over *all*
   gestures (a new one can break an old one), plus ball separation vs
   `min_sep_ratio`. Deterministic and O(|V|) arithmetic, not O(|V|) retraining.
3. **Null collision** *(new)* — is the candidate sitting inside the background
   manifold, i.e. something the user does accidentally all day? Measured as a
   local-density ratio: candidate's mean k-NN distance into background over
   background's own median k-NN distance.

   This cannot be done as "what fraction of background falls inside the
   candidate's ball" — a tightly-performed gesture has a tiny ball, so that
   fraction is ~0 even when the gesture sits exactly on the background manifold.

## Rejection at inference

```
c* = argmin_c ||z - p_c||          # class decision: RAW distance
u  = ||z - p_c*|| / r_c*           # reject statistic: radius-normalized
reject if u > tau_null
```

The class decision must not be radius-normalized. Dividing by `r_c` before the
argmin makes a high-variance class a black hole that swallows the vocabulary.

`tau_null` is set by `calibrate_reject()` to hit a target background
false-accept rate, so the operating point is explicit rather than implicit in a
softmax.

## Thresholds are calibrated, never hardcoded

`tau_rep`, `tau_null_ratio` and `min_sep_ratio` are all percentiles of real
gesture behaviour measured on users the head neither trained on nor is tested
on (`--n_calib_users`, default 2, withheld from episode sampling entirely).
Each therefore carries an explicit false-rejection budget: at the defaults,
about 5% of genuine gestures trip each check.

**This matters more than it sounds.** Calibrating on users the head trained on
gives degenerate thresholds — radii collapse on memorized classes, `tau_rep`
lands around 0.01, and the shipped validator rejects nearly every gesture a new
user demonstrates. `min_sep_ratio` was the worst offender: the hardcoded 1.25
sat at the *median* of real within-user gesture separation, so it rejected half
of every vocabulary. There is a floor and a loud warning for the degenerate case.

## Reading the results

Never read `acc_full` without `accepted` beside it. A validator that rejects
nine gestures scores 1.00. The `no_head` ablation demonstrates the pathology
directly: it reaches `acc_full` 0.96 by accepting only 4.0/10.

The bar is **Table 7.5's 0.8057 unfiltered**, not the 0.9614 post-filter, since
keeping more of the gestures users actually chose is the goal.

## Meta-augmentation

Apple's splice (two one-handed gestures -> a synthetic two-handed class) does
not transfer to a single wrist IMU. The principle does: synthesize the class
families customization actually fails on. Ch 7.4.3 says rejections cluster into
repetition-count variants (tap/double/triple) and same-motion-different-direction.
`MetaAugmentor` synthesizes exactly those (`repeat_nucleus`, `flip_axis`,
`time_warp`, `scale_amplitude`) at the raw-IMU level, then re-encodes — doing it
in embedding space would be wrong, since a time-warp in embedding space is not
the embedding of a time-warped signal.

If this works, the headline changes from "we filter out bad gestures" to "we
made the space good enough that you keep the gesture you wanted."

## Known gaps

- `BACKGROUND_SETS` points at HHAR/MotionSense/UCI/Shoaib, which are *activity*
  data. Walking is an easy negative; the hard negatives are idle hand movement.
  Apple curated 600 ADL clips (touching hair, typing, dishes) for exactly this.
  If `bg_fpr` looks too good, that is why.
- Single prototype per class assumes unimodality. A user who performs a gesture
  two distinct ways breaks it; either allow multiple prototypes or rely on the
  repeatability check to catch it.
- `g_phi` is meta-trained on other people, while Ch 4's central finding is that
  blind users have unusually high inter-user variance. Expect the tail users to
  be the story.

## Separately: repo/thesis mismatch worth fixing

- `gesture_validator.py` uses `LogisticRegression` + cosine on mean-pooled
  embeddings, not the text-guided classifier the thesis describes.
- `evaluate_gesture_quality_real.py` loads embeddings but trains on
  `raw_data` (`input_dim = raw_train.shape[-1]` = 6), bypassing the pre-trained
  backbone entirely.
- Its split is a global random 70/10/20 shuffle, not the per-gesture
  "7 train / 3 test" Ch 7 describes.

If the Table 7.4/7.5 numbers came from these scripts, "built on GestureLens" is
not what the code does.
