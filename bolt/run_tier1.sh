#!/usr/bin/env bash
# Tier 1 job body: the text-description ablation.
#
# Lives in a script rather than inline in config.yaml because a YAML `>` folded
# scalar keeps more-indented lines literal instead of folding them, which turned
# a wrapped `python ... \n --confusion ...` into two commands and killed the run
# with exit 127 after the sweep had already finished.
#
# Datasets are pinned to the two the Tier 0 grid uses, so both tiers report over
# the same data: sony_watch (Hand SU) and blind_user_filtered (Hand BU).
set -eux

ART="${BOLT_ARTIFACT_DIR:-./artifacts}"
OUT="$ART/text_ablation"
mkdir -p "$OUT" saved

# Only sony_watch (Hand SU). blind_user_filtered was already completed by an
# earlier run with the same arms and the same three seeds, so re-running it would
# just burn compute -- combine the two tables by hand.
#
# The code has changed since that run (sims cache, variable-length merge,
# use_contrastive switch), but none of it alters the arithmetic for a 120-step
# embedding: the cached similarity matrix is the same matrix BERT produced, and
# the merge path is identical when seq_len == 120. So those numbers remain valid.
DATASETS="sony_watch"

# Prefer the vendored BERT; fall back to the hub only if it is absent.
export GESTURELENS_BERT_PATH="${GESTURELENS_BERT_PATH:-$PWD/assets/bert-base-uncased}"
if [ ! -d "$GESTURELENS_BERT_PATH" ]; then
  echo "No vendored BERT at $GESTURELENS_BERT_PATH, falling back to the hub"
  unset GESTURELENS_BERT_PATH
fi

python tests/test_semantic_loss_vectorized.py

python run_text_ablation.py \
  --datasets $DATASETS \
  --embed_dir new_embed \
  --out_dir "$OUT"

# --out_dir must point INSIDE the artifact dir; the script's default is
# results/semantic_structure, which does not get synced and was lost last run.
python analyze_semantic_structure.py \
  --datasets $DATASETS \
  --confusion "$OUT/details.json" \
  --out_dir "$ART/semantic_structure"

echo "=== Tier 1 text ablation ==="
cat "$OUT/table.md"
