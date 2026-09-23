#!/usr/bin/env bash
# Tier 0 job body: extract frozen embeddings for every baseline encoder, then
# run the encoder-vs-classifier grid. Self-contained -- no network access needed,
# because BERT and harnet weights are vendored into assets/ by stage_tier0.sh.
set -eux

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/tier0"
mkdir -p "$OUT" saved new_embed

# Guard the whole run: if the vectorised semantic loss ever diverges from the
# original nested-loop reference, every `full` cell in the grid is suspect.
python tests/test_semantic_loss_vectorized.py

DATASETS="sony_watch blind_user_filtered"
SEEDS="3431 1234 2024"

# Preflight: the `full` classifier reads a precomputed [C, C] similarity matrix
# instead of loading BERT. A missing file falls back to BERT, which has no
# weights and no network here -- and that would surface hours into the run.
# Fail immediately instead.
python - <<PREFLIGHT
import os, sys
missing = [p for d in "$DATASETS".split() for s in "$SEEDS".split()
           for p in [f"assets/semantic_sims/sims_{d}_real_cls_seed{s}.npy"]
           if not os.path.exists(p)]
if missing:
    sys.exit("Missing similarity matrices (run precompute_semantic_sims.py "
             "and re-stage):\n  " + "\n  ".join(missing))
print("preflight ok: all similarity matrices present, BERT not required")
PREFLIGHT

# --- Extraction -------------------------------------------------------------
# Each extractor refuses to write if zero pretrained tensors matched, so the
# silent no-op in contrasense_imp/run_full.py (0/48 keys) cannot recur here.
#
# Extraction failures are NOT fatal. The ContrastSense and harnet extractors
# import those projects' own source, so they depend on those repos' transitive
# requirements being present -- and a single missing package would otherwise
# throw away the whole run. Drop the failed encoder and carry on; the summary at
# the end says plainly which encoders are in the grid and which are missing.
ENCODERS=""
set +e

python extract_unihar_embeddings.py \
  --ckpt unihar_impl/unihar_bert_fed_d2.pt --datasets $DATASETS \
  && ENCODERS="$ENCODERS unihar" || echo "!!! UniHAR extraction FAILED, dropping it"

python extract_contrastsense_embeddings.py \
  --repo third_party/ContrastSense-Public \
  --ckpt contrasense_imp/HHAR/model_best.pt --datasets $DATASETS \
  && ENCODERS="$ENCODERS contrastsense" || echo "!!! ContrastSense extraction FAILED, dropping it"

python extract_yuan_embeddings.py \
  --repo third_party/ssl-wearables \
  --weights assets/harnet10.pt --datasets $DATASETS \
  && ENCODERS="$ENCODERS yuan" || echo "!!! Yuan extraction FAILED, dropping it"

python extract_biopm_embeddings.py \
  --repo third_party/biopm --datasets $DATASETS \
  && ENCODERS="$ENCODERS biopm" || echo "!!! BioPM extraction FAILED, dropping it"

set -e
ENCODERS=$(echo $ENCODERS | xargs)
if [ -z "$ENCODERS" ]; then
  echo "Every extractor failed; nothing to run."
  exit 1
fi
echo "=== encoders entering the grid: $ENCODERS ==="

# Embeddings stay OUT of the artifact dir: they are ~130MB, fully regenerable
# from the shipped checkpoints, and BOLT_README section 5.3 warns against
# syncing throwaway intermediates. Only the tables and CSVs are artifacts.
ls -la new_embed/

# --- Grid -------------------------------------------------------------------
# GestureLens (limu_v1) is intentionally absent; its cells come from Table 4.
python run_encoder_grid.py \
  --encoders $ENCODERS \
  --classifiers simple full \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --embed_dir new_embed \
  --out_dir "$OUT/grid"

echo "=== Tier 0 grid ==="
cat "$OUT/grid/table.md"
