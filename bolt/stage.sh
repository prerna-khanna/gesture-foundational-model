#!/usr/bin/env bash
# Build a minimal staging directory for `bolt task submit --tar .`
#
# The repo root is ~2.6GB (dataset/ 554M, saved/ 512M, contrasense_imp/ 116M,
# ...), almost none of which the ablation needs. Tarring the root would upload
# all of it and appear to hang. Per BOLT_README section 9.2, the robust fix is to
# submit from a directory containing only what the job actually uses -- roughly
# 100MB here, virtually all of it the frozen encoder embeddings.
#
#   bash bolt/stage.sh
#   cd /tmp/gesturelens_stage
#   bolt task submit --tar . --config bolt/config.yaml --max-retries 2
set -eux

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-/tmp/gesturelens_stage}"
# Pinned to the Tier 0 datasets so both tiers report over the same data.
DATASETS=(sony_watch blind_user_filtered)

# Refuse to delete the staging directory if the caller's shell is inside it.
# `rm -rf "$STAGE"` would otherwise pull the ground out from under the current
# working directory, and every later command fails with
# "getcwd: cannot access parent directories".
case "$PWD/" in
  "$STAGE"/*)
    echo "ERROR: you are inside $STAGE, which this script deletes and rebuilds."
    echo "       Run it from the repo instead:"
    echo "         cd $REPO && bash bolt/$(basename "${BASH_SOURCE[0]}")"
    exit 1
    ;;
esac

rm -rf "$STAGE"
mkdir -p "$STAGE"/{contrastive,tests,bolt,config,dataset,new_embed}

# Import closure of run_text_ablation.py / analyze_semantic_structure.py.
for f in run_text_ablation.py analyze_semantic_structure.py \
         train.py utils.py config.py statistic.py plot.py datasets_common.py; do
  cp "$REPO/$f" "$STAGE/$f"
done

cp "$REPO"/contrastive/*.py "$STAGE/contrastive/"
cp "$REPO"/tests/test_semantic_loss_vectorized.py "$STAGE/tests/"
cp "$REPO"/bolt/config.yaml "$REPO"/bolt/task_setup.sh "$REPO"/bolt/run_tier1.sh "$STAGE/bolt/"
cp "$REPO"/config/*.json "$STAGE/config/"

# Only the label arrays are needed -- the raw IMU data is not read, because the
# classifier trains on the frozen embeddings in new_embed/.
cp "$REPO/dataset/data_config.json" "$STAGE/dataset/"
for d in "${DATASETS[@]}"; do
  mkdir -p "$STAGE/dataset/$d"
  cp "$REPO/dataset/$d/label_20_120.npy" "$STAGE/dataset/$d/"
  cp "$REPO/new_embed/embed_limu_v1_${d}_20_120.npy" "$STAGE/new_embed/"
done

# Vendored BERT, if it has been prefetched. Optional: without it the job falls
# back to downloading from the hub, which only works if the compute node has
# outbound internet.
if [ -d "$REPO/assets/bert-base-uncased" ]; then
  mkdir -p "$STAGE/assets"
  cp -R "$REPO/assets/bert-base-uncased" "$STAGE/assets/"
  echo "Staged vendored BERT."
else
  echo "WARNING: no assets/bert-base-uncased -- the real/shuffled/mismatched arms"
  echo "         will try to reach huggingface.co from the compute node."
  echo "         Run 'python bolt/prefetch_bert.py' first to vendor it."
fi

# Bolt filters the tar by .gitignore; give it one scoped to the staging dir.
cat > "$STAGE/.gitignore" <<'EOF'
__pycache__/
*.pyc
.DS_Store
saved/
artifacts/
EOF

echo
echo "Staged to $STAGE ($(du -sh "$STAGE" | cut -f1))"
du -sh "$STAGE"/* | sort -rh
echo
echo "Next:"
echo "  cd $STAGE && bolt task submit --tar . --config bolt/config.yaml --ignore-file .gitignore --max-retries 2"
