#!/usr/bin/env bash
# Build a minimal staging directory for the Tier 0 encoder grid.
#
# Ships encoder CHECKPOINTS + RAW DATA rather than precomputed embeddings, and
# vendors the two baseline repos (we import their real encoder classes rather
# than reconstructing architectures from tensor shapes) plus the BERT and harnet
# weights, so the job needs no network access.
#
#   bash bolt/stage_tier0.sh
#   cd /tmp/gesturelens_tier0
#   bolt task submit --tar . --config bolt/config_tier0.yaml \
#     --ignore-file .gitignore --max-retries 2
set -eux

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-/tmp/gesturelens_tier0}"
CONTRASTSENSE_SRC="${CONTRASTSENSE_SRC:-$HOME/Desktop/ContrastSense-Public}"
SSLWEARABLES_SRC="${SSLWEARABLES_SRC:-$HOME/Desktop/ssl-wearables}"
BIOPM_SRC="${BIOPM_SRC:-$HOME/Desktop/biopm}"
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
mkdir -p "$STAGE"/{contrastive,tests,bolt,config,dataset,new_embed,assets,third_party,unihar_impl,contrasense_imp}

# --- our code ---------------------------------------------------------------
for f in run_encoder_grid.py run_text_ablation.py analyze_semantic_structure.py \
         extract_unihar_embeddings.py extract_contrastsense_embeddings.py \
         extract_yuan_embeddings.py extract_biopm_embeddings.py \
         train.py utils.py config.py statistic.py plot.py models.py datasets_common.py; do
  cp "$REPO/$f" "$STAGE/$f"
done
cp "$REPO"/contrastive/*.py "$STAGE/contrastive/"
cp "$REPO"/tests/test_semantic_loss_vectorized.py "$STAGE/tests/"
cp "$REPO"/bolt/config_tier0.yaml "$REPO"/bolt/task_setup.sh "$REPO"/bolt/run_tier0.sh "$STAGE/bolt/"
cp "$REPO"/config/*.json "$STAGE/config/"

# --- raw data (inputs to extraction, far smaller than the embeddings) -------
cp "$REPO/dataset/data_config.json" "$STAGE/dataset/"
for d in "${DATASETS[@]}"; do
  mkdir -p "$STAGE/dataset/$d"
  cp "$REPO/dataset/$d/data_20_120.npy" "$REPO/dataset/$d/label_20_120.npy" "$STAGE/dataset/$d/"
done

# --- encoder checkpoints ----------------------------------------------------
cp "$REPO"/unihar_impl/unihar_bert_fed_d2.pt "$STAGE/unihar_impl/"
# The ContrastSense checkpoint is 114MB of MoCo pair + queues + optimizer state,
# of which we need only the 42 `encoder_q.encoder.*` tensors. Slim it here rather
# than uploading the rest.
mkdir -p "$STAGE/contrasense_imp/HHAR"
python3 - "$REPO/contrasense_imp/HHAR/model_best.pth" "$STAGE/contrasense_imp/HHAR/model_best.pt" <<'SLIM'
import os, sys, tempfile, zipfile, torch
src, dst = sys.argv[1], sys.argv[2]
if os.path.isdir(src):
    tmp = tempfile.mktemp(suffix=".pt")
    with zipfile.ZipFile(tmp, "w") as z:
        for root, _, files in os.walk(src):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.join("model_best", os.path.relpath(full, src)))
    src = tmp
sd = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
keep = {k: v for k, v in sd.items() if k.startswith("encoder_q.encoder.")}
assert keep, "no encoder_q.encoder.* tensors found"
torch.save({"state_dict": keep}, dst)
print(f"slimmed ContrastSense: {len(sd)} -> {len(keep)} tensors, "
      f"{os.path.getsize(dst)/1e6:.1f}MB")
SLIM

# --- vendored weights -------------------------------------------------------
# BERT contributes exactly one thing to the semantic loss: a [C, C] similarity
# matrix. Shipping the precomputed matrices (7KB) instead of bert-base-uncased
# (439MB) keeps the payload small and the job offline.
for a in semantic_sims harnet10.pt; do
  if [ ! -e "$REPO/assets/$a" ]; then
    echo "MISSING $REPO/assets/$a"
    echo "  semantic_sims -> python precompute_semantic_sims.py"
    echo "  harnet10.pt   -> see extract_yuan_embeddings.py (needs internet)"
    exit 1
  fi
  cp -R "$REPO/assets/$a" "$STAGE/assets/"
done

# --- baseline repos ---------------------------------------------------------
# We import ContrastSense_encoder and harnet's Resnet from their own source.
# Reconstructing them from checkpoint shapes would risk silently-wrong
# embeddings, which is exactly the class of bug this grid is meant to avoid.
for pair in "ContrastSense-Public:$CONTRASTSENSE_SRC" "ssl-wearables:$SSLWEARABLES_SRC" "biopm:$BIOPM_SRC"; do
  name="${pair%%:*}"; src="${pair#*:}"
  if [ ! -d "$src" ]; then
    echo "MISSING $name at $src. Clone it, or set CONTRASTSENSE_SRC / SSLWEARABLES_SRC."
    exit 1
  fi
  # model_check_point/ alone is 96MB of their own weights; we vendored harnet10
  # separately. Only the source is needed, since we just import their classes.
  # BioPM ships its three pretrained checkpoints INSIDE the repo, so the blanket
  # *.pt exclusion must not apply to it or we would strip what we came for.
  if [ "$name" = "biopm" ]; then
    rsync -a --exclude='.git' --exclude='__pycache__' "$src/" "$STAGE/third_party/$name/"
  else
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pt' --exclude='*.pth' \
          --exclude='model_check_point' --exclude='interpretability' --exclude='plots' \
          --exclude='data' --exclude='data_parsing' --exclude='tests' \
          "$src/" "$STAGE/third_party/$name/"
  fi
done

cat > "$STAGE/.gitignore" <<'EOF'
__pycache__/
*.pyc
.DS_Store
saved/
artifacts/
new_embed/
EOF

echo
echo "Staged to $STAGE ($(du -sh "$STAGE" | cut -f1))"
du -sh "$STAGE"/* | sort -rh
echo
echo "Next:"
echo "  cd $STAGE && bolt task submit --tar . --config bolt/config_tier0.yaml --ignore-file .gitignore --max-retries 2"
