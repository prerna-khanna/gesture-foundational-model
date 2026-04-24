#!/bin/bash
# Simple ablation test - train 4 models for 10 epochs each + embeddings

cd /home/prerna/LIMU-BERT-blind-users

echo "===== ABLATION TEST: 4 MODELS x 10 EPOCHS ====="
echo ""
echo "Using 10-epoch config: config/pretrain_10epochs.json"
echo ""

# Config
export DATASET="hhar"
export VERSION="20_120"
export MODEL_VERSION="v1"
export CONFIG="-t config/pretrain_10epochs.json"

# Ablation 1: Both (nucleus + sig_axis)
echo "[1/4] Training: Both (nucleus + sig_axis)"
conda run -n limu-bert-env python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_ablation_both $CONFIG
echo "[1/4] Generating embeddings for Both..."
conda run -n limu-bert-env python embedding.py $MODEL_VERSION $DATASET $VERSION -f limu_v1_ablation_both
echo ""

# Ablation 2: Nucleus only
echo "[2/4] Training: Nucleus only"
conda run -n limu-bert-env python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_ablation_nucleus_only $CONFIG
echo "[2/4] Generating embeddings for Nucleus-only..."
conda run -n limu-bert-env python embedding.py $MODEL_VERSION $DATASET $VERSION -f limu_v1_ablation_nucleus_only
echo ""

# Ablation 3: Sig_axis only
echo "[3/4] Training: Sig_axis only"
conda run -n limu-bert-env python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_ablation_sig_axis_only $CONFIG
echo "[3/4] Generating embeddings for Sig_axis-only..."
conda run -n limu-bert-env python embedding.py $MODEL_VERSION $DATASET $VERSION -f limu_v1_ablation_sig_axis_only
echo ""

# Ablation 4: Baseline (neither)
echo "[4/4] Training: Baseline (neither)"
conda run -n limu-bert-env python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_ablation_baseline $CONFIG
echo "[4/4] Generating embeddings for Baseline..."
conda run -n limu-bert-env python embedding.py $MODEL_VERSION $DATASET $VERSION -f limu_v1_ablation_baseline
echo ""

echo "===== ABLATION TEST COMPLETE ====="
ls -lh saved/pretrain_base_hhar_20_120/limu_v1_ablation*.pt
