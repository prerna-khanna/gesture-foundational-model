#!/bin/bash
# Test embedding generation for ablation models

cd /home/prerna/LIMU-BERT-blind-users

echo "===== EMBEDDING GENERATION TEST ====="
echo ""

DATASET="hhar"
VERSION="20_120"
MODEL_VERSION="v1"

# Test with the "Both" model first
echo "[TEST] Generating embeddings for: limu_v1_ablation_both"
echo "Model file: saved/pretrain_base_${DATASET}_${VERSION}/limu_v1_ablation_both.pt"
echo ""

timeout 300 conda run -n limu-bert-env python embedding.py $MODEL_VERSION $DATASET $VERSION -f limu_v1_ablation_both

echo ""
echo "===== CHECK RESULTS ====="
ls -lh embeddings/*ablation* 2>/dev/null || echo "No ablation embeddings found"
