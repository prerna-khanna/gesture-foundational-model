#!/bin/bash
# Run classifier with contrastive loss on all ablation embeddings

cd /home/prerna/LIMU-BERT-blind-users

echo "===== CLASSIFIER WITH CONTRASTIVE LOSS ON ABLATION EMBEDDINGS ====="
echo ""
echo "Command: python classifier_with_contrastive.py v2 hhar 20_120 -f MODEL -s MODEL_gru_v2 -l 0"
echo ""

DATASET="hhar"
VERSION="20_120"
MODEL_VERSION="v2"
LABEL_INDEX="0"

# Define ablation configurations
configs=(
    "limu_v1_ablation_both"
    "limu_v1_ablation_nucleus_only"
    "limu_v1_ablation_sig_axis_only"
    "limu_v1_ablation_baseline"
)

config_names=(
    "Both (nucleus + sig_axis)"
    "Nucleus only"
    "Sig_axis only"
    "Baseline (neither)"
)

# Run classifier for each ablation model
for idx in "${!configs[@]}"; do
    config="${configs[$idx]}"
    name="${config_names[$idx]}"
    
    echo ""
    echo "================================================================================"
    echo "[$((idx+1))/4] Testing: $name"
    echo "================================================================================"
    echo "Config: $config"
    echo "Embeddings: embed/embed_${config}_${DATASET}_${VERSION}.npy"
    echo ""
    
    # Create model save name
    save_model="${config}_gru_v2"
    
    # Run classifier
    timeout 600 conda run -n limu-bert-env python classifier_with_contrastive.py $MODEL_VERSION $DATASET $VERSION \
        -f "$config" \
        -s "$save_model" \
        -l $LABEL_INDEX
    
    echo ""
done

echo ""
echo "===== CLASSIFIER EVALUATION COMPLETE ====="
echo ""
echo "Checking saved models and results..."
ls -lh saved/classifier*_${VERSION}/ 2>/dev/null | grep "gru_v2" | tail -10
