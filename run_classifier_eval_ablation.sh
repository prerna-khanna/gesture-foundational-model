#!/bin/bash
# Run standard classifier on all ablation embeddings

cd /home/prerna/LIMU-BERT-blind-users

echo "===== CLASSIFIER EVALUATION ON ABLATION EMBEDDINGS ====="
echo ""
echo "Using: python classifier.py v2 hhar 20_120 -f MODEL -s MODEL_gru_v2 -l 0"
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

results_file="ablation_classification_results.txt"
echo "Results Summary" > "$results_file"
echo "================" >> "$results_file"
echo "Date: $(date)" >> "$results_file"
echo "" >> "$results_file"

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
    
    # Run classifier with standard classifier.py
    echo "Running: python classifier.py $MODEL_VERSION $DATASET $VERSION -f $config -s $save_model -l $LABEL_INDEX"
    timeout 600 conda run -n limu-bert-env python classifier.py $MODEL_VERSION $DATASET $VERSION \
        -f "$config" \
        -s "$save_model" \
        -l $LABEL_INDEX 2>&1 | tee -a "$results_file"
    
    echo ""
    echo "---" >> "$results_file"
    echo "" >> "$results_file"
done

echo ""
echo "===== CLASSIFIER EVALUATION COMPLETE ====="
echo ""
echo "Results saved to: $results_file"
echo ""
echo "Checking saved models..."
ls -lh saved/classifier_*_${VERSION}/ 2>/dev/null | grep "gru_v2" | tail -10
echo ""
echo "Summary of test results:"
grep -E "Test Accuracy|Test F1|Test Result" "$results_file" || echo "No summary available"
