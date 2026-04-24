#!/bin/bash

# Run contrastive classifier on all HHAR ablation embeddings with correct label index
cd /home/prerna/LIMU-BERT-blind-users

echo "Running contrastive classifier on HHAR ablation embeddings..."
echo "Using activity labels (label_index=2 for HHAR)"
echo "=================================================="

# Array of ablation configurations
ablation_configs=("limu_v1_ablation_both" "limu_v1_ablation_nucleus_only" "limu_v1_ablation_sig_axis_only" "limu_v1_ablation_baseline")

for config in "${ablation_configs[@]}"; do
    save_model="${config}_gru_v2"
    echo ""
    echo "Running classifier for: $config"
    echo "Save model: $save_model"
    echo "---"
    # Use -l 2 for HHAR activity labels (index 2, size 6)
    conda run -n limu-bert-env python classifier_with_contrastive.py v2 hhar 20_120 -f "$config" -s "$save_model" -l 2
    echo "Completed: $config"
done

echo ""
echo "=================================================="
echo "All classifier evaluations completed!"
