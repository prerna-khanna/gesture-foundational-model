#!/bin/bash
# Script to run a complete ablation study for LIMU-BERT
# Testing different combinations of nucleus and significant axis features

# Set the dataset and version to use
DATASET="hhar"
VERSION="20_120"
LABEL_INDEX=0  # activity label
LABEL_RATE=0.01  # proportion of labeled data to use
MODEL_VERSION="v1"

echo "Starting LIMU-BERT ablation study for $DATASET dataset ($VERSION)"
echo "=============================================================="

# Create directories if they don't exist
mkdir -p embed
mkdir -p saved/pretrain_base_${DATASET}_${VERSION}

# Step 1: Run the original pretraining (with both features)
#echo "Step 1: Pretraining original LIMU-BERT model (with both features)"
#python pretrain_ablation.py $MODEL_VERSION $DATASET $VERSION -s limu_original --ablation "" -t config/pretrain.json -g 0

# Step 2: Generate embeddings with the original model
#echo "Step 2: Generating embeddings with the original model"
#python embedding_ablation.py $MODEL_VERSION $DATASET $VERSION -f limu_original --ablation "" -t config/pretrain.json -g 0


# Step 4: Run the ablation pretraining (without nucleus, with sig_axis)
echo "Step 4: Pretraining model without nucleus, with sig_axis"
python pretrain_ablation.py $MODEL_VERSION $DATASET $VERSION -s limu_no_nucleus --ablation  "" -t config/pretrain.json -g 0

# Step 5: Generate embeddings (without nucleus, with sig_axis)
echo "Step 5: Generating embeddings without nucleus, with sig_axis"
python embedding_ablation.py $MODEL_VERSION $DATASET $VERSION -f limu_no_nucleus --ablation "" -t config/pretrain.json 

# Step 7: Run the ablation pretraining (with nucleus, without sig_axis)
echo "Step 7: Pretraining model with nucleus, without sig_axis"
python pretrain_ablation.py $MODEL_VERSION $DATASET $VERSION -s limu_no_sigaxis --ablation  "" -t config/pretrain.json -g 0

# Step 8: Generate embeddings (with nucleus, without sig_axis)
echo "Step 8: Generating embeddings with nucleus, without sig_axis"
python embedding_ablation.py $MODEL_VERSION $DATASET $VERSION -f limu_no_sigaxis --ablation "" -t config/pretrain.json


# Step 10: Run the ablation pretraining (without both features)
#echo "Step 10: Pretraining model without both features"
#python pretrain_ablation.py $MODEL_VERSION $DATASET $VERSION -s limu_no_both --ablation "nucleus,sig_axis" -t config/pretrain.json

# Step 11: Generate embeddings (without both features)
#echo "Step 11: Generating embeddings without both features"
#python embedding_ablation.py $MODEL_VERSION $DATASET $VERSION -f limu_no_both --ablation "nucleus,sig_axis" -t config/pretrain.json

echo "Ablation study complete!"
echo "=============================================================="
echo "Results saved"
