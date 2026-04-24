#!/bin/bash
# Script to run pretraining with different nucleus masking ratios
# Usage: bash pretrain_multiple_ratios.sh

# Activate conda environment
source ~/.bashrc
conda activate limu-bert-env

# Define the ratios to test
RATIOS=(0.5 0.7 0.8 0.9)
DATASET="hhar"
VERSION="20_120"
MODEL_VERSION="v1"

# Create a tmux session
SESSION_NAME="pretrain_nucleus_ratios"
tmux new-session -d -s $SESSION_NAME

# Function to run pretraining with a specific ratio
run_pretraining() {
    local ratio=$1
    local window_name="pretrain_${ratio}"
    
    echo "Starting pretraining with nucleus_prob=$ratio"
    
    # Create a new window for this ratio
    if [ "$ratio" = "0.5" ]; then
        # Use the first window for the first ratio
        tmux send-keys -t $SESSION_NAME "cd /home/prerna/LIMU-BERT-blind-users && python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_nucleus_${ratio}" C-m
    else
        # Create new windows for subsequent ratios
        tmux new-window -t $SESSION_NAME -n $window_name
        tmux send-keys -t $SESSION_NAME:$window_name "cd /home/prerna/LIMU-BERT-blind-users && python pretrain.py $MODEL_VERSION $DATASET $VERSION -s limu_v1_nucleus_${ratio}" C-m
    fi
}

# Run pretraining for each ratio
for ratio in "${RATIOS[@]}"; do
    run_pretraining $ratio
    echo "Submitted job for nucleus_prob=$ratio"
    sleep 2  # Small delay between submissions
done

echo "All pretraining jobs submitted!"
echo "Monitor progress with: tmux attach-session -t $SESSION_NAME"
echo ""
echo "Available windows:"
tmux list-windows -t $SESSION_NAME
