#!/bin/bash
# Quick srun command for same-action evaluation
# Usage: bash scripts/srun_same_action_eval.sh [dataset] [num_pairs]

DATASET=${1:-ntu_cv}
NUM_PAIRS=${2:-100}

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                    SAME-ACTION EVALUATION - SRUN                             ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Number of Pairs: $NUM_PAIRS"
echo ""
echo "Submitting job to SLURM..."
echo ""

# Create logs directory
mkdir -p logs

# Run with srun
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=32G \
     --time=01:00:00 \
     --job-name=same_action_eval \
     --output=logs/same_action_eval_%j.out \
     --error=logs/same_action_eval_%j.err \
     bash -c "
         source ~/.bashrc
         conda activate pytorch-2.3.0 || conda activate base
         cd /users/tcarr23/Transformer-Retargeting
         python scripts/eval_same_action.py \
             --dataset $DATASET \
             --num_pairs $NUM_PAIRS \
             --device cuda \
             --seed 42
     "

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully!"
else
    echo "❌ Job failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

