#!/bin/bash
#SBATCH --job-name=same_action_eval
#SBATCH --output=logs/same_action_eval_%j.out
#SBATCH --error=logs/same_action_eval_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Same-Action Evaluation SLURM Script
# This script evaluates TMR on pairs where both actors perform the same action

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                    SAME-ACTION EVALUATION - SLURM JOB                        ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Load modules (adjust based on your HPC environment)
echo "Loading modules..."
module load pytorch/2.3.0-cuda12.1
echo ""

# The pytorch module should set up the environment automatically
echo "Python environment:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Change to project directory
cd /users/tcarr23/Transformer-Retargeting || exit 1
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Parse command line arguments (with defaults)
DATASET=${1:-ntu_cv}
NUM_PAIRS=${2:-100}
DEVICE=${3:-cuda}

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         EVALUATION CONFIGURATION                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Number of Pairs: $NUM_PAIRS"
echo "Device: $DEVICE"
echo ""

# Run the evaluation
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         STARTING EVALUATION                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/eval_same_action.py \
    --dataset "$DATASET" \
    --num_pairs "$NUM_PAIRS" \
    --device "$DEVICE" \
    --seed 42

EXIT_CODE=$?

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         EVALUATION COMPLETE                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Exit code: $EXIT_CODE"
echo "Results saved to: results/same_action_evaluation.json"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    
    # Print results if available
    if [ -f "results/same_action_evaluation.json" ]; then
        echo ""
        echo "Results:"
        cat results/same_action_evaluation.json | python -m json.tool 2>/dev/null || cat results/same_action_evaluation.json
    fi
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi

echo ""
echo "Job finished at: $(date)"
echo ""

exit $EXIT_CODE

