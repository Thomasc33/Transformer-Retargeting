#!/bin/bash
#SBATCH --job-name=pipeline_evaluate_ntu_cv_interactive
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/pipeline_evaluate_ntu_cv_interactive_%j.out
#SBATCH --error=logs/pipeline_evaluate_ntu_cv_interactive_%j.err
#SBATCH --dependency=afterok:4791386

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p evaluation_results

# Run the evaluation command
echo "Executing: python evaluation_suite/run_experiments.py --experiments sgn_action_recognition,sgn_re_identification,sgn_gesture_classification,mixformer_action_recognition,mixformer_re_identification,mixformer_gesture_classification,privacy_utility_analysis"
python evaluation_suite/run_experiments.py --experiments sgn_action_recognition,sgn_re_identification,sgn_gesture_classification,mixformer_action_recognition,mixformer_re_identification,mixformer_gesture_classification,privacy_utility_analysis

echo "Step evaluate completed at $(date)"
