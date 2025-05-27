#!/bin/bash
#SBATCH --job-name="ntu60_to_ntu120"
#SBATCH --partition=GPU
#SBATCH --time=180:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Create output directory
mkdir -p experiments/cross_dataset/results/ntu60_to_ntu120

# Run cross-dataset evaluation
python cross_dataset_eval.py \
    --train-dataset ntu60 \
    --test-dataset ntu120 \
    --model-path model.pth \
    --output-dir experiments/cross_dataset/results/ntu60_to_ntu120 \
    --hpc
