#!/bin/bash
#SBATCH --job-name="mask_temporal_0.7_spatial_0.5"
#SBATCH --partition=GPU
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Create output directory
mkdir -p experiments/masking/results/temporal_0.7_spatial_0.5

# Run MLM pretraining with specific masking ratios
python pretrain.py \
    --dataset ntu \
    --setting cv \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 200 \
    --temporal-mask-ratio 0.7 \
    --spatial-mask-ratio 0.5 \
    --output-dir experiments/masking/results/temporal_0.7_spatial_0.5 \
    --hpc
