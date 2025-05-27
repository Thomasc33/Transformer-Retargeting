#!/bin/bash
#SBATCH --job-name="robust_seed_789"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Create output directory
mkdir -p experiments/robustness/results/seed_789

# Run training with specific seed
torchrun --nproc_per_node=1 --master_port=29589 main.py \
    --dataset ntu \
    --setting cv \
    --batch-size 32 \
    --lr 1e-5 \
    --epochs 100 \
    --train-samples 10000 \
    --test-samples 2000 \
    --seed 789 \
    --data-path data/ntu_cv_paired_10000_2000.pt \
    --output-model-path experiments/robustness/results/seed_789/model.pth \
    --run-eval \
    --hpc
