#!/bin/bash
#SBATCH --job-name="efficiency"
#SBATCH --partition=GPU
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Create output directory
mkdir -p experiments/efficiency/results

# Run efficiency analysis
python efficiency_analysis.py \
    --model-path model.pth \
    --data-path data/ntu_cv_paired_10000_2000.pt \
    --output-dir experiments/efficiency/results \
    --batch-sizes 1,8,16,32,64 \
    --num-runs 100 \
    --hpc
