#!/bin/bash
#SBATCH --job-name=per_class_analysis
#SBATCH --output=slurm_out/experiments/per_class_analysis_%j.out
#SBATCH --error=slurm_out/experiments/per_class_analysis_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Create output directory
mkdir -p experiments/analysis/per_class/results

# Run per-class analysis using the main transformer model
python evaluation_suite/run_experiments.py --experiment per_class_analysis

echo "Per-class analysis completed!"
