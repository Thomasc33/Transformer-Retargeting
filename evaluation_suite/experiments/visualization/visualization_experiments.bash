#!/bin/bash
#SBATCH --job-name=visualization_experiments
#SBATCH --output=slurm_out/experiments/visualization_%j.out
#SBATCH --error=slurm_out/experiments/visualization_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Create output directory
mkdir -p experiments/visualization/results

# Run motion visualizations
python evaluation_suite/run_experiments.py --experiment motion_visualizations

# Run attention visualizations
python evaluation_suite/run_experiments.py --experiment attention_visualization

echo "Visualization experiments completed!"
