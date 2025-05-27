#!/bin/bash
#SBATCH --job-name=privacy_utility_sgn
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --output=slurm_out/privacy_utility_sgn_%j.out
#SBATCH --error=slurm_out/privacy_utility_sgn_%j.err

# Load modules and activate environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Set environment variables
export OMP_NUM_THREADS=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p slurm_out
mkdir -p evaluation_suite/results/experiments

# Run the experiment
python evaluation_suite/run_experiments.py --experiment privacy_utility_sgn

echo "Experiment completed at $(date)"