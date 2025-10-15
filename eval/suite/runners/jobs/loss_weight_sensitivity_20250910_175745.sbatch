#!/bin/bash
#SBATCH --job-name=loss_weight_sensitivity
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/slurm/loss_weight_sensitivity_%j.out
#SBATCH --error=logs/slurm/loss_weight_sensitivity_%j.err

# Initialize module environment
source /etc/profile || true
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi

# Load modules and activate environment
module purge || true
module load pytorch/2.3.0-cuda12.1

# Diagnostics
which python || true
python -V || true
echo "Is CUDA Available?"
python -c 'import torch, sys; print(sys.executable); print(torch.cuda.is_available())' || true
echo ""
echo "nvidia-smi output:"
nvidia-smi || true

# Set environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p logs/slurm
mkdir -p results/experiments

# Run the experiment
python tmr.py eval --one loss_weight_sensitivity

# Refresh dashboard
python tmr.py dash || true

echo "Experiment completed at $(date)"