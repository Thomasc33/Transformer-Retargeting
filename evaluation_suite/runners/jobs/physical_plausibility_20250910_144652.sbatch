#!/bin/bash
#SBATCH --job-name=physical_plausibility
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=128GB
#SBATCH --output=logs/slurm/physical_plausibility_%j.out
#SBATCH --error=logs/slurm/physical_plausibility_%j.err

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
python pmr.py eval --one physical_plausibility

# Refresh dashboard
python pmr.py dash || true

echo "Experiment completed at $(date)"