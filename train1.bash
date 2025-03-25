#!/bin/bash
#SBATCH --job-name="transformer-retargeting"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#
#   ===== Main =====


module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

export MASTER_ADDR=$(hostname -s)   # or an actual IP address
export MASTER_PORT=29500
export HPC_MODE=1

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Starting torchrun..."

srun torchrun --nproc_per_node=4 main.py --hpc