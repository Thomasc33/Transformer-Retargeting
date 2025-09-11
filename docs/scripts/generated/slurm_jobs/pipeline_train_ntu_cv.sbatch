#!/bin/bash
#SBATCH --job-name=pipeline_train_ntu_cv
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --output=logs/pipeline_train_ntu_cv_%j.out
#SBATCH --error=logs/pipeline_train_ntu_cv_%j.err
#SBATCH --dependency=afterok:4832517

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Set environment variables for distributed training
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Run the command
echo "Executing: torchrun --standalone --nproc_per_node=4 main.py --dataset ntu --setting cv --hpc --gpus 4 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5"
torchrun --standalone --nproc_per_node=4 main.py --dataset ntu --setting cv --hpc --gpus 4 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5

echo "Step train completed at $(date)"