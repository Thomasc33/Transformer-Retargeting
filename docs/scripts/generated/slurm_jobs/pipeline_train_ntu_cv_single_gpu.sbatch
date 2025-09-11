#!/bin/bash
#SBATCH --job-name=pipeline_train_ntu_cv_single
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --output=logs/pipeline_train_ntu_cv_single_%j.out
#SBATCH --error=logs/pipeline_train_ntu_cv_single_%j.err

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

# Set environment variables for single GPU training
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Run the command with single GPU setup and larger batch size
echo "Executing: python main.py --dataset ntu --setting cv --gpus 1 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5 --batch-size 32 --mixed-precision"
python main.py --dataset ntu --setting cv --gpus 1 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5 --batch-size 32 --mixed-precision

echo "Step train completed at $(date)"
