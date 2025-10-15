#!/bin/bash
#SBATCH --job-name=pipeline_train_ntu_cv_6gpu
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --mem=384GB
#SBATCH --output=logs/pipeline_train_ntu_cv_6gpu_%j.out
#SBATCH --error=logs/pipeline_train_ntu_cv_6gpu_%j.err
#SBATCH --exclusive

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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# NCCL optimizations for better communication
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200000  # 2 hours timeout

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_NCCL_BLOCKING_WAIT=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Run the command with proper torchrun setup and optimized batch size for 6 GPUs
echo "Executing: torchrun --standalone --nproc_per_node=6 main.py --dataset ntu --setting cv --hpc --gpus 6 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5 --batch-size 12 --gradient-accumulation-steps 2"
torchrun --standalone --nproc_per_node=6 main.py --dataset ntu --setting cv --hpc --gpus 6 --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5 --batch-size 12 --gradient-accumulation-steps 2

echo "Step train completed at $(date)"
