#!/bin/bash
#SBATCH --job-name=train_transformer_ntu_cv
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --output=logs/train_transformer_ntu_cv_%j.out
#SBATCH --error=logs/train_transformer_ntu_cv_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

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
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4

# Set CUDA_VISIBLE_DEVICES to use all allocated GPUs
if [ 4 -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    # For multi-GPU training, use all available GPUs
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE-1)))
    # Alternative: explicitly set for common cases
    if [ "$SLURM_GPUS_ON_NODE" = "4" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$SLURM_GPUS_ON_NODE" = "2" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    fi
fi

echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p output

# Run training with optimized hyperparameters
echo "Executing: torchrun --nproc_per_node=4 main.py --dataset ntu --setting cv --gpus 4 --hpc"
torchrun --nproc_per_node=4 main.py \
    --dataset ntu \
    --setting cv \
    --gpus 4 \
    --data-path data/ntu_cv_paired_comprehensive.pt \
    --train-samples 999999999 \
    --test-samples 10000 \
    --epochs 5 \
    --hpc \
    --mixed-precision \
    --use-checkpoint \
    --log-dir logs/train_transformer_ntu_cv \

echo "Training completed at $(date)"
