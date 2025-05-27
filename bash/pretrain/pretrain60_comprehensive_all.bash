#!/bin/bash
#
#SBATCH --job-name="pt-ntu60-all"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200GB
#
#   ===== Main =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# cd ..
# cd ..

# Run the pretraining script with NTU-60 dataset using torchrun
# Using cross-view setting to match the comprehensive dataset
# Using ALL available data
torchrun --nproc_per_node=2 pretrain.py \
    --dataset ntu \
    --setting cv \
    --distributed \
    --cudnn-enabled \
    --batch-size 32 \
    --epochs 100
