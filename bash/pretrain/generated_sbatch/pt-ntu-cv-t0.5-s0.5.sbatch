#!/bin/bash
#
#SBATCH --job-name="pt-ntu-cv-t0.5-s0.5"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_out/pt-ntu-cv-t0.5-s0.5.out
#SBATCH --error=slurm_out/pt-ntu-cv-t0.5-s0.5.err
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

# Run the pretraining script
torchrun --nproc_per_node=1 --master_port=21890 pretrain.py \
    --dataset ntu \
    --setting cv \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --patience 10 \
    --temporal_masking_ratio 0.5 \
    --spatial_masking_ratio 0.5 \
    --distributed \
    --cudnn-enabled
