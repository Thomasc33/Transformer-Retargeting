#!/bin/bash
#
#SBATCH --job-name="pt-ntu-cs-t0.3-s0.7"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_out/pt-ntu-cs-t0.3-s0.7.out
#SBATCH --error=slurm_out/pt-ntu-cs-t0.3-s0.7.err
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
torchrun --nproc_per_node=1 --master_port=34345 pretrain.py \
    --dataset ntu \
    --setting cs \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 100 \
    --patience 10 \
    --temporal_masking_ratio 0.3 \
    --spatial_masking_ratio 0.7 \
    --distributed \
    --cudnn-enabled
