#!/bin/bash
#
#SBATCH --job-name="mixformer-all"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#
#   ===== Main =====


module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Run the OPTIMIZED training script with torchrun to utilize 4 GPUs
# Using the comprehensive dataset with cross-view setting
# Using ALL available data (no sample limit)
# OPTIMIZED with mixed precision, gradient accumulation, and robust checkpointing
# FIXED: AMP scaler state management, DataLoader workers, DDP warnings, scheduler deprecation
torchrun --nproc_per_node=4 main.py --dataset ntu --setting cv --hpc \
    --batch-size 8 \
    --lr 3e-05 \
    --decoder-dropout 0.11551063114920847 \
    --loss-mse 2.0 \
    --loss-ee 1.5 \
    --loss-smoothing 1.0 \
    --loss-inception 1.0 \
    --loss-fid-vel 0.8 \
    --loss-bone 2.5 \
    --loss-foot 0.3 \
    --loss-joint-limit 0.5 \
    --data-path data/ntu_cv_paired_comprehensive.pt \
    --train-samples 999999999 \
    --test-samples 10000 \
    --output-model-path model_comprehensive_all.pth \
    --epochs 5 \
    --mixed-precision \
    --gradient-accumulation-steps 4 \
    --max-grad-norm 0.5 \
    --nccl-timeout 3600 \
    --save-every 1 \
    --log-dir logs/comprehensive_all
