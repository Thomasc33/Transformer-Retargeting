#!/bin/bash
#
#SBATCH --job-name="resume-mixformer"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --mem=200GB
#
#   ===== Resume Training Script =====

module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 logs/comprehensive_all/checkpoints/checkpoint_latest.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming training from checkpoint: $CHECKPOINT_PATH"

# Resume training with the same optimized settings
torchrun --nproc_per_node=2 main.py --dataset ntu --setting cv --hpc \
    --batch-size 8 \
    --lr 9.43062936149491e-05 \
    --decoder-dropout 0.11551063114920847 \
    --loss-mse 5.323284271000699 \
    --loss-ee 4.7250245075017725 \
    --loss-smoothing 2.8747697025246937 \
    --loss-inception 3.0656068713914353 \
    --loss-fid-vel 2.2608337791442894 \
    --loss-bone 6.185377286837532 \
    --loss-foot 0.544125368059208 \
    --loss-joint-limit 1.768344443841404 \
    --data-path data/ntu_cv_paired_comprehensive.pt \
    --train-samples 999999999 \
    --test-samples 10000 \
    --output-model-path model_comprehensive_all_resumed.pth \
    --epochs 20 \
    --mixed-precision \
    --gradient-accumulation-steps 4 \
    --max-grad-norm 1.0 \
    --nccl-timeout 3600 \
    --save-every 1 \
    --log-dir logs/comprehensive_all_resumed \
    --resume-from "$CHECKPOINT_PATH"
