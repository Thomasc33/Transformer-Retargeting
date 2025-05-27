#!/bin/bash
#
#SBATCH --job-name="mixformer"
#SBATCH --partition=GPU
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#
#   ===== Main =====


module load pytorch/2.3.0-cuda12.1

echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Run the training script with torchrun to utilize all 4 GPUs
torchrun --nproc_per_node=1 main.py --dataset ntu --setting cv --hpc \
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
    --output-model-path test.pth \
    --train-samples 10000 \
    --test-samples 2000 \
    --no-checkpoint \
    --epochs 5