#!/bin/bash
#
#SBATCH --job-name="mlm-cls-ntu-cv-t0.3-s0.7"
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=evaluation_suite/slurm_out/mlm-cls-ntu-cv-t0.3-s0.7.out
#SBATCH --error=evaluation_suite/slurm_out/mlm-cls-ntu-cv-t0.3-s0.7.err
#
#   ===== MLM Feature Classification =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi
echo ""

# Print job information
echo "Job: mlm-cls-ntu-cv-t0.3-s0.7"
echo "Dataset: ntu"
echo "Setting: cv"
echo "Temporal Ratio: 0.3"
echo "Spatial Ratio: 0.7"
echo "Model Directory: eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.7"
echo ""

# Run the comprehensive MLM evaluation
python evaluation_suite/comprehensive_mlm_evaluation.py \
    --model-dir eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.7 \
    --dataset ntu \
    --setting cv \
    --temporal-ratio 0.3 \
    --spatial-ratio 0.7 \
    --seq-len 64 \
    --batch-size 32 \
    --output-dir results/comprehensive_mlm_evaluation \
    --train-samples 10000 \
    --test-samples 2000

echo ""
echo "MLM feature classification completed for mlm-cls-ntu-cv-t0.3-s0.7"
