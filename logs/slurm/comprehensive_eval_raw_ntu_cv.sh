#!/bin/bash
#SBATCH --job-name=comprehensive_eval_raw_ntu_cv
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/comprehensive_eval_raw_ntu_cv_%j.out
#SBATCH --error=logs/comprehensive_eval_raw_ntu_cv_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs results/comprehensive

# Run comprehensive evaluation
python eval_model.py \
    --dataset ntu \
    --setting cv \
    --model_type raw \
    --eval_model mixformer \
    --output_dir results/comprehensive/ntu_cv/raw_mixformer \
    --ar_model_weights eval/mixformer/pretrained/ntu/cv_ar.pth \
    --ri_model_weights eval/mixformer/pretrained/ntu/cv_ri.pth \
    --gc_model_weights eval/mixformer/pretrained/ntu/cv_gc.pth

echo ""
echo "Comprehensive evaluation completed at $(date)"
