#!/bin/bash
#SBATCH --job-name=train_pmr
#SBATCH --output=slurm_out/training/train_pmr_%j.out
#SBATCH --error=slurm_out/training/train_pmr_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=240:00:00

cd ..
cd ..

# Create output directories if needed
mkdir -p slurm_out/training
mkdir -p trained_models

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Run the training script
python train_anonymizer.py \
    --model_type pmr \
    --dataset ntu \
    --setting cv \
    --batch_size 32 \
    --paired_batch_size 8 \
    --epochs 100 \
    --lr 1e-5 \
    --adv_lr 1e-4 \
    --train_samples 10000 \
    --test_samples 1000 \
    --T 75 \
    --workers 4 \
    --output_dir trained_models \
    --eval_interval 5 \
    --save_interval 10 
