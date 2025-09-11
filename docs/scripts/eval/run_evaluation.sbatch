#!/bin/bash
#SBATCH --job-name="eval_transformer"
#SBATCH --partition=GPU
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/eval_transformer_%j.out
#SBATCH --error=logs/eval_transformer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

module load pytorch/2.3.0-cuda12.1

python -u eval_model.py \
    --dataset ntu \
    --setting cv \
    --model_type transformer \
    --transformer_model_path model.pth \
    --eval_model mixformer \
    --test_samples 1000 \
    --output_dir results/advisor_eval_final

