#!/bin/bash
#SBATCH --job-name="comprehensive_eval"
#SBATCH --partition=GPU
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/comprehensive_eval_%j.out
#SBATCH --error=logs/comprehensive_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

module load pytorch/2.3.0-cuda12.1

python -u scripts/comprehensive_eval.py \
    --dataset ntu \
    --setting cv \
    --model-type all \
    --eval-model mixformer \
    --output-dir results/comprehensive_final \
    --visualizations

