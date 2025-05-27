#!/bin/bash
#SBATCH --job-name=ntu120_ri_cview
#SBATCH --output=slurm_out/training/ntu120_ri_cview_%j.out
#SBATCH --error=slurm_out/training/ntu120_ri_cview_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module load pytorch/2.3.0-cuda12.1

cd ..
cd ..
cd ..

python train_sgn.py \
    --dataset NTU120 \
    --case 1 \
    --tag ri \
    --batch_size 64 \
    --max_epochs 120 \
    --workers 4 \
    --seg 20 \
    --lr 0.001 