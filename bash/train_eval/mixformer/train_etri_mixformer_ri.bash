#!/bin/bash
#SBATCH --job-name=etri_mixf_ri
#SBATCH --output=slurm_out/training/etri_mixf_ri_%j.out
#SBATCH --error=slurm_out/training/etri_mixf_ri_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module load pytorch/2.3.0-cuda12.1

# Create output directory
mkdir -p output/etri_mixformer_ri

cd ..
cd ..
cd ..

python eval/mixformer/train_mixformer.py \
    --dataset ETRI \
    --case 0 \
    --tag ri \
    --batch_size 32 \
    --max_epochs 100 \
    --workers 4 \
    --seg 64 \
    --lr 0.1 \
    --weight_decay 0.0001 \
    --lr_decay_interval 30 \
    --lr_factor 0.1 \
    --output_dir ./output/etri_mixformer_ri
