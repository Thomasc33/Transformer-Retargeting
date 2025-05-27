#!/bin/bash
#SBATCH --job-name=optuna_hp
#SBATCH --output=slurm_out/optuna/optuna_%j.out
#SBATCH --error=slurm_out/optuna/optuna_%j.err
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=GPU
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00

# Create output directories if they don't exist
mkdir -p slurm_out/optuna
mkdir -p experiments/3_hyperparameter/results
mkdir -p experiments/3_hyperparameter/logs

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Run the Optuna hyperparameter tuning script
python experiments/3_hyperparameter/optuna_tuning.py \
    --dataset ntu \
    --setting cv \
    --epochs 50 \
    --train-samples 10000 \
    --test-samples 2000 \
    --teacher-forcing-ratio 1.0 \
    --teacher-forcing-decay 0.0 \
    --use-pretrained \
    --freeze-encoder \
    --n-trials 25 \
    --output-dir experiments/3_hyperparameter/results \
    --log-dir experiments/3_hyperparameter/logs \
    --study-name hyperparameter_tuning \
    --seed 42
