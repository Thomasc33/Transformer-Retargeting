#!/bin/bash
#SBATCH --job-name=tmr-train-retarget
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=/users/tcarr23/Transformer-Retargeting/logs/slurm/tmr-train-retarget-%j.out
#SBATCH --mail-user=tcarr23@charlotte.edu
#SBATCH --mail-type=BEGIN,END,FAIL


set -euo pipefail
module load pytorch/2.3.0-cuda12.1 || true
cd "/users/tcarr23/Transformer-Retargeting"
echo "[PIPELINE] tmr-train-retarget start $(date)"
python src/training/main.py --resume-from /users/tcarr23/Transformer-Retargeting/data/models_output/model.pth --output-model-path /users/tcarr23/Transformer-Retargeting/data/models_output/model_resume_run-20250910-193557.pth --teacher-forcing-ratio 0.0 --epochs 5
echo "[PIPELINE] tmr-train-retarget done $(date)"
