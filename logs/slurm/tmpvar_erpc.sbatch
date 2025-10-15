#!/bin/bash
#SBATCH --job-name=tmr-dash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/users/tcarr23/Transformer-Retargeting/logs/slurm/tmr-dash-%j.out
#SBATCH --partition=GPU
#SBATCH --mail-user=tcarr23@charlotte.edu
#SBATCH --mail-type=BEGIN,END,FAIL


set -euo pipefail
module load pytorch/2.3.0-cuda12.1 || true
cd "/users/tcarr23/Transformer-Retargeting"
echo "[PIPELINE] tmr-dash start $(date)"
python eval/suite/generate_dashboard.py
echo "[PIPELINE] tmr-dash done $(date)"
