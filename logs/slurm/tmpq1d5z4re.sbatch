#!/bin/bash
#SBATCH --job-name=tmr-dash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/users/tcarr23/Transformer-Retargeting/logs/slurm/tmr-dash-%j.out
#SBATCH --partition=GPU
#SBATCH --mail-user=thomas.carr@incertastrategy.com
#SBATCH --mail-type=BEGIN,END,FAIL


set -euo pipefail
module load pytorch/2.3.0-cuda12.1 || true
cd "/users/tcarr23/Transformer-Retargeting"
echo "[PIPELINE] tmr-dash start $(date)"
python eval/suite/generate_dashboard.py
echo "[PIPELINE] tmr-dash done $(date)"
