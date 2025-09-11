#!/bin/bash
#SBATCH --job-name=tmr-eval-critical
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=/users/tcarr23/Transformer-Retargeting/logs/slurm/tmr-eval-critical-%j.out
#SBATCH --partition=GPU
#SBATCH --mail-user=tcarr23@charlotte.edu
#SBATCH --mail-type=BEGIN,END,FAIL


set -euo pipefail
module load pytorch/2.3.0-cuda12.1 || true
cd "/users/tcarr23/Transformer-Retargeting"
echo "[PIPELINE] tmr-eval-critical start $(date)"
python eval/suite/run_all_evaluations.py
echo "[PIPELINE] tmr-eval-critical done $(date)"
