#!/bin/bash
#
#SBATCH --job-name="ntu_pairs"
#SBATCH --partition=Orion
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#
# ===== Main =====

module load pytorch/2.3.0-cuda12.1

echo "Starting comprehensive data sampling for NTU dataset..."
python sample_data_pairs.py --dataset ntu
