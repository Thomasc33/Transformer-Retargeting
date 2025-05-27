#!/bin/bash
#
#SBATCH --job-name="etri_pairs"
#SBATCH --partition=Orion
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#
# ===== Main =====

module load pytorch/2.3.0-cuda12.1

cd /users/tcarr23/Transformer-Retargeting

echo "Starting comprehensive data sampling for ETRI dataset..."
python /users/tcarr23/Transformer-Retargeting/sample_data_pairs.py --dataset etri
