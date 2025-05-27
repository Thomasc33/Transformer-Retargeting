#!/bin/bash
#
#SBATCH --job-name="ntu120_sample"
#SBATCH --partition=Orion
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#
# ===== Main =====


module load pytorch/2.3.0-cuda12.1

cd ..
cd ..

echo "Starting data sampling..."
python sample_data.py --dataset ntu120 --cpus-per-task 32 --setting cv