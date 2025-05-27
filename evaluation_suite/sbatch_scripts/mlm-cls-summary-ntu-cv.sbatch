#!/bin/bash
#
#SBATCH --job-name="mlm-cls-summary-ntu-cv"
#SBATCH --partition=GPU
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=evaluation_suite/slurm_out/mlm-cls-summary-ntu-cv.out
#SBATCH --error=evaluation_suite/slurm_out/mlm-cls-summary-ntu-cv.err
#
#   ===== MLM Classification Summary and Visualization =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

echo "Generating MLM feature classification summary and visualizations..."
echo "Dataset: ntu"
echo "Setting: cv"
echo ""

# Run the comprehensive summary and visualization script
python evaluation_suite/generate_comprehensive_mlm_report.py \
    --results-dir results/comprehensive_mlm_evaluation \
    --dataset ntu \
    --setting cv \
    --output-dir results/comprehensive_mlm_evaluation/reports

echo ""
echo "MLM feature classification summary completed!"
