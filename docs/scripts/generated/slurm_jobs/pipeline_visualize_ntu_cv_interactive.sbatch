#!/bin/bash
#SBATCH --job-name=pipeline_visualize_ntu_cv_interactive
#SBATCH --partition=GPU
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=logs/pipeline_visualize_ntu_cv_interactive_%j.out
#SBATCH --error=logs/pipeline_visualize_ntu_cv_interactive_%j.err
#SBATCH --dependency=afterok:4791387

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p visualizations

# Run the visualization command
echo "Executing: python evaluation_suite/run_visualization.py --visualizations skeleton_animations,motion_visualizations,attention_visualization,comparison_visualizations,sensitivity_analysis,anonymization_showcase --dataset ntu --setting cv"
python evaluation_suite/run_visualization.py --visualizations skeleton_animations,motion_visualizations,attention_visualization,comparison_visualizations,sensitivity_analysis,anonymization_showcase --dataset ntu --setting cv

echo "Step visualize completed at $(date)"
