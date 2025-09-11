#!/bin/bash
#SBATCH --job-name=pipeline_visualize_ntu_cv
#SBATCH --partition=GPU
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/pipeline_visualize_ntu_cv_%j.out
#SBATCH --error=logs/pipeline_visualize_ntu_cv_%j.err

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Set environment variables for distributed training
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p checkpoints

# Run the command
echo "Executing: python evaluation_suite/run_visualization.py --visualizations skeleton_animations,motion_visualizations,attention_visualization,comparison_visualizations,sensitivity_analysis,anonymization_showcase,mlm_pretraining --dataset ntu --setting cv --temporal-ratio 0.5 --spatial-ratio 0.5"
python evaluation_suite/run_visualization.py --visualizations skeleton_animations,motion_visualizations,attention_visualization,comparison_visualizations,sensitivity_analysis,anonymization_showcase,mlm_pretraining --dataset ntu --setting cv --temporal-ratio 0.5 --spatial-ratio 0.5

echo "Step visualize completed at $(date)"