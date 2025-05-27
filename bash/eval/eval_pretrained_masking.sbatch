#!/bin/bash
#
#SBATCH --job-name="eval-masking"
#SBATCH --partition=GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=slurm_out/eval-masking_%j.out
#SBATCH --error=slurm_out/eval-masking_%j.err
#
#   ===== Main =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Create output directories
mkdir -p results/masking
mkdir -p results/masking/plots
mkdir -p slurm_out

# Define dataset and setting
DATASET="ntu"
SETTING="cv"

# Define recognition model paths
AR_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_ar.pth"
RI_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_ri.pth"
GC_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_gc.pth"

# Define temporal and spatial masking ratios to evaluate
TEMPORAL_RATIOS=(0.3 0.5 0.7)
SPATIAL_RATIOS=(0.3 0.5 0.7)

# Evaluate each combination of temporal and spatial masking ratios
for TEMPORAL_RATIO in "${TEMPORAL_RATIOS[@]}"; do
    for SPATIAL_RATIO in "${SPATIAL_RATIOS[@]}"; do
        echo "===== Evaluating model with temporal_ratio=${TEMPORAL_RATIO}, spatial_ratio=${SPATIAL_RATIO} ====="

        # Define model directory
        MODEL_DIR="eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_${TEMPORAL_RATIO}_spatial_${SPATIAL_RATIO}"

        # Check if model directory exists
        if [ ! -d "$MODEL_DIR" ]; then
            echo "Model directory not found: $MODEL_DIR"
            continue
        fi

        # Run evaluation
        python eval_pretrained.py \
            --dataset ${DATASET} \
            --setting ${SETTING} \
            --model-dir ${MODEL_DIR} \
            --temporal-ratio ${TEMPORAL_RATIO} \
            --spatial-ratio ${SPATIAL_RATIO} \
            --ar-model-weights ${AR_MODEL_WEIGHTS} \
            --ri-model-weights ${RI_MODEL_WEIGHTS} \
            --gc-model-weights ${GC_MODEL_WEIGHTS} \
            --batch-size 32 \
            --test-samples 2000 \
            --calculate-fid \
            --output-dir results/masking

        echo "Evaluation completed for temporal_ratio=${TEMPORAL_RATIO}, spatial_ratio=${SPATIAL_RATIO}"
        echo ""
    done
done

# Generate visualizations
echo "===== Generating visualizations ====="
python visualize_results.py \
    --results-dir results/masking \
    --output-dir results/masking/plots \
    --dataset ${DATASET} \
    --setting ${SETTING}

echo "All evaluations and visualizations completed!"
