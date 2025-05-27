#!/bin/bash
#
# Script to evaluate a single pretrained model with specific temporal and spatial masking ratios
#

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Check if temporal and spatial ratios are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <temporal_ratio> <spatial_ratio>"
    echo "Example: $0 0.3 0.5"
    exit 1
fi

# Get temporal and spatial ratios from arguments
TEMPORAL_RATIO=$1
SPATIAL_RATIO=$2

# Define dataset and setting
DATASET="ntu"
SETTING="cv"

# Define recognition model paths
AR_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_ar.pth"
RI_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_ri.pth"
GC_MODEL_WEIGHTS="eval/sgn/pretrained/ntu/cview_gc.pth"

# Define model directory
MODEL_DIR="eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_${TEMPORAL_RATIO}_spatial_${SPATIAL_RATIO}"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory not found: $MODEL_DIR"
    exit 1
fi

# Create output directories
mkdir -p results/masking
mkdir -p results/masking/plots
mkdir -p slurm_out

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
