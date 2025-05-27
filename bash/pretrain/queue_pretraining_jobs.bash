#!/bin/bash

# Default values
DATASET="ntu"
SETTING="cv"
BATCH_SIZE=32
LR=1e-4
EPOCHS=50
PATIENCE=10
DISTRIBUTED="--distributed"
CUDNN_ENABLED="--cudnn-enabled"
SLURM_PARTITION="GPU"
SLURM_TIME="240:00:00"
SLURM_MEM="64GB"
GPU_COUNT=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --setting)
      SETTING="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --gpu-count)
      GPU_COUNT="$2"
      shift 2
      ;;
    --no-distributed)
      DISTRIBUTED=""
      shift
      ;;
    --no-cudnn)
      CUDNN_ENABLED=""
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory for job scripts
mkdir -p bash/pretrain/generated_sbatch

# Define masking ratios to use
TEMPORAL_RATIOS=(0.3 0.5 0.7)
SPATIAL_RATIOS=(0.3 0.5 0.7)

echo "Queuing pretraining jobs with the following settings:"
echo "Dataset: $DATASET"
echo "Setting: $SETTING"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Distributed: ${DISTRIBUTED:-false}"
echo "GPU count: $GPU_COUNT"
echo

# Loop through masking ratio combinations
for TEMPORAL_RATIO in "${TEMPORAL_RATIOS[@]}"; do
  for SPATIAL_RATIO in "${SPATIAL_RATIOS[@]}"; do
    echo "Submitting job with temporal masking $TEMPORAL_RATIO and spatial masking $SPATIAL_RATIO"
    
    # Call the submission script with the current parameters
    bash bash/pretrain/submit_pretrain_job.bash \
      --dataset "$DATASET" \
      --setting "$SETTING" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --temporal-ratio "$TEMPORAL_RATIO" \
      --spatial-ratio "$SPATIAL_RATIO" \
      --partition "$SLURM_PARTITION" \
      --time "$SLURM_TIME" \
      --mem "$SLURM_MEM" \
      --gpu-count "$GPU_COUNT" \
      $DISTRIBUTED \
      $CUDNN_ENABLED
    
    echo "Waiting 2 seconds before submitting next job..."
    sleep 2
  done
done

echo "All jobs have been queued!"
