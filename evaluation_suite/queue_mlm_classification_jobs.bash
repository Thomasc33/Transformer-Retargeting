#!/bin/bash

# Queue MLM Feature Classification Jobs for HPC
# This script submits jobs for all 9 masking ratio combinations

# Default values
DATASET="ntu"
SETTING="cv"
PARTITION="GPU"
TIME="12:00:00"
MEM="64GB"
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
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --time)
      TIME="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directories
mkdir -p evaluation_suite/slurm_out
mkdir -p evaluation_suite/sbatch_scripts
mkdir -p results/comprehensive_mlm_evaluation

# Define masking ratios to evaluate
TEMPORAL_RATIOS=(0.3 0.5 0.7)
SPATIAL_RATIOS=(0.3 0.5 0.7)

echo "Queuing MLM feature classification jobs with the following settings:"
echo "Dataset: $DATASET"
echo "Setting: $SETTING"
echo "Partition: $PARTITION"
echo "Time limit: $TIME"
echo "Memory: $MEM"
echo "GPU count: $GPU_COUNT"
echo

# Counter for job dependencies
JOB_IDS=()

# Loop through masking ratio combinations
for TEMPORAL_RATIO in "${TEMPORAL_RATIOS[@]}"; do
  for SPATIAL_RATIO in "${SPATIAL_RATIOS[@]}"; do
    echo "Submitting job for temporal masking $TEMPORAL_RATIO and spatial masking $SPATIAL_RATIO"

    # Create job name
    JOB_NAME="mlm-cls-${DATASET}-${SETTING}-t${TEMPORAL_RATIO}-s${SPATIAL_RATIO}"

    # Define model directory
    MODEL_DIR="eval/mixformer/pretrained/${DATASET}/epochs_${SETTING}_comprehensive_temporal_${TEMPORAL_RATIO}_spatial_${SPATIAL_RATIO}"

    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
      echo "Warning: Model directory not found: $MODEL_DIR"
      echo "Skipping this combination..."
      continue
    fi

    # Create SBATCH script
    SBATCH_FILE="evaluation_suite/sbatch_scripts/${JOB_NAME}.sbatch"

    cat > "$SBATCH_FILE" << EOF
#!/bin/bash
#
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --mem=${MEM}
#SBATCH --output=evaluation_suite/slurm_out/${JOB_NAME}.out
#SBATCH --error=evaluation_suite/slurm_out/${JOB_NAME}.err
#
#   ===== MLM Feature Classification =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi
echo ""

# Print job information
echo "Job: ${JOB_NAME}"
echo "Dataset: ${DATASET}"
echo "Setting: ${SETTING}"
echo "Temporal Ratio: ${TEMPORAL_RATIO}"
echo "Spatial Ratio: ${SPATIAL_RATIO}"
echo "Model Directory: ${MODEL_DIR}"
echo ""

# Run the comprehensive MLM evaluation
python evaluation_suite/comprehensive_mlm_evaluation.py \\
    --model-dir ${MODEL_DIR} \\
    --dataset ${DATASET} \\
    --setting ${SETTING} \\
    --temporal-ratio ${TEMPORAL_RATIO} \\
    --spatial-ratio ${SPATIAL_RATIO} \\
    --seq-len 64 \\
    --batch-size 32 \\
    --output-dir results/comprehensive_mlm_evaluation \\
    --train-samples 10000 \\
    --test-samples 2000

echo ""
echo "MLM feature classification completed for ${JOB_NAME}"
EOF

    # Submit job and capture job ID
    JOB_OUTPUT=$(sbatch "$SBATCH_FILE")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*')
    JOB_IDS+=($JOB_ID)

    echo "Job submitted: $JOB_NAME (ID: $JOB_ID)"
    echo "Script saved at: $SBATCH_FILE"
    echo ""

    # Wait 2 seconds before submitting next job
    sleep 2
  done
done

echo "All MLM feature classification jobs have been queued!"
echo "Job IDs: ${JOB_IDS[*]}"

# Create a summary script that will run after all jobs complete
SUMMARY_JOB_NAME="mlm-cls-summary-${DATASET}-${SETTING}"
SUMMARY_SBATCH_FILE="evaluation_suite/sbatch_scripts/${SUMMARY_JOB_NAME}.sbatch"

# Create dependency string
DEPENDENCY_STRING=""
if [ ${#JOB_IDS[@]} -gt 0 ]; then
  DEPENDENCY_STRING="--dependency=afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
fi

cat > "$SUMMARY_SBATCH_FILE" << EOF
#!/bin/bash
#
#SBATCH --job-name="${SUMMARY_JOB_NAME}"
#SBATCH --partition=${PARTITION}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=evaluation_suite/slurm_out/${SUMMARY_JOB_NAME}.out
#SBATCH --error=evaluation_suite/slurm_out/${SUMMARY_JOB_NAME}.err
#
#   ===== MLM Classification Summary and Visualization =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

echo "Generating MLM feature classification summary and visualizations..."
echo "Dataset: ${DATASET}"
echo "Setting: ${SETTING}"
echo ""

# Run the comprehensive summary and visualization script
python evaluation_suite/generate_comprehensive_mlm_report.py \\
    --results-dir results/comprehensive_mlm_evaluation \\
    --dataset ${DATASET} \\
    --setting ${SETTING} \\
    --output-dir results/comprehensive_mlm_evaluation/reports

echo ""
echo "MLM feature classification summary completed!"
EOF

# Submit summary job with dependencies
if [ ${#JOB_IDS[@]} -gt 0 ]; then
  SUMMARY_OUTPUT=$(sbatch $DEPENDENCY_STRING "$SUMMARY_SBATCH_FILE")
  SUMMARY_JOB_ID=$(echo "$SUMMARY_OUTPUT" | grep -o '[0-9]*')
  echo ""
  echo "Summary job submitted: $SUMMARY_JOB_NAME (ID: $SUMMARY_JOB_ID)"
  echo "Will run after all classification jobs complete"
  echo "Summary script saved at: $SUMMARY_SBATCH_FILE"
else
  echo ""
  echo "No jobs were submitted, skipping summary job"
fi

echo ""
echo "=== Job Queue Summary ==="
echo "Classification jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
if [ ${#JOB_IDS[@]} -gt 0 ]; then
  echo "Summary job ID: $SUMMARY_JOB_ID"
fi
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check results in: results/mlm_classification/"
