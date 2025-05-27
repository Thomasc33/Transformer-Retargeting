#!/bin/bash

# Default values
DATASET="ntu"
SETTING="cv"
BATCH_SIZE=32
LR=1e-4
EPOCHS=50
PATIENCE=10
TEMPORAL_RATIO=0.5
SPATIAL_RATIO=0.5
DISTRIBUTED=""
CUDNN_ENABLED=""
PARTITION="GPU"
TIME="240:00:00"
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
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --temporal-ratio)
      TEMPORAL_RATIO="$2"
      shift 2
      ;;
    --spatial-ratio)
      SPATIAL_RATIO="$2"
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
    --gpu-count)
      GPU_COUNT="$2"
      shift 2
      ;;
    --distributed)
      DISTRIBUTED="--distributed"
      shift
      ;;
    --cudnn-enabled)
      CUDNN_ENABLED="--cudnn-enabled"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create name for the job
JOB_NAME="pt-${DATASET}-${SETTING}-t${TEMPORAL_RATIO}-s${SPATIAL_RATIO}"

# Create output directory
mkdir -p slurm_out

# Generate a random port to avoid conflicts
RANDOM_PORT=$((10000 + RANDOM % 55000))

# Create SBATCH script
SBATCH_FILE="bash/pretrain/generated_sbatch/${JOB_NAME}.sbatch"

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
#SBATCH --output=slurm_out/${JOB_NAME}.out
#SBATCH --error=slurm_out/${JOB_NAME}.err
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

# Run the pretraining script
torchrun --nproc_per_node=${GPU_COUNT} --master_port=${RANDOM_PORT} pretrain.py \\
    --dataset ${DATASET} \\
    --setting ${SETTING} \\
    --batch-size ${BATCH_SIZE} \\
    --lr ${LR} \\
    --epochs ${EPOCHS} \\
    --patience ${PATIENCE} \\
    --temporal_masking_ratio ${TEMPORAL_RATIO} \\
    --spatial_masking_ratio ${SPATIAL_RATIO} \\
EOF

# Add optional flags
if [ -n "$DISTRIBUTED" ]; then
  echo "    --distributed \\" >> "$SBATCH_FILE"
fi

if [ -n "$CUDNN_ENABLED" ]; then
  echo "    --cudnn-enabled" >> "$SBATCH_FILE"
else
  # Remove trailing backslash from last line if no CUDNN flag
  sed -i '$ s/\\$//' "$SBATCH_FILE"
fi

# Submit job
echo "Submitting job: $JOB_NAME"
sbatch "$SBATCH_FILE"
echo "Job submitted. Script saved at $SBATCH_FILE"
