#!/bin/bash
#SBATCH --job-name=motion_retarget_optimized
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/training_optimized_%j.out
#SBATCH --error=logs/training_optimized_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

# UPDATED: Comprehensive optimized training script with all fixes

echo "🚀 OPTIMIZED MOTION RETARGETING TRAINING"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules
module load pytorch/2.3.0-cuda12.1

# Set optimized environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200  # 2 hours
export NCCL_BUFFSIZE=8388608  # 8MB

# FIXED: NCCL_NTHREADS must be multiple of 32 for optimal performance
export NCCL_NTHREADS=128  # Optimal for 4 GPUs (32 threads per GPU)

# Additional NCCL optimizations for 4 GPU setup
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ALGO=Ring  # Use ring algorithm for 4 GPUs
export NCCL_MIN_NCHANNELS=4  # Minimum channels for better bandwidth
export NCCL_MAX_NCHANNELS=16  # Maximum channels
export NCCL_NSOCKS_PERTHREAD=4  # Sockets per thread for better throughput
export NCCL_SOCKET_NTHREADS=8  # Socket threads for network communication

# CUDA optimizations
export CUDNN_DETERMINISTIC=1
export CUDNN_BENCHMARK=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Change to project directory
cd /users/tcarr23/Transformer-Retargeting

# Create logs directory
mkdir -p logs

echo "Environment setup complete"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "NCCL timeout: $NCCL_TIMEOUT seconds"
echo ""

# Function to run training with error handling
run_training() {
    echo "🎯 Starting optimized distributed training..."
    echo "Configuration:"
    echo "  - 4 GPUs with DDP"
    echo "  - Batch size: 8 per GPU (32 total effective) - OPTIMIZED from hyperparameter tuning"
    echo "  - Learning rate: 9.43e-05 - OPTIMIZED from hyperparameter tuning"
    echo "  - Decoder dropout: 0.1155 - OPTIMIZED from hyperparameter tuning"
    echo "  - Loss weights: OPTIMIZED from hyperparameter tuning trial 17"
    echo "  - Gradient accumulation: 4 steps"
    echo "  - Mixed precision: auto-disabled for multi-GPU (prevents NaN gradients)"
    echo "  - Validation: every 5 epochs"
    echo "  - Checkpointing: enabled"
    echo "  - NCCL optimized: 128 threads, Ring algorithm"
    echo ""
    
    # Run with torchrun for proper distributed setup
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        main.py \
        --hpc \
        --dataset ntu \
        --setting cv \
        --epochs 5 \
        --batch-size 8 \
        --lr 9.43062936149491e-05 \
        --train-samples 999999999 \
        --test-samples 10000 \
        --config configs/main_config.yaml \
        --gradient-accumulation-steps 4 \
        --mixed-precision \
        --validate-every 5 \
        --progress-every 100 \
        --use-checkpoint \
        --nccl-timeout 7200 \
        --decoder-dropout 0.11551063114920847 \
        --save-every 1 \
        --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404
}

# Function to monitor training progress
monitor_progress() {
    echo "📊 Monitoring training progress..."
    
    # Check if training is running
    while pgrep -f "main.py" > /dev/null; do
        echo "Training is running... $(date)"
        
        # Check GPU utilization
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
        fi
        
        # Check for recent log updates
        if [ -f "logs/training_optimized_${SLURM_JOB_ID}.out" ]; then
            echo "Recent progress:"
            tail -n 5 "logs/training_optimized_${SLURM_JOB_ID}.out" | grep -E "(Epoch|Batch|Loss|GPU)"
        fi
        
        echo "---"
        sleep 300  # Check every 5 minutes
    done
}

# Function to handle cleanup
cleanup() {
    echo "🧹 Cleaning up..."
    
    # Kill any remaining processes
    pkill -f "main.py" || true
    
    # Clear CUDA cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    echo "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT
trap cleanup SIGTERM
trap cleanup SIGINT

# Main execution
echo "🚀 Starting optimized training pipeline..."

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""

# Run training with monitoring
{
    run_training
} &

# Get the PID of the training process
TRAINING_PID=$!

# Start monitoring in background
monitor_progress &
MONITOR_PID=$!

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "🏁 Training completed with exit code: $TRAINING_EXIT_CODE"
echo "End time: $(date)"

# Final status report
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Check if model was saved
    if [ -d "output" ]; then
        echo "📁 Output directory contents:"
        ls -la output/
    fi
    
    # Check for checkpoints
    if [ -d "checkpoints" ]; then
        echo "💾 Checkpoint directory contents:"
        ls -la checkpoints/
    fi
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Show recent error logs
    echo "Recent error logs:"
    tail -n 20 "logs/training_optimized_${SLURM_JOB_ID}.err" 2>/dev/null || echo "No error log found"
fi

echo ""
echo "📊 Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "🎯 Job summary:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Duration: $(date)"
echo "  Exit code: $TRAINING_EXIT_CODE"

exit $TRAINING_EXIT_CODE
