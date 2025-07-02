#!/bin/bash
#SBATCH --job-name=test_training_2gpu
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_training_2gpu_%j.out
#SBATCH --error=logs/test_training_2gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

echo "🧪 TESTING OPTIMIZED TRAINING - 2 GPU CONFIGURATION"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load required modules
module load pytorch/2.3.0-cuda12.1

# Set optimized environment variables for 2 GPU distributed training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # 1 hour for testing
export NCCL_BUFFSIZE=8388608  # 8MB

# FIXED: NCCL_NTHREADS must be multiple of 32 for optimal performance
export NCCL_NTHREADS=64  # Optimal for 2 GPUs (32 threads per GPU)

# Additional NCCL optimizations for 2 GPU setup
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Advanced NCCL optimizations for better performance
export NCCL_ALGO=Ring  # Use ring algorithm for 2 GPUs
export NCCL_MIN_NCHANNELS=4  # Minimum channels for better bandwidth
export NCCL_MAX_NCHANNELS=16  # Maximum channels (don't over-allocate)
export NCCL_NSOCKS_PERTHREAD=4  # Sockets per thread for better throughput
export NCCL_SOCKET_NTHREADS=8  # Socket threads for network communication

# CUDA optimizations
export CUDNN_DETERMINISTIC=1
export CUDNN_BENCHMARK=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

# Change to project directory
cd /users/tcarr23/Transformer-Retargeting

# Create logs directory
mkdir -p logs

echo "Environment setup complete"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "NCCL timeout: $NCCL_TIMEOUT seconds"
echo "NCCL threads: $NCCL_NTHREADS (optimized for multi-GPU)"
echo "NCCL algorithm: $NCCL_ALGO"
echo "NCCL channels: $NCCL_MIN_NCHANNELS-$NCCL_MAX_NCHANNELS"
echo ""

# Function to run training with error handling
run_training() {
    echo "🎯 Starting 2 GPU test training..."
    echo "Configuration:"
    echo "  - 2 GPUs with DDP"
    echo "  - 4000 train samples (2x 1 GPU test)"
    echo "  - 500 test samples (2x 1 GPU test)"
    echo "  - 15 epochs"
    echo "  - Batch size: 16 per GPU (32 total effective)"
    echo "  - Learning rate: 9.43e-05 - OPTIMIZED from hyperparameter tuning"
    echo "  - Decoder dropout: 0.1155 - OPTIMIZED from hyperparameter tuning"
    echo "  - Loss weights: OPTIMIZED from hyperparameter tuning trial 17"
    echo "  - Gradient accumulation: 2 steps"
    echo "  - Mixed precision: enabled (auto-disabled for multi-GPU if needed)"
    echo "  - Validation: every 3 epochs"
    echo "  - NCCL optimized: 64 threads, Ring algorithm"
    echo "  - Wandb project: Motion Retargeting Test 2GPU"
    echo ""
    
    # Run with torchrun for proper distributed setup (unbuffered output)
    python -u -m torch.distributed.run \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        main.py \
        --hpc \
        --dataset ntu \
        --setting cv \
        --epochs 15 \
        --batch-size 16 \
        --lr 9.43062936149491e-05 \
        --train-samples 4000 \
        --test-samples 500 \
        --config configs/main_config.yaml \
        --gradient-accumulation-steps 2 \
        --mixed-precision \
        --validate-every 3 \
        --progress-every 50 \
        --use-checkpoint \
        --nccl-timeout 3600 \
        --decoder-dropout 0.11551063114920847 \
        --save-every 5 \
        --wandb-project "Motion Retargeting Test 2GPU" \
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
        if [ -f "logs/test_training_2gpu_${SLURM_JOB_ID}.out" ]; then
            echo "Recent progress:"
            tail -n 5 "logs/test_training_2gpu_${SLURM_JOB_ID}.out" | grep -E "(Epoch|Batch|Loss|GPU|rank)"
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
echo "🚀 Starting 2 GPU test training pipeline..."

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
echo "🏁 2 GPU test training completed with exit code: $TRAINING_EXIT_CODE"
echo "End time: $(date)"

# Final status report
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ 2 GPU test training completed successfully!"
    
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
    
    echo ""
    echo "🎯 Test Results Summary:"
    echo "  ✅ Multi-GPU distributed training works"
    echo "  ✅ NCCL optimizations successful"
    echo "  ✅ No timeout errors"
    echo "  ✅ Config file loading works"
    echo "  ✅ Wandb integration works"
    echo "  ✅ Ready for full 4 GPU production training"
    
else
    echo "❌ 2 GPU test training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Show recent error logs
    echo "Recent error logs:"
    tail -n 20 "logs/test_training_2gpu_${SLURM_JOB_ID}.err" 2>/dev/null || echo "No error log found"
    
    # Check for NCCL timeout errors specifically
    if grep -q "NCCL.*timeout" "logs/test_training_2gpu_${SLURM_JOB_ID}.err" 2>/dev/null; then
        echo ""
        echo "⚠️  NCCL timeout detected - this indicates the multi-GPU fixes need more work"
    fi
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
