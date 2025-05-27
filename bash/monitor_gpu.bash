#!/bin/bash
#
# Script to monitor GPU utilization during training
# Run this in a separate terminal while your training is running
#

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Function to print a timestamp
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Create output directory if it doesn't exist
mkdir -p logs

# Output file
OUTPUT_FILE="logs/gpu_utilization_$(date +%Y%m%d_%H%M%S).log"

echo "Monitoring GPU utilization. Press Ctrl+C to stop."
echo "Logging to $OUTPUT_FILE"
echo "========================================================"

# Monitor GPU utilization every 5 seconds
while true; do
    # Get current timestamp
    current_time=$(timestamp)
    
    # Print to terminal
    echo "Time: $current_time"
    echo "------------------------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo "========================================================"
    
    # Also log to file
    echo "Time: $current_time" >> "$OUTPUT_FILE"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> "$OUTPUT_FILE"
    echo "========================================================" >> "$OUTPUT_FILE"
    
    # Wait 5 seconds before next check
    sleep 5
done
