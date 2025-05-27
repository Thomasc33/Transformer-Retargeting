#!/bin/bash
#
# Script to monitor GPU utilization
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

# Monitor GPU utilization every 5 seconds
echo "Monitoring GPU utilization (Press Ctrl+C to stop)..."
echo "========================================================"

while true; do
    echo "Time: $(timestamp)"
    echo "------------------------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo "========================================================"
    sleep 5
done
