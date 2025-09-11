#!/bin/bash
#
# This script analyzes the results of the Optuna hyperparameter tuning
#

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Check if a results file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_study_results.json>"
    echo "Example: $0 experiments/hyperparameter/results/study_results_20230101_120000.json"
    exit 1
fi

RESULTS_FILE=$1
OUTPUT_DIR="experiments/hyperparameter/analysis"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Install required packages if not already installed
pip install --user pandas matplotlib seaborn tabulate

# Run the analysis script
python experiments/hyperparameter/analyze_results.py \
    --results-file $RESULTS_FILE \
    --output-dir $OUTPUT_DIR

echo "Analysis complete. Results saved to $OUTPUT_DIR"
