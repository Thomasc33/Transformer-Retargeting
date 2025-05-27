#!/bin/bash
#
# This script analyzes the results of the pretraining experiments
#

# Create output directory if it doesn't exist
mkdir -p experiments/pretraining/analysis

# Install required packages if not already installed
pip install --user pandas matplotlib seaborn tabulate

# Run the analysis script
python experiments/pretraining/analyze_results.py \
    --results-dir experiments/pretraining/results \
    --output-dir experiments/pretraining/analysis

echo "Analysis complete. Results saved to experiments/pretraining/analysis"
