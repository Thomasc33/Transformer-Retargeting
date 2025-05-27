#!/bin/bash

# Submit all ETRI training scripts for Mixformer via sbatch

# Create output directories
mkdir -p output/etri_mixformer_ar
mkdir -p output/etri_mixformer_ri
mkdir -p results/mixformer

# ETRI action recognition
echo "Submitting ETRI action recognition training job..."
sbatch train_etri_mixformer_ar.bash

# ETRI re-identification
echo "Submitting ETRI re-identification training job..."
sbatch train_etri_mixformer_ri.bash

echo "All ETRI Mixformer training jobs submitted!"
