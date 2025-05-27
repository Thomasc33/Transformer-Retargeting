#!/bin/bash

# Submit all re-identification training scripts for Mixformer via sbatch

# Create output directories
mkdir -p output/ntu_mixformer_ri_cview
mkdir -p output/ntu120_mixformer_ri_cview
mkdir -p output/ntu120_mixformer_ri_cset
mkdir -p output/etri_mixformer_ri
mkdir -p results/mixformer

# NTU-60 re-identification
echo "Submitting NTU-60 re-identification training jobs..."
sbatch train_ntu_mixformer_ri_cview.bash

# NTU-120 re-identification
echo "Submitting NTU-120 re-identification training jobs..."
sbatch train_ntu120_mixformer_ri_cview.bash
sbatch train_ntu120_mixformer_ri_cset.bash

# ETRI re-identification
echo "Submitting ETRI re-identification training job..."
sbatch train_etri_mixformer_ri.bash

echo "All Mixformer re-identification training jobs submitted!"
