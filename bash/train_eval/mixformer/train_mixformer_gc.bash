#!/bin/bash

# Submit all gender classification training scripts for Mixformer via sbatch

# Create output directories
mkdir -p output/ntu_mixformer_gc_cview
mkdir -p output/ntu_mixformer_gc_csub
mkdir -p output/ntu120_mixformer_gc_cview
mkdir -p output/ntu120_mixformer_gc_csub
mkdir -p output/ntu120_mixformer_gc_cset
mkdir -p results/mixformer

# NTU-60 gender classification
echo "Submitting NTU-60 gender classification training jobs..."
sbatch train_ntu_mixformer_gc_cview.bash
sbatch train_ntu_mixformer_gc_csub.bash

# NTU-120 gender classification
echo "Submitting NTU-120 gender classification training jobs..."
sbatch train_ntu120_mixformer_gc_cview.bash
sbatch train_ntu120_mixformer_gc_csub.bash
sbatch train_ntu120_mixformer_gc_cset.bash

echo "All Mixformer gender classification training jobs submitted!"
