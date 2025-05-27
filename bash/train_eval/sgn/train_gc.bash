#!/bin/bash

# Submit all gender classification training scripts via sbatch

# Create output directories
mkdir -p output/ntu_gc_cview
mkdir -p output/ntu_gc_csub
mkdir -p output/ntu120_gc_cview
mkdir -p output/ntu120_gc_csub
mkdir -p output/ntu120_gc_cset

# NTU-60 gender classification
sbatch train_ntu_sgn_gc_cview.bash
sbatch train_ntu_sgn_gc_csub.bash

# NTU-120 gender classification
sbatch train_ntu120_sgn_gc_cview.bash
sbatch train_ntu120_sgn_gc_csub.bash
sbatch train_ntu120_sgn_gc_cset.bash

echo "All gender classification training jobs submitted!"
