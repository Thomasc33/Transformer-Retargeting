#!/bin/bash

# Submit all action recognition training scripts for Mixformer via sbatch

# Create output directories
mkdir -p output/ntu_mixformer_ar_cview
mkdir -p output/ntu_mixformer_ar_csub
mkdir -p output/ntu120_mixformer_ar_cview
mkdir -p output/ntu120_mixformer_ar_csub
mkdir -p output/ntu120_mixformer_ar_cset
mkdir -p output/etri_mixformer_ar
mkdir -p results/mixformer

# NTU-60 action recognition
echo "Submitting NTU-60 action recognition training jobs..."
sbatch train_ntu_mixformer_ar_cview.bash
sbatch train_ntu_mixformer_ar_csub.bash

# NTU-120 action recognition
echo "Submitting NTU-120 action recognition training jobs..."
sbatch train_ntu120_mixformer_ar_cview.bash
sbatch train_ntu120_mixformer_ar_csub.bash
sbatch train_ntu120_mixformer_ar_cset.bash

# ETRI action recognition
echo "Submitting ETRI action recognition training job..."
sbatch train_etri_mixformer_ar.bash

echo "All Mixformer action recognition training jobs submitted!"
