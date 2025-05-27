#!/bin/bash

# Submit all training scripts for Mixformer (AR, RI, GC) via sbatch

# Create output directories
mkdir -p output/ntu_mixformer_ar_cview
mkdir -p output/ntu_mixformer_ar_csub
mkdir -p output/ntu_mixformer_ri_cview
mkdir -p output/ntu_mixformer_gc_cview
mkdir -p output/ntu_mixformer_gc_csub

mkdir -p output/ntu120_mixformer_ar_cview
mkdir -p output/ntu120_mixformer_ar_csub
mkdir -p output/ntu120_mixformer_ar_cset
mkdir -p output/ntu120_mixformer_ri_cview
mkdir -p output/ntu120_mixformer_ri_cset
mkdir -p output/ntu120_mixformer_gc_cview
mkdir -p output/ntu120_mixformer_gc_csub
mkdir -p output/ntu120_mixformer_gc_cset

mkdir -p output/etri_mixformer_ar
mkdir -p output/etri_mixformer_ri

mkdir -p results/mixformer

# NTU-60 action recognition
echo "Submitting NTU-60 action recognition training jobs..."
sbatch train_ntu_mixformer_ar_cview.bash
sbatch train_ntu_mixformer_ar_csub.bash

# NTU-60 re-identification
echo "Submitting NTU-60 re-identification training jobs..."
sbatch train_ntu_mixformer_ri_cview.bash

# NTU-60 gender classification
echo "Submitting NTU-60 gender classification training jobs..."
sbatch train_ntu_mixformer_gc_cview.bash
sbatch train_ntu_mixformer_gc_csub.bash

# NTU-120 action recognition
echo "Submitting NTU-120 action recognition training jobs..."
sbatch train_ntu120_mixformer_ar_cview.bash
sbatch train_ntu120_mixformer_ar_csub.bash
sbatch train_ntu120_mixformer_ar_cset.bash

# NTU-120 re-identification
echo "Submitting NTU-120 re-identification training jobs..."
sbatch train_ntu120_mixformer_ri_cview.bash
sbatch train_ntu120_mixformer_ri_cset.bash

# NTU-120 gender classification
echo "Submitting NTU-120 gender classification training jobs..."
sbatch train_ntu120_mixformer_gc_cview.bash
sbatch train_ntu120_mixformer_gc_csub.bash
sbatch train_ntu120_mixformer_gc_cset.bash

# ETRI action recognition and re-identification
echo "Submitting ETRI training jobs..."
sbatch train_etri_mixformer_ar.bash
sbatch train_etri_mixformer_ri.bash

echo "All Mixformer training jobs submitted!"
