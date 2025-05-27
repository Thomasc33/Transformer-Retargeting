# Transformer Retargeting Experiments

This document outlines the planned experiments for our "Achieving Motion Privacy Through Transformer-based Motion Retargeting for Skeleton-based Data" research project.

## 1. Model Development & Initial Training

- [X] Implement Skeleton-MixFormer encoder with masked autoencoder pretraining
  
- [X] Implement autoregressive decoder with cross-attention mechanism
  - [X] Design for simultaneous attention to source motion and dummy skeleton
  - [X] Implement teacher forcing with scheduled decay

- [X] Implement all loss functions
  - [X] MSE loss
  - [X] End-effector loss
  - [X] Smoothing loss
  - [X] FID velocity loss
  - [X] Bone-length loss
  - [X] Foot-contact loss
  - [X] Joint limit loss

## 2. Pretraining Strategy Analysis

- [ ] Train encoder with different masking configurations
  - [ ] Test temporal masking ratios (30%, 50%, 70%)
  - [ ] Test spatial (joint) masking ratios (30%, 50%, 70%)
  - [ ] Evaluate different combinations of temporal/spatial masking

- [ ] Compare pretraining approaches
  - [ ] Pretraining with freezing encoder weights during retargeting
  - [ ] Pretraining with fine-tuning during retargeting
  - [ ] No pretraining (training from scratch)

## 3. Loss Function Analysis

- [ ] Single loss component ablation studies
  - [ ] Zero out each loss component individually
    - [ ] Bone-length
    - [ ] Foot-contact
    - [ ] Joint limit
    - [ ] FID velocity
  - [ ] Measure impact on physical plausibility metrics
  - [ ] Measure impact on privacy-utility tradeoff

- [X] Loss weight sensitivity analysis
  - [X] Use Optuna for hyperparameter optimization of loss weights
  - [X] Identify optimal configuration balancing privacy and utility

## 4. Primary Evaluation - Privacy vs. Utility

- [X] Train baseline models
  - [X] Raw skeleton data (no anonymization)
  - [X] Deep Motion Retargeting (DMR)
  - [X] Privacy-preserving Motion Retargeting (PMR)

- [X] Task performance evaluation with SGN
  - [X] Action Recognition (AR) accuracy
  - [X] Re-identification (RI) accuracy
  - [X] Gender Classification (GC) metrics (GC-Orig, GC-Ret, GC-Cross)

- [X] Task performance evaluation with Skeleton-MixFormer
  - [X] Action Recognition (AR) accuracy
  - [X] Re-identification (RI) accuracy
  - [X] Gender Classification (GC) metrics

- [X] Physical plausibility metrics
  - [X] Bone Length Consistency (BLC)
  - [X] Joint Angle Limits (JAL) compliance
  - [X] Temporal Smoothness (TS)
  - [X] Velocity Consistency (VC)
  - [X] Foot Contact Consistency (FCC)
  - [X] Fréchet Inception Distance (FID)

## 5. Robustness Analysis

- [ ] Training stability evaluation
  - [ ] Repeat key experiments with 5 different random seeds
  - [ ] Report mean and standard deviation for all metrics

- [ ] Teacher forcing analysis
  - [ ] Compare different teacher forcing decay schedules
  - [ ] Analyze impact on generation quality and convergence

- [ ] Per-class performance analysis
  - [ ] Identify actions that maintain or lose recognition accuracy
  - [ ] Identify patterns among action types (e.g., fine vs. gross motor)

- [ ] Per-subject anonymization effectiveness
  - [ ] Determine which subjects are easier or harder to anonymize
  - [ ] Analyze factors contributing to anonymization difficulty

## 6. Generalization & Efficiency

- [ ] Cross-dataset validation
  - [ ] Train on NTU-60, test on NTU-120
  - [ ] Evaluate generalization to ETRI-Activity3D

- [ ] Efficiency analysis
  - [ ] Measure inference speed (FPS)
  - [ ] Measure memory requirements
  - [ ] Compare computational efficiency to baseline methods

## 7. Qualitative Analysis

- [ ] Motion visualizations
  - [ ] Create animated sequences showing source-to-retargeted transfers
  - [ ] Generate overlays of source versus retargeted joint positions
  - [ ] Visualize key actions with varying privacy-utility tradeoffs

- [ ] Attention visualization
  - [ ] Create heatmaps showing attention patterns from source motion
  - [ ] Create heatmaps showing attention patterns from dummy skeleton
  - [ ] Analyze cross-attention behavior during different action phases