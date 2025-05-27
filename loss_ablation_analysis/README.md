# Loss Ablation Analysis

This directory contains the analysis of loss ablation studies conducted to evaluate the impact of different loss functions on model performance.

## Contents

- `summary_table.csv`: Summary of key metrics for each model
- `detailed_metrics_table.csv`: Detailed metrics for each model and evaluation method
- `loss_impact_analysis.md`: Analysis of the impact of different loss functions
- `plots/`: Visualizations of the results
  - `summary/`: Plots of summary metrics
  - `metrics/`: Plots of detailed metrics

## How to Interpret

The analysis focuses on the following key metrics:

- **Action Recognition (AR)**: Higher is better, indicates utility preservation
- **Re-identification (RI)**: Lower is better, indicates privacy protection
- **Mean Squared Error (MSE)**: Lower is better, indicates motion fidelity
- **Bone Length Error**: Lower is better, indicates physical plausibility
- **Foot Contact Error**: Lower is better, indicates physical plausibility
- **Privacy-Utility Score**: Higher is better, calculated as (AR - RI)

The loss functions evaluated include:

- **MSE Loss**: Direct difference between generated and target poses
- **Bone Length Loss**: Ensures consistent bone lengths
- **Foot Contact Loss**: Prevents foot sliding and floating artifacts
- **Smoothing Loss**: Encourages temporal consistency
- **Joint Limit Loss**: Ensures anatomically plausible joint angles

## Models

- **Transformer (Ours)**: Our proposed transformer-based motion privacy model
- **PMR**: Pose Motion Retargeting baseline
- **DMR**: Deep Motion Retargeting baseline
- **Raw**: Raw skeleton (no anonymization)

