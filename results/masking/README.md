# Pretrained MLM Model Evaluation

This directory contains evaluation results and visualizations for pretrained MLM models with different temporal and spatial masking ratios.

## Directory Structure

- `*.json`: Individual evaluation result files for each model
- `plots/`: Visualization plots comparing different models
  - `*_heatmap.png`: Heatmap visualizations of metrics across masking ratios
  - `*_surface.png`: 3D surface plots of metrics across masking ratios
  - `*_temporal_bar.png`: Bar charts comparing temporal masking ratios
  - `*_spatial_bar.png`: Bar charts comparing spatial masking ratios
  - `summary_table.csv`: Summary table of key metrics
  - `summary_table.html`: Styled HTML version of the summary table

## Evaluation Metrics

The evaluation includes the following metrics:

1. **Reconstruction MSE**: Mean squared error between the original and reconstructed skeletons
2. **Action Recognition Accuracy (AR)**: Accuracy of action recognition on reconstructed skeletons
3. **Re-identification Accuracy (RI)**: Accuracy of person re-identification on reconstructed skeletons
4. **Gender Classification Accuracy (GC)**: Accuracy of gender classification on reconstructed skeletons
5. **Physical Plausibility Metrics**:
   - **Bone Length Consistency (BLC)**: Variance of bone lengths across frames
   - **Joint Angle Violation Rate (JAL)**: Rate of joint angles outside anatomical limits
   - **Temporal Smoothness (TS)**: Measure of jerk in the motion
   - **Velocity Consistency (VC)**: Difference in velocity patterns between original and reconstructed
   - **Foot Contact Consistency (FCC)**: Consistency of foot contact with the ground
6. **FID Score**: Fréchet Inception Distance between original and reconstructed motion features

## Running Evaluations

To run evaluations for all models:

```bash
sbatch bash/eval/eval_pretrained_masking.sbatch
```

To evaluate a single model:

```bash
bash bash/eval/eval_single_pretrained.bash <temporal_ratio> <spatial_ratio>
```

Example:
```bash
bash bash/eval/eval_single_pretrained.bash 0.3 0.5
```

## Generating Visualizations

After running evaluations, visualizations can be generated with:

```bash
python visualize_results.py --results-dir results/masking --output-dir results/masking/plots
```

## Interpretation

When interpreting the results, consider the following:

1. **Reconstruction Quality**: Lower MSE indicates better reconstruction
2. **Privacy Protection**: Lower RI accuracy indicates better privacy protection
3. **Utility Preservation**: Higher AR accuracy indicates better utility preservation
4. **Physical Plausibility**: Lower values for BLC, JAL, TS, VC, and higher FCC indicate more physically plausible motion
5. **Overall Quality**: Lower FID score indicates more realistic motion

The optimal masking ratios balance these factors, with different applications potentially prioritizing different metrics.
