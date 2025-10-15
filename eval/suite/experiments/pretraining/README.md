# Pretraining Experiment

This directory contains scripts for running experiments to evaluate the impact of pretraining and freezing the encoder on model performance.

## Experiment Setup

Three configurations are tested:

1. **Pretrained + Frozen Encoder** (Default): Uses a pretrained encoder with weights frozen during training
2. **Pretrained + Unfrozen Encoder**: Uses a pretrained encoder but allows the weights to be updated during training
3. **No Pretraining**: Trains the encoder from scratch (no pretrained weights)

All experiments use the same hyperparameters:
- Dataset: NTU
- Setting: Cross-view (cv)
- Batch Size: 128
- Learning Rate: 1e-5
- Epochs: 100
- Training Samples: 10,000
- Test Samples: 2,000
- Loss Weights: Default values

## Directory Structure

```
experiments/pretraining/
├── pretrained_frozen.bash    # Script for pretrained + frozen encoder
├── pretrained_unfrozen.bash  # Script for pretrained + unfrozen encoder
├── no_pretrained.bash        # Script for no pretraining
├── analyze_results.py        # Script to analyze and compare results
├── run_analysis.bash         # Script to run the analysis
├── results/                  # Directory for storing experiment results
└── analysis/                 # Directory for storing analysis results
```

## Running the Experiments

To run the experiments, use the following commands:

```bash
# Run experiment with pretrained + frozen encoder (default)
sbatch experiments/pretraining/pretrained_frozen.bash

# Run experiment with pretrained + unfrozen encoder
sbatch experiments/pretraining/pretrained_unfrozen.bash

# Run experiment with no pretraining
sbatch experiments/pretraining/no_pretrained.bash
```

Note: Each experiment uses a different port for PyTorch's distributed training to avoid conflicts:
- pretrained_frozen: port 29501
- pretrained_unfrozen: port 29502
- no_pretrained: port 29503

If you encounter port conflicts, you can modify the `--master_port` parameter in each script.

## Analyzing the Results

After all experiments have completed, run the analysis script to compare the results:

```bash
bash experiments/pretraining/run_analysis.bash
```

This will generate:
- A comparison table in CSV and Markdown formats
- Comparison plots for key metrics
- Individual plots for each metric

The analysis results will be saved to `experiments/pretraining/analysis/`.

## Expected Outcomes

The analysis will help determine:
1. Whether using pretrained weights improves performance
2. Whether freezing the encoder during training is beneficial
3. Which configuration provides the best balance of:
   - Action recognition accuracy (utility preservation)
   - Re-identification accuracy (privacy protection)
   - Motion quality metrics (bone length consistency, temporal smoothness, etc.)

## Metrics Evaluated

The following metrics are compared across configurations:
- Action Recognition Accuracy (higher is better)
- Re-identification Accuracy (lower is better for privacy)
- MSE with Ground Truth (lower is better)
- Bone Length Consistency (lower is better)
- Joint Angle Limits (higher is better)
- Temporal Smoothness (lower is better)
- Velocity Consistency (higher is better)
- Foot Contact Consistency (higher is better)
- FID Score (lower is better)
