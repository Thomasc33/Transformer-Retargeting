# Hyperparameter Tuning with Optuna

This directory contains scripts for hyperparameter tuning using Optuna. The tuning process optimizes the following parameters:

- Batch size
- Learning rate
- Decoder dropout
- Loss weights (for all non-zero losses)

## Directory Structure

```
experiments/3_hyperparameter/
├── optuna_tuning.py       # Main Optuna optimization script
├── run_optuna.bash        # Bash script to run the optimization
├── analyze_results.py     # Script to analyze and visualize results
├── run_analysis.bash      # Bash script to run the analysis
├── results/               # Directory for storing optimization results
└── logs/                  # Directory for storing logs
```

## Running Hyperparameter Tuning

To run the hyperparameter tuning process, use the following command:

```bash
sbatch experiments/3_hyperparameter/run_optuna.bash
```

This will submit a job to the HPC system that:

1. Installs Optuna if needed
2. Runs the optimization process with 20 trials
3. Saves results to `experiments/3_hyperparameter/results/`
4. Logs output to `experiments/3_hyperparameter/logs/`

The optimization process uses the following default settings:
- Dataset: NTU
- Setting: Cross-view (cv)
- 50 epochs per trial
- 10,000 training samples
- 2,000 test samples
- Teacher forcing ratio: 1.0 (no decay)
- Pretrained encoder (frozen)

## Analyzing Results

After the optimization is complete, you can analyze the results using:

```bash
bash experiments/3_hyperparameter/run_analysis.bash experiments/3_hyperparameter/results/study_results_TIMESTAMP.json
```

Replace `TIMESTAMP` with the actual timestamp in the filename. This will:

1. Generate visualizations of parameter importance
2. Create plots showing relationships between parameters
3. Produce a summary of the best hyperparameters
4. Save all analysis results to `experiments/3_hyperparameter/analysis/`

### Combined Score for Optimization

The optimization process uses a combined score that balances multiple metrics:

- **Action Recognition Accuracy** (50% weight): Higher is better, measures utility preservation
- **Re-identification Accuracy** (20% weight): Lower is better, measures privacy protection
- **MSE with Ground Truth** (5% weight): Lower is better, measures reconstruction quality
- **Bone Length Consistency** (5% weight): Lower is better, measures physical plausibility
- **Joint Angle Limits** (5% weight): Higher is better, measures physical plausibility
- **Temporal Smoothness** (5% weight): Lower is better, measures motion quality
- **Velocity Consistency** (5% weight): Higher is better, measures motion quality
- **Foot Contact Consistency** (3% weight): Higher is better, measures motion quality
- **Validation Loss** (2% weight): Lower is better, measures training quality

Each metric is normalized to a 0-1 range and combined with appropriate weights. The optimization process aims to minimize this combined score, which represents a balance between privacy protection, motion quality, and utility preservation.

## Reproducing the Best Trial

The optimization process automatically generates a script to reproduce the best trial:

```bash
bash experiments/3_hyperparameter/results/reproduce_best_trial_TIMESTAMP.sh
```

This script will run the training process with the optimal hyperparameters found during the tuning process.

## Hyperparameter Search Spaces

The following hyperparameters are tuned with these search spaces:

- **batch_size**: [32, 64, 128, 256]
- **lr**: 1e-6 to 1e-4 (log scale)
- **decoder_dropout**: 0.0 to 0.3
- **loss_mse**: 1.0 to 10.0 (default: 7.0)
- **loss_ee**: 1.0 to 10.0 (default: 5.0)
- **loss_smoothing**: 0.01 to 5.0 (default: 0.075)
- **loss_inception**: 0.01 to 5.0 (default: 0.05)
- **loss_fid_vel**: 0.1 to 10.0 (default: 1.0)
- **loss_bone**: 1.0 to 15.0 (default: 10.0)
- **loss_foot**: 0.5 to 5.0 (default: 3.0)
- **loss_joint_limit**: 0.1 to 3.0 (default: 1.0)

## Customizing the Tuning Process

To modify the hyperparameter search spaces or other aspects of the tuning process, edit the `optuna_tuning.py` script. The search spaces are defined in the `objective` function.

To change the number of trials or other settings, edit the `run_optuna.bash` script and modify the command-line arguments passed to `optuna_tuning.py`.
