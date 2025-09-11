# Evaluation Suite (deprecated entrypoints)

This package is being consolidated under the top-level eval/ package and the pmr.py orchestrator.

Preferred commands:

- Interactive menu: `python pmr.py`
- Run critical set: `python pmr.py eval --set critical`
- Run one experiment: `python pmr.py eval --one baseline_comparison`
- Rebuild dashboard: `python pmr.py dash`

Legacy modules remain available for backward compatibility, but new work should use pmr.py or `python -m eval`.

## 📁 Structure

```
evaluation_suite/
├── core/                    # Core evaluation modules
│   ├── __init__.py
│   ├── evaluator.py        # Main evaluation engine
│   ├── metrics.py          # All metrics calculations
│   ├── models.py           # Model loading utilities
│   └── data_loader.py      # Data loading utilities
├── experiments/             # Experiment definitions
│   ├── __init__.py
│   ├── primary.py          # Primary evaluation experiments
│   ├── ablation.py         # Ablation studies
│   ├── robustness.py       # Robustness analysis
│   ├── efficiency.py       # Efficiency analysis
│   └── visualization.py    # Visualization experiments
├── runners/                 # HPC job management
│   ├── __init__.py
│   ├── slurm_runner.py     # Slurm job submission
│   ├── job_monitor.py      # Job monitoring
│   └── templates/          # Slurm job templates
├── analysis/                # Analysis and visualization
│   ├── __init__.py
│   ├── visualizer.py       # Plotting and visualization
│   ├── comparator.py       # Result comparison
│   └── report_generator.py # Report generation
├── results/                 # Centralized results storage
│   ├── experiments/        # Raw experiment results
│   ├── analysis/           # Analysis outputs
│   └── cache/              # Cached computations
├── reports/                 # Generated reports
│   ├── templates/          # Report templates
│   └── outputs/            # Generated reports
└── configs/                 # Configuration files
    ├── experiments.yaml    # Experiment configurations
    ├── models.yaml         # Model configurations
    └── hpc.yaml            # HPC configurations
```

## 🎯 Features

### Unified Evaluation Interface
- Single command to run any experiment from experiments.md
- Automatic dependency resolution
- Progress tracking and logging

### HPC Integration
- Automatic Slurm job submission
- Job monitoring and management
- Resource optimization

### Comprehensive Metrics
- All metrics from experiments.md implemented
- Physical plausibility metrics
- Privacy-utility tradeoff analysis
- Statistical significance testing

### Visualization & Reporting
- Publication-ready plots and tables
- Interactive dashboards
- LaTeX report generation
- Comparison tools

### Reproducibility
- Full experiment provenance tracking
- Automatic result caching
- Configuration management
- Random seed control

## 📊 Supported Experiments

Based on experiments.md, the suite supports:

1. **Primary Evaluation** - Privacy vs. Utility analysis
2. **Loss Function Analysis** - Ablation studies and weight sensitivity
3. **Pretraining Strategy Analysis** - Different masking and training approaches
4. **Robustness Analysis** - Stability, per-class, per-subject analysis
5. **Generalization & Efficiency** - Cross-dataset validation and speed analysis
6. **Qualitative Analysis** - Motion and attention visualizations

## 🔧 Configuration

All experiments are configured via YAML files in `configs/`. This makes it easy to:
- Modify experiment parameters
- Add new experiments
- Reproduce results
- Share configurations

## 📈 Results Management

Results are automatically organized by:
- Experiment type
- Dataset
- Model configuration
- Timestamp

This ensures easy comparison and prevents result conflicts.

## 🎨 Visualization

The suite generates:
- Heatmaps for metric comparisons
- Line plots for trends
- Bar charts for categorical comparisons
- 3D surface plots for parameter sweeps
- Interactive HTML dashboards

## 📝 Report Generation

Automatically generates:
- Executive summaries
- Detailed technical reports
- Comparison tables
- Statistical analysis
- Publication-ready figures

Perfect for sharing with your advisor!

## 🚀 Getting Started

### 1. Quick Setup

```bash
# Make scripts executable
chmod +x evaluation_suite/run_experiments.py
chmod +x evaluation_suite/generate_report.py
chmod +x evaluation_suite/monitor_jobs.py

# Check what experiments are available
python evaluation_suite/run_experiments.py --list-experiments

# Check current status
python evaluation_suite/run_experiments.py --status
```

### 2. Run Your First Experiment

```bash
# Run a quick test experiment
python evaluation_suite/run_experiments.py --experiment-set quick

# Or run a specific experiment
python evaluation_suite/run_experiments.py --experiment baseline_comparison
```

### 3. Monitor Progress

```bash
# Check job status
python evaluation_suite/monitor_jobs.py --status

# Wait for experiments to complete
python evaluation_suite/monitor_jobs.py --wait baseline_comparison physical_plausibility
```

### 4. Generate Reports

```bash
# Generate comprehensive report
python evaluation_suite/generate_report.py --experiment-set critical

# Generate executive summary for your advisor
python evaluation_suite/generate_report.py --all --type executive --output reports/advisor_summary
```

## 📋 Complete Usage Examples

### Running Experiments

```bash
# List all available experiments and their status
python evaluation_suite/run_experiments.py --list-experiments

# Run critical experiments (most important ones)
python evaluation_suite/run_experiments.py --experiment-set critical

# Run all experiments for paper submission
python evaluation_suite/run_experiments.py --experiment-set paper_ready

# Run specific experiment locally
python evaluation_suite/run_experiments.py --experiment privacy_utility_sgn

# Submit experiments to HPC/Slurm
python evaluation_suite/run_experiments.py --experiment-set critical --slurm
```

### Monitoring Jobs

```bash
# Show current status of all experiments
python evaluation_suite/monitor_jobs.py --status

# Wait for specific experiments to complete (with 12-hour timeout)
python evaluation_suite/monitor_jobs.py --wait privacy_utility_sgn baseline_comparison --timeout 12

# Export job history for analysis
python evaluation_suite/monitor_jobs.py --export results/job_history.csv

# Clean up old job files
python evaluation_suite/monitor_jobs.py --cleanup
```

### Generating Reports

```bash
# Generate technical report for specific experiment
python evaluation_suite/generate_report.py --experiment privacy_utility_sgn --type technical

# Generate executive summary for all completed experiments
python evaluation_suite/generate_report.py --all --type executive --output reports/executive_summary

# Generate comparison report for critical experiments
python evaluation_suite/generate_report.py --experiment-set critical --type comparison

# Generate report with custom output location
python evaluation_suite/generate_report.py --experiment baseline_comparison --output reports/baseline_analysis
```

## 🎯 Experiment Sets

The suite includes predefined experiment sets for different purposes:

### `critical` - Essential experiments (17 hours)
- `privacy_utility_sgn` - Privacy vs utility with SGN models
- `privacy_utility_mixformer` - Privacy vs utility with MixFormer models
- `baseline_comparison` - Compare all baseline models
- `physical_plausibility` - Comprehensive physical metrics

### `paper_ready` - For paper submission (35 hours)
- All critical experiments plus:
- `cross_dataset_validation` - Generalization analysis
- `per_class_analysis` - Per-action performance

### `complete` - All experiments (65 hours)
- Everything from experiments.md
- Includes robustness analysis and efficiency studies

### `quick` - For testing (9 hours)
- `baseline_comparison` and `physical_plausibility`
- Perfect for validating the setup

## 🔧 Configuration

### Experiment Configuration (`configs/experiments.yaml`)

```yaml
# Add custom experiment
custom_experiments:
  my_experiment:
    name: "My Custom Experiment"
    description: "Custom evaluation for specific use case"
    priority: 1
    estimated_time: "2 hours"
```

### HPC Configuration

```yaml
hpc:
  default_partition: "GPU"
  default_time: "24:00:00"
  default_mem: "32GB"

  job_templates:
    quick:
      time: "4:00:00"
      mem: "16GB"
```

## 📊 Results Organization

Results are automatically organized as:

```
evaluation_suite/results/
├── experiments/           # Raw experiment results
│   ├── privacy_utility_sgn/
│   │   └── exp_20240115_143022/
│   │       ├── results.json
│   │       ├── config.json
│   │       └── metrics.json
│   └── baseline_comparison/
├── analysis/             # Analysis outputs
│   ├── visualizations/
│   └── comparisons/
└── cache/               # Cached computations
```

## 📈 Visualization Examples

The suite automatically generates:

- **Privacy-Utility Plots**: Scatter plots showing tradeoffs
- **Metrics Heatmaps**: Comprehensive comparison across models
- **Physical Metrics Radar**: Physical plausibility visualization
- **Summary Tables**: CSV and HTML tables for easy sharing

## 🎨 Report Types

### Executive Summary
- 4-page overview perfect for advisors
- Key findings and recommendations
- High-level performance metrics

### Technical Report
- Detailed methodology and results
- Statistical analysis and significance tests
- Complete experimental details

### Comparison Report
- Model rankings and comparisons
- Statistical significance testing
- Effect size analysis

## 🚨 Troubleshooting

### Common Issues

1. **"No results found"**
   ```bash
   # Check if experiments have been run
   python evaluation_suite/run_experiments.py --status
   ```

2. **"Configuration file not found"**
   ```bash
   # Verify config file exists
   ls evaluation_suite/configs/experiments.yaml
   ```

3. **Slurm submission fails**
   ```bash
   # Check Slurm configuration
   squeue  # Verify Slurm is available
   ```

### Getting Help

```bash
# Get help for any script
python evaluation_suite/run_experiments.py --help
python evaluation_suite/generate_report.py --help
python evaluation_suite/monitor_jobs.py --help
```

## 🎉 Success Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Check what's available
python evaluation_suite/run_experiments.py --list-experiments

# 2. Run critical experiments
python evaluation_suite/run_experiments.py --experiment-set critical --slurm

# 3. Monitor progress
python evaluation_suite/monitor_jobs.py --status

# 4. Wait for completion
python evaluation_suite/monitor_jobs.py --wait privacy_utility_sgn baseline_comparison

# 5. Generate comprehensive report
python evaluation_suite/generate_report.py --experiment-set critical --output reports/final_results

# 6. Generate executive summary for advisor
python evaluation_suite/generate_report.py --all --type executive --output reports/advisor_summary

# 7. Check final results
ls reports/final_results/
```

## 🎯 Tips for Success

1. **Start Small**: Use `--experiment-set quick` to test everything works
2. **Monitor Regularly**: Use the monitor script to track progress
3. **Use HPC**: Submit long experiments with `--slurm` for faster completion
4. **Generate Reports Early**: Create reports as experiments complete
5. **Share Results**: Use executive summaries for advisor meetings

This evaluation suite makes it easy to run comprehensive experiments, monitor progress, and generate publication-ready results. Perfect for impressing your advisor and getting your paper ready! 🎓✨
