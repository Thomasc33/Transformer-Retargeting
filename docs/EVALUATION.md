# Evaluation Suite Usage

This guide explains how to run evaluations, monitor them on Slurm, and view results in the unified dashboard.

## One-command: run everything

- Submit the full set and refresh dashboard:

```bash
python -m evaluation_suite.run_all_evaluations
```

- Submit a specific set (critical, quick, paper_ready):

```bash
python -m evaluation_suite.run_all_evaluations --set critical
```

## Run a single experiment (no YAML on cluster)

```bash
python -m evaluation_suite.run_single_experiment --experiment baseline_comparison
```

This avoids PyYAML issues by sourcing experiment configs directly from the classes.

## Dashboard (auto-refreshes)

- Generated at the repo root:

```
results.html
```

- Regenerate manually:

```bash
python -m evaluation_suite.generate_dashboard
```

The dashboard shows:
- Status per experiment (Completed/Pending)
- Links to results.json, metrics.json, config.json
- Embedded GIFs/images/videos from results/experiments/<exp>/<run>/{visualizations,artifacts}
- Standalone visuals from results/visualizations
- Notices for any missing files/visuals

## Slurm

- Job files and outputs:
  - Scripts: evaluation_suite/runners/jobs/*.sbatch
  - Logs: slurm_out/<experiment>_<jobid>.out/.err
- Emails are disabled for evaluation jobs (training emails remain separate)
- The job script loads the PyTorch module and prints diagnostics

## Results Layout

```
results/
  experiments/
    <experiment_name>/
      <exp_id>/
        results.json
        metrics.json
        config.json
        metadata.json
        visualizations/
        artifacts/
        logs/
reports/
  latest/
results.html
```

## Tips

- Open results.html to navigate everything
- If something is missing, the dashboard will display a Missing notice
- Use slurm_out/*.err to debug any job issues

