#!/usr/bin/env python3
"""
Convenience wrapper to queue/run everything possible:
- Submits the 'complete' experiment set to Slurm (preferred) or runs locally if --local
- After submission, generates/refreshes the results.html dashboard at repo root

Usage:
  # Submit all via Slurm
  python evaluation_suite/run_all_evaluations.py

  # Run locally (not recommended for heavy jobs)
  python evaluation_suite/run_all_evaluations.py --local
"""

import argparse
import sys
from pathlib import Path

from eval.suite.generate_dashboard import main as generate_dashboard
from eval.suite.runners.slurm_runner import SlurmRunner

# Import experiment classes directly to avoid YAML dependency locally
from eval.suite.experiments.primary import PrimaryExperiments
try:
    from eval.suite.experiments.ablation import AblationExperiments
except Exception:
    AblationExperiments = None
try:
    from eval.suite.experiments.pretraining import PretrainingExperiments
except Exception:
    PretrainingExperiments = None
try:
    from eval.suite.experiments.robustness import RobustnessExperiments
except Exception:
    RobustnessExperiments = None
try:
    from eval.suite.experiments.generalization import GeneralizationExperiments
except Exception:
    GeneralizationExperiments = None
try:
    from eval.suite.experiments.qualitative import QualitativeExperiments
except Exception:
    QualitativeExperiments = None


def collect_all_experiments(set_name: str = "complete") -> dict:
    """Return a mapping of experiment_name -> config across categories.
    Falls back to union of available experiment classes when YAML isn't available.
    """
    agg = {}
    def merge(d):
        if not d:
            return
        for k, v in d.items():
            agg[k] = v

    # Primary
    merge(PrimaryExperiments.get_experiment_configs())

    # Other groups if available
    if AblationExperiments:
        merge(AblationExperiments.get_experiment_configs())
    if PretrainingExperiments:
        merge(PretrainingExperiments.get_experiment_configs())
    if RobustnessExperiments:
        merge(RobustnessExperiments.get_experiment_configs())
    if GeneralizationExperiments:
        merge(GeneralizationExperiments.get_experiment_configs())
    if QualitativeExperiments:
        merge(QualitativeExperiments.get_experiment_configs())

    # For now, all sets map to the full union. Specific set selection can be added later if needed.
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run all evaluations and refresh dashboard")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to Slurm")
    parser.add_argument("--set", type=str, default="complete", help="Experiment set to run (complete, critical, quick, paper_ready)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Log level")
    args = parser.parse_args()

    # Build experiment list directly from classes
    all_exps = collect_all_experiments(args.set)
    exp_names = list(all_exps.keys())
    print(f"Submitting {len(exp_names)} experiments in set '{args.set}'...")

    use_slurm = not args.local
    ok = True

    if use_slurm:
        # Minimal HPC defaults aligned with repo config defaults
        hpc_config = {
            'default_partition': 'GPU',
            'default_time': '240:00:00',
            'default_nodes': 1,
            'default_ntasks_per_node': 1,
            'default_gres': 'gpu:1',
            'default_mem': '64GB',
            'job_templates': {
                'standard': {'time': '12:00:00', 'mem': '64GB', 'gres': 'gpu:1', 'ntasks_per_node': 1},
                'quick': {'time': '4:00:00', 'mem': '64GB', 'gres': 'gpu:1', 'ntasks_per_node': 1},
                'long': {'time': '48:00:00', 'mem': '128GB', 'gres': 'gpu:4', 'ntasks_per_node': 4},
                'multi_seed': {'time': '72:00:00', 'mem': '128GB', 'gres': 'gpu:4', 'ntasks_per_node': 4, 'array': '1-5'},
            }
        }
        runner = SlurmRunner(hpc_config)
        submitted = []
        for name in exp_names:
            cfg = all_exps.get(name, {})
            job_id = runner.submit_experiment(name, cfg)
            if job_id:
                submitted.append(job_id)
            else:
                ok = False
        print(f"Submitted {len(submitted)} jobs.")
    else:
        # Local execution fallback (may require torch and full env); skip heavy local runs by default
        print("Local run selected. Warning: local execution requires full environment (torch, data). Skipping actual runs.")
        ok = False

    # Always refresh dashboard so it's available immediately
    try:
        generate_dashboard()
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

