#!/usr/bin/env python3
"""
Run a single evaluation experiment without relying on YAML config.
This avoids PyYAML dependency on HPC nodes.

Usage:
  python evaluation_suite/run_single_experiment.py --experiment privacy_utility_sgn
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.suite.core.evaluator import ComprehensiveEvaluator

# Experiment classes
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
    from eval.suite.experiments.efficiency import EfficiencyExperiments
except Exception:
    EfficiencyExperiments = None
try:
    from eval.suite.experiments.generalization import GeneralizationExperiments
except Exception:
    GeneralizationExperiments = None
try:
    from eval.suite.experiments.visualization import VisualizationExperiments
except Exception:
    VisualizationExperiments = None
try:
    from eval.suite.experiments.qualitative import QualitativeExperiments
except Exception:
    QualitativeExperiments = None


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def resolve_experiment_config(experiment_name: str):
    classes = [
        PrimaryExperiments,
        AblationExperiments,
        PretrainingExperiments,
        RobustnessExperiments,
        EfficiencyExperiments,
        GeneralizationExperiments,
        VisualizationExperiments,
        QualitativeExperiments,
    ]
    for cls in classes:
        if not cls:
            continue
        cfgs = cls.get_experiment_configs()
        if experiment_name in cfgs:
            return cfgs[experiment_name]
    raise ValueError(f"Experiment '{experiment_name}' not found in experiment classes")


def main():
    parser = argparse.ArgumentParser(description="Run a single evaluation experiment")
    parser.add_argument('--experiment', required=True, type=str, help='Experiment name to run')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Minimal config placeholder for evaluator
    config = {}

    evaluator = ComprehensiveEvaluator(config)
    exp_cfg = resolve_experiment_config(args.experiment)

    results = evaluator.run_experiment(args.experiment, exp_cfg)
    print(f"Completed experiment: {args.experiment}")
    print(f"Saved under: {evaluator.current_exp_dir}")


if __name__ == '__main__':
    main()

