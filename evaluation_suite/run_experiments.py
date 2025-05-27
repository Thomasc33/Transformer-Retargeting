#!/usr/bin/env python3
"""
Main runner script for the comprehensive evaluation suite.

Usage:
    python evaluation_suite/run_experiments.py --experiment-set critical
    python evaluation_suite/run_experiments.py --experiment privacy_utility_sgn
    python evaluation_suite/run_experiments.py --list-experiments
    python evaluation_suite/run_experiments.py --status
"""

import argparse
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation_suite.core import ComprehensiveEvaluator
from evaluation_suite.runners.slurm_runner import SlurmRunner

# Import experiment classes with error handling
try:
    from evaluation_suite.experiments.primary import PrimaryExperiments
except ImportError:
    PrimaryExperiments = None

try:
    from evaluation_suite.experiments.ablation import AblationExperiments
except ImportError:
    AblationExperiments = None

try:
    from evaluation_suite.experiments.pretraining import PretrainingExperiments
except ImportError:
    PretrainingExperiments = None

try:
    from evaluation_suite.experiments.robustness import RobustnessExperiments
except ImportError:
    RobustnessExperiments = None

try:
    from evaluation_suite.experiments.efficiency import EfficiencyExperiments
except ImportError:
    EfficiencyExperiments = None

try:
    from evaluation_suite.experiments.generalization import GeneralizationExperiments
except ImportError:
    GeneralizationExperiments = None

try:
    from evaluation_suite.experiments.visualization import VisualizationExperiments
except ImportError:
    VisualizationExperiments = None

try:
    from evaluation_suite.experiments.qualitative import QualitativeExperiments
except ImportError:
    QualitativeExperiments = None


def load_config(config_path: str = "evaluation_suite/configs/experiments.yaml") -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir = Path("evaluation_suite/results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "run_experiments.log"),
            logging.StreamHandler()
        ]
    )


def list_experiments(config: Dict[str, Any]):
    """List all available experiments."""
    print("\n🧪 Available Experiments:")
    print("=" * 50)

    # Primary experiments
    print("\n📊 Primary Experiments:")
    for exp_name, exp_config in config.get('primary_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Loss analysis experiments
    print("\n🔍 Loss Analysis Experiments:")
    for exp_name, exp_config in config.get('loss_analysis_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Pretraining experiments
    print("\n🎯 Pretraining Experiments:")
    for exp_name, exp_config in config.get('pretraining_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Robustness experiments
    print("\n💪 Robustness Experiments:")
    for exp_name, exp_config in config.get('robustness_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Generalization experiments
    print("\n🌐 Generalization & Efficiency Experiments:")
    for exp_name, exp_config in config.get('generalization_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Qualitative experiments
    print("\n🎨 Qualitative Analysis Experiments:")
    for exp_name, exp_config in config.get('qualitative_experiments', {}).items():
        priority = exp_config.get('priority', 'N/A')
        time_est = exp_config.get('estimated_time', 'N/A')
        status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
        print(f"  • {exp_name}: {exp_config.get('description', 'No description')}")
        print(f"    Priority: {priority}, Time: {time_est}, Status: {status}")

    # Experiment sets
    print("\n📦 Experiment Sets:")
    for set_name, set_config in config.get('experiment_sets', {}).items():
        time_est = set_config.get('estimated_time', 'N/A')
        num_exps = len(set_config.get('experiments', []))
        print(f"  • {set_name}: {set_config.get('description', 'No description')}")
        print(f"    Experiments: {num_exps}, Total Time: {time_est}")


def show_status(config: Dict[str, Any]):
    """Show status of all experiments."""
    print("\n📈 Experiment Status:")
    print("=" * 50)

    results_dir = Path("evaluation_suite/results/experiments")

    total_experiments = 0
    completed_experiments = 0

    categories = [
        'primary_experiments',
        'loss_analysis_experiments',
        'pretraining_experiments',
        'robustness_experiments',
        'generalization_experiments',
        'qualitative_experiments'
    ]

    for category in categories:
        if category in config:
            print(f"\n{category.replace('_', ' ').title()}:")
            for exp_name, exp_config in config[category].items():
                total_experiments += 1

                # Check if experiment has been run
                exp_dir = results_dir / exp_name
                if exp_dir.exists() and list(exp_dir.glob("*/results.json")):
                    status = "✅ Completed"
                    completed_experiments += 1
                elif exp_config.get('completed', False):
                    status = "✅ Completed (Legacy)"
                    completed_experiments += 1
                else:
                    status = "⏳ Pending"

                print(f"  • {exp_name}: {status}")

    completion_rate = (completed_experiments / total_experiments) * 100 if total_experiments > 0 else 0
    print(f"\n📊 Overall Progress: {completed_experiments}/{total_experiments} ({completion_rate:.1f}%)")


def run_single_experiment(experiment_name: str, config: Dict[str, Any], use_slurm: bool = False):
    """Run a single experiment."""
    logging.info(f"Starting experiment: {experiment_name}")

    # Find experiment configuration
    experiment_config = None
    for category in ['primary_experiments', 'loss_analysis_experiments',
                    'pretraining_experiments', 'robustness_experiments']:
        if category in config and experiment_name in config[category]:
            experiment_config = config[category][experiment_name]
            break

    if experiment_config is None:
        logging.error(f"Experiment '{experiment_name}' not found in configuration")
        return False

    # Get detailed experiment configuration
    detailed_config = None

    # Try to get detailed config from appropriate experiment class
    experiment_classes = [
        (PrimaryExperiments, "Primary"),
        (AblationExperiments, "Ablation"),
        (PretrainingExperiments, "Pretraining"),
        (RobustnessExperiments, "Robustness"),
        (EfficiencyExperiments, "Efficiency"),
        (GeneralizationExperiments, "Generalization"),
        (VisualizationExperiments, "Visualization"),
        (QualitativeExperiments, "Qualitative")
    ]

    for exp_class, class_name in experiment_classes:
        if exp_class and experiment_name in exp_class.get_experiment_configs():
            detailed_config = exp_class.get_experiment_configs()[experiment_name]
            logging.info(f"Using {class_name} experiment configuration for {experiment_name}")
            break

    if detailed_config is None:
        # Fall back to basic config from YAML
        logging.warning(f"Using basic configuration for {experiment_name}")
        detailed_config = experiment_config

    if use_slurm:
        # Submit to Slurm
        slurm_runner = SlurmRunner(config.get('hpc', {}))
        job_id = slurm_runner.submit_experiment(experiment_name, detailed_config)
        if job_id:
            logging.info(f"Submitted experiment {experiment_name} to Slurm with job ID: {job_id}")
            return True
        else:
            logging.error(f"Failed to submit experiment {experiment_name} to Slurm")
            return False
    else:
        # Run locally
        evaluator = ComprehensiveEvaluator(config)
        try:
            evaluator.run_experiment(experiment_name, detailed_config)
            logging.info(f"Completed experiment: {experiment_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to run experiment {experiment_name}: {str(e)}")
            return False


def run_experiment_set(set_name: str, config: Dict[str, Any], use_slurm: bool = False):
    """Run a set of experiments."""
    if set_name not in config.get('experiment_sets', {}):
        logging.error(f"Experiment set '{set_name}' not found")
        return False

    experiment_set = config['experiment_sets'][set_name]
    experiments = experiment_set.get('experiments', [])

    logging.info(f"Running experiment set: {set_name}")
    logging.info(f"Total experiments: {len(experiments)}")
    logging.info(f"Estimated time: {experiment_set.get('estimated_time', 'Unknown')}")

    success_count = 0
    for experiment_name in experiments:
        if run_single_experiment(experiment_name, config, use_slurm):
            success_count += 1
        else:
            logging.warning(f"Experiment {experiment_name} failed, continuing with next experiment")

    logging.info(f"Completed experiment set: {set_name}")
    logging.info(f"Success rate: {success_count}/{len(experiments)} ({success_count/len(experiments)*100:.1f}%)")

    return success_count == len(experiments)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Suite for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available experiments
  python evaluation_suite/run_experiments.py --list-experiments

  # Show experiment status
  python evaluation_suite/run_experiments.py --status

  # Run critical experiments
  python evaluation_suite/run_experiments.py --experiment-set critical

  # Run specific experiment
  python evaluation_suite/run_experiments.py --experiment privacy_utility_sgn

  # Submit to Slurm
  python evaluation_suite/run_experiments.py --experiment-set critical --slurm
        """
    )

    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--experiment-set', type=str, help='Run experiment set (critical, complete, quick, paper_ready)')
    parser.add_argument('--list-experiments', action='store_true', help='List all available experiments')
    parser.add_argument('--status', action='store_true', help='Show experiment status')
    parser.add_argument('--config', type=str, default='evaluation_suite/configs/experiments.yaml',
                       help='Path to configuration file')
    parser.add_argument('--slurm', action='store_true', help='Submit jobs to Slurm instead of running locally')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    config = load_config(args.config)

    # Handle different commands
    if args.list_experiments:
        list_experiments(config)
    elif args.status:
        show_status(config)
    elif args.experiment:
        success = run_single_experiment(args.experiment, config, args.slurm)
        sys.exit(0 if success else 1)
    elif args.experiment_set:
        success = run_experiment_set(args.experiment_set, config, args.slurm)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --list-experiments to see what's available!")


if __name__ == "__main__":
    main()
