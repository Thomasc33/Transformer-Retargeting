#!/usr/bin/env python3
"""
Unified Evaluation Script for Transformer Retargeting Project

This is the main entry point for all evaluation tasks. It provides:
- Interactive mode for experiment selection
- Batch mode for automated evaluation
- HPC job generation with --slurm flag
- Cross-platform support with --windows flag

Usage:
    # Interactive mode
    python scripts/evaluate.py --interactive
    
    # Run specific experiment
    python scripts/evaluate.py --experiment privacy_utility_sgn
    
    # Run experiment set
    python scripts/evaluate.py --experiment-set critical
    
    # Generate HPC jobs
    python scripts/evaluate.py --experiment-set critical --slurm
    
    # Generate Windows batch files
    python scripts/evaluate.py --experiment baseline_comparison --windows
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/evaluate.log')
        ]
    )

def load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'python scripts/setup.py' to initialize the project")
        sys.exit(1)

def load_experiments_config(config_path: str = "evaluation_suite/configs/experiments.yaml") -> Dict[str, Any]:
    """Load experiments configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Experiments configuration not found: {config_path}")
        sys.exit(1)

def validate_environment() -> bool:
    """Validate that required files and directories exist."""
    required_paths = [
        "src/model",
        "src/data", 
        "src/training",
        "src/evaluation",
        "evaluation_suite",
        "configs/main_config.yaml"
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("⚠️  Missing required files/directories:")
        for path in missing:
            print(f"   - {path}")
        print("\n💡 Run 'python scripts/setup.py' to initialize the project")
        return False
    
    return True

def list_experiments(experiments_config: Dict[str, Any]) -> None:
    """Display all available experiments in an organized way."""
    print("\n🧪 AVAILABLE EXPERIMENTS")
    print("=" * 60)
    
    categories = [
        ("primary_experiments", "📊 Primary Experiments"),
        ("loss_analysis_experiments", "🔬 Loss Analysis"),
        ("pretraining_experiments", "🏗️  Pretraining Analysis"),
        ("robustness_experiments", "🛡️  Robustness Analysis"),
        ("generalization_experiments", "🌐 Generalization"),
        ("qualitative_experiments", "🎨 Qualitative Analysis")
    ]
    
    for category_key, category_name in categories:
        if category_key in experiments_config:
            print(f"\n{category_name}:")
            for exp_name, exp_config in experiments_config[category_key].items():
                status = "✅ Completed" if exp_config.get('completed', False) else "⏳ Pending"
                priority = exp_config.get('priority', 'N/A')
                time_est = exp_config.get('estimated_time', 'N/A')
                
                print(f"  • {exp_name}")
                print(f"    {exp_config.get('description', 'No description')}")
                print(f"    Priority: {priority} | Time: {time_est} | Status: {status}")
    
    # Show experiment sets
    if 'experiment_sets' in experiments_config:
        print(f"\n🎯 EXPERIMENT SETS:")
        for set_name, set_config in experiments_config['experiment_sets'].items():
            time_est = set_config.get('estimated_time', 'N/A')
            num_experiments = len(set_config.get('experiments', []))
            print(f"  • {set_name}: {set_config.get('description', 'No description')}")
            print(f"    {num_experiments} experiments | Time: {time_est}")

def check_model_dependencies(evaluations: List[str], config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Check which models need to be trained for the selected evaluations."""
    missing_models = {
        'sgn': [],
        'mixformer': [],
        'transformer': []
    }

    # Get available datasets from config
    datasets = list(config['datasets'].keys())

    for evaluation in evaluations:
        for dataset in datasets:
            for setting in ['cs', 'cv']:
                # Check SGN model dependencies
                if 'sgn' in evaluation.lower():
                    for task in ['ar', 'ri', 'gc']:
                        if task in evaluation.lower() or 'action' in evaluation.lower() and task == 'ar':
                            model_files = [
                                f"trained_models/sgn_{dataset}_{setting}_{task}_pretrained.pth",
                                f"output/{dataset}_sgn_{task}_{setting}/model_best.pth",
                                f"eval/sgn/{dataset}_{setting}_{task}.pth"
                            ]

                            if not any(os.path.exists(f) for f in model_files):
                                missing_models['sgn'].append(f"{task}_{dataset}_{setting}")

                # Check MixFormer model dependencies
                if 'mixformer' in evaluation.lower():
                    for task in ['ar', 'ri', 'gc']:
                        if task in evaluation.lower() or 'action' in evaluation.lower() and task == 'ar':
                            model_files = [
                                f"trained_models/mixformer_{dataset}_{setting}_{task}_pretrained.pth",
                                f"output/{dataset}_mixformer_{task}_{setting}/model_best.pth",
                                f"eval/mixformer/{dataset}_{setting}_{task}.pth"
                            ]

                            if not any(os.path.exists(f) for f in model_files):
                                missing_models['mixformer'].append(f"{task}_{dataset}_{setting}")

                # Check transformer model dependencies
                if 'transformer' in evaluation.lower() or 'autoencoder' in evaluation.lower():
                    model_files = [
                        f"trained_models/{dataset}_{setting}_best.pth",
                        f"model.pth",
                        f"checkpoints/checkpoint_best.pth"
                    ]

                    if not any(os.path.exists(f) for f in model_files):
                        missing_models['transformer'].append(f"{dataset}_{setting}")

    # Remove duplicates and empty lists
    for key in missing_models:
        missing_models[key] = list(set(missing_models[key]))
    return {k: v for k, v in missing_models.items() if v}

def prompt_train_missing_models(missing_models: Dict[str, List[str]]) -> bool:
    """Prompt user whether to train missing models."""
    if not missing_models:
        return False

    print(f"\n⚠️  Some evaluations require models that don't exist yet:")

    for model_type, missing_list in missing_models.items():
        print(f"\n🏗️  Missing {model_type.upper()} models:")
        for missing in missing_list:
            print(f"   • {missing}")

    while True:
        choice = input(f"\n🏋️  Do you want to automatically train the missing models first? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print(f"🏋️  Will train missing models before evaluation")
            return True
        elif choice in ['n', 'no']:
            print(f"⏭️  Will skip evaluations that require missing models")
            return False
        else:
            print("❌ Please enter 'y' or 'n'")

def interactive_mode(experiments_config: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Interactive mode for selecting experiments."""
    print("\n🎮 INTERACTIVE EXPERIMENT SELECTION")
    print("=" * 50)

    while True:
        print("\nWhat would you like to do?")
        print("1. 📋 List all experiments")
        print("2. 🎯 Run experiment set")
        print("3. 🧪 Run single experiment")
        print("4. 🔧 Run custom evaluation selection")
        print("5. ⚙️  Configure experiment")
        print("6. 🚪 Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            list_experiments(experiments_config)
        elif choice == "2":
            return select_experiment_set(experiments_config, config)
        elif choice == "3":
            return select_single_experiment(experiments_config, config)
        elif choice == "4":
            return select_custom_evaluations(config)
        elif choice == "5":
            return configure_experiment(experiments_config)
        elif choice == "6":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def select_custom_evaluations(config: Dict[str, Any]) -> Optional[str]:
    """Let user select custom evaluations with model dependency checking."""
    print("\n🔧 CUSTOM EVALUATION SELECTION")
    print("=" * 40)

    available_evaluations = [
        "transformer_autoencoder",
        "sgn_action_recognition",
        "sgn_re_identification",
        "sgn_gesture_classification",
        "mixformer_action_recognition",
        "mixformer_re_identification",
        "mixformer_gesture_classification",
        "privacy_utility_analysis",
        "loss_component_ablation",
        "masking_ratio_analysis",
        "comprehensive_evaluation"
    ]

    print("Available evaluations:")
    for i, evaluation in enumerate(available_evaluations, 1):
        print(f"{i:2d}. {evaluation.replace('_', ' ').title()}")

    print("\nSelect evaluations:")
    print("  - Enter numbers separated by commas (e.g., 1,2,3)")
    print("  - Enter 'all' for all evaluations")
    print("  - Enter 'critical' for essential evaluations")

    eval_choice = input("\nEvaluations to run: ").strip()

    if eval_choice.lower() == 'all':
        selected_evaluations = available_evaluations
    elif eval_choice.lower() == 'critical':
        selected_evaluations = [
            "transformer_autoencoder",
            "sgn_action_recognition",
            "mixformer_action_recognition",
            "privacy_utility_analysis"
        ]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in eval_choice.split(',')]
            selected_evaluations = [available_evaluations[i] for i in indices if 0 <= i < len(available_evaluations)]
        except:
            print("❌ Invalid evaluation selection, using critical evaluations")
            selected_evaluations = [
                "transformer_autoencoder",
                "sgn_action_recognition",
                "privacy_utility_analysis"
            ]

    if not selected_evaluations:
        print("❌ No evaluations selected")
        return None

    print(f"\n✅ Selected {len(selected_evaluations)} evaluations:")
    for eval_name in selected_evaluations:
        print(f"   • {eval_name.replace('_', ' ').title()}")

    # Check model dependencies
    missing_models = check_model_dependencies(selected_evaluations, config)

    if missing_models:
        if prompt_train_missing_models(missing_models):
            print("\n🏋️  Training missing models...")
            # TODO: Implement model training logic
            print("💡 Model training will be implemented in the next phase")
        else:
            print("\n⏭️  Proceeding with available models only")

    return f"custom:{','.join(selected_evaluations)}"

def select_experiment_set(experiments_config: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Let user select an experiment set."""
    if 'experiment_sets' not in experiments_config:
        print("❌ No experiment sets found in configuration")
        return None

    sets = list(experiments_config['experiment_sets'].keys())

    print("\n🎯 Available Experiment Sets:")
    for i, set_name in enumerate(sets, 1):
        set_config = experiments_config['experiment_sets'][set_name]
        time_est = set_config.get('estimated_time', 'N/A')
        num_exp = len(set_config.get('experiments', []))
        print(f"{i}. {set_name} ({num_exp} experiments, ~{time_est})")
        print(f"   {set_config.get('description', 'No description')}")

    while True:
        try:
            choice = input(f"\nSelect experiment set (1-{len(sets)}) or 'b' to go back: ").strip()
            if choice.lower() == 'b':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(sets):
                selected_set = sets[idx]

                # Check model dependencies for this experiment set
                set_config = experiments_config['experiment_sets'][selected_set]
                experiments = set_config.get('experiments', [])

                missing_models = check_model_dependencies(experiments, config)
                if missing_models:
                    if prompt_train_missing_models(missing_models):
                        print("\n🏋️  Training missing models...")
                        print("💡 Model training will be implemented in the next phase")

                print(f"✅ Selected experiment set: {selected_set}")
                return f"set:{selected_set}"
            else:
                print(f"❌ Please enter a number between 1 and {len(sets)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def select_single_experiment(experiments_config: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Let user select a single experiment."""
    # Collect all experiments from all categories
    all_experiments = {}

    categories = [
        "primary_experiments", "loss_analysis_experiments", "pretraining_experiments",
        "robustness_experiments", "generalization_experiments", "qualitative_experiments"
    ]

    for category in categories:
        if category in experiments_config:
            all_experiments.update(experiments_config[category])

    if not all_experiments:
        print("❌ No experiments found in configuration")
        return None

    exp_names = list(all_experiments.keys())

    print("\n🧪 Available Experiments:")
    for i, exp_name in enumerate(exp_names, 1):
        exp_config = all_experiments[exp_name]
        status = "✅" if exp_config.get('completed', False) else "⏳"
        time_est = exp_config.get('estimated_time', 'N/A')
        print(f"{i:2d}. {status} {exp_name} (~{time_est})")
        print(f"     {exp_config.get('description', 'No description')}")

    while True:
        try:
            choice = input(f"\nSelect experiment (1-{len(exp_names)}) or 'b' to go back: ").strip()
            if choice.lower() == 'b':
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(exp_names):
                selected_exp = exp_names[idx]

                # Check model dependencies for this experiment
                missing_models = check_model_dependencies([selected_exp], config)
                if missing_models:
                    if prompt_train_missing_models(missing_models):
                        print("\n🏋️  Training missing models...")
                        print("💡 Model training will be implemented in the next phase")

                print(f"✅ Selected experiment: {selected_exp}")
                return f"exp:{selected_exp}"
            else:
                print(f"❌ Please enter a number between 1 and {len(exp_names)}")
        except ValueError:
            print("❌ Please enter a valid number or 'b'")

def configure_experiment(experiments_config: Dict[str, Any]) -> Optional[str]:
    """Allow user to configure experiment parameters."""
    print("\n⚙️  EXPERIMENT CONFIGURATION")
    print("This feature will be implemented in the next version.")
    print("For now, you can modify configs/main_config.yaml directly.")
    return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/evaluate.py --interactive
  
  # List experiments
  python scripts/evaluate.py --list
  
  # Run specific experiment
  python scripts/evaluate.py --experiment privacy_utility_sgn
  
  # Run experiment set
  python scripts/evaluate.py --experiment-set critical
  
  # Generate HPC jobs
  python scripts/evaluate.py --experiment-set critical --slurm
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--experiment', type=str,
                       help='Run specific experiment')
    parser.add_argument('--experiment-set', type=str,
                       help='Run experiment set')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM job scripts instead of running directly')
    parser.add_argument('--windows', action='store_true',
                       help='Generate Windows batch files instead of running directly')
    parser.add_argument('--config', type=str, default='configs/main_config.yaml',
                       help='Path to main configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Load configurations
    config = load_config(args.config)
    experiments_config = load_experiments_config()
    
    print("🚀 TRANSFORMER RETARGETING EVALUATION SUITE")
    print("=" * 50)
    
    # Handle different modes
    if args.list:
        list_experiments(experiments_config)
    elif args.interactive:
        selection = interactive_mode(experiments_config, config)
        if selection:
            print(f"\n🎯 Would execute: {selection}")
            print("💡 Full execution will be implemented in the next phase")
    elif args.experiment:
        # Check model dependencies for single experiment
        missing_models = check_model_dependencies([args.experiment], config)
        if missing_models:
            if prompt_train_missing_models(missing_models):
                print("\n🏋️  Training missing models...")
                print("💡 Model training will be implemented in the next phase")

        print(f"🧪 Running experiment: {args.experiment}")
        print("💡 Experiment execution will be implemented in the next phase")
    elif args.experiment_set:
        # Check model dependencies for experiment set
        if 'experiment_sets' in experiments_config and args.experiment_set in experiments_config['experiment_sets']:
            set_config = experiments_config['experiment_sets'][args.experiment_set]
            experiments = set_config.get('experiments', [])
            missing_models = check_model_dependencies(experiments, config)
            if missing_models:
                if prompt_train_missing_models(missing_models):
                    print("\n🏋️  Training missing models...")
                    print("💡 Model training will be implemented in the next phase")

        print(f"🎯 Running experiment set: {args.experiment_set}")
        print("💡 Experiment set execution will be implemented in the next phase")
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --interactive to explore available experiments!")

if __name__ == "__main__":
    main()
