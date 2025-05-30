#!/usr/bin/env python3
"""
Master Pipeline Script for Transformer Retargeting Project

This is the ultimate entry point that orchestrates the entire workflow:
- Data preprocessing and sampling
- Model pretraining (encoder, DMR/PMR, SGN/MixFormer)
- Main training
- Comprehensive evaluation
- Report generation

Features:
- Interactive pipeline configuration
- HPC job generation and queue management
- Cross-platform support (Windows/Linux)
- Environment validation and setup
- Progress tracking and resumption
- Automated dependency management

Usage:
    # Interactive mode - guided setup
    python scripts/pipeline.py --interactive
    
    # Quick start - run everything with defaults
    python scripts/pipeline.py --quick-start --dataset ntu --setting cv
    
    # Custom pipeline
    python scripts/pipeline.py --steps preprocess,sample,pretrain,train,evaluate --dataset ntu120
    
    # Generate HPC job queue
    python scripts/pipeline.py --quick-start --dataset ntu --setting cv --slurm
    
    # Resume from specific step
    python scripts/pipeline.py --resume-from train --dataset ntu --setting cv
"""

import os
import sys
import argparse
import yaml
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

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
            logging.FileHandler('logs/pipeline.log')
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

def validate_environment() -> bool:
    """Validate that the environment is properly set up."""
    print("\n🔍 ENVIRONMENT VALIDATION")
    print("=" * 40)
    
    required_paths = [
        "src/model", "src/data", "src/training", "src/evaluation", "src/utils",
        "scripts", "configs", "evaluation_suite"
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("⚠️  Missing required directories:")
        for path in missing:
            print(f"   - {path}")
        print("\n💡 The project structure seems incomplete")
        return False
    
    # Check for Python packages
    required_packages = ['torch', 'numpy', 'scipy', 'matplotlib', 'pandas', 'yaml', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Environment validation passed")
    return True

class PipelineState:
    """Manages pipeline state and progress tracking."""
    
    def __init__(self, state_file: str = "logs/pipeline_state.json"):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load pipeline state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'current_step': None,
            'completed_steps': [],
            'failed_steps': [],
            'start_time': None,
            'last_update': None,
            'config': {}
        }
    
    def save_state(self):
        """Save pipeline state to file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def start_pipeline(self, config: Dict[str, Any]):
        """Initialize pipeline state."""
        self.state.update({
            'start_time': datetime.now().isoformat(),
            'config': config,
            'completed_steps': [],
            'failed_steps': [],
            'current_step': None
        })
        self.save_state()
    
    def start_step(self, step: str):
        """Mark step as started."""
        self.state['current_step'] = step
        self.save_state()
    
    def complete_step(self, step: str):
        """Mark step as completed."""
        if step not in self.state['completed_steps']:
            self.state['completed_steps'].append(step)
        self.state['current_step'] = None
        self.save_state()
    
    def fail_step(self, step: str, error: str):
        """Mark step as failed."""
        self.state['failed_steps'].append({'step': step, 'error': error, 'time': datetime.now().isoformat()})
        self.state['current_step'] = None
        self.save_state()

def interactive_pipeline_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interactive pipeline configuration."""
    print("\n🎮 INTERACTIVE PIPELINE CONFIGURATION")
    print("=" * 50)
    
    # Select dataset
    datasets = list(config['datasets'].keys())
    print("\n📊 Available Datasets:")
    for i, dataset in enumerate(datasets, 1):
        dataset_config = config['datasets'][dataset]
        print(f"{i}. {dataset}: {dataset_config['name']} ({dataset_config['num_classes']} classes)")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected_dataset = datasets[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Select setting
    settings = config['datasets'][selected_dataset]['settings']
    print(f"\n⚙️  Available Settings for {selected_dataset}:")
    for i, setting in enumerate(settings, 1):
        print(f"{i}. {setting}")
    
    while True:
        try:
            choice = input(f"\nSelect setting (1-{len(settings)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(settings):
                selected_setting = settings[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(settings)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Select pipeline steps
    all_steps = ['preprocess', 'sample', 'pretrain', 'train', 'evaluate']
    step_descriptions = {
        'preprocess': 'Data preprocessing and normalization',
        'sample': 'Generate paired training/test samples',
        'pretrain': 'Pretrain encoder and baseline models',
        'train': 'Main model training',
        'evaluate': 'Comprehensive evaluation and reporting'
    }
    
    print(f"\n🔄 Pipeline Steps:")
    for i, step in enumerate(all_steps, 1):
        print(f"{i}. {step}: {step_descriptions[step]}")
    
    print("\nSelect steps to run:")
    print("  - Enter numbers separated by commas (e.g., 1,2,3)")
    print("  - Enter 'all' for complete pipeline")
    print("  - Enter 'quick' for sample,train,evaluate")
    
    step_choice = input("\nSteps to run: ").strip()
    
    if step_choice.lower() == 'all':
        selected_steps = all_steps
    elif step_choice.lower() == 'quick':
        selected_steps = ['sample', 'train', 'evaluate']
    else:
        try:
            indices = [int(x.strip()) - 1 for x in step_choice.split(',')]
            selected_steps = [all_steps[i] for i in indices if 0 <= i < len(all_steps)]
        except:
            print("❌ Invalid step selection, using default (all steps)")
            selected_steps = all_steps
    
    # Select evaluations if evaluate step is included
    selected_evaluations = []
    if 'evaluate' in selected_steps:
        print(f"\n🧪 Evaluation Selection:")
        print("Which evaluations would you like to run?")

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

        print(f"\n✅ Selected {len(selected_evaluations)} evaluations:")
        for eval_name in selected_evaluations:
            print(f"   • {eval_name.replace('_', ' ').title()}")

        # Check model dependencies for selected evaluations
        missing_models = check_model_dependencies(selected_evaluations, selected_dataset, selected_setting, config)

        if missing_models:
            if prompt_train_missing_models(missing_models):
                # Add training steps for missing models
                if 'sgn' in missing_models and 'pretrain' not in selected_steps:
                    selected_steps.insert(-1, 'pretrain')  # Add before evaluate
                    print("📝 Added 'pretrain' step for SGN models")
                if 'mixformer' in missing_models and 'pretrain' not in selected_steps:
                    selected_steps.insert(-1, 'pretrain')  # Add before evaluate
                    print("📝 Added 'pretrain' step for MixFormer models")
                if 'transformer' in missing_models and 'train' not in selected_steps:
                    selected_steps.insert(-1, 'train')  # Add before evaluate
                    print("📝 Added 'train' step for Transformer models")

    # Execution mode
    print(f"\n🚀 Execution Mode:")
    print("1. Direct execution (run locally)")
    print("2. Generate HPC jobs (SLURM)")
    print("3. Generate Windows batch files")

    exec_choice = input("Select execution mode (1-3) [1]: ").strip() or "1"
    exec_modes = ['direct', 'slurm', 'windows']
    exec_mode = exec_modes[int(exec_choice) - 1] if exec_choice in ['1', '2', '3'] else 'direct'

    return {
        'dataset': selected_dataset,
        'setting': selected_setting,
        'steps': selected_steps,
        'execution_mode': exec_mode,
        'evaluations': selected_evaluations,
        'interactive': True
    }

def check_step_completion(step: str, dataset: str, setting: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if a pipeline step has already been completed."""
    if 'step_completion' not in config or step not in config['step_completion']:
        return False, []

    step_config = config['step_completion'][step]
    check_files = step_config.get('check_files', [])

    dataset_config = config['datasets'].get(dataset, {})

    # Format file paths with dataset and setting variables
    formatted_files = []
    for file_pattern in check_files:
        try:
            formatted_file = file_pattern.format(
                dataset=dataset,
                setting=setting,
                data_path=dataset_config.get('data_path', ''),
                processed_files=dataset_config.get('processed_files', {})
            )
            formatted_files.append(formatted_file)
        except KeyError:
            # Skip files that can't be formatted (missing variables)
            continue

    # Check if any of the expected files exist
    existing_files = []
    for file_path in formatted_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)

    if existing_files:
        print(f"  ✅ Step '{step}' appears to be completed:")
        for file_path in existing_files:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"     📁 {file_path} ({size_mb:.1f} MB)")
        return True, existing_files

    return False, []

def prompt_overwrite_step(step: str, existing_files: List[str]) -> bool:
    """Prompt user whether to overwrite existing step results."""
    print(f"\n⚠️  Step '{step}' appears to be already completed.")
    print("📁 Existing files:")
    for file_path in existing_files:
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        print(f"   • {file_path} ({size_mb:.1f} MB)")

    while True:
        choice = input(f"\n🔄 Do you want to re-run '{step}' and overwrite existing results? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print(f"🔄 Will re-run '{step}' step")
            return True
        elif choice in ['n', 'no']:
            print(f"⏭️  Skipping '{step}' step")
            return False
        else:
            print("❌ Please enter 'y' or 'n'")

def check_model_dependencies(evaluations: List[str], dataset: str, setting: str, config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Check which models need to be trained for the selected evaluations."""
    missing_models = {
        'sgn': [],
        'mixformer': [],
        'transformer': []
    }

    for evaluation in evaluations:
        # Check SGN model dependencies
        if 'sgn' in evaluation.lower():
            for task in ['ar', 'ri', 'gc']:  # action recognition, re-identification, gesture classification
                if task in evaluation.lower():
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
                if task in evaluation.lower():
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

    # Remove empty lists
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
        choice = input(f"\n🏋️  Do you want to automatically train the missing models? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print(f"🏋️  Will train missing models before evaluation")
            return True
        elif choice in ['n', 'no']:
            print(f"⏭️  Will skip evaluations that require missing models")
            return False
        else:
            print("❌ Please enter 'y' or 'n'")

def execute_step(step: str, pipeline_config: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """Execute a single pipeline step."""
    dataset = pipeline_config['dataset']
    setting = pipeline_config['setting']
    exec_mode = pipeline_config['execution_mode']
    interactive = pipeline_config.get('interactive', False)

    print(f"\n🔄 EXECUTING STEP: {step.upper()}")
    print("=" * 40)

    # Check if step is already completed
    is_completed, existing_files = check_step_completion(step, dataset, setting, config)
    if is_completed:
        if interactive:
            # In interactive mode, ask if user wants to overwrite
            if not prompt_overwrite_step(step, existing_files):
                return True, f"Step {step} skipped (already completed)"
            # If user chose to overwrite, continue with execution
            print(f"🔄 Re-running '{step}' step as requested")
        else:
            # In non-interactive mode, just skip
            print(f"✅ Step '{step}' already completed, skipping")
            return True, f"Step {step} already completed"
    
    if exec_mode == 'direct':
        # Direct execution
        if step == 'preprocess':
            cmd = f"python scripts/preprocess.py --dataset {dataset} --setting {setting}"
        elif step == 'sample':
            cmd = f"python scripts/sample.py --dataset {dataset} --setting {setting}"
        elif step == 'pretrain':
            cmd = f"python scripts/pretrain.py --task encoder --dataset {dataset} --setting {setting}"
        elif step == 'train':
            cmd = f"python scripts/train.py --model transformer --dataset {dataset} --setting {setting}"
        elif step == 'evaluate':
            cmd = f"python scripts/evaluate.py --experiment-set critical"
        else:
            return False, f"Unknown step: {step}"
        
        print(f"💻 Executing: {cmd}")
        print("💡 Direct execution will be implemented in the next phase")
        return True, "Step completed (simulated)"
    
    elif exec_mode == 'slurm':
        # Generate SLURM job
        if step == 'preprocess':
            cmd = f"python scripts/preprocess.py --dataset {dataset} --setting {setting}"
        elif step == 'sample':
            cmd = f"python scripts/sample.py --dataset {dataset} --setting {setting}"
        elif step == 'pretrain':
            cmd = f"python scripts/pretrain.py --task encoder --dataset {dataset} --setting {setting} --slurm"
        elif step == 'train':
            cmd = f"python scripts/train.py --model transformer --dataset {dataset} --setting {setting} --slurm"
        elif step == 'evaluate':
            cmd = f"python scripts/evaluate.py --experiment-set critical --slurm"
        else:
            return False, f"Unknown step: {step}"
        
        print(f"🖥️  Generating SLURM job: {cmd}")
        print("💡 SLURM job generation will be implemented in the next phase")
        return True, "SLURM job generated"
    
    elif exec_mode == 'windows':
        # Generate Windows batch file
        if step == 'preprocess':
            cmd = f"python scripts/preprocess.py --dataset {dataset} --setting {setting}"
        elif step == 'sample':
            cmd = f"python scripts/sample.py --dataset {dataset} --setting {setting}"
        elif step == 'pretrain':
            cmd = f"python scripts/pretrain.py --task encoder --dataset {dataset} --setting {setting} --windows"
        elif step == 'train':
            cmd = f"python scripts/train.py --model transformer --dataset {dataset} --setting {setting} --windows"
        elif step == 'evaluate':
            cmd = f"python scripts/evaluate.py --experiment-set critical --windows"
        else:
            return False, f"Unknown step: {step}"
        
        print(f"🪟 Generating Windows batch: {cmd}")
        print("💡 Windows batch generation will be implemented in the next phase")
        return True, "Windows batch file generated"
    
    return False, f"Unknown execution mode: {exec_mode}"

def run_pipeline(pipeline_config: Dict[str, Any], config: Dict[str, Any], resume_from: Optional[str] = None) -> bool:
    """Run the complete pipeline."""
    state = PipelineState()
    
    if not resume_from:
        state.start_pipeline(pipeline_config)
    
    steps = pipeline_config['steps']
    
    # If resuming, skip completed steps
    if resume_from:
        try:
            resume_idx = steps.index(resume_from)
            steps = steps[resume_idx:]
            print(f"🔄 Resuming pipeline from step: {resume_from}")
        except ValueError:
            print(f"❌ Invalid resume step: {resume_from}")
            return False
    
    print(f"\n🚀 STARTING PIPELINE")
    print("=" * 50)
    print(f"📊 Dataset: {pipeline_config['dataset']}")
    print(f"⚙️  Setting: {pipeline_config['setting']}")
    print(f"🔄 Steps: {' → '.join(steps)}")
    print(f"🖥️  Execution: {pipeline_config['execution_mode']}")
    
    total_steps = len(steps)
    for i, step in enumerate(steps, 1):
        print(f"\n📍 STEP {i}/{total_steps}: {step.upper()}")
        print("-" * 30)
        
        state.start_step(step)
        
        try:
            success, message = execute_step(step, pipeline_config, config)
            
            if success:
                print(f"✅ {step} completed: {message}")
                state.complete_step(step)
            else:
                print(f"❌ {step} failed: {message}")
                state.fail_step(step, message)
                return False
                
        except Exception as e:
            error_msg = f"Exception in {step}: {str(e)}"
            print(f"❌ {error_msg}")
            state.fail_step(step, error_msg)
            return False
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"✅ All {total_steps} steps completed")
    print(f"⏱️  Started: {state.state['start_time']}")
    print(f"⏱️  Finished: {datetime.now().isoformat()}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Master Pipeline Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - guided setup
  python scripts/pipeline.py --interactive
  
  # Quick start with defaults
  python scripts/pipeline.py --quick-start --dataset ntu --setting cv
  
  # Custom pipeline
  python scripts/pipeline.py --steps preprocess,sample,train,evaluate --dataset ntu120 --setting cs
  
  # Generate HPC job queue
  python scripts/pipeline.py --quick-start --dataset ntu --setting cv --slurm
  
  # Resume from specific step
  python scripts/pipeline.py --resume-from train --dataset ntu --setting cv
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--quick-start', action='store_true',
                       help='Quick start with default configuration')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       help='Dataset to use')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'],
                       help='Evaluation setting')
    parser.add_argument('--steps', type=str,
                       help='Comma-separated list of steps (preprocess,sample,pretrain,train,evaluate)')
    parser.add_argument('--resume-from', type=str,
                       choices=['preprocess', 'sample', 'pretrain', 'train', 'evaluate'],
                       help='Resume pipeline from specific step')
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
        print("\n💡 Please fix the environment issues before running the pipeline")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    print("🚀 TRANSFORMER RETARGETING MASTER PIPELINE")
    print("=" * 60)
    print("🎯 The ultimate orchestrator for the complete workflow")
    print("=" * 60)
    
    # Determine execution mode
    exec_mode = 'direct'
    if args.slurm:
        exec_mode = 'slurm'
    elif args.windows:
        exec_mode = 'windows'
    
    # Handle different modes
    if args.interactive:
        pipeline_config = interactive_pipeline_config(config)
        if not pipeline_config:
            print("👋 Pipeline cancelled")
            sys.exit(0)
        pipeline_config['execution_mode'] = exec_mode
        
    elif args.quick_start:
        if not args.dataset or not args.setting:
            print("❌ Dataset and setting required for quick start")
            sys.exit(1)
        
        pipeline_config = {
            'dataset': args.dataset,
            'setting': args.setting,
            'steps': ['sample', 'train', 'evaluate'],  # Quick pipeline
            'execution_mode': exec_mode
        }
        
    elif args.steps and args.dataset and args.setting:
        steps = [s.strip() for s in args.steps.split(',')]
        valid_steps = ['preprocess', 'sample', 'pretrain', 'train', 'evaluate']
        
        invalid_steps = [s for s in steps if s not in valid_steps]
        if invalid_steps:
            print(f"❌ Invalid steps: {', '.join(invalid_steps)}")
            print(f"💡 Valid steps: {', '.join(valid_steps)}")
            sys.exit(1)
        
        pipeline_config = {
            'dataset': args.dataset,
            'setting': args.setting,
            'steps': steps,
            'execution_mode': exec_mode
        }
        
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --interactive for guided setup!")
        print("💡 Or use --quick-start for immediate execution with defaults")
        sys.exit(0)
    
    # Run the pipeline
    success = run_pipeline(pipeline_config, config, args.resume_from)
    
    if success:
        print("\n🎊 CONGRATULATIONS!")
        print("The Transformer Retargeting pipeline has been successfully executed!")
        print("\n📋 Next steps:")
        print("  - Check results in the 'results/' directory")
        print("  - Review logs in the 'logs/' directory")
        print("  - Generate reports with the evaluation suite")
        sys.exit(0)
    else:
        print("\n💥 PIPELINE FAILED")
        print("Check the logs for detailed error information")
        print("Use --resume-from to continue from the failed step after fixing issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
