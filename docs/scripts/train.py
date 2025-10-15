#!/usr/bin/env python3
"""
Unified Training Script for Transformer Retargeting Project

This is the main entry point for all training tasks. It provides:
- Interactive mode for training configuration
- Support for different model architectures (Transformer, SGN, MixFormer)
- HPC job generation with --slurm flag
- Cross-platform support with --windows flag
- Environment validation

Usage:
    # Interactive mode
    python scripts/train.py --interactive
    
    # Train transformer model
    python scripts/train.py --model transformer --dataset ntu --setting cv
    
    # Train SGN model
    python scripts/train.py --model sgn --dataset ntu120 --setting cs
    
    # Generate HPC job
    python scripts/train.py --model transformer --dataset ntu --setting cv --slurm
    
    # Generate Windows batch file
    python scripts/train.py --model sgn --dataset etri --setting cv --windows
"""

import os
import sys
import argparse
import yaml
import logging
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_best_hyperparameters():
    """Load the best hyperparameters from the study results."""
    study_results_path = "experiments/hyperparameter/results/study_results_20250430_013324.json"
    if os.path.exists(study_results_path):
        try:
            with open(study_results_path, 'r') as f:
                study_data = json.load(f)
                return study_data['best_params']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load hyperparameters from {study_results_path}: {e}")
            return {}
    else:
        print(f"Warning: Study results file not found at {study_results_path}")
        return {}

def _get_hyperparameter_args(best_params):
    """Convert hyperparameters to command line arguments."""
    if not best_params:
        return ""

    args = []
    if 'batch_size' in best_params:
        args.append(f"--batch-size {best_params['batch_size']}")
    if 'lr' in best_params:
        args.append(f"--lr {best_params['lr']}")
    if 'decoder_dropout' in best_params:
        args.append(f"--decoder-dropout {best_params['decoder_dropout']}")

    # Add loss weights
    loss_weights = []
    for loss_name, value in best_params.items():
        if loss_name.startswith('loss_'):
            loss_type = loss_name[5:]  # Remove 'loss_' prefix
            loss_weights.append(f"{loss_type}:{value}")

    if loss_weights:
        args.append(f"--loss-weights {','.join(loss_weights)}")

    return " \\\n    " + " \\\n    ".join(args) if args else ""

def get_gpu_count_interactive():
    """Ask user for number of GPUs to use."""
    try:
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"\n🖥️  GPU Information:")
        print(f"   Available GPUs: {available_gpus}")
        if available_gpus > 0:
            for i in range(available_gpus):
                try:
                    name = torch.cuda.get_device_name(i)
                    print(f"   GPU {i}: {name}")
                except:
                    print(f"   GPU {i}: <name unavailable>")
        else:
            print("   No CUDA GPUs available")
    except ImportError:
        available_gpus = 0
        print(f"\n🖥️  GPU Information:")
        print("   PyTorch not available, cannot detect GPUs")

    while True:
        try:
            default_gpus = min(4, available_gpus) if available_gpus > 0 else 1
            gpu_input = input(f"\n🎯 How many GPUs to use? (default: {default_gpus}): ").strip()

            if not gpu_input:
                return default_gpus

            gpu_count = int(gpu_input)
            if gpu_count < 1:
                print("❌ GPU count must be at least 1")
                continue
            elif gpu_count > available_gpus and available_gpus > 0:
                print(f"⚠️  Requested {gpu_count} GPUs but only {available_gpus} available")
                use_anyway = input("   Continue anyway? (y/n): ").strip().lower()
                if use_anyway in ['y', 'yes']:
                    return gpu_count
                else:
                    continue
            else:
                return gpu_count

        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return None

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/train.log')
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
    """Validate that required files and directories exist."""
    required_paths = [
        "src/training",
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

def check_data_availability(dataset: str, setting: str, config: Dict[str, Any]) -> bool:
    """Check if required data files are available."""
    dataset_config = config['datasets'].get(dataset)
    if not dataset_config:
        print(f"❌ Unknown dataset: {dataset}")
        return False
    
    data_path = dataset_config['data_path']
    if not os.path.exists(data_path):
        print(f"⚠️  Data directory not found: {data_path}")
        print(f"💡 Run 'python scripts/preprocess.py --dataset {dataset}' to prepare data")
        return False
    
    return True

def interactive_mode(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interactive mode for training configuration."""
    print("\n🎮 INTERACTIVE TRAINING CONFIGURATION")
    print("=" * 50)
    
    # Select model
    models = list(config['models'].keys())
    print("\n🏗️  Available Models:")
    for i, model in enumerate(models, 1):
        model_config = config['models'][model]
        print(f"{i}. {model}: {model_config['name']}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Select dataset
    datasets = list(config['datasets'].keys())
    print(f"\n📊 Available Datasets:")
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
    
    # Training parameters
    training_config = config['training']
    print(f"\n🏋️  Training Configuration:")
    print(f"Default epochs: {training_config['default_epochs']}")
    print(f"Default batch size: {training_config['default_batch_size']}")
    print(f"Default learning rate: {training_config['default_lr']}")
    
    use_defaults = input("\nUse default training parameters? (y/n): ").strip().lower()
    
    if use_defaults == 'y':
        epochs = training_config['default_epochs']
        batch_size = training_config['default_batch_size']
        lr = training_config['default_lr']
    else:
        epochs = int(input(f"Epochs [{training_config['default_epochs']}]: ") or training_config['default_epochs'])
        batch_size = int(input(f"Batch size [{training_config['default_batch_size']}]: ") or training_config['default_batch_size'])
        lr = float(input(f"Learning rate [{training_config['default_lr']}]: ") or training_config['default_lr'])
    
    return {
        'model': selected_model,
        'dataset': selected_dataset,
        'setting': selected_setting,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr
    }

def generate_slurm_script(train_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate SLURM job script for training."""
    hpc_config = config['hpc']

    # Load best hyperparameters
    best_params = load_best_hyperparameters()

    # Get GPU count from train_config
    gpu_count = train_config.get('gpus', 4)

    # Get appropriate job template based on model and GPU count
    templates = hpc_config.get('job_templates', {})
    if gpu_count == 1:
        template = templates.get('quick', {})
    elif gpu_count == 2:
        template = templates.get('standard', {})
    else:
        template = templates.get('long', {})

    # Merge with defaults, using GPU count from train_config
    defaults = {
        'partition': hpc_config.get('default_partition', 'GPU'),
        'time': hpc_config.get('default_time', '12:00:00'),
        'nodes': hpc_config.get('default_nodes', 1),
        'ntasks_per_node': gpu_count,
        'gres': f'gpu:{gpu_count}',
        'mem': hpc_config.get('default_mem', '64GB')
    }
    defaults.update(template)

    # Override with GPU-specific settings
    defaults['ntasks_per_node'] = gpu_count
    defaults['gres'] = f'gpu:{gpu_count}'

    # Generate training command based on GPU count
    if gpu_count > 1:
        train_cmd = f"torchrun --nproc_per_node={gpu_count} main.py"
    else:
        train_cmd = "python main.py"

    script_content = f"""#!/bin/bash
#SBATCH --job-name=train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}
#SBATCH --partition={defaults['partition']}
#SBATCH --time={defaults['time']}
#SBATCH --nodes={defaults['nodes']}
#SBATCH --ntasks-per-node={defaults['ntasks_per_node']}
#SBATCH --gres={defaults['gres']}
#SBATCH --mem={defaults['mem']}
#SBATCH --output=logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}_%j.out
#SBATCH --error=logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}_%j.err

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Set environment variables for distributed training
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE={gpu_count}

# Set CUDA_VISIBLE_DEVICES to use all allocated GPUs
if [ {gpu_count} -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
else
    # For multi-GPU training, use all available GPUs
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE-1)))
    # Alternative: explicitly set for common cases
    if [ "$SLURM_GPUS_ON_NODE" = "4" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$SLURM_GPUS_ON_NODE" = "2" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    fi
fi

echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p output

# Run training with optimized hyperparameters
echo "Executing: {train_cmd} --dataset {train_config['dataset']} --setting {train_config['setting']} --gpus {gpu_count} --hpc"
{train_cmd} \\
    --dataset {train_config['dataset']} \\
    --setting {train_config['setting']} \\
    --gpus {gpu_count} \\
    --data-path data/ntu_cv_paired_comprehensive.pt \\
    --train-samples 999999999 \\
    --test-samples 10000 \\
    --epochs 5 \\
    --hpc \\
    --mixed-precision \\
    --use-checkpoint \\
    --log-dir logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']} \\{_get_hyperparameter_args(best_params)}

echo "Training completed at $(date)"
"""

    # Save script
    script_path = f"scripts/generated/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}.sbatch"
    os.makedirs("scripts/generated", exist_ok=True)

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)

    return script_path

def generate_windows_script(train_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate Windows batch script for training."""
    gpu_count = train_config.get('gpus', 4)

    script_content = f"""@echo off
REM Training script for {train_config['model']} on {train_config['dataset']} ({train_config['setting']})
REM Generated automatically by Transformer Retargeting training system

echo Starting training at %date% %time%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

REM Run training with optimized hyperparameters
python main.py ^
    --dataset {train_config['dataset']} ^
    --setting {train_config['setting']} ^
    --gpus {gpu_count} ^
    --data-path data/ntu_cv_paired_comprehensive.pt ^
    --train-samples 999999999 ^
    --test-samples 10000 ^
    --epochs 5 ^
    --mixed-precision ^
    --use-checkpoint ^
    --log-dir logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}

echo Training completed at %date% %time%
pause
"""
    
    # Save script
    script_path = f"scripts/generated/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}.bat"
    os.makedirs("scripts/generated", exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/train.py --interactive
  
  # Train transformer model
  python scripts/train.py --model transformer --dataset ntu --setting cv
  
  # Generate HPC job
  python scripts/train.py --model transformer --dataset ntu --setting cv --slurm
  
  # Generate Windows batch file
  python scripts/train.py --model sgn --dataset etri --setting cv --windows
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--model', type=str, choices=['transformer', 'sgn', 'mixformer'],
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       help='Dataset to use')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'],
                       help='Evaluation setting')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (default: ask user, fallback to 4)')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM job script instead of running directly')
    parser.add_argument('--windows', action='store_true',
                       help='Generate Windows batch file instead of running directly')
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
    
    # Load configuration
    config = load_config(args.config)
    
    print("🚀 TRANSFORMER RETARGETING TRAINING SYSTEM")
    print("=" * 50)
    
    # Handle different modes
    if args.interactive:
        train_config = interactive_mode(config)
        if not train_config:
            print("👋 Training cancelled")
            sys.exit(0)
    else:
        if not all([args.model, args.dataset, args.setting]):
            print("❌ Model, dataset, and setting are required in non-interactive mode")
            parser.print_help()
            sys.exit(1)
        
        training_config = config['training']
        train_config = {
            'model': args.model,
            'dataset': args.dataset,
            'setting': args.setting,
            'epochs': args.epochs or training_config['default_epochs'],
            'batch_size': args.batch_size or training_config['default_batch_size'],
            'lr': args.lr or training_config['default_lr'],
            'gpus': args.gpus or get_gpu_count_interactive()
        }
    
    # Validate data availability
    if not check_data_availability(train_config['dataset'], train_config['setting'], config):
        sys.exit(1)
    
    # Generate scripts or run training
    if args.slurm:
        script_path = generate_slurm_script(train_config, config)
        print(f"✅ SLURM script generated: {script_path}")

        # Ask if user wants to submit immediately
        submit = input("🚀 Submit job immediately? (y/n): ").strip().lower()
        if submit in ['y', 'yes']:
            try:
                result = subprocess.run(['sbatch', script_path], capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                print(f"✅ Job submitted: {output}")

                # Extract job ID for monitoring
                job_id_match = re.search(r'Submitted batch job (\d+)', output)
                if job_id_match:
                    job_id = job_id_match.group(1)
                    print(f"💡 Monitor with: squeue -j {job_id}")
                    print(f"💡 View logs: tail -f logs/train_*_{job_id}.out")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to submit job: {e.stderr}")
            except FileNotFoundError:
                print("❌ sbatch command not found. Submit manually with:")
                print(f"   sbatch {script_path}")
        else:
            print(f"💡 Submit manually with: sbatch {script_path}")
    elif args.windows:
        script_path = generate_windows_script(train_config, config)
        print(f"✅ Windows batch script generated: {script_path}")
        print(f"💡 Run with: {script_path}")
    else:
        print(f"🏋️  Training configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        print("\n💡 Direct training execution will be implemented in the next phase")
        print("💡 For now, use --slurm or --windows to generate execution scripts")

if __name__ == "__main__":
    main()
