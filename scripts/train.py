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
        "src/model",
        "src/data", 
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
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}
#SBATCH --partition={hpc_config['default_partition']}
#SBATCH --time={hpc_config['default_time']}
#SBATCH --nodes={hpc_config['default_nodes']}
#SBATCH --ntasks-per-node={hpc_config['default_ntasks_per_node']}
#SBATCH --gres={hpc_config['default_gres']}
#SBATCH --mem={hpc_config['default_mem']}
#SBATCH --output=logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}_%j.out
#SBATCH --error=logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}_%j.err

# Load modules (adjust as needed for your HPC system)
module load python/3.8
module load cuda/11.8
module load pytorch/1.12

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=1

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run training
python main.py \\
    --dataset {train_config['dataset']} \\
    --setting {train_config['setting']} \\
    --epochs {train_config['epochs']} \\
    --batch-size {train_config['batch_size']} \\
    --lr {train_config['lr']} \\
    --hpc \\
    --mixed-precision \\
    --use-checkpoint \\
    --log-dir logs/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}

echo "Training completed at $(date)"
"""
    
    # Save script
    script_path = f"scripts/generated/train_{train_config['model']}_{train_config['dataset']}_{train_config['setting']}.sbatch"
    os.makedirs("scripts/generated", exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def generate_windows_script(train_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate Windows batch script for training."""
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

REM Run training
python main.py ^
    --dataset {train_config['dataset']} ^
    --setting {train_config['setting']} ^
    --epochs {train_config['epochs']} ^
    --batch-size {train_config['batch_size']} ^
    --lr {train_config['lr']} ^
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
            'lr': args.lr or training_config['default_lr']
        }
    
    # Validate data availability
    if not check_data_availability(train_config['dataset'], train_config['setting'], config):
        sys.exit(1)
    
    # Generate scripts or run training
    if args.slurm:
        script_path = generate_slurm_script(train_config, config)
        print(f"✅ SLURM script generated: {script_path}")
        print(f"💡 Submit with: sbatch {script_path}")
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
