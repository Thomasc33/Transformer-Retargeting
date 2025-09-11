#!/usr/bin/env python3
"""
Unified Pretraining Script for Transformer Retargeting Project

This script handles all pretraining tasks:
- Encoder pretraining with MLM (Masked Language Modeling)
- DMR/PMR pretraining
- SGN/MixFormer pretraining for AR/RI/GC tasks
- Interactive configuration
- HPC job generation

Usage:
    # Interactive mode
    python scripts/pretrain.py --interactive
    
    # Pretrain encoder
    python scripts/pretrain.py --task encoder --dataset ntu --setting cv
    
    # Pretrain DMR/PMR
    python scripts/pretrain.py --task dmr_pmr --dataset ntu120 --setting cs
    
    # Pretrain SGN for action recognition
    python scripts/pretrain.py --task sgn --target ar --dataset etri --setting cv
    
    # Generate HPC jobs
    python scripts/pretrain.py --task encoder --dataset ntu --setting cv --slurm
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
            logging.FileHandler('logs/pretrain.log')
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

def check_step_completion(step: str, dataset: str, setting: str, config: Dict[str, Any]) -> bool:
    """Check if a pipeline step has already been completed."""
    if 'step_completion' not in config or step not in config['step_completion']:
        return False

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
        return True

    return False

def check_pretrain_completion(task: str, dataset: str, setting: str, config: Dict[str, Any], **kwargs) -> bool:
    """Check if specific pretraining task is completed."""
    if task == 'encoder':
        # Check for encoder pretraining files
        temporal_ratio = kwargs.get('temporal_ratio', 0.5)
        spatial_ratio = kwargs.get('spatial_ratio', 0.5)

        encoder_files = [
            f"eval/mixformer/pretrained/{dataset}/encoder_{setting}.pth",
            f"eval/mixformer/pretrained/{dataset}/encoder_{setting}_comprehensive.pth",
            f"eval/mixformer/pretrained/{dataset}/encoder_{setting}_t{temporal_ratio}_s{spatial_ratio}.pth"
        ]

        for file_path in encoder_files:
            if os.path.exists(file_path):
                size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                print(f"  ✅ Encoder pretraining completed: {file_path} ({size_mb:.1f} MB)")
                return True

    elif task == 'dmr_pmr':
        # Check for DMR/PMR files
        dmr_file = f"eval/dmr/{dataset}.pt"
        pmr_file = f"eval/pmr/{dataset}.pt"

        if os.path.exists(dmr_file) and os.path.exists(pmr_file):
            dmr_size = Path(dmr_file).stat().st_size / (1024 * 1024)
            pmr_size = Path(pmr_file).stat().st_size / (1024 * 1024)
            print(f"  ✅ DMR/PMR pretraining completed:")
            print(f"     📁 {dmr_file} ({dmr_size:.1f} MB)")
            print(f"     📁 {pmr_file} ({pmr_size:.1f} MB)")
            return True

    elif task in ['sgn', 'mixformer']:
        target = kwargs.get('target', 'ar')
        # Check for SGN/MixFormer pretrained models
        model_files = [
            f"trained_models/{task}_{dataset}_{setting}_{target}_pretrained.pth",
            f"output/{dataset}_{task}_{target}_{setting}/model_best.pth"
        ]

        for file_path in model_files:
            if os.path.exists(file_path):
                size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                print(f"  ✅ {task.upper()} pretraining completed: {file_path} ({size_mb:.1f} MB)")
                return True

    return False

def interactive_mode(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Interactive mode for pretraining configuration."""
    print("\n🎮 INTERACTIVE PRETRAINING CONFIGURATION")
    print("=" * 50)
    
    # Select pretraining task
    tasks = {
        'encoder': 'Encoder pretraining with MLM (Masked Language Modeling)',
        'dmr_pmr': 'DMR/PMR pretraining for motion retargeting',
        'sgn': 'SGN pretraining for action recognition/re-identification/gesture classification',
        'mixformer': 'MixFormer pretraining for action recognition/re-identification/gesture classification'
    }
    
    print("\n🏗️  Available Pretraining Tasks:")
    task_list = list(tasks.keys())
    for i, task in enumerate(task_list, 1):
        print(f"{i}. {task}: {tasks[task]}")
    
    while True:
        try:
            choice = input(f"\nSelect pretraining task (1-{len(task_list)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(task_list):
                selected_task = task_list[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(task_list)}")
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
    
    # Task-specific configuration
    result = {
        'task': selected_task,
        'dataset': selected_dataset,
        'setting': selected_setting
    }
    
    if selected_task in ['sgn', 'mixformer']:
        # Select target task for SGN/MixFormer
        targets = ['ar', 'ri', 'gc']  # Action Recognition, Re-Identification, Gesture Classification
        target_names = {
            'ar': 'Action Recognition',
            'ri': 'Re-Identification', 
            'gc': 'Gesture Classification'
        }
        
        print(f"\n🎯 Target Task for {selected_task.upper()}:")
        for i, target in enumerate(targets, 1):
            print(f"{i}. {target}: {target_names[target]}")
        
        while True:
            try:
                choice = input(f"\nSelect target task (1-{len(targets)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(targets):
                    result['target'] = targets[idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(targets)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    if selected_task == 'encoder':
        # Configure masking ratios for encoder pretraining
        masking_config = config['pretraining']['encoder']['masking_ratios']
        print(f"\n🎭 Masking Configuration:")
        print(f"Available temporal ratios: {masking_config['temporal']}")
        print(f"Available spatial ratios: {masking_config['spatial']}")
        
        temporal_ratio = float(input(f"Temporal masking ratio [0.5]: ") or "0.5")
        spatial_ratio = float(input(f"Spatial masking ratio [0.5]: ") or "0.5")
        
        result['temporal_ratio'] = temporal_ratio
        result['spatial_ratio'] = spatial_ratio
    
    # Training parameters
    pretrain_config = config['pretraining'][selected_task]
    print(f"\n🏋️  Pretraining Configuration for {selected_task}:")
    print(f"Default epochs: {pretrain_config['epochs']}")
    print(f"Default batch size: {pretrain_config['batch_size']}")
    print(f"Default learning rate: {pretrain_config['lr']}")
    
    use_defaults = input("\nUse default pretraining parameters? (y/n): ").strip().lower()
    
    if use_defaults == 'y':
        result.update({
            'epochs': pretrain_config['epochs'],
            'batch_size': pretrain_config['batch_size'],
            'lr': pretrain_config['lr']
        })
    else:
        result.update({
            'epochs': int(input(f"Epochs [{pretrain_config['epochs']}]: ") or pretrain_config['epochs']),
            'batch_size': int(input(f"Batch size [{pretrain_config['batch_size']}]: ") or pretrain_config['batch_size']),
            'lr': float(input(f"Learning rate [{pretrain_config['lr']}]: ") or pretrain_config['lr'])
        })
    
    return result

def generate_slurm_script(pretrain_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate SLURM job script for pretraining."""
    hpc_config = config['hpc']
    task = pretrain_config['task']
    
    # Determine job template based on task
    if task == 'encoder':
        template = hpc_config['job_templates']['standard']
    elif task == 'dmr_pmr':
        template = hpc_config['job_templates']['long']
    else:  # sgn, mixformer
        template = hpc_config['job_templates']['long']
    
    job_name = f"pretrain_{task}_{pretrain_config['dataset']}_{pretrain_config['setting']}"
    if 'target' in pretrain_config:
        job_name += f"_{pretrain_config['target']}"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={hpc_config['default_partition']}
#SBATCH --time={template['time']}
#SBATCH --nodes={hpc_config['default_nodes']}
#SBATCH --ntasks-per-node={template['ntasks_per_node']}
#SBATCH --gres={template['gres']}
#SBATCH --mem={template['mem']}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

# Load modules (adjust as needed for your HPC system)
module load python/3.8
module load cuda/11.8
module load pytorch/1.12

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=1

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create output directory
mkdir -p eval/{task}/pretrained/{pretrain_config['dataset']}

# Run pretraining based on task
"""
    
    if task == 'encoder':
        script_content += f"""
# Encoder pretraining with MLM
python pretrain.py \\
    --dataset {pretrain_config['dataset']} \\
    --setting {pretrain_config['setting']} \\
    --epochs {pretrain_config['epochs']} \\
    --batch-size {pretrain_config['batch_size']} \\
    --lr {pretrain_config['lr']} \\
    --temporal-ratio {pretrain_config.get('temporal_ratio', 0.5)} \\
    --spatial-ratio {pretrain_config.get('spatial_ratio', 0.5)} \\
    --hpc \\
    --mixed-precision \\
    --save-path eval/mixformer/pretrained/{pretrain_config['dataset']}/encoder_{pretrain_config['setting']}_t{pretrain_config.get('temporal_ratio', 0.5)}_s{pretrain_config.get('spatial_ratio', 0.5)}.pth
"""
    elif task == 'dmr_pmr':
        script_content += f"""
# DMR/PMR pretraining
python eval/dmr/dmr.py --dataset {pretrain_config['dataset']} --setting {pretrain_config['setting']} --epochs {pretrain_config['epochs']}
python eval/pmr/pmr.py --dataset {pretrain_config['dataset']} --setting {pretrain_config['setting']} --epochs {pretrain_config['epochs']}
"""
    else:  # sgn, mixformer
        target = pretrain_config.get('target', 'ar')
        script_content += f"""
# {task.upper()} pretraining for {target.upper()}
python train_{task}.py \\
    --dataset {pretrain_config['dataset']} \\
    --setting {pretrain_config['setting']} \\
    --task {target} \\
    --epochs {pretrain_config['epochs']} \\
    --batch-size {pretrain_config['batch_size']} \\
    --lr {pretrain_config['lr']} \\
    --hpc \\
    --mixed-precision
"""
    
    script_content += f"""
echo "Pretraining completed at $(date)"
"""
    
    # Save script
    script_path = f"scripts/generated/{job_name}.sbatch"
    os.makedirs("scripts/generated", exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def generate_windows_script(pretrain_config: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate Windows batch script for pretraining."""
    task = pretrain_config['task']
    job_name = f"pretrain_{task}_{pretrain_config['dataset']}_{pretrain_config['setting']}"
    if 'target' in pretrain_config:
        job_name += f"_{pretrain_config['target']}"
    
    script_content = f"""@echo off
REM Pretraining script for {task} on {pretrain_config['dataset']} ({pretrain_config['setting']})
REM Generated automatically by Transformer Retargeting pretraining system

echo Starting pretraining at %date% %time%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

REM Create output directory
mkdir eval\\{task}\\pretrained\\{pretrain_config['dataset']} 2>nul

REM Run pretraining based on task
"""
    
    if task == 'encoder':
        script_content += f"""
REM Encoder pretraining with MLM
python pretrain.py ^
    --dataset {pretrain_config['dataset']} ^
    --setting {pretrain_config['setting']} ^
    --epochs {pretrain_config['epochs']} ^
    --batch-size {pretrain_config['batch_size']} ^
    --lr {pretrain_config['lr']} ^
    --temporal-ratio {pretrain_config.get('temporal_ratio', 0.5)} ^
    --spatial-ratio {pretrain_config.get('spatial_ratio', 0.5)} ^
    --mixed-precision ^
    --save-path eval/mixformer/pretrained/{pretrain_config['dataset']}/encoder_{pretrain_config['setting']}_t{pretrain_config.get('temporal_ratio', 0.5)}_s{pretrain_config.get('spatial_ratio', 0.5)}.pth
"""
    elif task == 'dmr_pmr':
        script_content += f"""
REM DMR/PMR pretraining
python eval/dmr/dmr.py --dataset {pretrain_config['dataset']} --setting {pretrain_config['setting']} --epochs {pretrain_config['epochs']}
python eval/pmr/pmr.py --dataset {pretrain_config['dataset']} --setting {pretrain_config['setting']} --epochs {pretrain_config['epochs']}
"""
    else:  # sgn, mixformer
        target = pretrain_config.get('target', 'ar')
        script_content += f"""
REM {task.upper()} pretraining for {target.upper()}
python train_{task}.py ^
    --dataset {pretrain_config['dataset']} ^
    --setting {pretrain_config['setting']} ^
    --task {target} ^
    --epochs {pretrain_config['epochs']} ^
    --batch-size {pretrain_config['batch_size']} ^
    --lr {pretrain_config['lr']} ^
    --mixed-precision
"""
    
    script_content += f"""
echo Pretraining completed at %date% %time%
pause
"""
    
    # Save script
    script_path = f"scripts/generated/{job_name}.bat"
    os.makedirs("scripts/generated", exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Pretraining Script for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/pretrain.py --interactive
  
  # Pretrain encoder
  python scripts/pretrain.py --task encoder --dataset ntu --setting cv
  
  # Pretrain SGN for action recognition
  python scripts/pretrain.py --task sgn --target ar --dataset ntu120 --setting cs
  
  # Generate HPC job
  python scripts/pretrain.py --task encoder --dataset ntu --setting cv --slurm
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--task', type=str, choices=['encoder', 'dmr_pmr', 'sgn', 'mixformer'],
                       help='Pretraining task')
    parser.add_argument('--target', type=str, choices=['ar', 'ri', 'gc'],
                       help='Target task for SGN/MixFormer (ar=action recognition, ri=re-identification, gc=gesture classification)')
    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'],
                       help='Dataset to use')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'],
                       help='Evaluation setting')
    parser.add_argument('--temporal-ratio', type=float, default=0.5,
                       help='Temporal masking ratio for encoder pretraining')
    parser.add_argument('--spatial-ratio', type=float, default=0.5,
                       help='Spatial masking ratio for encoder pretraining')
    parser.add_argument('--epochs', type=int,
                       help='Number of pretraining epochs')
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
    
    print("🚀 TRANSFORMER RETARGETING PRETRAINING SYSTEM")
    print("=" * 50)
    
    # Handle different modes
    if args.interactive:
        pretrain_config = interactive_mode(config)
        if not pretrain_config:
            print("👋 Pretraining cancelled")
            sys.exit(0)
    else:
        if not all([args.task, args.dataset, args.setting]):
            print("❌ Task, dataset, and setting are required in non-interactive mode")
            parser.print_help()
            sys.exit(1)
        
        pretrain_defaults = config['pretraining'][args.task]
        pretrain_config = {
            'task': args.task,
            'dataset': args.dataset,
            'setting': args.setting,
            'epochs': args.epochs or pretrain_defaults['epochs'],
            'batch_size': args.batch_size or pretrain_defaults['batch_size'],
            'lr': args.lr or pretrain_defaults['lr']
        }
        
        if args.target:
            pretrain_config['target'] = args.target
        
        if args.task == 'encoder':
            pretrain_config.update({
                'temporal_ratio': args.temporal_ratio,
                'spatial_ratio': args.spatial_ratio
            })
    
    # Generate scripts or run pretraining
    if args.slurm:
        script_path = generate_slurm_script(pretrain_config, config)
        print(f"✅ SLURM script generated: {script_path}")
        print(f"💡 Submit with: sbatch {script_path}")
    elif args.windows:
        script_path = generate_windows_script(pretrain_config, config)
        print(f"✅ Windows batch script generated: {script_path}")
        print(f"💡 Run with: {script_path}")
    else:
        print(f"🏗️  Pretraining configuration:")
        for key, value in pretrain_config.items():
            print(f"  {key}: {value}")
        print("\n💡 Direct pretraining execution will be implemented in the next phase")
        print("💡 For now, use --slurm or --windows to generate execution scripts")

if __name__ == "__main__":
    main()
