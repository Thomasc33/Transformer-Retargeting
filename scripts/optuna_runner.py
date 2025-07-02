#!/usr/bin/env python3
"""
Interactive Optuna hyperparameter tuning runner for motion retargeting.
Updated with fixed loss weights to prevent numerical instability.
"""

import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_user_input(prompt, default=None, choices=None):
    """Get user input with validation."""
    while True:
        if default:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if choices and user_input not in choices:
            print(f"Invalid choice. Please select from: {', '.join(choices)}")
            continue
        
        return user_input

def get_int_input(prompt, default=None, min_val=None, max_val=None):
    """Get integer input with validation."""
    while True:
        try:
            if default:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
                value = int(user_input)
            else:
                value = int(input(f"{prompt}: ").strip())
            
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            
            return value
        except ValueError:
            print("Please enter a valid integer.")

def interactive_optuna_setup():
    """Interactive setup for Optuna hyperparameter tuning."""
    
    print("🔬 INTERACTIVE OPTUNA HYPERPARAMETER TUNING SETUP")
    print("=" * 60)
    print("This will help you configure and run hyperparameter optimization")
    print("with numerically stable loss weight ranges.")
    print()
    
    # Basic configuration
    print("📊 BASIC CONFIGURATION:")
    dataset = get_user_input("Dataset", default="ntu", choices=["ntu", "ntu120", "etri"])
    setting = get_user_input("Setting", default="cv", choices=["cv", "cs"])
    
    # Study configuration
    print("\n🔬 STUDY CONFIGURATION:")
    study_name = get_user_input("Study name", default=f"optuna_{dataset}_{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    n_trials = get_int_input("Number of trials", default=50, min_val=1, max_val=1000)
    
    # Training configuration
    print("\n🏋️ TRAINING CONFIGURATION:")
    train_samples = get_int_input("Training samples", default=10000, min_val=1000)
    test_samples = get_int_input("Test samples", default=2000, min_val=100)
    epochs = get_int_input("Epochs per trial", default=20, min_val=1, max_val=100)
    
    # Resource configuration
    print("\n💻 RESOURCE CONFIGURATION:")
    use_slurm = get_user_input("Use SLURM for HPC execution?", default="y", choices=["y", "n"]) == "y"
    
    if use_slurm:
        gpus = get_int_input("Number of GPUs", default=1, min_val=1, max_val=8)
        time_limit = get_user_input("Time limit (HH:MM:SS)", default="48:00:00")
        partition = get_user_input("SLURM partition", default="gpu")
        memory = get_user_input("Memory allocation", default="64G")
    else:
        gpus = 1
        time_limit = None
        partition = None
        memory = None
    
    # Advanced options
    print("\n⚙️ ADVANCED OPTIONS:")
    use_pruning = get_user_input("Enable pruning for faster optimization?", default="y", choices=["y", "n"]) == "y"
    wandb_project = get_user_input("Wandb project name", default=f"Optuna {dataset.upper()} {setting.upper()}")
    
    return {
        'dataset': dataset,
        'setting': setting,
        'study_name': study_name,
        'n_trials': n_trials,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'epochs': epochs,
        'use_slurm': use_slurm,
        'gpus': gpus,
        'time_limit': time_limit,
        'partition': partition,
        'memory': memory,
        'use_pruning': use_pruning,
        'wandb_project': wandb_project
    }

def generate_optuna_script(config):
    """Generate the Optuna execution script."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_name = f"optuna_{config['study_name']}_{timestamp}.bash"
    script_path = Path("scripts/generated") / script_name
    
    # Create generated directory
    script_path.parent.mkdir(exist_ok=True)
    
    if config['use_slurm']:
        # Generate SLURM script
        script_content = f"""#!/bin/bash
#SBATCH --job-name=optuna_{config['study_name']}
#SBATCH --partition={config['partition']}
#SBATCH --time={config['time_limit']}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={config['gpus']}
#SBATCH --gres=gpu:{config['gpus']}
#SBATCH --mem={config['memory']}
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/optuna_{config['study_name']}_%j.out
#SBATCH --error=logs/optuna_{config['study_name']}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

echo "🔬 OPTUNA HYPERPARAMETER TUNING"
echo "================================"
echo "Study: {config['study_name']}"
echo "Dataset: {config['dataset']} {config['setting']}"
echo "Trials: {config['n_trials']}"
echo "Start time: $(date)"
echo ""

# Load required modules
module load pytorch/2.3.0-cuda12.1

# Set environment variables
export CUDA_VISIBLE_DEVICES={','.join(map(str, range(config['gpus'])))}
export WANDB_PROJECT="{config['wandb_project']}"

# Change to project directory
cd /users/tcarr23/Transformer-Retargeting

# Create output directory
mkdir -p experiments/hyperparameter/optuna_results/{config['study_name']}

echo "🚀 Starting Optuna optimization..."
"""
    else:
        # Generate local script
        script_content = f"""#!/bin/bash

echo "🔬 OPTUNA HYPERPARAMETER TUNING (LOCAL)"
echo "======================================="
echo "Study: {config['study_name']}"
echo "Dataset: {config['dataset']} {config['setting']}"
echo "Trials: {config['n_trials']}"
echo "Start time: $(date)"
echo ""

# Set environment variables
export WANDB_PROJECT="{config['wandb_project']}"

# Change to project directory
cd /users/tcarr23/Transformer-Retargeting

# Create output directory
mkdir -p experiments/hyperparameter/optuna_results/{config['study_name']}

echo "🚀 Starting Optuna optimization..."
"""
    
    # Add the Python command
    python_cmd = f"""
# Run Optuna optimization
python evaluation_suite/experiments/hyperparameter/optuna_tuning.py \\
    --dataset {config['dataset']} \\
    --setting {config['setting']} \\
    --study-name {config['study_name']} \\
    --n-trials {config['n_trials']} \\
    --train-samples {config['train_samples']} \\
    --test-samples {config['test_samples']} \\
    --epochs {config['epochs']} \\
    --output-dir experiments/hyperparameter/optuna_results/{config['study_name']}"""
    
    if config['use_pruning']:
        python_cmd += " \\\n    --enable-pruning"
    
    if config['use_slurm']:
        python_cmd += " \\\n    --hpc"
    
    script_content += python_cmd
    
    script_content += f"""

echo ""
echo "✅ Optuna optimization completed at: $(date)"
echo "📊 Results saved to: experiments/hyperparameter/optuna_results/{config['study_name']}"
echo "🏆 Check best_trial.json for optimal hyperparameters"
"""
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive Optuna hyperparameter tuning runner")
    parser.add_argument('--config-file', type=str, help='Load configuration from JSON file')
    parser.add_argument('--non-interactive', action='store_true', help='Run with default settings')
    
    args = parser.parse_args()
    
    if args.config_file and os.path.exists(args.config_file):
        print(f"📄 Loading configuration from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    elif args.non_interactive:
        print("🤖 Running with default configuration...")
        config = {
            'dataset': 'ntu',
            'setting': 'cv',
            'study_name': f"optuna_ntu_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'n_trials': 50,
            'train_samples': 10000,
            'test_samples': 2000,
            'epochs': 20,
            'use_slurm': True,
            'gpus': 1,
            'time_limit': '48:00:00',
            'partition': 'gpu',
            'memory': '64G',
            'use_pruning': True,
            'wandb_project': 'Optuna NTU CV'
        }
    else:
        config = interactive_optuna_setup()
    
    # Save configuration
    config_path = Path("scripts/generated") / f"optuna_config_{config['study_name']}.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n💾 Configuration saved to: {config_path}")
    
    # Generate script
    script_path = generate_optuna_script(config)
    print(f"📝 Execution script generated: {script_path}")
    
    # Show summary
    print(f"\n📋 OPTUNA STUDY SUMMARY:")
    print(f"   Study Name: {config['study_name']}")
    print(f"   Dataset: {config['dataset']} {config['setting']}")
    print(f"   Trials: {config['n_trials']}")
    print(f"   Training: {config['train_samples']} samples, {config['epochs']} epochs")
    print(f"   Resources: {config['gpus']} GPU(s)")
    if config['use_slurm']:
        print(f"   SLURM: {config['partition']} partition, {config['time_limit']} time limit")
    print(f"   Wandb: {config['wandb_project']}")
    
    # Ask to run
    if not args.non_interactive:
        run_now = get_user_input("\nRun optimization now?", default="y", choices=["y", "n"]) == "y"
    else:
        run_now = True
    
    if run_now:
        print(f"\n🚀 Starting Optuna optimization...")
        if config['use_slurm']:
            cmd = ["sbatch", str(script_path)]
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Job submitted successfully!")
                print(f"Job ID: {result.stdout.strip()}")
                print(f"Monitor with: tail -f logs/optuna_{config['study_name']}_*.out")
            else:
                print(f"❌ Job submission failed: {result.stderr}")
        else:
            cmd = ["bash", str(script_path)]
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd)
    else:
        print(f"\n📝 To run later, execute: {'sbatch' if config['use_slurm'] else 'bash'} {script_path}")

if __name__ == "__main__":
    main()
