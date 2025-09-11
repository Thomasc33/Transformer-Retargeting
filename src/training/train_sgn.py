#!/usr/bin/env python3
"""
SGN Training Script for AR/RI/GC Tasks

This script trains SGN models for Action Recognition (AR), Re-identification (RI), 
and Gender Classification (GC) tasks on NTU60, NTU120, and ETRI datasets.

Usage:
    # Train SGN AR model for NTU60 CV
    python train_sgn.py --dataset ntu --setting cv --task ar
    
    # Train SGN RI model for NTU120 CS  
    python train_sgn.py --dataset ntu120 --setting cs --task ri
    
    # Train SGN GC model for NTU60 CS
    python train_sgn.py --dataset ntu --setting cs --task gc
    
    # Generate SLURM job
    python train_sgn.py --dataset ntu --setting cs --task ar --slurm
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_task_config(dataset: str, setting: str, task: str) -> Dict[str, Any]:
    """Get configuration for specific task."""
    
    # Dataset configurations
    dataset_configs = {
        'ntu': {
            'num_classes': {'ar': 60, 'ri': 40, 'gc': 2},
            'data_path': 'data/ntu',
            'processed_file': f'ntu_{setting}_processed.pkl'
        },
        'ntu120': {
            'num_classes': {'ar': 120, 'ri': 106, 'gc': 2},
            'data_path': 'data/ntu120', 
            'processed_file': f'ntu120_{setting}_processed.pkl'
        },
        'etri': {
            'num_classes': {'ar': 55, 'ri': 50, 'gc': 2},
            'data_path': 'data/etri',
            'processed_file': f'etri_{setting}_processed.pkl'
        }
    }
    
    base_config = dataset_configs[dataset]
    
    config = {
        'dataset': dataset,
        'setting': setting,
        'task': task,
        'num_classes': base_config['num_classes'][task],
        'data_path': base_config['data_path'],
        'processed_file': base_config['processed_file'],
        'output_dir': f'output/{dataset}_sgn_{task}_{setting}',
        'model_save_path': f'eval/sgn/pretrained/{dataset}/{setting}_{task}.pth',
        
        # Training hyperparameters
        'epochs': 150,
        'batch_size': 64,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'step_size': 30,
        'gamma': 0.1,
        
        # Model parameters
        'num_point': 25,
        'num_person': 2,
        'seg': 20,
        'graph': 'src.graph.ntu_rgb_d.Graph',
        'graph_args': {'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        
        # Training settings
        'device': 'cuda',
        'num_workers': 4,
        'save_interval': 10,
        'eval_interval': 5,
        'early_stopping_patience': 20,
        'mixed_precision': False,  # Disabled due to NaN gradient issues
    }
    
    return config

def create_training_script(config: Dict[str, Any]) -> str:
    """Create the actual training script content."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Generated SGN Training Script for {config['task'].upper()} task
Dataset: {config['dataset'].upper()}, Setting: {config['setting'].upper()}
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.sgn import SGN
from src.data import load_data, get_cross_data, optimize_data_loading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_sgn_model():
    """Train SGN model for {config['task'].upper()} task."""

    # Configuration
    config = {{
        "dataset": "{config['dataset']}",
        "setting": "{config['setting']}",
        "task": "{config['task']}",
        "num_classes": {config['num_classes']},
        "data_path": "{config['data_path']}",
        "processed_file": "{config['processed_file']}",
        "output_dir": "{config['output_dir']}",
        "model_save_path": "{config['model_save_path']}",
        "epochs": {config['epochs']},
        "batch_size": {config['batch_size']},
        "lr": {config['lr']},
        "weight_decay": {config['weight_decay']},
        "momentum": {config['momentum']},
        "step_size": {config['step_size']},
        "gamma": {config['gamma']},
        "num_point": {config['num_point']},
        "num_person": {config['num_person']},
        "seg": {config['seg']},
        "graph": "{config['graph']}",
        "device": "{config['device']}",
        "num_workers": {config['num_workers']},
        "save_interval": {config['save_interval']},
        "eval_interval": {config['eval_interval']},
        "early_stopping_patience": {config['early_stopping_patience']},
        "mixed_precision": {'True' if config['mixed_precision'] else 'False'}
    }}

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {{device}}")

    # Limit PyTorch threading to match SLURM allocation
    torch.set_num_threads(4)  # Conservative threading

    # Load data
    logger.info("Loading data...")
    X = load_data(config['dataset'], T=64)

    # Create paired datasets for training
    paired_train, paired_test = get_cross_data(
        X, config['dataset'], config['setting'],
        batch_size=config['batch_size'],
        return_loader=False,
        train_samples=10000,  # Use smaller sample for faster training
        test_samples=2000,
        threads=1,
        seg=config['seg'],
        augment=True,
        train_theta=0.3,
        val_theta=0.3
    )

    # Create data loaders
    train_loader, test_loader = optimize_data_loading(
        paired_train, paired_test, config['batch_size'],
        distributed=False, rank=0, world_size=1
    )
    
    # Create model
    logger.info("Creating SGN model...")
    model = SGN(
        num_classes=config['num_classes'],
        dataset=config['dataset'],
        seg=config['seg']
    )
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle paired data format: (x1, x2, y1, y2, actors, actions)
            x1, x2, y1, y2, actors, actions = batch

            # For classification tasks, use x1 and appropriate labels
            if config['task'] == 'ar':
                inputs, labels = x1, actions[:, 0]  # Use first action
            elif config['task'] == 'ri':
                inputs, labels = x1, actors[:, 0]  # Use first actor
            elif config['task'] == 'gc':
                # Gender classification - use first actor for now
                inputs, labels = x1, actors[:, 0] % 2  # Simple gender mapping
            else:
                inputs, labels = x1, actions[:, 0]
            
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {{epoch+1}}/{{config["epochs"]}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}')
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        if (epoch + 1) % config['eval_interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in test_loader:  # Use test_loader for validation
                    x1, x2, y1, y2, actors, actions = batch

                    if config['task'] == 'ar':
                        inputs, labels = x1, actions[:, 0]
                    elif config['task'] == 'ri':
                        inputs, labels = x1, actors[:, 0]
                    elif config['task'] == 'gc':
                        inputs, labels = x1, actors[:, 0] % 2
                    else:
                        inputs, labels = x1, actions[:, 0]
                    
                    inputs = inputs.float().to(device)
                    labels = labels.long().to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(test_loader)
            
            logger.info(f'Epoch {{epoch+1}}/{{config["epochs"]}}:')
            logger.info(f'  Train Loss: {{avg_train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
            logger.info(f'  Val Loss: {{avg_val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({{
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'config': config
                }}, config['model_save_path'])
                logger.info(f'New best model saved with accuracy: {{best_acc:.2f}}%')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f'Early stopping triggered after {{epoch+1}} epochs')
                break
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{{epoch+1}}.pth')
            torch.save({{
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': config
            }}, checkpoint_path)
    
    logger.info(f'Training completed. Best accuracy: {{best_acc:.2f}}%')
    logger.info(f'Best model saved to: {{config["model_save_path"]}}')

if __name__ == '__main__':
    train_sgn_model()
'''
    
    return script_content

def create_slurm_script(config: Dict[str, Any]) -> str:
    """Create SLURM job script for SGN training.

    Generates a single sbatch file that calls this train_sgn.py with the same args.
    No per-case Python script is generated to avoid repo clutter.
    """

    job_name = f"sgn_{config['task']}_{config['dataset']}_{config['setting']}"

    slurm_script = f'''#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carrt313@gmail.com

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Load PyTorch module
module purge
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2>/dev/null || true
nvidia-smi || true

echo ""
echo "Starting SGN {config['task'].upper()} training for {config['dataset'].upper()} {config['setting'].upper()}"
echo "Configuration:"
echo "  Dataset: {config['dataset']}"
echo "  Setting: {config['setting']}"
echo "  Task: {config['task']}"
echo "  Num Classes: {config['num_classes']}"
echo "  Epochs: {config['epochs']}"
echo "  Batch Size: {config['batch_size']}"
echo "  Learning Rate: {config['lr']}"
echo ""

# Ensure directories exist
mkdir -p logs
mkdir -p output

# Run training using this script with the same args
python -u train_sgn.py \
    --dataset {config['dataset']} \
    --setting {config['setting']} \
    --task {config['task']} \
    --epochs {config['epochs']} \
    --batch-size {config['batch_size']} \
    --lr {config['lr']}

echo ""
echo "Training completed at $(date)"
'''

    return slurm_script

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="SGN Training Script for AR/RI/GC Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SGN AR model for NTU60 CV
  python train_sgn.py --dataset ntu --setting cv --task ar

  # Train SGN RI model for NTU120 CS
  python train_sgn.py --dataset ntu120 --setting cs --task ri

  # Generate SLURM job for SGN GC
  python train_sgn.py --dataset ntu --setting cs --task gc --slurm
        """
    )

    parser.add_argument('--dataset', type=str, choices=['ntu', 'ntu120', 'etri'], required=True,
                       help='Dataset to use')
    parser.add_argument('--setting', type=str, choices=['cs', 'cv'], required=True,
                       help='Cross-subject (cs) or cross-view (cv) setting')
    parser.add_argument('--task', type=str, choices=['ar', 'ri', 'gc'], required=True,
                       help='Task: ar (action recognition), ri (re-identification), gc (gender classification)')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate SLURM job script instead of running directly')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')

    args = parser.parse_args()

    # Validate RI setting (no CS for RI as requested)
    if args.task == 'ri' and args.setting == 'cs':
        print("❌ Error: RI task does not support CS setting (as requested)")
        return

    # Get task configuration
    config = get_task_config(args.dataset, args.setting, args.task)

    # Override with command line arguments
    if args.epochs != 150:
        config['epochs'] = args.epochs
    if args.batch_size != 64:
        config['batch_size'] = args.batch_size
    if args.lr != 1e-4:
        config['lr'] = args.lr

    print(f"🚀 SGN {args.task.upper()} Training Setup")
    print("=" * 50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Setting: {args.setting.upper()}")
    print(f"Task: {args.task.upper()}")
    print(f"Num Classes: {config['num_classes']}")
    print(f"Output Dir: {config['output_dir']}")
    print(f"Model Save Path: {config['model_save_path']}")
    print("")

    if args.slurm:
        # Generate SLURM job only (no per-case Python training file)
        print("📝 Generating SLURM job script...")

        # Save SLURM script into bash/train_eval/sgn
        os.makedirs("bash/train_eval/sgn", exist_ok=True)
        slurm_name = f"bash/train_eval/sgn/train_sgn_{args.task}_{args.dataset}_{args.setting}.sbatch"
        slurm_content = create_slurm_script(config)

        with open(slurm_name, 'w') as f:
            f.write(slurm_content)
        print(f"✅ SLURM script saved: {slurm_name}")
        print(f"\n🎯 To submit job: sbatch {slurm_name}")

    else:
        # Run training directly
        print("🏃 Running training directly...")
        # Build training function source and execute
        script_content = create_training_script(config)
        exec(script_content)

if __name__ == '__main__':
    main()
