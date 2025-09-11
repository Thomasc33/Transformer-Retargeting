#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Weight Manager

This module handles detection and management of missing AR/RI model weights
for SGN and Mixformer models. It provides alternatives and training options
when required models are not available.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ModelWeightManager:
    """Manage model weights and handle missing models."""
    
    def __init__(self, dataset, setting):
        self.dataset = dataset
        self.setting = setting
        self.available_models = {}
        self.missing_models = {}
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan for available model weights."""
        print("🔍 Scanning for available model weights...")
        
        # SGN models
        sgn_base = f"eval/sgn/pretrained/{self.dataset}"
        if os.path.exists(sgn_base):
            sgn_files = os.listdir(sgn_base)
            self.available_models['sgn'] = {}
            
            # Check for AR models
            ar_patterns = [f"{self.setting}_ar.pth", f"cview_ar.pth", f"cv_ar.pth"]
            for pattern in ar_patterns:
                if pattern in sgn_files:
                    self.available_models['sgn']['ar'] = os.path.join(sgn_base, pattern)
                    break
            
            # Check for RI models
            ri_patterns = [f"{self.setting}_ri.pth", f"cview_ri.pth", f"cv_ri.pth"]
            for pattern in ri_patterns:
                if pattern in sgn_files:
                    self.available_models['sgn']['ri'] = os.path.join(sgn_base, pattern)
                    break
            
            # Check for GC models
            gc_patterns = [f"{self.setting}_gc.pth", f"cview_gc.pth", f"cv_gc.pth"]
            for pattern in gc_patterns:
                if pattern in sgn_files:
                    self.available_models['sgn']['gc'] = os.path.join(sgn_base, pattern)
                    break
        
        # Mixformer models - check both pretrained and old_pretrained directories
        mixformer_bases = [
            f"eval/mixformer/pretrained/{self.dataset}",
            f"eval/mixformer/old_pretrained/{self.dataset}"
        ]

        self.available_models['mixformer'] = {}

        for mixformer_base in mixformer_bases:
            if os.path.exists(mixformer_base):
                mixformer_files = []
                for root, dirs, files in os.walk(mixformer_base):
                    mixformer_files.extend([os.path.join(root, f) for f in files if f.endswith('.pth')])

                # Look for AR/RI models
                for file_path in mixformer_files:
                    filename = os.path.basename(file_path)

                    # Check for AR models
                    if filename.lower() in ['ar.pth', f'ar_{self.setting}.pth'] or \
                       ('ar' in filename.lower() and self.setting in filename.lower()):
                        if 'ar' not in self.available_models['mixformer']:
                            self.available_models['mixformer']['ar'] = file_path

                    # Check for RI models
                    elif filename.lower() in ['ri.pth', f'ri_{self.setting}.pth'] or \
                         ('ri' in filename.lower() and self.setting in filename.lower()):
                        if 'ri' not in self.available_models['mixformer']:
                            self.available_models['mixformer']['ri'] = file_path
        
        # Identify missing models
        required_models = {
            'sgn': ['ar', 'ri'],
            'mixformer': ['ar', 'ri']
        }
        
        for model_type, tasks in required_models.items():
            if model_type not in self.available_models:
                self.available_models[model_type] = {}
            
            for task in tasks:
                if task not in self.available_models[model_type]:
                    if model_type not in self.missing_models:
                        self.missing_models[model_type] = []
                    self.missing_models[model_type].append(task)
    
    def print_model_status(self):
        """Print status of available and missing models."""
        print("\n📋 Model Weight Status:")
        print("=" * 50)
        
        for model_type in ['sgn', 'mixformer']:
            print(f"\n{model_type.upper()} Models:")
            
            for task in ['ar', 'ri', 'gc']:
                if model_type in self.available_models and task in self.available_models[model_type]:
                    path = self.available_models[model_type][task]
                    print(f"  ✅ {task.upper()}: {path}")
                else:
                    print(f"  ❌ {task.upper()}: Missing")
        
        if self.missing_models:
            print(f"\n⚠️  Missing Models:")
            for model_type, tasks in self.missing_models.items():
                print(f"  {model_type.upper()}: {', '.join(tasks)}")
    
    def get_alternative_models(self):
        """Get alternative models when primary ones are missing."""
        alternatives = {}
        
        # For missing Mixformer models, suggest using SGN
        if 'mixformer' in self.missing_models:
            for task in self.missing_models['mixformer']:
                if 'sgn' in self.available_models and task in self.available_models['sgn']:
                    if 'mixformer' not in alternatives:
                        alternatives['mixformer'] = {}
                    alternatives['mixformer'][task] = {
                        'type': 'sgn',
                        'path': self.available_models['sgn'][task],
                        'note': f"Using SGN {task.upper()} model as alternative to Mixformer"
                    }
        
        # For missing SGN models, check if we can use different settings
        if 'sgn' in self.missing_models:
            sgn_base = f"eval/sgn/pretrained/{self.dataset}"
            if os.path.exists(sgn_base):
                all_files = os.listdir(sgn_base)
                
                for task in self.missing_models['sgn']:
                    # Look for any AR/RI model regardless of setting
                    for file in all_files:
                        if task in file.lower() and file.endswith('.pth'):
                            if 'sgn' not in alternatives:
                                alternatives['sgn'] = {}
                            alternatives['sgn'][task] = {
                                'type': 'sgn',
                                'path': os.path.join(sgn_base, file),
                                'note': f"Using alternative SGN {task.upper()} model from different setting"
                            }
                            break
        
        return alternatives
    
    def suggest_training_commands(self):
        """Suggest commands to train missing models."""
        if not self.missing_models:
            return []
        
        commands = []
        
        for model_type, tasks in self.missing_models.items():
            for task in tasks:
                if model_type == 'sgn':
                    # SGN training command
                    cmd = f"""
# Train SGN {task.upper()} model for {self.dataset} {self.setting}
python main.py \\
    --config config/{self.dataset}-cross-{'subject' if self.setting == 'cs' else 'view'}/sgn_{task}.yaml \\
    --work-dir work_dir/{self.dataset}/{self.setting}/sgn_{task} \\
    --device 0
"""
                elif model_type == 'mixformer':
                    # Mixformer training command
                    cmd = f"""
# Train Mixformer {task.upper()} model for {self.dataset} {self.setting}
python main.py \\
    --config config/{self.dataset}-cross-{'subject' if self.setting == 'cs' else 'view'}/mixformer_{task}.yaml \\
    --work-dir work_dir/{self.dataset}/{self.setting}/mixformer_{task} \\
    --device 0
"""
                commands.append(cmd.strip())
        
        return commands
    
    def create_training_script(self, output_path="scripts/train_missing_models.sh"):
        """Create a script to train missing models."""
        commands = self.suggest_training_commands()
        
        if not commands:
            print("✅ No missing models to train!")
            return None
        
        script_content = f"""#!/bin/bash
# Auto-generated script to train missing AR/RI models
# Generated for {self.dataset} {self.setting}

set -e

echo "🚀 Training missing AR/RI models for {self.dataset} {self.setting}"
echo "=================================================="

# Load required modules (adjust for your HPC environment)
module load pytorch/2.3.0-cuda12.1

"""
        
        for i, cmd in enumerate(commands, 1):
            script_content += f"""
echo "📈 Training model {i}/{len(commands)}..."
{cmd}

"""
        
        script_content += """
echo "✅ All missing models trained successfully!"
"""
        
        # Write script
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"📝 Training script created: {output_path}")
        return output_path
    
    def interactive_model_selection(self):
        """Interactive selection of models to use."""
        print("\n🎯 Model Selection for Evaluation")
        print("=" * 40)
        
        selected_models = {}
        alternatives = self.get_alternative_models()
        
        for model_type in ['sgn', 'mixformer']:
            print(f"\n{model_type.upper()} Models:")
            selected_models[model_type] = {}
            
            for task in ['ar', 'ri']:
                print(f"\n  {task.upper()} Task:")
                
                options = []
                
                # Add primary model if available
                if model_type in self.available_models and task in self.available_models[model_type]:
                    options.append({
                        'name': f"Primary {model_type.upper()} {task.upper()}",
                        'path': self.available_models[model_type][task],
                        'type': model_type
                    })
                
                # Add alternatives
                if model_type in alternatives and task in alternatives[model_type]:
                    alt = alternatives[model_type][task]
                    options.append({
                        'name': f"Alternative: {alt['note']}",
                        'path': alt['path'],
                        'type': alt['type']
                    })
                
                # Add skip option
                options.append({
                    'name': "Skip this model",
                    'path': None,
                    'type': None
                })
                
                # Display options
                for i, option in enumerate(options, 1):
                    print(f"    {i}. {option['name']}")
                
                # Get user choice
                while True:
                    try:
                        choice = int(input(f"    Select option (1-{len(options)}): ")) - 1
                        if 0 <= choice < len(options):
                            selected_models[model_type][task] = options[choice]
                            break
                        else:
                            print("    Invalid choice. Please try again.")
                    except ValueError:
                        print("    Please enter a number.")
        
        return selected_models
    
    def get_model_config(self, model_type, task):
        """Get model configuration for evaluation."""
        if model_type in self.available_models and task in self.available_models[model_type]:
            return {
                'available': True,
                'path': self.available_models[model_type][task],
                'type': model_type
            }
        
        # Check alternatives
        alternatives = self.get_alternative_models()
        if model_type in alternatives and task in alternatives[model_type]:
            alt = alternatives[model_type][task]
            return {
                'available': True,
                'path': alt['path'],
                'type': alt['type'],
                'note': alt['note']
            }
        
        return {
            'available': False,
            'path': None,
            'type': None
        }


def main():
    """Test the model weight manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Weight Manager')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'])
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'])
    parser.add_argument('--interactive', action='store_true', help='Interactive model selection')
    parser.add_argument('--create-training-script', action='store_true', help='Create training script for missing models')
    
    args = parser.parse_args()
    
    manager = ModelWeightManager(args.dataset, args.setting)
    manager.print_model_status()
    
    if args.interactive:
        selected = manager.interactive_model_selection()
        print("\n📋 Selected Models:")
        for model_type, tasks in selected.items():
            for task, config in tasks.items():
                if config['path']:
                    print(f"  {model_type.upper()} {task.upper()}: {config['name']}")
                else:
                    print(f"  {model_type.upper()} {task.upper()}: Skipped")
    
    if args.create_training_script:
        manager.create_training_script()
    
    alternatives = manager.get_alternative_models()
    if alternatives:
        print("\n🔄 Available Alternatives:")
        for model_type, tasks in alternatives.items():
            for task, alt in tasks.items():
                print(f"  {model_type.upper()} {task.upper()}: {alt['note']}")


if __name__ == "__main__":
    main()
