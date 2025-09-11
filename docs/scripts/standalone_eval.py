#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Evaluation Script

This script provides a standalone way to run evaluation and visualization on trained models.
It supports both command-line arguments and interactive mode for user convenience.

Features:
- Evaluate models with SGN/MixFormer on AR/RI/GC tasks
- Run visualizations (skeleton animations, comparisons, etc.)
- Interactive mode with path validation and recursive search
- Support for all model types (transformer, PMR, DMR, raw)
- Comprehensive result reporting

Usage:
    # Command line mode
    python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv

    # Interactive mode
    python scripts/standalone_eval.py --interactive

Author: Generated for Transformer-Retargeting project
"""

import os
import sys
import argparse
import glob
import fnmatch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/standalone_eval.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class StandaloneEvaluator:
    """Standalone evaluator for trained models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.supported_datasets = ['ntu', 'ntu120', 'etri']
        self.supported_settings = ['cs', 'cv']
        self.supported_model_types = ['transformer', 'pmr', 'dmr', 'raw', 'all']
        self.supported_eval_models = ['sgn', 'mixformer', 'both']
        self.supported_visualizations = [
            'skeleton_animations',
            'comparison_visualization',
            'motion_visualization',
            'attention_visualization',
            'mlm_pretraining',
            'all'
        ]

        # Default model paths for different types
        self.default_model_paths = {
            'transformer': [
                'model_all.pth',  # Fully trained transformer model (primary)
                'model.pth',
                'output/ntu_transformer_cv/model_best.pth.tar',
                'trained_models/transformer_ntu_cv.pth'
            ],
            'pmr': [
                'trained_models/pmr_ntu_cv_best.pth',
                'trained_models/pmr_ntu_cv_final.pth',
                'output/ntu_pmr_cv/model_best.pth.tar'
            ],
            'dmr': [
                'trained_models/dmr_ntu_cv_best.pth',
                'trained_models/dmr_ntu_cv_final.pth',
                'output/ntu_dmr_cv/model_best.pth.tar'
            ]
        }
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def find_default_model(self, model_type: str, dataset: str = 'ntu', setting: str = 'cv') -> Optional[str]:
        """Find the default model path for a given type."""
        if model_type not in self.default_model_paths:
            return None

        # Try dataset/setting specific paths first
        specific_paths = []
        for base_path in self.default_model_paths[model_type]:
            # Replace generic patterns with specific dataset/setting
            specific_path = base_path.replace('ntu', dataset).replace('cv', setting)
            specific_paths.append(specific_path)

        # Check all paths in order of preference
        all_paths = specific_paths + self.default_model_paths[model_type]

        for path in all_paths:
            if self.validate_model_path(path):
                return path

        return None

    def get_all_available_models(self, dataset: str = 'ntu', setting: str = 'cv') -> Dict[str, List[str]]:
        """Get all available models organized by type."""
        available_models = {}

        for model_type in ['transformer', 'pmr', 'dmr']:
            models = []

            # Check default paths
            default_model = self.find_default_model(model_type, dataset, setting)
            if default_model:
                models.append(default_model)

            # Search for additional models
            search_patterns = [
                f"trained_models/{model_type}*{dataset}*{setting}*.pth",
                f"output/{dataset}*{model_type}*{setting}*/model_best.pth*",
                f"trained_models/*{model_type}*.pth"
            ]

            for pattern in search_patterns:
                found_files = glob.glob(pattern)
                for file_path in found_files:
                    if self.validate_model_path(file_path) and file_path not in models:
                        models.append(file_path)

            if models:
                available_models[model_type] = models

        # Always include raw data option
        available_models['raw'] = ['raw_data']

        return available_models

    def find_default_model(self, model_type: str, dataset: str = 'ntu', setting: str = 'cv') -> Optional[str]:
        """Find the default model path for a given type."""
        if model_type not in self.default_model_paths:
            return None

        # Try dataset/setting specific paths first
        specific_paths = []
        for base_path in self.default_model_paths[model_type]:
            # Replace generic patterns with specific dataset/setting
            specific_path = base_path.replace('ntu', dataset).replace('cv', setting)
            specific_paths.append(specific_path)

        # Check all paths in order of preference
        all_paths = specific_paths + self.default_model_paths[model_type]

        for path in all_paths:
            if self.validate_model_path(path):
                return path

        return None

    def get_all_available_models(self, dataset: str = 'ntu', setting: str = 'cv') -> Dict[str, List[str]]:
        """Get all available models organized by type."""
        available_models = {}

        for model_type in ['transformer', 'pmr', 'dmr']:
            models = []

            # Check default paths
            default_model = self.find_default_model(model_type, dataset, setting)
            if default_model:
                models.append(default_model)

            # Search for additional models
            search_patterns = [
                f"trained_models/{model_type}*{dataset}*{setting}*.pth",
                f"output/{dataset}*{model_type}*{setting}*/model_best.pth*",
                f"trained_models/*{model_type}*.pth"
            ]

            for pattern in search_patterns:
                found_files = glob.glob(pattern)
                for file_path in found_files:
                    if self.validate_model_path(file_path) and file_path not in models:
                        models.append(file_path)

            if models:
                available_models[model_type] = models

        # Always include raw data option
        available_models['raw'] = ['raw_data']

        return available_models
        
    def validate_model_path(self, model_path: str) -> bool:
        """Validate if model path exists and is a valid model file."""
        if not os.path.exists(model_path):
            return False
            
        # Check if it's a valid model file
        valid_extensions = ['.pth', '.pt', '.tar', '.pkl']
        return any(model_path.endswith(ext) for ext in valid_extensions)
    
    def search_model_files(self, search_dir: str = '.', pattern: str = '*.pth') -> List[str]:
        """Recursively search for model files in directory."""
        model_files = []

        # Search patterns for different model types
        patterns = [
            '**/*.pth',
            '**/*.pt',
            '**/*.tar',
            '**/model_best.pth.tar',
            '**/model.pth',
            '**/checkpoint*.pth'
        ]

        # Exclude patterns for data files and other non-model files
        exclude_patterns = [
            '**/data/**',
            '**/wandb/**',
            '**/logs/**',
            '**/*_data.pt',
            '**/*_paired*.pt',
            '**/pretraining_data.pt',
            '**/test_data.pt',
            '**/train_data.pt'
        ]

        for pattern in patterns:
            search_path = os.path.join(search_dir, pattern)
            found_files = glob.glob(search_path, recursive=True)

            # Filter out excluded files
            filtered_files = []
            for file_path in found_files:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if glob.fnmatch.fnmatch(file_path, exclude_pattern):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_files.append(file_path)

            model_files.extend(filtered_files)

        # Remove duplicates and sort
        model_files = sorted(list(set(model_files)))

        return model_files
    
    def interactive_model_selection(self) -> Tuple[str, str]:
        """Interactive model path selection with validation."""
        print("\n🔍 MODEL SELECTION")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Enter model path directly")
            print("2. Search for models in current directory")
            print("3. Search for models in specific directory")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                model_path = input("Enter model path: ").strip()
                if self.validate_model_path(model_path):
                    model_type = self._infer_model_type(model_path, interactive=True)
                    return model_path, model_type
                else:
                    print(f"❌ Invalid model path: {model_path}")
                    continue
                    
            elif choice == '2':
                return self._search_and_select_model('.')
                
            elif choice == '3':
                search_dir = input("Enter directory to search: ").strip()
                if os.path.exists(search_dir):
                    return self._search_and_select_model(search_dir)
                else:
                    print(f"❌ Directory not found: {search_dir}")
                    continue
            else:
                print("❌ Invalid choice. Please select 1-3.")
    
    def _search_and_select_model(self, search_dir: str) -> Tuple[str, str]:
        """Search and select model from directory."""
        print(f"\n🔍 Searching for models in: {search_dir}")
        model_files = self.search_model_files(search_dir)
        
        if not model_files:
            print("❌ No model files found.")
            return self.interactive_model_selection()
        
        print(f"\n📁 Found {len(model_files)} model files:")
        for i, model_file in enumerate(model_files, 1):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"{i:2d}. {model_file} ({file_size:.1f} MB)")
        
        while True:
            try:
                selection = input(f"\nSelect model (1-{len(model_files)}): ").strip()
                idx = int(selection) - 1
                if 0 <= idx < len(model_files):
                    selected_path = model_files[idx]
                    model_type = self._infer_model_type(selected_path, interactive=True)
                    return selected_path, model_type
                else:
                    print(f"❌ Invalid selection. Please choose 1-{len(model_files)}")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def _infer_model_type(self, model_path: str, interactive: bool = True) -> str:
        """Infer model type from path."""
        path_lower = model_path.lower()

        if 'dmr' in path_lower:
            return 'dmr'
        elif 'pmr' in path_lower:
            return 'pmr'
        elif 'transformer' in path_lower or 'model.pth' in path_lower or 'model_best.pth' in path_lower:
            return 'transformer'
        elif 'output/' in path_lower and 'model_best.pth' in path_lower:
            return 'transformer'
        elif 'trained_models/' in path_lower:
            return 'transformer'
        else:
            if not interactive:
                return 'unknown'

            # Ask user to specify
            print(f"\n❓ Could not infer model type from path: {model_path}")
            while True:
                model_type = input(f"Please specify model type ({'/'.join(self.supported_model_types)}): ").strip().lower()
                if model_type in self.supported_model_types:
                    return model_type
                print(f"❌ Invalid model type. Choose from: {', '.join(self.supported_model_types)}")
    
    def interactive_config_selection(self) -> Dict[str, Any]:
        """Interactive configuration selection."""
        print("\n⚙️  CONFIGURATION")
        print("=" * 50)
        
        config = {}
        
        # Dataset selection
        print(f"\n📊 Available datasets: {', '.join(self.supported_datasets)}")
        while True:
            dataset = input("Select dataset [ntu]: ").strip().lower() or 'ntu'
            if dataset in self.supported_datasets:
                config['dataset'] = dataset
                break
            print(f"❌ Invalid dataset. Choose from: {', '.join(self.supported_datasets)}")
        
        # Setting selection
        print(f"\n🎯 Available settings: {', '.join(self.supported_settings)}")
        while True:
            setting = input("Select setting [cv]: ").strip().lower() or 'cv'
            if setting in self.supported_settings:
                config['setting'] = setting
                break
            print(f"❌ Invalid setting. Choose from: {', '.join(self.supported_settings)}")
        
        # Evaluation model selection
        print(f"\n🧠 Available evaluation models: {', '.join(self.supported_eval_models)}")
        while True:
            eval_model = input("Select evaluation model [sgn]: ").strip().lower() or 'sgn'
            if eval_model in self.supported_eval_models:
                config['eval_model'] = eval_model
                break
            print(f"❌ Invalid evaluation model. Choose from: {', '.join(self.supported_eval_models)}")
        
        # Output directory
        default_output = f"results/standalone_eval_{config['dataset']}_{config['setting']}"
        output_dir = input(f"Output directory [{default_output}]: ").strip() or default_output
        config['output_dir'] = output_dir
        
        # Run evaluation
        run_eval = input("Run evaluation? [y]: ").strip().lower() or 'y'
        config['run_evaluation'] = run_eval.startswith('y')
        
        # Run visualization
        run_viz = input("Run visualization? [n]: ").strip().lower() or 'n'
        config['run_visualization'] = run_viz.startswith('y')
        
        if config['run_visualization']:
            print(f"\n🎨 Available visualizations: {', '.join(self.supported_visualizations)}")
            viz_types = input("Select visualizations (comma-separated) [skeleton_animations]: ").strip() or 'skeleton_animations'
            config['visualizations'] = [v.strip() for v in viz_types.split(',')]
        
        return config

    def run_evaluation(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation using eval_model.py."""
        print("\n🧪 RUNNING EVALUATION")
        print("=" * 50)

        # Handle "all" model type
        if model_type == 'all':
            return self.run_all_models_evaluation(config)

        # Handle single model evaluation
        return self.run_single_model_evaluation(model_path, model_type, config)

    def run_single_model_evaluation(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation for a single model."""
        # Create temporary args for eval_model.py
        original_argv = sys.argv.copy()

        try:
            # Use SLURM if requested
            if config.get('use_slurm', False):
                return self.run_slurm_evaluation(model_path, model_type, config)

            # Import evaluation functions
            from eval_model import main as eval_main

            # Build arguments for eval_model.py
            eval_args = [
                'eval_model.py',
                '--dataset', config['dataset'],
                '--setting', config['setting'],
                '--model_type', model_type,
                '--eval_model', config['eval_model']
            ]

            # Add model path based on type
            if model_type == 'transformer':
                eval_args.extend(['--transformer_model_path', model_path])
            elif model_type in ['pmr', 'dmr']:
                eval_args.extend(['--transformer_model_path', model_path])

            # Set sys.argv for eval_model.py
            sys.argv = eval_args

            print(f"📝 Running evaluation with args: {' '.join(eval_args[1:])}")

            # Run evaluation and catch SystemExit
            try:
                eval_main()
            except SystemExit as e:
                if e.code != 0:
                    raise Exception(f"Evaluation exited with code {e.code}")

            # Restore original argv
            sys.argv = original_argv

            print("✅ Evaluation completed successfully!")

            # Try to find and return results
            results_file = f"results/{config['dataset']}_{config['setting']}_evaluation_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                return {"status": "completed", "message": "Evaluation finished but results file not found"}

        except Exception as e:
            # Restore original argv in case of error
            sys.argv = original_argv
            logger.error(f"Evaluation failed: {str(e)}")
            print(f"❌ Evaluation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def run_all_models_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation on all available models."""
        print("🔄 Running evaluation on all available models...")

        available_models = self.get_all_available_models(config['dataset'], config['setting'])
        all_results = {}

        for model_type, model_paths in available_models.items():
            print(f"\n📊 Evaluating {model_type.upper()} models...")

            for model_path in model_paths:
                print(f"  🔍 Evaluating: {model_path}")

                # Create model-specific config
                model_config = config.copy()
                model_config['output_dir'] = f"{config['output_dir']}/{model_type}"

                # Run evaluation
                if model_type == 'raw':
                    result = self.run_single_model_evaluation('raw', 'raw', model_config)
                else:
                    result = self.run_single_model_evaluation(model_path, model_type, model_config)

                # Store results with better naming
                model_name = self._get_model_display_name(model_path, model_type)
                all_results[f"{model_type}_{model_name}"] = result

        return {"status": "completed", "all_results": all_results}

    def _get_model_display_name(self, model_path: str, model_type: str) -> str:
        """Get a clean display name for the model."""
        if model_path == 'raw' or model_path == 'raw_data':
            return 'raw_data'

        # Extract meaningful name from path
        filename = os.path.basename(model_path)

        # Remove common prefixes/suffixes
        name = filename.replace('.pth', '').replace('.tar', '').replace('.pt', '')

        # Handle specific cases
        if 'model_all' in name:
            return 'fully_trained'
        elif 'model_best' in name:
            return 'best_checkpoint'
        elif 'model.pth' in filename:
            return 'main_model'
        elif 'best' in name:
            return 'best'
        elif 'final' in name:
            return 'final'
        else:
            return name

    def run_slurm_evaluation(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation using SLURM."""
        print(f"🖥️  Submitting SLURM job for {model_type} evaluation...")

        try:
            # Create SLURM script
            slurm_script = self._create_slurm_script(model_path, model_type, config)
            script_path = f"slurm_out/standalone_eval_{model_type}_{config['dataset']}_{config['setting']}.sh"

            os.makedirs('slurm_out', exist_ok=True)
            with open(script_path, 'w') as f:
                f.write(slurm_script)

            # Submit job
            import subprocess
            result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)

            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"✅ SLURM job submitted: {job_id}")
                return {
                    "status": "submitted",
                    "job_id": job_id,
                    "script_path": script_path,
                    "message": f"Job {job_id} submitted to SLURM"
                }
            else:
                raise Exception(f"SLURM submission failed: {result.stderr}")

        except Exception as e:
            logger.error(f"SLURM evaluation failed: {str(e)}")
            print(f"❌ SLURM evaluation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _create_slurm_script(self, model_path: str, model_type: str, config: Dict[str, Any]) -> str:
        """Create SLURM script for evaluation."""
        job_name = f"standalone_eval_{model_type}_{config['dataset']}_{config['setting']}"

        # Get user email from config or use default
        email = config.get('email', 'carrt313@gmail.com')

        # Use same SLURM configuration as pipeline.py for evaluation jobs
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={email}

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

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p results

# Check if safe_model_loading.py exists
if [ ! -f "safe_model_loading.py" ]; then
    echo "Error: safe_model_loading.py not found. Please ensure it exists in the project root."
    exit 1
fi

# Run evaluation
python eval_model.py \\
    --dataset {config['dataset']} \\
    --setting {config['setting']} \\
    --model_type {model_type} \\
    --eval_model {config['eval_model']}"""

        # Add model path argument
        if model_type == 'transformer':
            script += f" \\\n    --transformer_model_path {model_path}"
        elif model_type in ['pmr', 'dmr']:
            script += f" \\\n    --transformer_model_path {model_path}"

        # Add validation set option if requested
        if config.get('use_validation', False):
            script += " \\\n    --use-validation"

        script += """

echo ""
echo "Evaluation completed at $(date)"
echo "Job finished successfully"
"""

        return script

    def run_visualization(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run visualization using evaluation_suite."""
        print("\n🎨 RUNNING VISUALIZATION")
        print("=" * 50)

        try:
            from evaluation_suite.run_visualization import main as viz_main

            # Create temporary args for visualization
            original_argv = sys.argv.copy()

            # Build arguments for run_visualization.py
            viz_args = [
                'run_visualization.py',
                '--visualizations', ','.join(config['visualizations']),
                '--dataset', config['dataset'],
                '--setting', config['setting'],
                '--output-dir', config['output_dir']
            ]

            # Add model-specific arguments if needed
            if 'mlm_pretraining' in config['visualizations']:
                viz_args.extend(['--temporal-ratio', '0.3', '--spatial-ratio', '0.3'])

            # Set sys.argv for visualization
            sys.argv = viz_args

            print(f"🎬 Running visualization with args: {' '.join(viz_args[1:])}")

            # Run visualization and catch SystemExit
            try:
                viz_main()
            except SystemExit as e:
                # Check if it's a successful exit (code 0)
                if e.code == 0:
                    print("✅ Visualization completed successfully!")
                else:
                    raise Exception(f"Visualization exited with code {e.code}")

            # Restore original argv
            sys.argv = original_argv

            return {"status": "completed", "output_dir": config['output_dir']}

        except Exception as e:
            # Restore original argv in case of error
            sys.argv = original_argv
            logger.error(f"Visualization failed: {str(e)}")
            print(f"❌ Visualization failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def generate_report(self, model_path: str, model_type: str, config: Dict[str, Any],
                       eval_results: Optional[Dict] = None, viz_results: Optional[Dict] = None) -> str:
        """Generate a comprehensive evaluation report."""
        print("\n📊 GENERATING REPORT")
        print("=" * 50)

        report_lines = []
        report_lines.append("# Standalone Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Configuration summary
        report_lines.append("## Configuration")
        report_lines.append(f"- **Dataset**: {config['dataset']}")
        report_lines.append(f"- **Setting**: {config['setting']}")
        report_lines.append(f"- **Evaluation Model**: {config['eval_model']}")
        report_lines.append(f"- **Use Validation Set**: {config.get('use_validation', False)}")
        report_lines.append(f"- **SLURM Mode**: {config.get('use_slurm', False)}")
        report_lines.append("")

        # Handle multiple models vs single model
        if model_type == 'all' and eval_results and 'all_results' in eval_results:
            self._add_multi_model_results(report_lines, eval_results['all_results'], config)
        else:
            self._add_single_model_results(report_lines, model_path, model_type, eval_results, config)

        # Visualization results
        if viz_results:
            self._add_visualization_results(report_lines, viz_results, config)

        # Default model paths summary
        self._add_default_paths_summary(report_lines, config)

        # Save report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(config['output_dir'], 'evaluation_report.md')
        os.makedirs(config['output_dir'], exist_ok=True)

        with open(report_file, 'w') as f:
            f.write(report_content)

        print(f"📄 Report saved to: {report_file}")
        return report_file

    def _add_single_model_results(self, report_lines: List[str], model_path: str, model_type: str,
                                 eval_results: Optional[Dict], config: Dict[str, Any]):
        """Add single model results to report."""
        report_lines.append("## Model Information")
        report_lines.append(f"- **Model Path**: {model_path}")
        report_lines.append(f"- **Model Type**: {model_type}")
        report_lines.append(f"- **Display Name**: {self._get_model_display_name(model_path, model_type)}")

        # File information
        if model_path != 'all' and model_path != 'raw' and os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            report_lines.append(f"- **File Size**: {file_size:.1f} MB")
        report_lines.append("")

        # Evaluation results
        if eval_results:
            report_lines.append("## Evaluation Results")
            if eval_results.get('status') == 'completed':
                report_lines.append("✅ **Status**: Completed successfully")
                self._add_metrics_to_report(report_lines, eval_results)
            elif eval_results.get('status') == 'submitted':
                report_lines.append("🚀 **Status**: Submitted to SLURM")
                report_lines.append(f"- **Job ID**: {eval_results.get('job_id', 'Unknown')}")
                report_lines.append(f"- **Script Path**: {eval_results.get('script_path', 'Unknown')}")
            else:
                report_lines.append(f"❌ **Status**: {eval_results.get('status', 'Unknown')}")
                if 'error' in eval_results:
                    report_lines.append(f"- **Error**: {eval_results['error']}")
            report_lines.append("")

    def _add_multi_model_results(self, report_lines: List[str], all_results: Dict[str, Any], config: Dict[str, Any]):
        """Add multi-model results to report."""
        report_lines.append("## Multi-Model Evaluation Results")
        report_lines.append(f"Evaluated {len(all_results)} models:")
        report_lines.append("")

        # Summary table
        report_lines.append("### Summary")
        report_lines.append("| Model Type | Model Name | Status | Key Metrics |")
        report_lines.append("|------------|------------|--------|-------------|")

        for model_key, result in all_results.items():
            model_type, model_name = model_key.split('_', 1)
            status = result.get('status', 'Unknown')

            # Extract key metrics
            key_metrics = "N/A"
            if result.get('status') == 'completed' and 'metrics' in result:
                metrics = result['metrics']
                if isinstance(metrics, dict):
                    # Get first few important metrics
                    metric_strs = []
                    for key, value in list(metrics.items())[:3]:
                        if isinstance(value, (int, float)):
                            metric_strs.append(f"{key}: {value:.3f}")
                    key_metrics = ", ".join(metric_strs) if metric_strs else "N/A"

            report_lines.append(f"| {model_type} | {model_name} | {status} | {key_metrics} |")

        report_lines.append("")

        # Detailed results for each model
        report_lines.append("### Detailed Results")
        for model_key, result in all_results.items():
            model_type, model_name = model_key.split('_', 1)
            report_lines.append(f"#### {model_type.upper()} - {model_name}")

            if result.get('status') == 'completed':
                report_lines.append("✅ **Status**: Completed successfully")
                self._add_metrics_to_report(report_lines, result)
            elif result.get('status') == 'submitted':
                report_lines.append("🚀 **Status**: Submitted to SLURM")
                report_lines.append(f"- **Job ID**: {result.get('job_id', 'Unknown')}")
            else:
                report_lines.append(f"❌ **Status**: {result.get('status', 'Unknown')}")
                if 'error' in result:
                    report_lines.append(f"- **Error**: {result['error']}")

            report_lines.append("")

    def _add_metrics_to_report(self, report_lines: List[str], result: Dict[str, Any]):
        """Add metrics to report."""
        if 'metrics' in result:
            metrics = result['metrics']
            report_lines.append("")
            report_lines.append("### Performance Metrics")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- **{metric_name}**: {value:.4f}")
                    else:
                        report_lines.append(f"- **{metric_name}**: {value}")
            else:
                report_lines.append(f"- **Metrics**: {metrics}")

    def _add_visualization_results(self, report_lines: List[str], viz_results: Dict[str, Any], config: Dict[str, Any]):
        """Add visualization results to report."""
        report_lines.append("## Visualization Results")
        if viz_results.get('status') == 'completed':
            report_lines.append("✅ **Status**: Completed successfully")
            report_lines.append(f"- **Output Directory**: {viz_results.get('output_dir', 'Unknown')}")
            report_lines.append(f"- **Visualizations**: {', '.join(config.get('visualizations', []))}")
        else:
            report_lines.append(f"❌ **Status**: {viz_results.get('status', 'Unknown')}")
            if 'error' in viz_results:
                report_lines.append(f"- **Error**: {viz_results['error']}")
        report_lines.append("")

    def _add_default_paths_summary(self, report_lines: List[str], config: Dict[str, Any]):
        """Add default model paths summary to report."""
        report_lines.append("## Default Model Paths")
        report_lines.append("The following default paths are checked for each model type:")
        report_lines.append("")

        for model_type, paths in self.default_model_paths.items():
            report_lines.append(f"### {model_type.upper()}")
            for path in paths:
                # Replace generic patterns with current dataset/setting
                specific_path = path.replace('ntu', config['dataset']).replace('cv', config['setting'])
                exists = "✅" if os.path.exists(specific_path) else "❌"
                report_lines.append(f"- {exists} `{specific_path}`")
            report_lines.append("")

        # Available models summary
        available_models = self.get_all_available_models(config['dataset'], config['setting'])
        report_lines.append("## Available Models Summary")
        for model_type, model_paths in available_models.items():
            report_lines.append(f"### {model_type.upper()}")
            if model_paths:
                for path in model_paths:
                    if path != 'raw_data':
                        file_size = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
                        report_lines.append(f"- `{path}` ({file_size:.1f} MB)")
                    else:
                        report_lines.append(f"- `{path}`")
            else:
                report_lines.append("- No models found")
            report_lines.append("")

        # Save report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(config['output_dir'], 'evaluation_report.md')
        os.makedirs(config['output_dir'], exist_ok=True)

        with open(report_file, 'w') as f:
            f.write(report_content)

        print(f"📄 Report saved to: {report_file}")
        return report_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone evaluation and visualization for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Command line mode - basic evaluation
  python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv

  # Command line mode - with visualization
  python scripts/standalone_eval.py --model-path model.pth --model-type transformer --dataset ntu --setting cv --visualize --viz-types skeleton_animations

  # Interactive mode
  python scripts/standalone_eval.py --interactive

  # Search for models and select interactively
  python scripts/standalone_eval.py --interactive --search-dir output/
        """
    )

    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    # Model configuration
    parser.add_argument('--model-path', type=str,
                       help='Path to model file (use "auto" to find default model)')
    parser.add_argument('--model-type', choices=['transformer', 'pmr', 'dmr', 'raw', 'all'],
                       help='Type of model (use "all" to evaluate all available models)')

    # Dataset configuration
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'etri'], default='ntu',
                       help='Dataset to evaluate on (default: ntu)')
    parser.add_argument('--setting', choices=['cs', 'cv'], default='cv',
                       help='Cross-subject (cs) or cross-view (cv) setting (default: cv)')

    # Evaluation configuration
    parser.add_argument('--eval-model', choices=['sgn', 'mixformer', 'both'], default='mixformer',
                       help='Evaluation model to use (default: mixformer)')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--use-validation', action='store_true',
                       help='Use validation set from comprehensive data based on camera')
    parser.add_argument('--slurm', action='store_true',
                       help='Submit evaluation job to SLURM')
    parser.add_argument('--email', type=str,
                       help='Email for SLURM notifications')

    # Visualization configuration
    parser.add_argument('--visualize', action='store_true',
                       help='Run visualization')
    parser.add_argument('--viz-types', nargs='+',
                       choices=['skeleton_animations', 'comparison_visualization',
                               'motion_visualization', 'attention_visualization',
                               'mlm_pretraining', 'all'],
                       default=['skeleton_animations'],
                       help='Types of visualizations to create')

    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')

    # Search configuration
    parser.add_argument('--search-dir', type=str, default='.',
                       help='Directory to search for models (for interactive mode)')

    # Utility options
    parser.add_argument('--list-models', action='store_true',
                       help='List available model files and exit')
    parser.add_argument('--validate-path', type=str,
                       help='Validate a model path and exit')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    evaluator = StandaloneEvaluator()

    print("🚀 STANDALONE EVALUATION TOOL")
    print("=" * 50)

    # Handle utility options
    if args.list_models:
        print(f"\n🔍 Searching for models in: {args.search_dir}")
        model_files = evaluator.search_model_files(args.search_dir)
        if model_files:
            print(f"\n📁 Found {len(model_files)} model files:")
            for i, model_file in enumerate(model_files, 1):
                file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                model_type = evaluator._infer_model_type(model_file, interactive=False)
                print(f"{i:2d}. {model_file} ({file_size:.1f} MB) - {model_type}")
        else:
            print("❌ No model files found.")
        return

    if args.validate_path:
        if evaluator.validate_model_path(args.validate_path):
            model_type = evaluator._infer_model_type(args.validate_path, interactive=False)
            print(f"✅ Valid model path: {args.validate_path} (type: {model_type})")
        else:
            print(f"❌ Invalid model path: {args.validate_path}")
        return

    # Interactive mode
    if args.interactive:
        print("\n🎯 INTERACTIVE MODE")
        print("Welcome to the interactive evaluation tool!")

        # Model selection
        model_path, model_type = evaluator.interactive_model_selection()

        # Configuration selection
        config = evaluator.interactive_config_selection()

    else:
        # Command line mode
        print("\n💻 COMMAND LINE MODE")

        # Handle model type and path
        if args.model_type == 'all':
            model_path = 'all'
            model_type = 'all'
        elif args.model_path == 'auto' or not args.model_path:
            # Find default model for the specified type
            if not args.model_type:
                print("❌ Error: --model-type is required when using auto model path")
                return

            model_path = evaluator.find_default_model(args.model_type, args.dataset, args.setting)
            if not model_path:
                print(f"❌ Error: No default model found for type: {args.model_type}")
                print("Available models:")
                available = evaluator.get_all_available_models(args.dataset, args.setting)
                for mtype, paths in available.items():
                    print(f"  {mtype}: {paths}")
                return
            model_type = args.model_type
            print(f"🔍 Using default model: {model_path}")
        else:
            # Validate provided model path
            if not evaluator.validate_model_path(args.model_path):
                print(f"❌ Error: Invalid model path: {args.model_path}")
                return

            model_path = args.model_path
            model_type = args.model_type or evaluator._infer_model_type(model_path, interactive=True)

        # Build configuration
        config = {
            'dataset': args.dataset,
            'setting': args.setting,
            'eval_model': args.eval_model,
            'output_dir': args.output_dir or f"results/standalone_eval_{args.dataset}_{args.setting}",
            'run_evaluation': not args.no_eval,
            'run_visualization': args.visualize,
            'visualizations': args.viz_types if args.visualize else [],
            'use_validation': args.use_validation,
            'use_slurm': args.slurm,
            'email': args.email
        }

    # Display configuration summary
    print(f"\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Model Path: {model_path}")
    print(f"Model Type: {model_type}")
    print(f"Dataset: {config['dataset']}")
    print(f"Setting: {config['setting']}")
    print(f"Evaluation Model: {config['eval_model']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Run Evaluation: {config['run_evaluation']}")
    print(f"Run Visualization: {config['run_visualization']}")
    if config['run_visualization']:
        print(f"Visualizations: {', '.join(config['visualizations'])}")

    # Confirm execution
    if args.interactive:
        confirm = input("\n🚀 Proceed with evaluation? [y]: ").strip().lower() or 'y'
        if not confirm.startswith('y'):
            print("❌ Evaluation cancelled.")
            return

    # Run evaluation and visualization
    eval_results = None
    viz_results = None

    try:
        if config['run_evaluation']:
            eval_results = evaluator.run_evaluation(model_path, model_type, config)

        if config['run_visualization']:
            viz_results = evaluator.run_visualization(model_path, model_type, config)

        # Generate report
        report_file = evaluator.generate_report(model_path, model_type, config, eval_results, viz_results)

        print(f"\n🎉 EVALUATION COMPLETED!")
        print(f"📄 Report: {report_file}")
        print(f"📁 Results: {config['output_dir']}")

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n💥 Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
