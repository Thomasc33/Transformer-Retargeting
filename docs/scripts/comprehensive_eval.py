#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Evaluation Script

This script provides a unified interface for comprehensive model evaluation including:
- AR (Action Recognition), RI (Re-identification), GC (Gender Classification) metrics
- Physical Plausibility assessment (all 5 required metrics)
- Per-actor and per-action breakdown of results
- Visualizations for both anonymized and raw data outputs
- Support for both SGN and Mixformer models
- Clean, well-organized code architecture

Usage:
    # Evaluate single model
    python scripts/comprehensive_eval.py --model-path model.pth --model-type transformer

    # Evaluate all available models
    python scripts/comprehensive_eval.py --model-type all

    # Interactive mode
    python scripts/comprehensive_eval.py --interactive

    # SLURM mode
    python scripts/comprehensive_eval.py --model-path model.pth --model-type transformer --slurm

Author: Generated for Transformer-Retargeting project
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/comprehensive_eval.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluator for trained models with full AR/RI/GC and physical metrics."""
    
    def __init__(self):
        """Initialize the comprehensive evaluator."""
        self.supported_datasets = ['ntu', 'ntu120', 'etri']
        self.supported_settings = ['cs', 'cv']
        self.supported_model_types = ['transformer', 'pmr', 'dmr', 'raw', 'all']
        self.supported_eval_models = ['sgn', 'mixformer', 'both']

        # Key model paths for evaluation
        self.key_models = {
            'transformer': [
                'model_all.pth',  # Fully trained transformer model
                'model.pth',      # Main model
            ],
            'pmr': [
                'trained_models/pmr_ntu_cv_best.pth',
                'trained_models/pmr_ntu_cv_final.pth'
            ],
            'dmr': [
                'trained_models/dmr_ntu_cv_best.pth',
                'trained_models/dmr_ntu_cv_final.pth'
            ]
        }

        # Available evaluation model weights
        self.eval_model_weights = {
            'mixformer': {
                'ntu': {
                    'cv': {
                        'ar': 'eval/mixformer/pretrained/ntu/cv_ar.pth',
                        'ri': 'eval/mixformer/pretrained/ntu/cv_ri.pth',
                        'gc': 'eval/mixformer/pretrained/ntu/cv_gc.pth'
                    }
                }
            },
            'sgn': {
                # SGN models not available - will be skipped
            }
        }

        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results/comprehensive', exist_ok=True)

    def find_available_models(self, dataset: str = 'ntu', setting: str = 'cv') -> Dict[str, List[str]]:
        """Find all available models for evaluation."""
        available_models = {}
        
        for model_type in ['transformer', 'pmr', 'dmr']:
            models = []
            
            # Check key models first
            for model_path in self.key_models.get(model_type, []):
                if os.path.exists(model_path):
                    models.append(model_path)
            
            # Add dataset-specific models from output directory
            output_patterns = [
                f"output/{dataset}_mixformer_*_{setting}/*/model_best.pth.tar",
                f"output/{dataset}_*_{setting}/*/model_best.pth.tar"
            ]
            
            import glob
            for pattern in output_patterns:
                found_files = glob.glob(pattern)
                for file_path in found_files:
                    if file_path not in models:
                        models.append(file_path)
            
            if models:
                available_models[model_type] = models
        
        # Always include raw data
        available_models['raw'] = ['raw_data']
        
        return available_models

    def get_available_eval_models(self, dataset: str, setting: str) -> List[str]:
        """Get list of available evaluation models based on existing weights."""
        available_models = []

        for eval_model in ['mixformer', 'sgn']:
            if (eval_model in self.eval_model_weights and
                dataset in self.eval_model_weights[eval_model] and
                setting in self.eval_model_weights[eval_model][dataset]):

                weights = self.eval_model_weights[eval_model][dataset][setting]
                # Check if all required weights exist
                if all(os.path.exists(path) for path in weights.values()):
                    available_models.append(eval_model)

        return available_models

    def run_comprehensive_evaluation(self, model_path: str, model_type: str,
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation including AR/RI/GC and physical metrics."""
        print(f"\n🧪 COMPREHENSIVE EVALUATION: {model_type.upper()}")
        print("=" * 60)

        results = {}

        # Get available evaluation models
        available_eval_models = self.get_available_eval_models(config['dataset'], config['setting'])

        if not available_eval_models:
            print(f"❌ No evaluation models available for {config['dataset']} {config['setting']}")
            return {"status": "failed", "error": "No evaluation models available"}

        # Determine which eval models to use
        if config['eval_model'] == 'both':
            eval_models = available_eval_models
        elif config['eval_model'] in available_eval_models:
            eval_models = [config['eval_model']]
        else:
            print(f"⚠️  Requested eval model '{config['eval_model']}' not available. Using: {available_eval_models}")
            eval_models = available_eval_models

        for eval_model in eval_models:
            print(f"\n📊 Evaluating with {eval_model.upper()} models...")

            # Create model-specific config
            eval_config = config.copy()
            eval_config['eval_model'] = eval_model
            eval_config['output_dir'] = f"{config['output_dir']}/{model_type}_{eval_model}"

            # Add model weights to config
            if eval_model in self.eval_model_weights:
                weights = self.eval_model_weights[eval_model][config['dataset']][config['setting']]
                eval_config['ar_model_weights'] = weights['ar']
                eval_config['ri_model_weights'] = weights['ri']
                eval_config['gc_model_weights'] = weights['gc']

            # Run evaluation using eval_model.py
            eval_result = self._run_eval_model(model_path, model_type, eval_config)
            results[f"{model_type}_{eval_model}"] = eval_result

            # Add visualization if requested
            if config.get('include_visualizations', False):
                print(f"🎨 Creating visualizations for {model_type}_{eval_model}...")
                viz_result = self._create_visualizations(model_path, model_type, eval_config)
                if viz_result:
                    eval_result['visualizations'] = viz_result

        return results

    def _run_eval_model(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run eval_model.py with specified parameters."""
        try:
            # Use SLURM if requested
            if config.get('use_slurm', False):
                return self._run_slurm_evaluation(model_path, model_type, config)
            
            # Import and run eval_model directly
            try:
                from eval_model import main as eval_main
            except ImportError as e:
                raise Exception(f"Failed to import eval_model: {e}")
            
            # Save original argv
            original_argv = sys.argv.copy()
            
            # Build arguments for eval_model.py
            eval_args = [
                'eval_model.py',
                '--dataset', config['dataset'],
                '--setting', config['setting'],
                '--model_type', model_type,
                '--eval_model', config['eval_model']
            ]
            
            # Add model path for non-raw models
            if model_type != 'raw':
                eval_args.extend(['--transformer_model_path', model_path])
            
            # Add output directory
            if 'output_dir' in config:
                eval_args.extend(['--output_dir', config['output_dir']])

            # Add model weights if provided
            if 'ar_model_weights' in config:
                eval_args.extend(['--ar_model_weights', config['ar_model_weights']])
            if 'ri_model_weights' in config:
                eval_args.extend(['--ri_model_weights', config['ri_model_weights']])
            if 'gc_model_weights' in config:
                eval_args.extend(['--gc_model_weights', config['gc_model_weights']])

            # Set sys.argv and run evaluation
            sys.argv = eval_args
            
            print(f"📝 Running: {' '.join(eval_args[1:])}")
            
            try:
                eval_main()
                result = {"status": "completed", "message": "Evaluation finished successfully"}
                
                # Try to load results if available
                results_file = os.path.join(config.get('output_dir', 'results'), f"{model_type}_metrics.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        metrics = json.load(f)
                    result['metrics'] = metrics
                
                return result
                
            except SystemExit as e:
                if e.code != 0:
                    raise Exception(f"Evaluation exited with code {e.code}")
                return {"status": "completed", "message": "Evaluation finished successfully"}
            
            finally:
                # Restore original argv
                sys.argv = original_argv
                
        except Exception as e:
            logger.error(f"Evaluation failed for {model_type}: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _run_slurm_evaluation(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit evaluation job to SLURM."""
        print(f"🖥️  Submitting SLURM job for {model_type} evaluation...")
        
        try:
            # Create SLURM script
            job_name = f"comprehensive_eval_{model_type}_{config['dataset']}_{config['setting']}"
            script_content = self._create_slurm_script(model_path, model_type, config, job_name)
            
            # Save script
            script_path = f"slurm_out/{job_name}.sh"
            os.makedirs('slurm_out', exist_ok=True)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Submit job
            import subprocess
            result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"✅ SLURM job submitted: {job_id}")
                return {
                    "status": "submitted",
                    "job_id": job_id,
                    "script_path": script_path
                }
            else:
                raise Exception(f"SLURM submission failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"SLURM evaluation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _create_slurm_script(self, model_path: str, model_type: str, config: Dict[str, Any], job_name: str) -> str:
        """Create SLURM script for evaluation."""

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# Load modules and set environment
module load pytorch/2.3.0-cuda12.1

# Check CUDA availability
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs results/comprehensive

# Run comprehensive evaluation
python eval_model.py \\
    --dataset {config['dataset']} \\
    --setting {config['setting']} \\
    --model_type {model_type} \\
    --eval_model {config['eval_model']}"""

        # Add model path for non-raw models
        if model_type != 'raw':
            script += f" \\\n    --transformer_model_path {model_path}"
        
        # Add output directory
        if 'output_dir' in config:
            script += f" \\\n    --output_dir {config['output_dir']}"

        # Add model weights if provided
        if 'ar_model_weights' in config:
            script += f" \\\n    --ar_model_weights {config['ar_model_weights']}"
        if 'ri_model_weights' in config:
            script += f" \\\n    --ri_model_weights {config['ri_model_weights']}"
        if 'gc_model_weights' in config:
            script += f" \\\n    --gc_model_weights {config['gc_model_weights']}"

        script += """

echo ""
echo "Comprehensive evaluation completed at $(date)"
"""

        return script

    def _create_visualizations(self, model_path: str, model_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create visualizations for the evaluated model."""
        try:
            # Import visualization functionality
            import sys
            eval_suite_path = os.path.join(project_root, 'evaluation_suite')
            if eval_suite_path not in sys.path:
                sys.path.insert(0, eval_suite_path)

            try:
                from run_visualization import run_visualizations
                from core.visualization_evaluator import VisualizationEvaluator
            except ImportError as e:
                logger.warning(f"Visualization modules not available: {e}")
                return None

            # Import torch only when needed
            import torch

            # Create visualization config
            viz_config = {
                'dataset': config['dataset'],
                'setting': config['setting'],
                'visualizations': ['skeleton_animations'],
                'max_samples': 5,
                'max_frames': 32,
                'output_dir': os.path.join(config['output_dir'], 'visualizations'),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }

            # Initialize visualization evaluator
            evaluator = VisualizationEvaluator(
                device=viz_config['device'],
                output_base_dir=viz_config['output_dir']
            )

            # Create visualization for the specific model
            viz_results = {}

            if model_type != 'raw':
                # Load the model for visualization
                from eval_model import load_anonymizer
                anonymizer_model = load_anonymizer(model_type, model_path, viz_config['device'])

                # Create model-specific visualizations
                viz_results['model_animations'] = self._create_model_animations(
                    anonymizer_model, model_type, viz_config
                )

            # Create raw data visualizations for comparison
            viz_results['raw_animations'] = self._create_model_animations(
                None, 'raw', viz_config
            )

            return viz_results

        except Exception as e:
            logger.warning(f"Visualization creation failed: {str(e)}")
            return None

    def _create_model_animations(self, model, model_type: str, config: Dict[str, Any]) -> List[str]:
        """Create skeleton animations for a specific model."""
        try:
            from evaluation_suite.experiments.visualization import VisualizationExperiments
            from data import get_dataloader

            # Get test data
            _, _, test_loader = get_dataloader(
                config['dataset'], config['setting'],
                batch_size=1, num_workers=1
            )

            # Sample a few test examples
            animations = []
            sample_count = 0
            max_samples = config.get('max_samples', 5)

            for batch in test_loader:
                if sample_count >= max_samples:
                    break

                # Extract skeleton data
                if isinstance(batch, (list, tuple)):
                    skeleton_data = batch[0]
                else:
                    skeleton_data = batch

                # Process with model if provided
                if model is not None:
                    import torch
                    with torch.no_grad():
                        model.eval()
                        processed_data = model(skeleton_data.to(config['device']))
                        if isinstance(processed_data, (list, tuple)):
                            processed_data = processed_data[0]
                else:
                    processed_data = skeleton_data

                # Create animation
                animation_path = VisualizationExperiments.create_skeleton_animation(
                    [processed_data.cpu()],
                    output_dir=config['output_dir'],
                    figure_type=f'{model_type}_sample_{sample_count}',
                    max_frames=config.get('max_frames', 32)
                )

                if animation_path:
                    animations.append(animation_path)

                sample_count += 1

            return animations

        except Exception as e:
            logger.warning(f"Model animation creation failed: {str(e)}")
            return []

    def generate_comprehensive_report(self, results: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        print("\n📊 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"results/comprehensive/evaluation_report_{timestamp}.md"
        
        report_lines = []
        report_lines.append("# Comprehensive Model Evaluation Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Configuration
        report_lines.append("## Configuration")
        report_lines.append(f"- **Dataset**: {config['dataset']}")
        report_lines.append(f"- **Setting**: {config['setting']}")
        report_lines.append(f"- **Evaluation Models**: {config['eval_model']}")
        report_lines.append(f"- **SLURM Mode**: {config.get('use_slurm', False)}")
        report_lines.append("")
        
        # Results summary
        report_lines.append("## Evaluation Results Summary")
        report_lines.append("")
        
        # Create summary table
        report_lines.append("| Model | Eval Model | Status | AR Acc | RI Acc | GC Acc | MSE | Physical Score |")
        report_lines.append("|-------|------------|--------|--------|--------|--------|-----|----------------|")
        
        for result_key, result in results.items():
            model_parts = result_key.split('_')
            model_type = model_parts[0]
            eval_model = model_parts[1] if len(model_parts) > 1 else 'unknown'
            
            status = result.get('status', 'unknown')
            
            # Extract key metrics if available
            metrics = result.get('metrics', {})
            ar_acc = f"{metrics.get('action_recognition_accuracy', 0):.3f}" if 'action_recognition_accuracy' in metrics else "N/A"
            ri_acc = f"{metrics.get('reidentification_accuracy', 0):.3f}" if 'reidentification_accuracy' in metrics else "N/A"
            gc_acc = f"{metrics.get('gender_classification_accuracy', 0):.3f}" if 'gender_classification_accuracy' in metrics else "N/A"
            mse = f"{metrics.get('mse', 0):.6f}" if 'mse' in metrics else "N/A"
            
            # Calculate physical plausibility score (average of all 5 metrics)
            physical_metrics = ['bone_length_consistency', 'joint_angle_limits', 'temporal_smoothness', 
                              'velocity_consistency', 'foot_contact_consistency']
            physical_scores = []
            for pm in physical_metrics:
                if pm in metrics:
                    physical_scores.append(metrics[pm])
            
            physical_score = f"{sum(physical_scores)/len(physical_scores):.3f}" if physical_scores else "N/A"
            
            report_lines.append(f"| {model_type} | {eval_model} | {status} | {ar_acc} | {ri_acc} | {gc_acc} | {mse} | {physical_score} |")
        
        report_lines.append("")
        
        # Detailed results for each model
        report_lines.append("## Detailed Results")
        for result_key, result in results.items():
            model_parts = result_key.split('_')
            model_type = model_parts[0]
            eval_model = model_parts[1] if len(model_parts) > 1 else 'unknown'
            
            report_lines.append(f"### {model_type.upper()} - {eval_model.upper()}")
            
            status = result.get('status', 'unknown')
            report_lines.append(f"**Status**: {status}")
            
            if status == 'completed' and 'metrics' in result:
                metrics = result['metrics']
                
                # Core metrics
                report_lines.append("")
                report_lines.append("#### Core Performance Metrics")
                for metric_name, value in metrics.items():
                    if metric_name in ['action_recognition_accuracy', 'reidentification_accuracy', 
                                     'gender_classification_accuracy', 'mse']:
                        if isinstance(value, (int, float)):
                            report_lines.append(f"- **{metric_name}**: {value:.6f}")
                        else:
                            report_lines.append(f"- **{metric_name}**: {value}")
                
                # Physical plausibility metrics
                report_lines.append("")
                report_lines.append("#### Physical Plausibility Metrics")
                physical_metrics = ['bone_length_consistency', 'joint_angle_limits', 'temporal_smoothness', 
                                  'velocity_consistency', 'foot_contact_consistency']
                for pm in physical_metrics:
                    if pm in metrics:
                        value = metrics[pm]
                        if isinstance(value, (int, float)):
                            report_lines.append(f"- **{pm}**: {value:.6f}")
                        else:
                            report_lines.append(f"- **{pm}**: {value}")
            
            elif status == 'submitted':
                report_lines.append(f"- **Job ID**: {result.get('job_id', 'Unknown')}")
                report_lines.append(f"- **Script Path**: {result.get('script_path', 'Unknown')}")
            
            elif status == 'failed':
                report_lines.append(f"- **Error**: {result.get('error', 'Unknown error')}")
            
            report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Comprehensive report saved to: {report_file}")
        return report_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for trained models with AR/RI/GC and physical metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model-path', type=str,
                       help='Path to model file (use "auto" to find default model)')
    parser.add_argument('--model-type', choices=['transformer', 'pmr', 'dmr', 'raw', 'all'],
                       default='transformer',
                       help='Type of model (use "all" to evaluate all available models)')
    
    # Dataset configuration
    parser.add_argument('--dataset', choices=['ntu', 'ntu120', 'etri'], default='ntu',
                       help='Dataset to evaluate on (default: ntu)')
    parser.add_argument('--setting', choices=['cs', 'cv'], default='cv',
                       help='Cross-subject (cs) or cross-view (cv) setting (default: cv)')
    
    # Evaluation configuration
    parser.add_argument('--eval-model', choices=['sgn', 'mixformer', 'both'], default='both',
                       help='Evaluation model to use (default: both)')
    parser.add_argument('--slurm', action='store_true',
                       help='Submit evaluation jobs to SLURM')
    parser.add_argument('--email', type=str, default='carrt313@gmail.com',
                       help='Email for SLURM notifications')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--visualizations', action='store_true',
                       help='Include visualization generation in evaluation')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    evaluator = ComprehensiveEvaluator()
    
    print("🚀 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    print("Features:")
    print("- AR (Action Recognition), RI (Re-identification), GC (Gender Classification)")
    print("- Physical Plausibility (5 metrics: bone length, joint angles, smoothness, velocity, foot contact)")
    print("- Per-actor and per-action breakdowns")
    print("- Automatic detection of available evaluation models (SGN/Mixformer)")
    print("- Optional skeleton visualizations and animations")
    print("- Clean, organized results and reports")
    print("")
    
    # Build configuration
    config = {
        'dataset': args.dataset,
        'setting': args.setting,
        'eval_model': args.eval_model,
        'output_dir': args.output_dir or f"results/comprehensive/{args.dataset}_{args.setting}",
        'use_slurm': args.slurm,
        'email': args.email,
        'include_visualizations': args.visualizations
    }
    
    # Handle model selection
    if args.model_type == 'all':
        print("🔄 Evaluating all available models...")
        available_models = evaluator.find_available_models(args.dataset, args.setting)
        
        all_results = {}
        for model_type, model_paths in available_models.items():
            print(f"\n📊 Evaluating {model_type.upper()} models...")
            
            for model_path in model_paths:
                print(f"  🔍 Model: {model_path}")
                
                # Create model-specific config
                model_config = config.copy()
                model_config['output_dir'] = f"{config['output_dir']}/{model_type}"
                
                # Run evaluation
                if model_type == 'raw':
                    result = evaluator.run_comprehensive_evaluation('raw', 'raw', model_config)
                else:
                    result = evaluator.run_comprehensive_evaluation(model_path, model_type, model_config)
                
                # Store results
                model_name = os.path.basename(model_path).replace('.pth', '').replace('.tar', '')
                all_results[f"{model_type}_{model_name}"] = result
        
        # Generate comprehensive report
        report_file = evaluator.generate_comprehensive_report(all_results, config)
        
    else:
        # Single model evaluation
        if not args.model_path or args.model_path == 'auto':
            # Find default model
            available_models = evaluator.find_available_models(args.dataset, args.setting)
            if args.model_type in available_models and available_models[args.model_type]:
                model_path = available_models[args.model_type][0]
                print(f"🔍 Using default model: {model_path}")
            else:
                print(f"❌ No default model found for type: {args.model_type}")
                return
        else:
            model_path = args.model_path
            if not os.path.exists(model_path) and args.model_type != 'raw':
                print(f"❌ Model file not found: {model_path}")
                return
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(model_path, args.model_type, config)
        
        # Generate report
        report_file = evaluator.generate_comprehensive_report(results, config)
    
    print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETED!")
    print(f"📄 Report: {report_file}")
    print(f"📁 Results: {config['output_dir']}")


if __name__ == "__main__":
    main()
