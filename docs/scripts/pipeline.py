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
import time
import re
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

class SlurmJobManager:
    """Manages SLURM job creation, submission, and dependency tracking."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hpc_config = config.get('hpc', {})
        self.job_dir = Path("scripts/generated/slurm_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.submitted_jobs = {}  # step_name -> job_id

    def get_job_template(self, step: str) -> Dict[str, Any]:
        """Get appropriate job template based on step type."""
        templates = self.hpc_config.get('job_templates', {})

        step_templates = {
            'preprocess': templates.get('quick', {}),
            'sample': templates.get('standard', {}),
            'pretrain': templates.get('long', {}),
            'train': templates.get('long', {}),
            'evaluate': templates.get('standard', {})
        }

        template = step_templates.get(step, templates.get('standard', {}))

        # Merge with defaults
        defaults = {
            'partition': self.hpc_config.get('default_partition', 'GPU'),
            'time': self.hpc_config.get('default_time', '12:00:00'),
            'nodes': self.hpc_config.get('default_nodes', 1),
            'ntasks_per_node': self.hpc_config.get('default_ntasks_per_node', 1),
            'gres': self.hpc_config.get('default_gres', 'gpu:1'),
            'mem': self.hpc_config.get('default_mem', '64GB')
        }

        # Update defaults with template values
        defaults.update(template)
        return defaults

    def generate_job_script(self, step: str, pipeline_config: Dict[str, Any], dependency_job_id: Optional[str] = None) -> Path:
        """Generate SLURM job script for a pipeline step."""
        dataset = pipeline_config['dataset']
        setting = pipeline_config['setting']
        job_name = f"pipeline_{step}_{dataset}_{setting}"

        template = self.get_job_template(step)

        # Extract GPU count from template for environment setup
        gres = template.get('gres', 'gpu:1')
        gpu_count = 1
        if ':' in gres:
            try:
                gpu_count = int(gres.split(':')[1])
            except (ValueError, IndexError):
                gpu_count = 1

        # Build SLURM directives
        slurm_directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={template['partition']}",
            f"#SBATCH --time={template['time']}",
            f"#SBATCH --nodes={template['nodes']}",
            f"#SBATCH --ntasks-per-node={template['ntasks_per_node']}",
            f"#SBATCH --gres={template['gres']}",
            f"#SBATCH --mem={template['mem']}",
            f"#SBATCH --output=logs/{job_name}_%j.out",
            f"#SBATCH --error=logs/{job_name}_%j.err",
            f"#SBATCH --mail-type=BEGIN,END,FAIL",
            f"#SBATCH --mail-user=carrt313@gmail.com"
        ]

        # Add dependency if specified
        if dependency_job_id:
            slurm_directives.append(f"#SBATCH --dependency=afterok:{dependency_job_id}")

        # Generate command based on step
        command = self._generate_step_command(step, pipeline_config)

        # Build complete script
        script_lines = slurm_directives + [
            "",
            "# Load modules and set environment",
            "module load pytorch/2.3.0-cuda12.1",
            "",
            "# Check CUDA availability",
            "echo \"Job started at $(date)\"",
            "echo \"Running on node: $(hostname)\"",
            "echo \"Is CUDA Available?\"",
            "python -c 'import torch; print(torch.cuda.is_available())'",
            "echo \"\"",
            "echo \"nvidia-smi output:\"",
            "nvidia-smi",
            "",
            "# Set environment variables for distributed training",
            "export OMP_NUM_THREADS=1",
            f"export CUDA_VISIBLE_DEVICES={','.join(map(str, range(gpu_count)))}",
            "export MASTER_ADDR=$(hostname)",
            "export MASTER_PORT=29500",
            "",
            "# Change to project directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
            "# Create necessary directories",
            "mkdir -p logs",
            "mkdir -p results",
            "mkdir -p checkpoints",
            "",
            "# Run the command",
            f"echo \"Executing: {command}\"",
            command,
            "",
            f"echo \"Step {step} completed at $(date)\""
        ]

        # Save script
        script_path = self.job_dir / f"{job_name}.sbatch"
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))

        # Make executable
        os.chmod(script_path, 0o755)

        return script_path

    def _generate_step_command(self, step: str, pipeline_config: Dict[str, Any]) -> str:
        """Generate the command to run for a specific step."""
        dataset = pipeline_config['dataset']
        setting = pipeline_config['setting']

        if step == 'preprocess':
            return f"python scripts/preprocess.py --dataset {dataset} --setting {setting}"
        elif step == 'sample':
            return f"python scripts/sample.py --dataset {dataset} --setting {setting}"
        elif step == 'pretrain':
            return f"python scripts/pretrain.py --task encoder --dataset {dataset} --setting {setting}"
        elif step == 'train':
            # Extract number of GPUs from the job template
            template = self.get_job_template(step)
            gres = template.get('gres', 'gpu:1')
            # Extract GPU count from gres (e.g., 'gpu:4' -> 4)
            gpu_count = 1
            if ':' in gres:
                try:
                    gpu_count = int(gres.split(':')[1])
                except (ValueError, IndexError):
                    gpu_count = 1

            # OPTIMIZED: Build training command with all performance optimizations
            if gpu_count > 1:
                base_cmd = f"torchrun --nproc_per_node={gpu_count} --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355"
            else:
                base_cmd = "python"

            train_cmd = f"{base_cmd} main.py --dataset {dataset} --setting {setting}"

            # Add HPC and performance optimizations
            if gpu_count > 1:
                train_cmd += " --hpc"

            # Add optimized training configuration
            train_cmd += f" --gpus {gpu_count}"
            train_cmd += " --data-path data/ntu_cv_paired_comprehensive.pt"
            train_cmd += " --train-samples 999999999"
            train_cmd += " --test-samples 10000"
            train_cmd += " --epochs 5"

            # OPTIMIZED: Use hyperparameter tuning results (trial 17)
            if gpu_count > 1:
                train_cmd += " --batch-size 8"  # Optimized for multi-GPU from hyperparameter tuning
            else:
                train_cmd += " --batch-size 32"  # Optimized for single GPU from hyperparameter tuning
            train_cmd += " --lr 9.43062936149491e-05"  # Optimized learning rate
            train_cmd += " --gradient-accumulation-steps 4"  # 4x gradient accumulation
            train_cmd += " --mixed-precision"  # Enable mixed precision (auto-disabled for multi-GPU)
            train_cmd += " --validate-every 5"  # Validate every 5 epochs
            train_cmd += " --progress-every 100"  # Progress every 100 batches
            train_cmd += " --use-checkpoint"  # Enable gradient checkpointing
            train_cmd += " --nccl-timeout 7200"  # 2 hour NCCL timeout
            train_cmd += " --decoder-dropout 0.11551063114920847"  # Optimized dropout
            train_cmd += " --save-every 1"

            # OPTIMIZED: Use hyperparameter tuning loss weights (trial 17)
            train_cmd += " --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404"

            return train_cmd
        elif step == 'evaluate':
            evaluations = pipeline_config.get('evaluations', [])
            if evaluations:
                if isinstance(evaluations, list):
                    # Convert evaluation names to proper format
                    eval_names = []
                    for eval_name in evaluations:
                        # Convert from display format back to internal format
                        eval_names.append(eval_name.lower().replace(' ', '_'))
                    eval_str = ','.join(eval_names)
                else:
                    eval_str = str(evaluations)
                return f"python evaluation_suite/run_experiments.py --experiments {eval_str}"
            else:
                # Default to critical evaluations
                return f"python scripts/evaluate.py --experiment-set critical"
        elif step == 'visualize':
            visualizations = pipeline_config.get('visualizations', [])
            if visualizations:
                if isinstance(visualizations, list):
                    viz_names = []
                    for viz_name in visualizations:
                        viz_names.append(viz_name.lower().replace(' ', '_'))
                    viz_str = ','.join(viz_names)
                else:
                    viz_str = str(visualizations)

                # Add MLM-specific arguments if MLM pretraining is selected
                mlm_args = ""
                if 'mlm_pretraining' in visualizations:
                    temporal_ratio = pipeline_config.get('temporal_ratio', 0.3)
                    spatial_ratio = pipeline_config.get('spatial_ratio', 0.3)
                    mlm_args = f" --temporal-ratio {temporal_ratio} --spatial-ratio {spatial_ratio}"

                return f"python evaluation_suite/run_visualization.py --visualizations {viz_str} --dataset {dataset} --setting {setting}{mlm_args}"
            else:
                # Default to skeleton animations
                return f"python evaluation_suite/run_visualization.py --visualizations skeleton_animations --dataset {dataset} --setting {setting}"
        else:
            return f"echo 'Unknown step: {step}'"

    def submit_job(self, script_path: Path) -> Optional[str]:
        """Submit a SLURM job and return the job ID."""
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Extract job ID from sbatch output (format: "Submitted batch job 12345")
            output = result.stdout.strip()
            job_id_match = re.search(r'Submitted batch job (\d+)', output)

            if job_id_match:
                job_id = job_id_match.group(1)
                print(f"✅ Job submitted: {script_path.name} (ID: {job_id})")
                return job_id
            else:
                print(f"⚠️  Job submitted but couldn't extract ID: {output}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit job {script_path.name}: {e.stderr}")
            return None
        except FileNotFoundError:
            print("❌ sbatch command not found. Are you on an HPC system with SLURM?")
            return None

    def check_job_status(self, job_id: str) -> str:
        """Check the status of a SLURM job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True,
                check=True
            )

            status = result.stdout.strip()
            return status if status else "UNKNOWN"

        except subprocess.CalledProcessError:
            # Job might be completed or not found
            try:
                # Check completed jobs
                result = subprocess.run(
                    ['sacct', '-j', job_id, '-n', '-o', 'State'],
                    capture_output=True,
                    text=True,
                    check=True
                )

                status = result.stdout.strip().split('\n')[0]
                return status if status else "COMPLETED"

            except subprocess.CalledProcessError:
                return "NOT_FOUND"
        except FileNotFoundError:
            print("⚠️  SLURM commands not available")
            return "UNKNOWN"

    def wait_for_job(self, job_id: str, timeout: int = 3600) -> bool:
        """Wait for a job to complete with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.check_job_status(job_id)

            if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                return status == 'COMPLETED'

            print(f"⏳ Job {job_id} status: {status}")
            time.sleep(30)  # Check every 30 seconds

        print(f"⏰ Timeout waiting for job {job_id}")
        return False

    def submit_pipeline_jobs(self, pipeline_config: Dict[str, Any]) -> Dict[str, str]:
        """Submit all pipeline jobs with proper dependencies."""
        steps = pipeline_config['steps']
        job_ids = {}

        print(f"\n🚀 SUBMITTING SLURM PIPELINE JOBS")
        print("=" * 50)
        print(f"📊 Dataset: {pipeline_config['dataset']}")
        print(f"⚙️  Setting: {pipeline_config['setting']}")
        print(f"🔄 Steps: {' → '.join(steps)}")

        previous_job_id = None

        for i, step in enumerate(steps, 1):
            print(f"\n📍 STEP {i}/{len(steps)}: {step.upper()}")
            print("-" * 30)

            # Generate job script
            script_path = self.generate_job_script(step, pipeline_config, previous_job_id)
            print(f"📝 Generated script: {script_path}")

            # Submit job
            job_id = self.submit_job(script_path)

            if job_id:
                job_ids[step] = job_id
                self.submitted_jobs[step] = job_id

                if previous_job_id:
                    print(f"🔗 Depends on job: {previous_job_id}")

                previous_job_id = job_id
            else:
                print(f"❌ Failed to submit job for step: {step}")
                break

        return job_ids

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
    all_steps = ['preprocess', 'sample', 'pretrain', 'train', 'evaluate', 'visualize']
    step_descriptions = {
        'preprocess': 'Data preprocessing and normalization',
        'sample': 'Generate paired training/test samples',
        'pretrain': 'Pretrain encoder and baseline models',
        'train': 'Main model training',
        'evaluate': 'Comprehensive evaluation and reporting',
        'visualize': 'Create skeleton animations and visualizations'
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
        selected_steps = ['sample', 'train', 'evaluate', 'visualize']
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

    # Select visualizations if visualize step is included
    selected_visualizations = []
    if 'visualize' in selected_steps:
        print(f"\n🎨 Visualization Selection:")
        print("Which visualizations would you like to create?")

        available_visualizations = [
            "skeleton_animations",
            "motion_visualizations",
            "attention_visualization",
            "comparison_visualizations",
            "sensitivity_analysis",
            "anonymization_showcase",
            "mlm_pretraining"
        ]

        for i, visualization in enumerate(available_visualizations, 1):
            print(f"{i:2d}. {visualization.replace('_', ' ').title()}")

        print("\nSelect visualizations:")
        print("  - Enter numbers separated by commas (e.g., 1,2,3)")
        print("  - Enter 'all' for all visualizations")
        print("  - Enter 'basic' for essential visualizations")

        viz_choice = input("\nVisualizations to create: ").strip()

        if viz_choice.lower() == 'all':
            selected_visualizations = available_visualizations
        elif viz_choice.lower() == 'basic':
            selected_visualizations = [
                "skeleton_animations",
                "comparison_visualizations",
                "mlm_pretraining"
            ]
        else:
            try:
                indices = [int(x.strip()) - 1 for x in viz_choice.split(',')]
                selected_visualizations = [available_visualizations[i] for i in indices if 0 <= i < len(available_visualizations)]
            except:
                print("❌ Invalid visualization selection, using basic visualizations")
                selected_visualizations = [
                    "skeleton_animations",
                    "comparison_visualizations",
                    "mlm_pretraining"
                ]

        print(f"\n✅ Selected {len(selected_visualizations)} visualizations:")
        for viz_name in selected_visualizations:
            print(f"   • {viz_name.replace('_', ' ').title()}")

    # MLM-specific configuration
    temporal_ratio = 0.3
    spatial_ratio = 0.3
    if 'visualize' in selected_steps and 'mlm_pretraining' in selected_visualizations:
        print(f"\n🧠 MLM Pretraining Configuration:")
        print("Configure masking ratios for MLM visualization:")

        temporal_input = input(f"Temporal masking ratio [0.3]: ").strip()
        if temporal_input:
            try:
                temporal_ratio = float(temporal_input)
                if not 0.0 <= temporal_ratio <= 1.0:
                    print("⚠️  Warning: Temporal ratio should be between 0.0 and 1.0")
                    temporal_ratio = 0.3
            except ValueError:
                print("⚠️  Invalid temporal ratio, using default 0.3")
                temporal_ratio = 0.3

        spatial_input = input(f"Spatial masking ratio [0.3]: ").strip()
        if spatial_input:
            try:
                spatial_ratio = float(spatial_input)
                if not 0.0 <= spatial_ratio <= 1.0:
                    print("⚠️  Warning: Spatial ratio should be between 0.0 and 1.0")
                    spatial_ratio = 0.3
            except ValueError:
                print("⚠️  Invalid spatial ratio, using default 0.3")
                spatial_ratio = 0.3

        print(f"✅ MLM Configuration: temporal={temporal_ratio}, spatial={spatial_ratio}")

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
        'visualizations': selected_visualizations,
        'temporal_ratio': temporal_ratio,
        'spatial_ratio': spatial_ratio,
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

def execute_step(step: str, pipeline_config: Dict[str, Any], config: Dict[str, Any], slurm_manager: Optional[SlurmJobManager] = None) -> Tuple[bool, str]:
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
            # OPTIMIZED: Use hyperparameter tuning results (trial 17)
            cmd = f"torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 main.py"
            cmd += f" --dataset {dataset} --setting {setting} --hpc --gpus 4"
            cmd += " --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5"
            cmd += " --batch-size 8 --lr 9.43062936149491e-05 --gradient-accumulation-steps 4 --mixed-precision"
            cmd += " --validate-every 5 --progress-every 100 --use-checkpoint --nccl-timeout 7200"
            cmd += " --decoder-dropout 0.11551063114920847 --save-every 1"
            cmd += " --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404"
            cmd += " --loss-mse 1.0 --loss-ee 1.0 --loss-smoothing 0.1 --loss-inception 0.1"
            cmd += " --loss-fid-vel 0.5 --loss-bone 1.0 --loss-foot 0.1 --loss-joint-limit 0.01"
        elif step == 'evaluate':
            # UPDATED: Use main eval_model.py and add MLM evaluation
            cmd = f"python eval_model.py --dataset {dataset} --setting {setting} --eval_model sgn --model_type transformer"
            # Add MLM evaluation if MLM pretraining was done
            if pipeline_config.get('include_mlm_evaluation', True):
                cmd += f" && python evaluation_suite/comprehensive_mlm_evaluation.py --dataset {dataset} --setting {setting}"
        elif step == 'visualize':
            visualizations = pipeline_config.get('visualizations', ['skeleton_animations'])
            if isinstance(visualizations, list):
                viz_str = ','.join(visualizations)
            else:
                viz_str = str(visualizations)

            # Add MLM-specific arguments if MLM pretraining is selected
            mlm_args = ""
            if 'mlm_pretraining' in visualizations:
                temporal_ratio = pipeline_config.get('temporal_ratio', 0.3)
                spatial_ratio = pipeline_config.get('spatial_ratio', 0.3)
                mlm_args = f" --temporal-ratio {temporal_ratio} --spatial-ratio {spatial_ratio}"

            cmd = f"python evaluation_suite/run_visualization.py --visualizations {viz_str} --dataset {dataset} --setting {setting}{mlm_args}"
        else:
            return False, f"Unknown step: {step}"

        print(f"💻 Executing: {cmd}")

        # Actually execute the command
        try:
            import shlex

            # For HPC environments, prepend module loading
            if 'python' in cmd and ('evaluation_suite' in cmd or 'visualiz' in cmd):
                # Add module loading for Python-based commands
                full_cmd = f"module load pytorch/2.3.0-cuda12.1 && {cmd}"
                print(f"🔧 HPC command: {full_cmd}")

                result = subprocess.run(
                    full_cmd,
                    shell=True,  # Use shell for module loading
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=os.getcwd()
                )
            else:
                result = subprocess.run(
                    shlex.split(cmd),  # Better command parsing
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise exception on non-zero exit
                    cwd=os.getcwd()
                )

            # Log the output (limit length for readability)
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                if len(stdout_lines) > 20:
                    print(f"📤 STDOUT (showing last 20 lines):")
                    for line in stdout_lines[-20:]:
                        print(f"   {line}")
                else:
                    print(f"📤 STDOUT:")
                    for line in stdout_lines:
                        print(f"   {line}")

            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                if len(stderr_lines) > 10:
                    print(f"📤 STDERR (showing last 10 lines):")
                    for line in stderr_lines[-10:]:
                        print(f"   {line}")
                else:
                    print(f"📤 STDERR:")
                    for line in stderr_lines:
                        print(f"   {line}")

            if result.returncode == 0:
                return True, f"Step completed successfully (exit code: {result.returncode})"
            else:
                return False, f"Step failed with exit code: {result.returncode}"

        except Exception as e:
            return False, f"Failed to execute command: {str(e)}"

    elif exec_mode == 'slurm':
        # This is now handled by the pipeline-level SLURM submission
        # Individual steps should not be executed in SLURM mode
        print(f"🖥️  SLURM mode: Step will be handled by pipeline job submission")
        return True, "SLURM job will be generated by pipeline"

    elif exec_mode == 'windows':
        # Generate Windows batch file
        if step == 'preprocess':
            cmd = f"python scripts/preprocess.py --dataset {dataset} --setting {setting}"
        elif step == 'sample':
            cmd = f"python scripts/sample.py --dataset {dataset} --setting {setting}"
        elif step == 'pretrain':
            cmd = f"python scripts/pretrain.py --task encoder --dataset {dataset} --setting {setting} --windows"
        elif step == 'train':
            # OPTIMIZED: Windows single GPU with hyperparameter tuning results (trial 17)
            cmd = f"python main.py --dataset {dataset} --setting {setting} --gpus 1"
            cmd += " --data-path data/ntu_cv_paired_comprehensive.pt --train-samples 999999999 --test-samples 10000 --epochs 5"
            cmd += " --batch-size 32 --lr 9.43062936149491e-05 --gradient-accumulation-steps 1 --mixed-precision"
            cmd += " --validate-every 5 --progress-every 100 --use-checkpoint"
            cmd += " --decoder-dropout 0.11551063114920847 --save-every 1"
            cmd += " --loss-weights mse:5.323284271000699,ee:4.7250245075017725,smoothing:2.8747697025246937,inception:3.0656068713914353,fid_vel:2.2608337791442894,bone:6.185377286837532,foot:0.544125368059208,joint_limit:1.768344443841404"
            cmd += " --loss-fid-vel 0.5 --loss-bone 1.0 --loss-foot 0.1 --loss-joint-limit 0.01"
        elif step == 'evaluate':
            # UPDATED: Use main eval_model.py and add MLM evaluation (Windows)
            cmd = f"python eval_model.py --dataset {dataset} --setting {setting} --eval_model sgn --model_type transformer"
            # Add MLM evaluation if MLM pretraining was done
            if pipeline_config.get('include_mlm_evaluation', True):
                cmd += f" && python evaluation_suite/comprehensive_mlm_evaluation.py --dataset {dataset} --setting {setting}"
        elif step == 'visualize':
            visualizations = pipeline_config.get('visualizations', ['skeleton_animations'])
            if isinstance(visualizations, list):
                viz_str = ','.join(visualizations)
            else:
                viz_str = str(visualizations)

            # Add MLM-specific arguments if MLM pretraining is selected
            mlm_args = ""
            if 'mlm_pretraining' in visualizations:
                temporal_ratio = pipeline_config.get('temporal_ratio', 0.3)
                spatial_ratio = pipeline_config.get('spatial_ratio', 0.3)
                mlm_args = f" --temporal-ratio {temporal_ratio} --spatial-ratio {spatial_ratio}"

            cmd = f"python evaluation_suite/run_visualization.py --visualizations {viz_str} --dataset {dataset} --setting {setting} --windows{mlm_args}"
        else:
            return False, f"Unknown step: {step}"

        print(f"🪟 Executing on Windows: {cmd}")

        # Execute the command on Windows
        try:
            import shlex

            # For Windows, just execute directly (no module loading needed)
            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                check=False,
                cwd=os.getcwd(),
                shell=True  # Use shell on Windows for better compatibility
            )

            # Log the output (limit length for readability)
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                if len(stdout_lines) > 20:
                    print(f"📤 STDOUT (showing last 20 lines):")
                    for line in stdout_lines[-20:]:
                        print(f"   {line}")
                else:
                    print(f"📤 STDOUT:")
                    for line in stdout_lines:
                        print(f"   {line}")

            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                if len(stderr_lines) > 10:
                    print(f"📤 STDERR (showing last 10 lines):")
                    for line in stderr_lines[-10:]:
                        print(f"   {line}")
                else:
                    print(f"📤 STDERR:")
                    for line in stderr_lines:
                        print(f"   {line}")

            if result.returncode == 0:
                return True, f"Step completed successfully (exit code: {result.returncode})"
            else:
                return False, f"Step failed with exit code: {result.returncode}"

        except Exception as e:
            return False, f"Failed to execute command: {str(e)}"

    return False, f"Unknown execution mode: {exec_mode}"

def run_pipeline(pipeline_config: Dict[str, Any], config: Dict[str, Any], resume_from: Optional[str] = None) -> bool:
    """Run the complete pipeline."""
    state = PipelineState()

    if not resume_from:
        state.start_pipeline(pipeline_config)

    steps = pipeline_config['steps']
    exec_mode = pipeline_config['execution_mode']

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
    print(f"🖥️  Execution: {exec_mode}")

    # Handle SLURM mode differently - submit all jobs with dependencies
    if exec_mode == 'slurm':
        slurm_manager = SlurmJobManager(config)
        job_ids = slurm_manager.submit_pipeline_jobs(pipeline_config)

        if job_ids:
            print(f"\n🎉 SLURM PIPELINE JOBS SUBMITTED SUCCESSFULLY!")
            print("=" * 60)
            print(f"✅ {len(job_ids)} jobs submitted with dependencies")

            print(f"\n📋 JOB SUMMARY:")
            for step, job_id in job_ids.items():
                print(f"  • {step}: Job ID {job_id}")

            print(f"\n💡 MONITORING COMMANDS:")
            print(f"  • Check queue: squeue -u $USER")
            print(f"  • Check specific job: squeue -j <job_id>")
            print(f"  • Cancel job: scancel <job_id>")
            print(f"  • View logs: tail -f logs/pipeline_*_<job_id>.out")

            # Save job IDs to state for monitoring
            state.state['slurm_jobs'] = job_ids
            state.save_state()

            return True
        else:
            print(f"\n❌ FAILED TO SUBMIT SLURM JOBS")
            return False

    # Handle direct and windows modes with step-by-step execution
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

def monitor_slurm_jobs(config: Dict[str, Any]) -> None:
    """Monitor SLURM jobs from the pipeline state."""
    state = PipelineState()

    if 'slurm_jobs' not in state.state or not state.state['slurm_jobs']:
        print("❌ No SLURM jobs found in pipeline state")
        print("💡 Run the pipeline with --slurm first")
        return

    job_ids = state.state['slurm_jobs']
    slurm_manager = SlurmJobManager(config)

    print(f"\n📊 SLURM JOB MONITORING")
    print("=" * 50)
    print(f"⏱️  Last updated: {state.state.get('last_update', 'Unknown')}")

    all_completed = True

    for step, job_id in job_ids.items():
        status = slurm_manager.check_job_status(job_id)

        if status == "COMPLETED":
            status_icon = "✅"
        elif status in ["RUNNING", "PENDING"]:
            status_icon = "⏳"
            all_completed = False
        elif status in ["FAILED", "CANCELLED", "TIMEOUT"]:
            status_icon = "❌"
            all_completed = False
        else:
            status_icon = "❓"
            all_completed = False

        print(f"  {status_icon} {step}: Job {job_id} - {status}")

    if all_completed:
        print(f"\n🎉 All jobs completed successfully!")
    else:
        print(f"\n⏳ Some jobs are still running or failed")
        print(f"💡 Use 'squeue -u $USER' for real-time status")

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

  # Submit SLURM jobs with dependencies
  python scripts/pipeline.py --quick-start --dataset ntu --setting cv --slurm

  # Monitor submitted SLURM jobs
  python scripts/pipeline.py --monitor

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
                       choices=['preprocess', 'sample', 'pretrain', 'train', 'evaluate', 'visualize'],
                       help='Resume pipeline from specific step')
    parser.add_argument('--slurm', action='store_true',
                       help='Generate and submit SLURM job scripts with dependencies')
    parser.add_argument('--windows', action='store_true',
                       help='Generate Windows batch files instead of running directly')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor status of submitted SLURM jobs')
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

    # Handle monitoring mode
    if args.monitor:
        monitor_slurm_jobs(config)
        sys.exit(0)

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

        # If command-line execution mode was specified, override the interactive choice
        if args.slurm or args.windows:
            pipeline_config['execution_mode'] = exec_mode
            print(f"🔧 Overriding execution mode with command-line argument: {exec_mode}")
        
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
        valid_steps = ['preprocess', 'sample', 'pretrain', 'train', 'evaluate', 'visualize']

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
