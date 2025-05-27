"""
Loss Component Ablation Analysis

This script provides framework and analysis for actual loss component ablation studies.
It can analyze results from individual loss component removal experiments.

Expected structure:
- experiments/losses/single_loss/results/no_[component]/
- Each directory should contain evaluation results from models trained without specific loss components
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LossComponentAblationAnalyzer:
    """Analyzes loss component ablation study results."""

    def __init__(self, base_dir="/users/tcarr23/Transformer-Retargeting"):
        self.base_dir = Path(base_dir)
        self.ablation_dir = self.base_dir / "experiments" / "losses"
        self.results_dir = self.base_dir / "evaluation_suite" / "results" / "experiments" / "loss_component_ablation"
        self.plots_dir = self.results_dir / "plots"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Define loss components
        self.loss_components = {
            'mse': 'MSE Loss',
            'ee': 'End-Effector Loss',
            'smoothing': 'Smoothing Loss',
            'bone': 'Bone Length Loss',
            'foot': 'Foot Contact Loss',
            'joint_limit': 'Joint Limit Loss',
            'fid_vel': 'FID Velocity Loss',
            'inception': 'Inception Loss'
        }

    def check_ablation_results(self):
        """Check what ablation results are available."""
        print("🔍 Checking for loss component ablation results...")

        available_results = {}

        # Check single loss ablations
        single_loss_dir = self.ablation_dir / "single_loss" / "results"
        if single_loss_dir.exists():
            for component in self.loss_components.keys():
                component_dir = single_loss_dir / f"no_{component}"
                if component_dir.exists():
                    # Look for evaluation results
                    eval_files = list(component_dir.glob("*results*.json")) + list(component_dir.glob("*metrics*.json"))
                    if eval_files:
                        available_results[f"no_{component}"] = {
                            'path': component_dir,
                            'files': eval_files,
                            'type': 'single_loss'
                        }

        # Check group ablations
        group_dir = self.ablation_dir / "group_ablation" / "results"
        if group_dir.exists():
            for group_name in ['no_kinematic', 'no_smoothness']:
                group_path = group_dir / group_name
                if group_path.exists():
                    eval_files = list(group_path.glob("*results*.json")) + list(group_path.glob("*metrics*.json"))
                    if eval_files:
                        available_results[group_name] = {
                            'path': group_path,
                            'files': eval_files,
                            'type': 'group_ablation'
                        }

        print(f"✅ Found {len(available_results)} ablation experiments:")
        for name, info in available_results.items():
            print(f"  📁 {name}: {len(info['files'])} result files")

        return available_results

    def load_ablation_results(self, available_results):
        """Load results from ablation experiments."""
        print("📊 Loading ablation results...")

        all_results = []

        for exp_name, info in available_results.items():
            print(f"  📄 Loading {exp_name}...")

            # Try to load evaluation results
            for file_path in info['files']:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Extract metrics
                    result = {
                        'experiment': exp_name,
                        'ablation_type': info['type'],
                        'file': file_path.name
                    }

                    # Add metrics (adapt based on your evaluation output format)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                result[key] = value

                    all_results.append(result)

                except Exception as e:
                    print(f"    ⚠️  Error loading {file_path}: {e}")

        if all_results:
            df = pd.DataFrame(all_results)
            print(f"✅ Loaded {len(df)} result entries")
            return df
        else:
            print("❌ No valid results found")
            return None

    def create_ablation_impact_plot(self, df):
        """Create plot showing impact of each ablated component."""
        print("📊 Creating ablation impact plot...")

        # Group by experiment and calculate mean metrics
        grouped = df.groupby('experiment').mean(numeric_only=True)

        # Define key metrics to plot
        key_metrics = ['accuracy', 'identity_accuracy', 'mse', 'bone_length_error', 'foot_contact_error']
        available_metrics = [m for m in key_metrics if m in grouped.columns]

        if not available_metrics:
            print("⚠️  No standard metrics found for plotting")
            return None

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]

        fig.suptitle('Loss Component Ablation Impact Analysis', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            # Create bar plot
            experiments = grouped.index
            values = grouped[metric]

            bars = ax.bar(range(len(experiments)), values)

            # Customize plot
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels([exp.replace('no_', '').replace('_', ' ').title() for exp in experiments],
                              rotation=45, ha='right')

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "loss_component_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Impact plot saved to {plot_path}")
        return plot_path

    def generate_ablation_scripts(self):
        """Generate SLURM scripts for loss component ablation experiments."""
        print("🔧 Generating loss component ablation scripts...")

        scripts_dir = self.results_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Generate individual component ablation scripts
        for component, name in self.loss_components.items():
            script_content = f"""#!/bin/bash
#
#SBATCH --job-name="no-{component}"
#SBATCH --partition=GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#
#   ===== No {name} Experiment =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Create output directory
mkdir -p experiments/losses/single_loss/results/no_{component}

# Run the training with ablated loss component
torchrun --nproc_per_node=1 main.py \\
    --dataset ntu \\
    --setting cv \\
    --batch-size 128 \\
    --lr 1e-5 \\
    --epochs 40 \\
    --train-samples 10000 \\
    --test-samples 100 \\
    --teacher-forcing-ratio 1.0 \\
    --teacher-forcing-decay 0.01 \\
    --data-path data/ntu_cv_paired_comprehensive.pt \\
    --output-model-path experiments/losses/single_loss/results/no_{component}/model.pth \\
    --run-eval \\
    --hpc \\
    --loss-{component.replace('_', '-')} 0.0

# Run repeated evaluation for better statistics
python experiments/repeat/run_repeated_eval.py \\
    --dataset ntu \\
    --setting cv \\
    --eval_model sgn \\
    --num_runs 5 \\
    --test_samples 100 \\
    --transformer_model_path experiments/losses/single_loss/results/no_{component}/model.pth \\
    --output_dir experiments/losses/single_loss/results/no_{component}
"""

            script_path = scripts_dir / f"run_no_{component}_ablation.bash"
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable
            script_path.chmod(0o755)

        # Generate baseline model script (all losses enabled)
        baseline_script = f"""#!/bin/bash
#
#SBATCH --job-name="baseline-full"
#SBATCH --partition=GPU
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#
#   ===== Baseline Model (All Losses) =====

# Load PyTorch module
module load pytorch/2.3.0-cuda12.1

# Print CUDA availability information
echo "Is CUDA Available?"
python -c 'import torch; print(torch.cuda.is_available())'
echo ""
echo "nvidia-smi output:"
nvidia-smi

# Create output directory
mkdir -p experiments/losses/baseline/results

# Run the training with all loss components (baseline)
torchrun --nproc_per_node=1 main.py \\
    --dataset ntu \\
    --setting cv \\
    --batch-size 128 \\
    --lr 1e-5 \\
    --epochs 40 \\
    --train-samples 10000 \\
    --test-samples 100 \\
    --teacher-forcing-ratio 1.0 \\
    --teacher-forcing-decay 0.01 \\
    --data-path data/ntu_cv_paired_comprehensive.pt \\
    --output-model-path experiments/losses/baseline/results/model.pth \\
    --run-eval \\
    --hpc \\
    --loss-mse 7.0 \\
    --loss-ee 5.0 \\
    --loss-smoothing 0.075 \\
    --loss-inception 0.05 \\
    --loss-fid-vel 1.0 \\
    --loss-bone 10.0 \\
    --loss-foot 3.0 \\
    --loss-joint-limit 1.0

# Run repeated evaluation for better statistics
python experiments/repeat/run_repeated_eval.py \\
    --dataset ntu \\
    --setting cv \\
    --eval_model sgn \\
    --num_runs 5 \\
    --test_samples 100 \\
    --transformer_model_path experiments/losses/baseline/results/model.pth \\
    --output_dir experiments/losses/baseline/results
"""

        baseline_path = scripts_dir / "run_baseline_full.bash"
        with open(baseline_path, 'w') as f:
            f.write(baseline_script)
        baseline_path.chmod(0o755)

        # Generate master script to run all ablations
        master_script = f"""#!/bin/bash
#
# Master script to run all loss component ablation experiments
#

echo "🚀 Starting Loss Component Ablation Studies"
echo "=================================================="

# Submit baseline job first
echo "Submitting baseline (all losses) job..."
sbatch {scripts_dir}/run_baseline_full.bash

# Submit all individual ablation jobs
"""

        for component in self.loss_components.keys():
            master_script += f'echo "Submitting no_{component} ablation..."\n'
            master_script += f'sbatch {scripts_dir}/run_no_{component}_ablation.bash\n\n'

        master_script += 'echo "✅ All ablation jobs submitted!"\n'
        master_script += f'echo "📊 Total jobs: {len(self.loss_components) + 1} (1 baseline + {len(self.loss_components)} ablations)"\n'

        master_path = scripts_dir / "run_all_ablations.bash"
        with open(master_path, 'w') as f:
            f.write(master_script)
        master_path.chmod(0o755)

        print(f"✅ Generated {len(self.loss_components)} ablation scripts + 1 baseline script")
        print(f"📁 Scripts saved to: {scripts_dir}")
        print(f"🚀 Run all experiments with: bash {master_path}")
        print(f"📊 Total training time estimate: ~{(len(self.loss_components) + 1) * 8} hours")
        print(f"💾 Each experiment uses: 10k train samples, 100 test samples, 40 epochs")
        print(f"🔧 Configuration: 1 GPU, 64GB RAM, 24h time limit per job")

        return scripts_dir

    def run_analysis(self):
        """Run complete loss component ablation analysis."""
        print("🚀 Starting Loss Component Ablation Analysis")
        print("=" * 50)

        # Check for existing results
        available_results = self.check_ablation_results()

        if available_results:
            # Analyze existing results
            df = self.load_ablation_results(available_results)
            if df is not None:
                self.create_ablation_impact_plot(df)

                # Save results
                df.to_csv(self.results_dir / "ablation_results.csv", index=False)
                print(f"✅ Results saved to {self.results_dir}")
        else:
            print("❌ No ablation results found")

        # Generate scripts for future experiments
        self.generate_ablation_scripts()

        return available_results


def main():
    """Main execution function."""
    analyzer = LossComponentAblationAnalyzer()
    results = analyzer.run_analysis()

    if not results:
        print("\n📝 Next Steps:")
        print("1. Run the generated ablation scripts to train models without specific loss components")
        print("2. Re-run this analysis after experiments complete")
        print("3. Generated scripts are ready to submit to SLURM")

    return results


if __name__ == "__main__":
    main()
