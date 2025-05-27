"""
Method Comparison & Loss Ablation Integration Script

This script integrates method comparison results (Transformer vs DMR vs PMR vs Raw)
and provides framework for loss component ablation analysis.

Current data structure:
- loss_ablation_analysis/: Method comparison results (Transformer, DMR, PMR, Raw)
- experimentsold/losses/: Loss component ablation scripts (not yet run)
- slurm_out/experiments/ablations/: Training logs (method comparison)
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

class MethodComparisonIntegrator:
    """Integrates method comparison results and provides loss ablation framework."""

    def __init__(self, base_dir="/users/tcarr23/Transformer-Retargeting"):
        self.base_dir = Path(base_dir)
        self.method_comparison_dir = self.base_dir / "loss_ablation_analysis"
        self.loss_ablation_dir = self.base_dir / "experimentsold" / "losses"
        self.eval_suite_dir = self.base_dir / "evaluation_suite"
        self.results_dir = self.eval_suite_dir / "results" / "experiments" / "method_comparison"
        self.plots_dir = self.results_dir / "plots"

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_method_comparison_data(self):
        """Load method comparison data (Transformer vs DMR vs PMR vs Raw)."""
        print("🔍 Loading method comparison data...")

        # Load summary data
        summary_file = self.method_comparison_dir / "summary_table.csv"
        detailed_file = self.method_comparison_dir / "detailed_metrics_table.csv"
        best_params_file = self.method_comparison_dir / "best_parameters.csv"

        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        self.summary_data = pd.read_csv(summary_file)

        if detailed_file.exists():
            self.detailed_data = pd.read_csv(detailed_file)
        else:
            print("⚠️  Detailed metrics file not found, using summary only")
            self.detailed_data = None

        if best_params_file.exists():
            self.best_params = pd.read_csv(best_params_file)
        else:
            print("⚠️  Best parameters file not found")
            self.best_params = None

        print(f"✅ Loaded data for {len(self.summary_data)} methods")
        return self.summary_data

    def create_performance_comparison_plot(self):
        """Create comprehensive performance comparison plot."""
        print("📊 Creating performance comparison plot...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Method Comparison: Performance Analysis', fontsize=16, fontweight='bold')

        # Define metrics and their display names
        metrics = [
            ('accuracy', 'Action Recognition Accuracy (%)'),
            ('identity_accuracy', 'Re-identification Accuracy (%)'),
            ('mse', 'Mean Squared Error'),
            ('bone_length_error', 'Bone Length Error'),
            ('foot_contact_error', 'Foot Contact Error'),
            ('privacy_utility_score', 'Privacy-Utility Score')
        ]

        # Create color palette
        colors = sns.color_palette("husl", len(self.summary_data))
        model_colors = dict(zip(self.summary_data['model_name'], colors))

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            # Create bar plot
            bars = ax.bar(range(len(self.summary_data)),
                         self.summary_data[metric],
                         color=[model_colors[name] for name in self.summary_data['model_name']])

            # Customize plot
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xticks(range(len(self.summary_data)))
            ax.set_xticklabels([name.replace(' ', '\n') for name in self.summary_data['model_name']],
                              rotation=0, ha='center', fontsize=10)

            # Add value labels on bars
            for bar, value in zip(bars, self.summary_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)

            # Highlight best performance (higher is better for accuracy and privacy score, lower for errors)
            if metric in ['accuracy', 'identity_accuracy', 'privacy_utility_score']:
                best_idx = self.summary_data[metric].idxmax()
            else:
                best_idx = self.summary_data[metric].idxmin()
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Performance comparison plot saved to {plot_path}")
        return plot_path

    def create_detailed_metrics_heatmap(self):
        """Create detailed metrics heatmap if detailed data is available."""
        if self.detailed_data is None:
            print("⚠️  Skipping detailed metrics heatmap - no detailed data available")
            return None

        print("🔥 Creating detailed metrics heatmap...")

        # Select numeric columns for heatmap
        numeric_cols = self.detailed_data.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['Unnamed: 0']]

        # Prepare data for heatmap
        heatmap_data = self.detailed_data[['model_name', 'eval_model'] + metric_cols].copy()

        # Create model-eval combination labels
        heatmap_data['model_eval'] = heatmap_data['model_name'] + ' + ' + heatmap_data['eval_model']

        # Pivot data for heatmap
        pivot_data = heatmap_data.set_index('model_eval')[metric_cols]

        # Normalize data for better visualization (0-1 scale)
        normalized_data = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min())

        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(normalized_data.T,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   square=False,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8})

        plt.title('Detailed Metrics Heatmap (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model + Evaluation Method', fontsize=12, fontweight='bold')
        plt.ylabel('Metrics', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plot_path = self.plots_dir / "detailed_metrics_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Detailed metrics heatmap saved to {plot_path}")
        return plot_path

    def create_privacy_utility_tradeoff_plot(self):
        """Create privacy-utility tradeoff analysis plot."""
        print("🎯 Creating privacy-utility tradeoff plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Accuracy vs Re-identification (Privacy-Utility Tradeoff)
        colors = sns.color_palette("husl", len(self.summary_data))

        scatter = ax1.scatter(self.summary_data['identity_accuracy'],
                            self.summary_data['accuracy'],
                            c=colors,
                            s=200,
                            alpha=0.7,
                            edgecolors='black',
                            linewidth=2)

        # Add model labels
        for idx, row in self.summary_data.iterrows():
            ax1.annotate(row['model_name'].replace(' ', '\n'),
                        (row['identity_accuracy'], row['accuracy']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')

        ax1.set_xlabel('Re-identification Accuracy (%) - Privacy Risk', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Action Recognition Accuracy (%) - Utility', fontsize=12, fontweight='bold')
        ax1.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add ideal region annotation
        ax1.axhspan(80, 100, alpha=0.1, color='green', label='High Utility')
        ax1.axvspan(0, 20, alpha=0.1, color='blue', label='High Privacy')
        ax1.legend()

        # Plot 2: Privacy-Utility Score comparison
        bars = ax2.bar(range(len(self.summary_data)),
                      self.summary_data['privacy_utility_score'],
                      color=colors)

        ax2.set_title('Privacy-Utility Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Privacy-Utility Score', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(self.summary_data)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in self.summary_data['model_name']],
                           rotation=0, ha='center')

        # Add value labels
        for bar, value in zip(bars, self.summary_data['privacy_utility_score']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "privacy_utility_tradeoff.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Privacy-utility tradeoff plot saved to {plot_path}")
        return plot_path

    def create_loss_component_impact_analysis(self):
        """Analyze the impact of each loss component."""
        print("🔬 Creating loss component impact analysis...")

        # Identify baseline (full model) and ablated models
        baseline_idx = self.summary_data[self.summary_data['model_name'].str.contains('Full|Baseline|Complete', case=False, na=False)].index
        if len(baseline_idx) == 0:
            # If no explicit baseline, use the first model
            baseline_idx = [0]
        baseline = self.summary_data.iloc[baseline_idx[0]]

        # Calculate impact of each loss component
        impact_data = []
        for idx, row in self.summary_data.iterrows():
            if idx == baseline_idx[0]:
                continue  # Skip baseline

            impact = {
                'model': row['model_name'],
                'accuracy_impact': baseline['accuracy'] - row['accuracy'],
                'identity_accuracy_impact': baseline['identity_accuracy'] - row['identity_accuracy'],
                'mse_impact': row['mse'] - baseline['mse'],
                'bone_length_impact': row['bone_length_error'] - baseline['bone_length_error'],
                'foot_contact_impact': row['foot_contact_error'] - baseline['foot_contact_error'],
                'privacy_utility_impact': baseline['privacy_utility_score'] - row['privacy_utility_score']
            }
            impact_data.append(impact)

        if not impact_data:
            print("⚠️  No ablated models found for impact analysis")
            return None

        impact_df = pd.DataFrame(impact_data)

        # Create impact visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Loss Component Impact Analysis\n(Positive = Performance Degradation)',
                    fontsize=16, fontweight='bold')

        impact_metrics = [
            ('accuracy_impact', 'Action Recognition Impact (%)'),
            ('identity_accuracy_impact', 'Re-identification Impact (%)'),
            ('mse_impact', 'MSE Impact'),
            ('bone_length_impact', 'Bone Length Error Impact'),
            ('foot_contact_impact', 'Foot Contact Error Impact'),
            ('privacy_utility_impact', 'Privacy-Utility Score Impact')
        ]

        for idx, (metric, title) in enumerate(impact_metrics):
            ax = axes[idx // 3, idx % 3]

            # Create horizontal bar plot for better readability
            bars = ax.barh(range(len(impact_df)), impact_df[metric])

            # Color bars based on impact (red for negative impact, green for positive)
            for bar, value in zip(bars, impact_df[metric]):
                if value > 0:
                    bar.set_color('red')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('green')
                    bar.set_alpha(0.7)

            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.set_yticks(range(len(impact_df)))
            ax.set_yticklabels([name.replace(' ', '\n') for name in impact_df['model']], fontsize=9)

            # Add value labels
            for i, value in enumerate(impact_df[metric]):
                ax.text(value + (0.01 * max(abs(impact_df[metric]))), i,
                       f'{value:.3f}', va='center', fontsize=9)

            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.plots_dir / "loss_component_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save impact analysis data
        impact_df.to_csv(self.results_dir / "loss_component_impact.csv", index=False)

        print(f"✅ Loss component impact analysis saved to {plot_path}")
        return plot_path, impact_df

    def create_comprehensive_table(self):
        """Create comprehensive results table."""
        print("📋 Creating comprehensive results table...")

        # Create formatted table
        table_data = self.summary_data.copy()

        # Round numeric columns for better presentation
        numeric_cols = table_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            table_data[col] = table_data[col].round(3)

        # Create LaTeX-style table
        latex_table = table_data.to_latex(index=False,
                                         caption="Loss Ablation Study Results",
                                         label="tab:loss_ablation",
                                         column_format='l' + 'c' * (len(table_data.columns) - 1))

        # Save tables
        table_data.to_csv(self.results_dir / "comprehensive_results.csv", index=False)

        with open(self.results_dir / "comprehensive_results.tex", 'w') as f:
            f.write(latex_table)

        # Create markdown table for easy viewing
        markdown_table = table_data.to_markdown(index=False, tablefmt="grid")

        with open(self.results_dir / "comprehensive_results.md", 'w') as f:
            f.write("# Loss Ablation Study Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(markdown_table)

        print(f"✅ Comprehensive table saved to {self.results_dir}")
        return table_data

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("📝 Generating summary report...")

        # Find best performing models for each metric
        best_models = {}
        metrics = ['accuracy', 'identity_accuracy', 'mse', 'bone_length_error',
                  'foot_contact_error', 'privacy_utility_score']

        for metric in metrics:
            if metric in ['accuracy', 'identity_accuracy', 'privacy_utility_score']:
                best_idx = self.summary_data[metric].idxmax()
            else:
                best_idx = self.summary_data[metric].idxmin()
            best_models[metric] = {
                'model': self.summary_data.iloc[best_idx]['model_name'],
                'value': self.summary_data.iloc[best_idx][metric]
            }

        # Create summary report
        report = f"""# Loss Ablation Study Summary Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Models Evaluated**: {len(self.summary_data)}
- **Evaluation Dataset**: NTU RGB+D Cross-View
- **Metrics Analyzed**: Action Recognition, Re-identification, Physical Plausibility

## Best Performing Models by Metric

### Action Recognition Accuracy
- **Best Model**: {best_models['accuracy']['model']}
- **Accuracy**: {best_models['accuracy']['value']:.1f}%

### Re-identification Accuracy (Privacy)
- **Best Model**: {best_models['identity_accuracy']['model']}
- **Accuracy**: {best_models['identity_accuracy']['value']:.1f}%

### Mean Squared Error
- **Best Model**: {best_models['mse']['model']}
- **MSE**: {best_models['mse']['value']:.4f}

### Bone Length Consistency
- **Best Model**: {best_models['bone_length_error']['model']}
- **Error**: {best_models['bone_length_error']['value']:.4f}

### Foot Contact Preservation
- **Best Model**: {best_models['foot_contact_error']['model']}
- **Error**: {best_models['foot_contact_error']['value']:.4f}

### Privacy-Utility Score
- **Best Model**: {best_models['privacy_utility_score']['model']}
- **Score**: {best_models['privacy_utility_score']['value']:.1f}

## Key Findings

### Performance Rankings
"""

        # Add performance rankings
        for metric in ['accuracy', 'privacy_utility_score']:
            if metric == 'accuracy':
                sorted_data = self.summary_data.sort_values(metric, ascending=False)
                report += f"\n#### Action Recognition Accuracy Rankings\n"
            else:
                sorted_data = self.summary_data.sort_values(metric, ascending=False)
                report += f"\n#### Privacy-Utility Score Rankings\n"

            for i, (_, row) in enumerate(sorted_data.iterrows(), 1):
                report += f"{i}. **{row['model_name']}**: {row[metric]:.1f}\n"

        # Save report
        with open(self.results_dir / "summary_report.md", 'w') as f:
            f.write(report)

        print(f"✅ Summary report saved to {self.results_dir / 'summary_report.md'}")
        return report

    def run_complete_analysis(self):
        """Run complete method comparison analysis."""
        print("🚀 Starting complete method comparison analysis...")

        # Load data
        self.load_method_comparison_data()

        # Generate all visualizations and analyses
        results = {}

        try:
            results['performance_plot'] = self.create_performance_comparison_plot()
        except Exception as e:
            print(f"❌ Error creating performance plot: {e}")

        try:
            results['heatmap_plot'] = self.create_detailed_metrics_heatmap()
        except Exception as e:
            print(f"❌ Error creating heatmap: {e}")

        try:
            results['tradeoff_plot'] = self.create_privacy_utility_tradeoff_plot()
        except Exception as e:
            print(f"❌ Error creating tradeoff plot: {e}")

        try:
            impact_plot, impact_data = self.create_loss_component_impact_analysis()
            results['impact_plot'] = impact_plot
            results['impact_data'] = impact_data
        except Exception as e:
            print(f"❌ Error creating impact analysis: {e}")

        try:
            results['table'] = self.create_comprehensive_table()
        except Exception as e:
            print(f"❌ Error creating table: {e}")

        try:
            results['report'] = self.generate_summary_report()
        except Exception as e:
            print(f"❌ Error generating report: {e}")

        print("✅ Complete method comparison analysis finished!")
        print(f"📁 Results saved to: {self.results_dir}")

        return results


def main():
    """Main execution function."""
    print("🔬 Method Comparison Analysis")
    print("=" * 50)

    integrator = MethodComparisonIntegrator()
    results = integrator.run_complete_analysis()

    print("\n📊 Analysis Complete!")
    print(f"📁 Results directory: {integrator.results_dir}")
    print(f"📈 Plots directory: {integrator.plots_dir}")

    return results


if __name__ == "__main__":
    main()
