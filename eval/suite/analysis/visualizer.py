"""
Comprehensive visualization module for creating publication-ready plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging


class ComprehensiveVisualizer:
    """
    Creates comprehensive visualizations for evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config.get('visualization', {})
        self.logger = logging.getLogger(__name__)
        
        # Set style
        style = self.config.get('default_style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            
        # Set default parameters
        self.figure_size = self.config.get('figure_size', [12, 8])
        self.dpi = self.config.get('dpi', 300)
        self.formats = self.config.get('formats', ['png', 'pdf'])
        
        # Color scheme
        self.colors = self.config.get('colors', {
            'raw': '#1f77b4',
            'transformer': '#ff7f0e',
            'dmr': '#2ca02c',
            'pmr': '#d62728'
        })
        
    def create_privacy_utility_plot(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Create privacy vs utility tradeoff plot."""
        output_paths = []
        
        try:
            # Extract data for plotting
            models = []
            ar_accuracy = []
            ri_accuracy = []
            mse_values = []
            
            for model_name, model_results in results.items():
                if model_name == 'aggregate':
                    continue
                    
                models.append(model_name)
                
                # Get metrics (with fallback values)
                ar_acc = 0
                ri_acc = 0
                mse_val = 0
                
                for data_name, data_results in model_results.items():
                    if 'action_recognition' in data_results:
                        ar_acc = data_results['action_recognition'].get('accuracy', 0)
                    if 'reidentification' in data_results:
                        ri_acc = data_results['reidentification'].get('identity_accuracy', 0)
                    if 'physical_plausibility' in data_results:
                        mse_val = data_results['physical_plausibility'].get('mse', {}).get('mean', 0)
                        
                ar_accuracy.append(ar_acc)
                ri_accuracy.append(ri_acc)
                mse_values.append(mse_val)
                
            if not models:
                self.logger.warning("No data available for privacy-utility plot")
                return output_paths
                
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Privacy vs Utility scatter
            colors = [self.colors.get(model, '#333333') for model in models]
            scatter = ax1.scatter(ri_accuracy, ar_accuracy, c=colors, s=100, alpha=0.7)
            
            # Add model labels
            for i, model in enumerate(models):
                ax1.annotate(model, (ri_accuracy[i], ar_accuracy[i]), 
                           xytext=(5, 5), textcoords='offset points')
                           
            ax1.set_xlabel('Re-identification Accuracy (%) - Lower is Better for Privacy')
            ax1.set_ylabel('Action Recognition Accuracy (%) - Higher is Better for Utility')
            ax1.set_title('Privacy vs Utility Tradeoff')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: MSE comparison
            bars = ax2.bar(models, mse_values, color=colors, alpha=0.7)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('MSE (Lower is Better)')
            ax2.set_title('Reconstruction Quality (MSE)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, mse_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
                        
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in self.formats:
                output_path = output_dir / f"privacy_utility_tradeoff.{fmt}"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                output_paths.append(output_path)
                
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating privacy-utility plot: {str(e)}")
            
        return output_paths
        
    def create_metrics_comparison_heatmap(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Create heatmap comparing all metrics across models."""
        output_paths = []
        
        try:
            # Prepare data for heatmap
            metrics_data = []
            models = []
            metric_names = []
            
            for model_name, model_results in results.items():
                if model_name == 'aggregate':
                    continue
                    
                models.append(model_name)
                model_metrics = []
                
                # Collect all metrics for this model
                for data_name, data_results in model_results.items():
                    # Action Recognition
                    if 'action_recognition' in data_results:
                        model_metrics.append(data_results['action_recognition'].get('accuracy', 0))
                        if len(metric_names) < len(model_metrics):
                            metric_names.append('AR Accuracy')
                            
                    # Re-identification
                    if 'reidentification' in data_results:
                        model_metrics.append(100 - data_results['reidentification'].get('identity_accuracy', 0))
                        if len(metric_names) < len(model_metrics):
                            metric_names.append('Anonymization Rate')
                            
                    # Physical metrics
                    if 'physical_plausibility' in data_results:
                        phys_metrics = data_results['physical_plausibility']
                        
                        if 'bone_length_consistency' in phys_metrics:
                            model_metrics.append(phys_metrics['bone_length_consistency'].get('mean', 0))
                            if len(metric_names) < len(model_metrics):
                                metric_names.append('Bone Length Consistency')
                                
                        if 'temporal_smoothness' in phys_metrics:
                            model_metrics.append(phys_metrics['temporal_smoothness'].get('mean', 0))
                            if len(metric_names) < len(model_metrics):
                                metric_names.append('Temporal Smoothness')
                                
                        if 'velocity_consistency' in phys_metrics:
                            model_metrics.append(phys_metrics['velocity_consistency'].get('mean', 0))
                            if len(metric_names) < len(model_metrics):
                                metric_names.append('Velocity Consistency')
                                
                    break  # Only use first dataset for now
                    
                metrics_data.append(model_metrics)
                
            if not metrics_data:
                self.logger.warning("No data available for metrics heatmap")
                return output_paths
                
            # Ensure all rows have the same length
            max_len = max(len(row) for row in metrics_data) if metrics_data else 0
            for row in metrics_data:
                while len(row) < max_len:
                    row.append(0)
                    
            # Create DataFrame
            df = pd.DataFrame(metrics_data, index=models, columns=metric_names[:max_len])
            
            # Create heatmap
            plt.figure(figsize=self.figure_size)
            sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=50, 
                       fmt='.2f', cbar_kws={'label': 'Score'})
            plt.title('Comprehensive Metrics Comparison')
            plt.xlabel('Metrics')
            plt.ylabel('Models')
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in self.formats:
                output_path = output_dir / f"metrics_comparison_heatmap.{fmt}"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                output_paths.append(output_path)
                
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating metrics heatmap: {str(e)}")
            
        return output_paths
        
    def create_physical_metrics_radar(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Create radar chart for physical plausibility metrics."""
        output_paths = []
        
        try:
            # Extract physical metrics
            models = []
            metrics_data = {}
            
            for model_name, model_results in results.items():
                if model_name == 'aggregate':
                    continue
                    
                models.append(model_name)
                
                for data_name, data_results in model_results.items():
                    if 'physical_plausibility' in data_results:
                        phys_metrics = data_results['physical_plausibility']
                        
                        # Collect metrics (normalize to 0-100 scale)
                        model_data = {}
                        
                        if 'bone_length_consistency' in phys_metrics:
                            model_data['BLC'] = phys_metrics['bone_length_consistency'].get('mean', 0) * 100
                            
                        if 'temporal_smoothness' in phys_metrics:
                            model_data['TS'] = phys_metrics['temporal_smoothness'].get('mean', 0) * 100
                            
                        if 'velocity_consistency' in phys_metrics:
                            model_data['VC'] = phys_metrics['velocity_consistency'].get('mean', 0) * 100
                            
                        if 'joint_angle_limits' in phys_metrics:
                            # Convert violation rate to compliance rate
                            violation_rate = phys_metrics['joint_angle_limits'].get('mean', 0)
                            model_data['JAL'] = (1 - violation_rate) * 100
                            
                        if 'foot_contact_consistency' in phys_metrics:
                            model_data['FCC'] = phys_metrics['foot_contact_consistency'].get('mean', 0) * 100
                            
                        metrics_data[model_name] = model_data
                        break
                        
            if not metrics_data:
                self.logger.warning("No physical metrics data available for radar chart")
                return output_paths
                
            # Get all metric names
            all_metrics = set()
            for model_data in metrics_data.values():
                all_metrics.update(model_data.keys())
            all_metrics = sorted(list(all_metrics))
            
            if not all_metrics:
                return output_paths
                
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for model_name in models:
                if model_name in metrics_data:
                    values = []
                    for metric in all_metrics:
                        values.append(metrics_data[model_name].get(metric, 0))
                    values += values[:1]  # Complete the circle
                    
                    color = self.colors.get(model_name, '#333333')
                    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
                    ax.fill(angles, values, alpha=0.25, color=color)
                    
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(all_metrics)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Score (0-100)', labelpad=30)
            ax.set_title('Physical Plausibility Metrics', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save in multiple formats
            for fmt in self.formats:
                output_path = output_dir / f"physical_metrics_radar.{fmt}"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                output_paths.append(output_path)
                
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating radar chart: {str(e)}")
            
        return output_paths
        
    def create_summary_table(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """Create summary table of all results."""
        try:
            # Prepare data for table
            table_data = []
            
            for model_name, model_results in results.items():
                if model_name == 'aggregate':
                    continue
                    
                row = {'Model': model_name}
                
                for data_name, data_results in model_results.items():
                    # Action Recognition
                    if 'action_recognition' in data_results:
                        row['AR Accuracy (%)'] = f"{data_results['action_recognition'].get('accuracy', 0):.2f}"
                        
                    # Re-identification
                    if 'reidentification' in data_results:
                        ri_acc = data_results['reidentification'].get('identity_accuracy', 0)
                        row['RI Accuracy (%)'] = f"{ri_acc:.2f}"
                        row['Anonymization Rate (%)'] = f"{100 - ri_acc:.2f}"
                        
                    # Physical metrics
                    if 'physical_plausibility' in data_results:
                        phys_metrics = data_results['physical_plausibility']
                        
                        if 'mse' in phys_metrics:
                            row['MSE'] = f"{phys_metrics['mse'].get('mean', 0):.4f}"
                            
                        if 'bone_length_consistency' in phys_metrics:
                            row['BLC'] = f"{phys_metrics['bone_length_consistency'].get('mean', 0):.3f}"
                            
                        if 'temporal_smoothness' in phys_metrics:
                            row['TS'] = f"{phys_metrics['temporal_smoothness'].get('mean', 0):.3f}"
                            
                    break  # Only use first dataset
                    
                table_data.append(row)
                
            # Create DataFrame and save as CSV
            df = pd.DataFrame(table_data)
            output_path = output_dir / "summary_table.csv"
            df.to_csv(output_path, index=False)
            
            # Also save as HTML for better formatting
            html_path = output_dir / "summary_table.html"
            df.to_html(html_path, index=False, table_id="results_table", 
                      classes="table table-striped table-hover")
                      
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating summary table: {str(e)}")
            return None
            
    def generate_all_visualizations(self, results: Dict[str, Any], output_dir: Path) -> Dict[str, List[Path]]:
        """Generate all visualizations for the results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_outputs = {}
        
        # Privacy-utility plot
        all_outputs['privacy_utility'] = self.create_privacy_utility_plot(results, output_dir)
        
        # Metrics comparison heatmap
        all_outputs['metrics_heatmap'] = self.create_metrics_comparison_heatmap(results, output_dir)
        
        # Physical metrics radar
        all_outputs['physical_radar'] = self.create_physical_metrics_radar(results, output_dir)
        
        # Summary table
        table_path = self.create_summary_table(results, output_dir)
        if table_path:
            all_outputs['summary_table'] = [table_path]
            
        self.logger.info(f"Generated visualizations in {output_dir}")
        return all_outputs
