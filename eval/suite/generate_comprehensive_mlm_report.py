#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate comprehensive reports for MLM evaluation including both classification and physical plausibility metrics.

This script creates publication-quality reports, tables, and visualizations for the complete MLM evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_comprehensive_results(results_dir, dataset, setting):
    """Load all comprehensive evaluation results from JSON files."""
    results = []
    
    temporal_ratios = [0.3, 0.5, 0.7]
    spatial_ratios = [0.3, 0.5, 0.7]
    
    for temporal_ratio in temporal_ratios:
        for spatial_ratio in spatial_ratios:
            filename = f"{dataset}_{setting}_temporal_{temporal_ratio}_spatial_{spatial_ratio}_comprehensive.json"
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
                print(f"Loaded: {filename}")
            else:
                print(f"Warning: Missing results file: {filename}")
    
    return results


def create_comprehensive_dataframe(results):
    """Convert comprehensive results to pandas DataFrame."""
    data = []
    
    for result in results:
        classification = result['results']['classification']
        plausibility = result['results']['physical_plausibility']
        
        row = {
            'temporal_ratio': result['temporal_ratio'],
            'spatial_ratio': result['spatial_ratio'],
            # Classification metrics
            'ar_accuracy': classification['action_recognition']['accuracy'],
            'ar_f1_score': classification['action_recognition']['f1_score'],
            'ri_accuracy': classification['re_identification']['accuracy'],
            'ri_f1_score': classification['re_identification']['f1_score'],
            # Physical plausibility metrics
            'reconstruction_mse': plausibility['reconstruction_mse'],
            'bone_length_consistency': plausibility['bone_length_consistency'],
            'joint_angle_violation': plausibility['joint_angle_violation'],
            'temporal_smoothness': plausibility['temporal_smoothness'],
            'velocity_consistency': plausibility['velocity_consistency'],
            'foot_contact_consistency': plausibility['foot_contact_consistency'],
            'fid_score': plausibility['fid_score']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_comprehensive_heatmaps(df, output_dir, dataset, setting):
    """Create heatmaps for all metrics."""
    
    # Define metric groups
    classification_metrics = {
        'ar_accuracy': 'Action Recognition Accuracy',
        'ar_f1_score': 'Action Recognition F1-Score',
        'ri_accuracy': 'Re-Identification Accuracy',
        'ri_f1_score': 'Re-Identification F1-Score'
    }
    
    plausibility_metrics = {
        'reconstruction_mse': 'Reconstruction MSE',
        'bone_length_consistency': 'Bone Length Consistency',
        'joint_angle_violation': 'Joint Angle Violation',
        'temporal_smoothness': 'Temporal Smoothness',
        'velocity_consistency': 'Velocity Consistency',
        'foot_contact_consistency': 'Foot Contact Consistency',
        'fid_score': 'FID Score'
    }
    
    all_metrics = {**classification_metrics, **plausibility_metrics}
    
    # Create individual heatmaps
    for metric, title in all_metrics.items():
        plt.figure(figsize=(10, 8))
        
        # Create pivot table
        pivot_data = df.pivot(index='temporal_ratio', columns='spatial_ratio', values=metric)
        
        # Choose colormap based on metric type
        if 'mse' in metric.lower() or 'violation' in metric.lower():
            cmap = 'viridis_r'  # Lower is better
        else:
            cmap = 'viridis'    # Higher is better
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.4f' if 'mse' in metric.lower() else '.3f', 
            cmap=cmap,
            cbar_kws={'label': title},
            square=True,
            linewidths=0.5
        )
        
        plt.title(f'{title}\n{dataset.upper()} {setting.upper()} - MLM Comprehensive Evaluation', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Spatial Masking Ratio', fontsize=14)
        plt.ylabel('Temporal Masking Ratio', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        filename = f'{metric}_heatmap_{dataset}_{setting}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {filename}")


def create_correlation_analysis(df, output_dir, dataset, setting):
    """Create correlation analysis between classification and plausibility metrics."""
    
    # Select relevant columns for correlation
    correlation_cols = [
        'ar_accuracy', 'ri_accuracy', 'reconstruction_mse', 
        'bone_length_consistency', 'temporal_smoothness', 
        'velocity_consistency', 'fid_score'
    ]
    
    corr_df = df[correlation_cols]
    
    # Calculate correlation matrix
    correlation_matrix = corr_df.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    
    plt.title(f'Metric Correlation Analysis\n{dataset.upper()} {setting.upper()}', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'correlation_analysis_{dataset}_{setting}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation analysis: {filename}")
    
    return correlation_matrix


def create_comprehensive_summary_plot(df, output_dir, dataset, setting):
    """Create a comprehensive summary plot with multiple subplots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MLM Comprehensive Evaluation Summary\n{dataset.upper()} {setting.upper()}', 
                 fontsize=20, fontweight='bold')
    
    # Configuration labels
    config_labels = [f"T{row['temporal_ratio']}_S{row['spatial_ratio']}" 
                    for _, row in df.iterrows()]
    
    # Plot 1: Classification Accuracy
    axes[0, 0].bar(range(len(df)), df['ar_accuracy'], alpha=0.7, label='Action Recognition')
    axes[0, 0].bar(range(len(df)), df['ri_accuracy'], alpha=0.7, label='Re-Identification')
    axes[0, 0].set_title('Classification Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(config_labels, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Reconstruction Quality
    axes[0, 1].plot(range(len(df)), df['reconstruction_mse'], 'o-', linewidth=2, markersize=6)
    axes[0, 1].set_title('Reconstruction MSE', fontweight='bold')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(config_labels, rotation=45)
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Physical Plausibility
    axes[0, 2].plot(range(len(df)), df['bone_length_consistency'], 'o-', label='Bone Length', linewidth=2)
    axes[0, 2].plot(range(len(df)), df['temporal_smoothness'], 's-', label='Temporal Smoothness', linewidth=2)
    axes[0, 2].set_title('Physical Plausibility', fontweight='bold')
    axes[0, 2].set_ylabel('Metric Value')
    axes[0, 2].set_xticks(range(len(df)))
    axes[0, 2].set_xticklabels(config_labels, rotation=45)
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Velocity and Motion Consistency
    axes[1, 0].plot(range(len(df)), df['velocity_consistency'], '^-', label='Velocity Consistency', linewidth=2)
    axes[1, 0].plot(range(len(df)), df['foot_contact_consistency'], 'v-', label='Foot Contact', linewidth=2)
    axes[1, 0].set_title('Motion Consistency', fontweight='bold')
    axes[1, 0].set_ylabel('Consistency Score')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(config_labels, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: FID Score
    axes[1, 1].plot(range(len(df)), df['fid_score'], 'D-', color='red', linewidth=2, markersize=6)
    axes[1, 1].set_title('FID Score', fontweight='bold')
    axes[1, 1].set_ylabel('FID Score')
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels(config_labels, rotation=45)
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Overall Performance Radar (placeholder for now)
    axes[1, 2].text(0.5, 0.5, 'Overall\nPerformance\nSummary', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    
    filename = f'comprehensive_summary_{dataset}_{setting}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive summary: {filename}")


def create_comprehensive_tables(df, output_dir, dataset, setting):
    """Create comprehensive tables for all metrics."""
    
    # Round values for better presentation
    display_df = df.copy()
    display_df = display_df.round(4)
    
    # Add configuration column
    display_df['config'] = display_df.apply(
        lambda row: f"T{row['temporal_ratio']}_S{row['spatial_ratio']}", axis=1
    )
    
    # Reorder columns for better presentation
    column_order = [
        'config', 'temporal_ratio', 'spatial_ratio',
        'ar_accuracy', 'ar_f1_score', 'ri_accuracy', 'ri_f1_score',
        'reconstruction_mse', 'bone_length_consistency', 'joint_angle_violation',
        'temporal_smoothness', 'velocity_consistency', 'foot_contact_consistency', 'fid_score'
    ]
    
    display_df = display_df[column_order]
    
    # Save comprehensive results table
    csv_filename = f'comprehensive_results_{dataset}_{setting}.csv'
    display_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    
    # Create summary statistics
    numeric_cols = [col for col in display_df.columns if col not in ['config', 'temporal_ratio', 'spatial_ratio']]
    summary_stats = display_df[numeric_cols].describe()
    
    summary_filename = f'summary_statistics_{dataset}_{setting}.csv'
    summary_stats.to_csv(os.path.join(output_dir, summary_filename))
    
    print(f"Saved comprehensive table: {csv_filename}")
    print(f"Saved summary statistics: {summary_filename}")
    
    return display_df, summary_stats


def find_best_configurations(df):
    """Find best performing configurations for each metric."""
    best_configs = {}
    
    # Metrics where higher is better
    higher_better = ['ar_accuracy', 'ar_f1_score', 'ri_accuracy', 'ri_f1_score', 
                    'bone_length_consistency', 'temporal_smoothness', 
                    'velocity_consistency', 'foot_contact_consistency']
    
    # Metrics where lower is better
    lower_better = ['reconstruction_mse', 'joint_angle_violation', 'fid_score']
    
    for metric in higher_better:
        best_idx = df[metric].idxmax()
        best_row = df.iloc[best_idx]
        best_configs[metric] = {
            'temporal_ratio': best_row['temporal_ratio'],
            'spatial_ratio': best_row['spatial_ratio'],
            'value': best_row[metric],
            'better': 'higher'
        }
    
    for metric in lower_better:
        best_idx = df[metric].idxmin()
        best_row = df.iloc[best_idx]
        best_configs[metric] = {
            'temporal_ratio': best_row['temporal_ratio'],
            'spatial_ratio': best_row['spatial_ratio'],
            'value': best_row[metric],
            'better': 'lower'
        }
    
    return best_configs


def generate_comprehensive_report(results, df, best_configs, correlation_matrix, output_dir, dataset, setting):
    """Generate comprehensive text report."""
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"COMPREHENSIVE MLM EVALUATION REPORT")
    report_lines.append(f"Dataset: {dataset.upper()}")
    report_lines.append(f"Setting: {setting.upper()}")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 50)
    report_lines.append(f"Total configurations evaluated: {len(df)}")
    report_lines.append(f"Evaluation combines feature-based classification and physical plausibility metrics")
    report_lines.append("")
    
    # Classification Performance Summary
    report_lines.append("CLASSIFICATION PERFORMANCE")
    report_lines.append("-" * 50)
    report_lines.append(f"Action Recognition Accuracy: {df['ar_accuracy'].mean():.3f} ± {df['ar_accuracy'].std():.3f}")
    report_lines.append(f"Action Recognition F1-Score: {df['ar_f1_score'].mean():.3f} ± {df['ar_f1_score'].std():.3f}")
    report_lines.append(f"Re-Identification Accuracy: {df['ri_accuracy'].mean():.3f} ± {df['ri_accuracy'].std():.3f}")
    report_lines.append(f"Re-Identification F1-Score: {df['ri_f1_score'].mean():.3f} ± {df['ri_f1_score'].std():.3f}")
    report_lines.append("")
    
    # Physical Plausibility Summary
    report_lines.append("PHYSICAL PLAUSIBILITY")
    report_lines.append("-" * 50)
    report_lines.append(f"Reconstruction MSE: {df['reconstruction_mse'].mean():.6f} ± {df['reconstruction_mse'].std():.6f}")
    report_lines.append(f"Bone Length Consistency: {df['bone_length_consistency'].mean():.6f} ± {df['bone_length_consistency'].std():.6f}")
    report_lines.append(f"Temporal Smoothness: {df['temporal_smoothness'].mean():.6f} ± {df['temporal_smoothness'].std():.6f}")
    report_lines.append(f"Velocity Consistency: {df['velocity_consistency'].mean():.6f} ± {df['velocity_consistency'].std():.6f}")
    report_lines.append(f"FID Score: {df['fid_score'].mean():.6f} ± {df['fid_score'].std():.6f}")
    report_lines.append("")
    
    # Best Configurations
    report_lines.append("BEST PERFORMING CONFIGURATIONS")
    report_lines.append("-" * 50)
    for metric, config in best_configs.items():
        metric_name = metric.replace('_', ' ').title()
        direction = "↑" if config['better'] == 'higher' else "↓"
        report_lines.append(
            f"{metric_name}: T{config['temporal_ratio']}_S{config['spatial_ratio']} "
            f"({config['value']:.4f}) {direction}"
        )
    report_lines.append("")
    
    # Key Insights
    report_lines.append("KEY INSIGHTS")
    report_lines.append("-" * 50)
    
    # Find correlations
    ar_recon_corr = correlation_matrix.loc['ar_accuracy', 'reconstruction_mse']
    ri_recon_corr = correlation_matrix.loc['ri_accuracy', 'reconstruction_mse']
    
    report_lines.append(f"• AR Accuracy vs Reconstruction MSE correlation: {ar_recon_corr:.3f}")
    report_lines.append(f"• RI Accuracy vs Reconstruction MSE correlation: {ri_recon_corr:.3f}")
    
    # Best overall configuration (balanced performance)
    df['combined_score'] = (df['ar_accuracy'] + df['ri_accuracy']) / 2 - df['reconstruction_mse'] * 100
    best_overall_idx = df['combined_score'].idxmax()
    best_overall = df.iloc[best_overall_idx]
    
    report_lines.append(f"• Best overall configuration: T{best_overall['temporal_ratio']}_S{best_overall['spatial_ratio']}")
    report_lines.append(f"  (Combined score: {best_overall['combined_score']:.3f})")
    report_lines.append("")
    
    report_lines.append("=" * 100)
    
    # Save report
    report_filename = f'comprehensive_evaluation_report_{dataset}_{setting}.txt'
    with open(os.path.join(output_dir, report_filename), 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved comprehensive report: {report_filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Comprehensive MLM Reports')
    
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing comprehensive evaluation results')
    parser.add_argument('--dataset', type=str, default='ntu',
                        help='Dataset name')
    parser.add_argument('--setting', type=str, default='cv',
                        help='Evaluation setting')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for reports')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating comprehensive MLM evaluation report...")
    print(f"Dataset: {args.dataset}")
    print(f"Setting: {args.setting}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Load results
    print("Loading comprehensive evaluation results...")
    results = load_comprehensive_results(args.results_dir, args.dataset, args.setting)
    
    if not results:
        print("Error: No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create DataFrame
    df = create_comprehensive_dataframe(results)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Find best configurations
    best_configs = find_best_configurations(df)
    
    # Generate visualizations
    print("\nGenerating comprehensive visualizations...")
    create_comprehensive_heatmaps(df, args.output_dir, args.dataset, args.setting)
    correlation_matrix = create_correlation_analysis(df, args.output_dir, args.dataset, args.setting)
    create_comprehensive_summary_plot(df, args.output_dir, args.dataset, args.setting)
    
    # Generate tables
    print("\nGenerating comprehensive tables...")
    display_df, summary_stats = create_comprehensive_tables(df, args.output_dir, args.dataset, args.setting)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    generate_comprehensive_report(results, df, best_configs, correlation_matrix, args.output_dir, args.dataset, args.setting)
    
    print(f"\nComprehensive report generation completed!")
    print(f"All outputs saved to: {args.output_dir}")
    
    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Best AR Accuracy: T{best_configs['ar_accuracy']['temporal_ratio']}_S{best_configs['ar_accuracy']['spatial_ratio']} ({best_configs['ar_accuracy']['value']:.3f})")
    print(f"Best RI Accuracy: T{best_configs['ri_accuracy']['temporal_ratio']}_S{best_configs['ri_accuracy']['spatial_ratio']} ({best_configs['ri_accuracy']['value']:.3f})")
    print(f"Best Reconstruction: T{best_configs['reconstruction_mse']['temporal_ratio']}_S{best_configs['reconstruction_mse']['spatial_ratio']} ({best_configs['reconstruction_mse']['value']:.6f})")
    print(f"Mean AR Accuracy: {df['ar_accuracy'].mean():.3f} ± {df['ar_accuracy'].std():.3f}")
    print(f"Mean RI Accuracy: {df['ri_accuracy'].mean():.3f} ± {df['ri_accuracy'].std():.3f}")


if __name__ == "__main__":
    main()
