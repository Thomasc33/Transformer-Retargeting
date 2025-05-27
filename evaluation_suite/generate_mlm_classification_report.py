#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate comprehensive reports and visualizations for MLM feature classification results.

This script analyzes results from all 9 masking ratio combinations and generates
publication-quality reports, tables, and visualizations.
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


def load_classification_results(results_dir, dataset, setting):
    """Load all classification results from JSON files."""
    results = []
    
    temporal_ratios = [0.3, 0.5, 0.7]
    spatial_ratios = [0.3, 0.5, 0.7]
    
    for temporal_ratio in temporal_ratios:
        for spatial_ratio in spatial_ratios:
            filename = f"{dataset}_{setting}_temporal_{temporal_ratio}_spatial_{spatial_ratio}_classification.json"
            filepath = os.path.join(results_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
                print(f"Loaded: {filename}")
            else:
                print(f"Warning: Missing results file: {filename}")
    
    return results


def create_results_dataframe(results):
    """Convert results to pandas DataFrame for analysis."""
    data = []
    
    for result in results:
        row = {
            'temporal_ratio': result['temporal_ratio'],
            'spatial_ratio': result['spatial_ratio'],
            'ar_accuracy': result['action_recognition']['accuracy'],
            'ar_f1_score': result['action_recognition']['f1_score'],
            'ri_accuracy': result['re_identification']['accuracy'],
            'ri_f1_score': result['re_identification']['f1_score']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_heatmaps(df, output_dir, dataset, setting):
    """Create heatmaps for all metrics."""
    metrics = {
        'ar_accuracy': 'Action Recognition Accuracy',
        'ar_f1_score': 'Action Recognition F1-Score',
        'ri_accuracy': 'Re-Identification Accuracy',
        'ri_f1_score': 'Re-Identification F1-Score'
    }
    
    # Create pivot tables for heatmaps
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 8))
        
        # Create pivot table
        pivot_data = df.pivot(index='temporal_ratio', columns='spatial_ratio', values=metric)
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='viridis',
            cbar_kws={'label': title},
            square=True,
            linewidths=0.5
        )
        
        plt.title(f'{title}\n{dataset.upper()} {setting.upper()} - MLM Feature Classification', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Spatial Masking Ratio', fontsize=14)
        plt.ylabel('Temporal Masking Ratio', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        filename = f'{metric}_heatmap_{dataset}_{setting}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {filename}")


def create_bar_plots(df, output_dir, dataset, setting):
    """Create bar plots comparing different masking strategies."""
    
    # AR vs RI Accuracy Comparison
    plt.figure(figsize=(14, 8))
    
    # Prepare data for grouped bar plot
    x_labels = [f"T{row['temporal_ratio']}_S{row['spatial_ratio']}" 
                for _, row in df.iterrows()]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    plt.bar(x - width/2, df['ar_accuracy'], width, label='Action Recognition', alpha=0.8)
    plt.bar(x + width/2, df['ri_accuracy'], width, label='Re-Identification', alpha=0.8)
    
    plt.xlabel('Masking Configuration (Temporal_Spatial)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'MLM Feature Classification Performance\n{dataset.upper()} {setting.upper()}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = f'accuracy_comparison_{dataset}_{setting}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bar plot: {filename}")


def create_summary_table(df, output_dir, dataset, setting):
    """Create summary table with statistics."""
    
    # Calculate summary statistics
    summary_stats = {
        'Metric': ['AR Accuracy', 'AR F1-Score', 'RI Accuracy', 'RI F1-Score'],
        'Mean': [
            df['ar_accuracy'].mean(),
            df['ar_f1_score'].mean(),
            df['ri_accuracy'].mean(),
            df['ri_f1_score'].mean()
        ],
        'Std': [
            df['ar_accuracy'].std(),
            df['ar_f1_score'].std(),
            df['ri_accuracy'].std(),
            df['ri_f1_score'].std()
        ],
        'Min': [
            df['ar_accuracy'].min(),
            df['ar_f1_score'].min(),
            df['ri_accuracy'].min(),
            df['ri_f1_score'].min()
        ],
        'Max': [
            df['ar_accuracy'].max(),
            df['ar_f1_score'].max(),
            df['ri_accuracy'].max(),
            df['ri_f1_score'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save as CSV
    csv_filename = f'summary_statistics_{dataset}_{setting}.csv'
    summary_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    
    # Create detailed results table
    detailed_df = df.copy()
    detailed_df = detailed_df.round(4)
    detailed_df['config'] = detailed_df.apply(
        lambda row: f"T{row['temporal_ratio']}_S{row['spatial_ratio']}", axis=1
    )
    
    # Reorder columns
    detailed_df = detailed_df[['config', 'temporal_ratio', 'spatial_ratio', 
                              'ar_accuracy', 'ar_f1_score', 'ri_accuracy', 'ri_f1_score']]
    
    detailed_csv_filename = f'detailed_results_{dataset}_{setting}.csv'
    detailed_df.to_csv(os.path.join(output_dir, detailed_csv_filename), index=False)
    
    print(f"Saved summary table: {csv_filename}")
    print(f"Saved detailed results: {detailed_csv_filename}")
    
    return summary_df, detailed_df


def find_best_configurations(df):
    """Find best performing configurations for each metric."""
    best_configs = {}
    
    metrics = ['ar_accuracy', 'ar_f1_score', 'ri_accuracy', 'ri_f1_score']
    
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_row = df.iloc[best_idx]
        best_configs[metric] = {
            'temporal_ratio': best_row['temporal_ratio'],
            'spatial_ratio': best_row['spatial_ratio'],
            'value': best_row[metric]
        }
    
    return best_configs


def generate_latex_table(df, output_dir, dataset, setting):
    """Generate LaTeX table for publication."""
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{MLM Feature Classification Results}")
    latex_content.append(f"\\label{{tab:mlm_classification_{dataset}_{setting}}}")
    latex_content.append("\\begin{tabular}{|c|c|c|c|c|c|}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Temporal} & \\textbf{Spatial} & \\textbf{AR Acc.} & \\textbf{AR F1} & \\textbf{RI Acc.} & \\textbf{RI F1} \\\\")
    latex_content.append("\\hline")
    
    for _, row in df.iterrows():
        latex_content.append(
            f"{row['temporal_ratio']:.1f} & {row['spatial_ratio']:.1f} & "
            f"{row['ar_accuracy']:.3f} & {row['ar_f1_score']:.3f} & "
            f"{row['ri_accuracy']:.3f} & {row['ri_f1_score']:.3f} \\\\"
        )
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Save LaTeX table
    latex_filename = f'results_table_{dataset}_{setting}.tex'
    with open(os.path.join(output_dir, latex_filename), 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"Saved LaTeX table: {latex_filename}")


def create_comprehensive_report(results, df, best_configs, output_dir, dataset, setting):
    """Create comprehensive text report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"MLM FEATURE CLASSIFICATION EVALUATION REPORT")
    report_lines.append(f"Dataset: {dataset.upper()}")
    report_lines.append(f"Setting: {setting.upper()}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total configurations evaluated: {len(df)}")
    report_lines.append(f"Action Recognition Accuracy: {df['ar_accuracy'].mean():.3f} ± {df['ar_accuracy'].std():.3f}")
    report_lines.append(f"Action Recognition F1-Score: {df['ar_f1_score'].mean():.3f} ± {df['ar_f1_score'].std():.3f}")
    report_lines.append(f"Re-Identification Accuracy: {df['ri_accuracy'].mean():.3f} ± {df['ri_accuracy'].std():.3f}")
    report_lines.append(f"Re-Identification F1-Score: {df['ri_f1_score'].mean():.3f} ± {df['ri_f1_score'].std():.3f}")
    report_lines.append("")
    
    # Best configurations
    report_lines.append("BEST PERFORMING CONFIGURATIONS")
    report_lines.append("-" * 40)
    for metric, config in best_configs.items():
        metric_name = metric.replace('_', ' ').title()
        report_lines.append(
            f"{metric_name}: T{config['temporal_ratio']}_S{config['spatial_ratio']} "
            f"({config['value']:.3f})"
        )
    report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 40)
    report_lines.append("Config\t\tAR Acc\tAR F1\tRI Acc\tRI F1")
    report_lines.append("-" * 50)
    
    for _, row in df.iterrows():
        config = f"T{row['temporal_ratio']}_S{row['spatial_ratio']}"
        report_lines.append(
            f"{config}\t\t{row['ar_accuracy']:.3f}\t{row['ar_f1_score']:.3f}\t"
            f"{row['ri_accuracy']:.3f}\t{row['ri_f1_score']:.3f}"
        )
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_filename = f'comprehensive_report_{dataset}_{setting}.txt'
    with open(os.path.join(output_dir, report_filename), 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved comprehensive report: {report_filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate MLM Classification Reports')
    
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing classification results')
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
    
    print(f"Generating MLM feature classification report...")
    print(f"Dataset: {args.dataset}")
    print(f"Setting: {args.setting}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Load results
    print("Loading classification results...")
    results = load_classification_results(args.results_dir, args.dataset, args.setting)
    
    if not results:
        print("Error: No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create DataFrame
    df = create_results_dataframe(results)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Find best configurations
    best_configs = find_best_configurations(df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_heatmaps(df, args.output_dir, args.dataset, args.setting)
    create_bar_plots(df, args.output_dir, args.dataset, args.setting)
    
    # Generate tables
    print("\nGenerating tables...")
    summary_df, detailed_df = create_summary_table(df, args.output_dir, args.dataset, args.setting)
    generate_latex_table(df, args.output_dir, args.dataset, args.setting)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    create_comprehensive_report(results, df, best_configs, args.output_dir, args.dataset, args.setting)
    
    print(f"\nReport generation completed!")
    print(f"All outputs saved to: {args.output_dir}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Best AR Accuracy: T{best_configs['ar_accuracy']['temporal_ratio']}_S{best_configs['ar_accuracy']['spatial_ratio']} ({best_configs['ar_accuracy']['value']:.3f})")
    print(f"Best RI Accuracy: T{best_configs['ri_accuracy']['temporal_ratio']}_S{best_configs['ri_accuracy']['spatial_ratio']} ({best_configs['ri_accuracy']['value']:.3f})")
    print(f"Mean AR Accuracy: {df['ar_accuracy'].mean():.3f} ± {df['ar_accuracy'].std():.3f}")
    print(f"Mean RI Accuracy: {df['ri_accuracy'].mean():.3f} ± {df['ri_accuracy'].std():.3f}")


if __name__ == "__main__":
    main()
