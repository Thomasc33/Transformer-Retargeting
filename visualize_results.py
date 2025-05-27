#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for pretrained MLM model evaluation results.
This script generates plots comparing the performance of models with
different temporal and spatial masking ratios.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualization of Pretrained MLM Model Evaluation Results')

    parser.add_argument('--results-dir', type=str, default='results/masking',
                        help='Directory containing evaluation results (default: results/masking)')
    parser.add_argument('--output-dir', type=str, default='results/masking/plots',
                        help='Directory to save visualization plots (default: results/masking/plots)')
    parser.add_argument('--dataset', type=str, default='ntu',
                        help='Dataset name for filtering results (default: ntu)')
    parser.add_argument('--setting', type=str, default='cv',
                        help='Setting (cs/cv) for filtering results (default: cv)')

    return parser.parse_args()

def load_results(results_dir, dataset, setting):
    """Load evaluation results from JSON files."""
    results_data = []

    # Find all result files
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and dataset in filename and setting in filename:
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    # Extract temporal and spatial ratios from the result
                    t_ratio = result.get('temporal_masking_ratio')
                    s_ratio = result.get('spatial_masking_ratio')

                    # Add to results data
                    results_data.append(result)
                    print(f"Loaded results from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not results_data:
        print(f"No results found in {results_dir} for dataset={dataset}, setting={setting}")
        print("Available files:")
        for filename in os.listdir(results_dir):
            print(f"  {filename}")
    else:
        print(f"Loaded {len(results_data)} result files")

    return results_data

def create_dataframe(results_data):
    """Convert results data to a pandas DataFrame."""
    df_data = []

    for result in results_data:
        row = {
            'temporal_ratio': result['temporal_masking_ratio'],
            'spatial_ratio': result['spatial_masking_ratio'],
            'reconstruction_mse': result['reconstruction_mse'],
            'ar_accuracy': result['ar_accuracy'],
            'ri_accuracy': result['ri_accuracy'],
            'gc_accuracy': result['gc_accuracy'] if result['gc_accuracy'] != -1 else np.nan,
            'bone_length_consistency': result['utility_metrics']['bone_len'],
            'joint_angle_violation': result['utility_metrics']['joint_angle'],
            'temporal_smoothness': result['utility_metrics']['smoothness'],
            'velocity_consistency': result['utility_metrics']['vel_cons'],
            'foot_contact_consistency': result['utility_metrics']['foot_contact'],
            'fid_score': result['fid_score'] if result['fid_score'] != -1 else np.nan
        }
        df_data.append(row)

    return pd.DataFrame(df_data)

def plot_heatmap(df, x_col, y_col, value_col, title, filename, output_dir, cmap='viridis', fmt='.3f'):
    """Create a heatmap plot."""
    plt.figure(figsize=(10, 8))

    # Create pivot table for heatmap
    pivot_table = df.pivot(index=y_col, columns=x_col, values=value_col)

    # Create heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        cmap=cmap,
        fmt=fmt,
        linewidths=.5,
        cbar_kws={'label': value_col.replace('_', ' ').title()}
    )

    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Temporal Masking Ratio', fontsize=14)
    plt.ylabel('Spatial Masking Ratio', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_surface(df, x_col, y_col, value_col, title, filename, output_dir, cmap='viridis'):
    """Create a 3D surface plot."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - import needed for 3D projection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create pivot table for surface plot
    pivot_table = df.pivot(index=y_col, columns=x_col, values=value_col)

    # Get x, y, z data
    x = pivot_table.columns.values
    y = pivot_table.index.values
    x_grid, y_grid = np.meshgrid(x, y)
    z = pivot_table.values

    # Create surface plot
    surf = ax.plot_surface(x_grid, y_grid, z, cmap=cmap, edgecolor='none', alpha=0.8)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(value_col.replace('_', ' ').title(), fontsize=12)

    # Set title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Temporal Masking Ratio', fontsize=14)
    ax.set_ylabel('Spatial Masking Ratio', fontsize=14)
    ax.set_zlabel(value_col.replace('_', ' ').title(), fontsize=14)

    # Set tick formatter for better readability
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(df, x_col, y_col, hue_col, title, filename, output_dir):
    """Create a grouped bar chart."""
    plt.figure(figsize=(12, 8))

    # Create bar chart
    ax = sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis')

    # Set title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=14)

    # Add legend
    plt.legend(title=hue_col.replace('_', ' ').title(), fontsize=12)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(df, output_dir):
    """Create various visualizations from the results data."""
    # Set Seaborn style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Heatmaps for key metrics
    metrics = {
        'reconstruction_mse': 'Reconstruction MSE',
        'ar_accuracy': 'Action Recognition Accuracy',
        'ri_accuracy': 'Re-identification Accuracy',
        'bone_length_consistency': 'Bone Length Consistency',
        'joint_angle_violation': 'Joint Angle Violation Rate',
        'temporal_smoothness': 'Temporal Smoothness',
        'velocity_consistency': 'Velocity Consistency',
        'foot_contact_consistency': 'Foot Contact Consistency'
    }

    for metric, title in metrics.items():
        plot_heatmap(
            df,
            'temporal_ratio',
            'spatial_ratio',
            metric,
            f'{title} by Masking Ratios',
            f'{metric}_heatmap.png',
            output_dir
        )

    # 2. 3D surface plots for key metrics
    for metric, title in metrics.items():
        plot_3d_surface(
            df,
            'temporal_ratio',
            'spatial_ratio',
            metric,
            f'{title} by Masking Ratios (3D)',
            f'{metric}_surface.png',
            output_dir
        )

    # 3. Bar charts comparing temporal ratios for each spatial ratio
    for metric, title in metrics.items():
        # Reshape data for grouped bar chart
        df_melted = df.melt(
            id_vars=['temporal_ratio', 'spatial_ratio'],
            value_vars=[metric],
            var_name='metric',
            value_name='value'
        )

        plot_bar_chart(
            df_melted,
            'temporal_ratio',
            'value',
            'spatial_ratio',
            f'{title} by Temporal Masking Ratio',
            f'{metric}_temporal_bar.png',
            output_dir
        )

    # 4. Bar charts comparing spatial ratios for each temporal ratio
    for metric, title in metrics.items():
        # Reshape data for grouped bar chart
        df_melted = df.melt(
            id_vars=['temporal_ratio', 'spatial_ratio'],
            value_vars=[metric],
            var_name='metric',
            value_name='value'
        )

        plot_bar_chart(
            df_melted,
            'spatial_ratio',
            'value',
            'temporal_ratio',
            f'{title} by Spatial Masking Ratio',
            f'{metric}_spatial_bar.png',
            output_dir
        )

    # 5. Summary table
    summary_df = df.pivot_table(
        index='spatial_ratio',
        columns='temporal_ratio',
        values=['reconstruction_mse', 'ar_accuracy', 'ri_accuracy']
    )

    # Save summary table as CSV
    summary_df.to_csv(os.path.join(output_dir, 'summary_table.csv'))

    # Also create a styled HTML table
    styled_table = summary_df.style.format(precision=4).background_gradient(cmap='viridis')
    styled_table.to_html(os.path.join(output_dir, 'summary_table.html'))

def main():
    """Main visualization function."""
    args = parse_args()

    # Load results
    results_data = load_results(args.results_dir, args.dataset, args.setting)

    if not results_data:
        print(f"No results found in {args.results_dir} for dataset={args.dataset}, setting={args.setting}")
        return

    # Create DataFrame
    df = create_dataframe(results_data)

    # Print the DataFrame for debugging
    print("\nResults DataFrame:")
    print(df)

    # Create visualizations
    create_visualizations(df, args.output_dir)

    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
