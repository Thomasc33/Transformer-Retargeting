#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_metrics(results_dir):
    """Load metrics from the results directory."""
    metrics_file = os.path.join(results_dir, "transformer_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Metrics file not found at {metrics_file}")
        return None

def create_comparison_table(metrics_dict):
    """Create a comparison table of metrics."""
    # Define the metrics to include in the table
    metrics_to_include = [
        ('Action Recognition Accuracy (%)', 'action_recognition_accuracy', 'higher'),
        ('Re-identification Accuracy (%)', 'reidentification_accuracy', 'lower'),
        ('MSE with Ground Truth', 'mse_gt', 'lower'),
        ('Bone Length Consistency', 'bone_length_consistency', 'lower'),
        ('Joint Angle Limits (%)', 'joint_angle_limits', 'higher'),
        ('Temporal Smoothness', 'temporal_smoothness', 'lower'),
        ('Velocity Consistency', 'velocity_consistency', 'higher'),
        ('Foot Contact Consistency (%)', 'foot_contact_consistency', 'higher'),
        ('FID Score', 'fid_score', 'lower')
    ]
    
    # Create a DataFrame for the comparison
    data = []
    for metric_name, metric_key, direction in metrics_to_include:
        row = {'Metric': metric_name, 'Better': direction}
        for config, metrics in metrics_dict.items():
            if metrics and metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, (int, float)):
                    row[config] = value
                else:
                    row[config] = 'N/A'
            else:
                row[config] = 'N/A'
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Highlight the best value in each row
    def highlight_best(row):
        if row['Better'] == 'higher':
            best_idx = row.drop(['Metric', 'Better']).astype(float).idxmax()
        else:  # lower is better
            best_idx = row.drop(['Metric', 'Better']).astype(float).idxmin()
        return [f'**{row[best_idx]}**' if col == best_idx else row[col] for col in row.index]
    
    # Apply highlighting
    for i, row in df.iterrows():
        try:
            highlighted = highlight_best(row)
            for j, val in enumerate(highlighted):
                if '**' in str(val):
                    df.iloc[i, j] = val
        except:
            # Skip if there's an error (e.g., non-numeric values)
            continue
    
    return df

def plot_comparison(metrics_dict, output_dir):
    """Create comparison plots for key metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define key metrics to plot
    key_metrics = [
        ('action_recognition_accuracy', 'Action Recognition Accuracy (%)', 'higher'),
        ('reidentification_accuracy', 'Re-identification Accuracy (%)', 'lower'),
        ('mse_gt', 'MSE with Ground Truth', 'lower'),
        ('bone_length_consistency', 'Bone Length Consistency', 'lower'),
        ('joint_angle_limits', 'Joint Angle Limits (%)', 'higher'),
        ('velocity_consistency', 'Velocity Consistency', 'higher')
    ]
    
    # Prepare data for plotting
    plot_data = []
    for config, metrics in metrics_dict.items():
        if metrics:
            for metric_key, metric_name, direction in key_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    if isinstance(value, (int, float)):
                        plot_data.append({
                            'Configuration': config,
                            'Metric': metric_name,
                            'Value': value,
                            'Direction': direction
                        })
    
    if not plot_data:
        print("No data available for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create a bar plot for each metric
    plt.figure(figsize=(15, 10))
    
    # Use a color palette that's good for publications
    colors = sns.color_palette("colorblind", n_colors=len(metrics_dict))
    
    # Create a bar plot
    g = sns.catplot(
        data=df, 
        kind="bar",
        x="Configuration", 
        y="Value", 
        hue="Configuration",
        col="Metric", 
        col_wrap=2,
        height=4, 
        aspect=1.5,
        palette=colors,
        sharex=False,
        sharey=False
    )
    
    # Customize the plot
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Value")
    g.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each metric for better visibility
    for metric_key, metric_name, direction in key_metrics:
        metric_data = df[df['Metric'] == metric_name]
        if len(metric_data) > 0:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=metric_data,
                x="Configuration",
                y="Value",
                palette=colors
            )
            plt.title(metric_name)
            plt.ylabel("Value")
            plt.xlabel("")
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                ax.annotate(
                    f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', 
                    va='bottom', 
                    fontsize=10
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric_key}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare pretraining experiment results")
    parser.add_argument("--results-dir", type=str, default="experiments/pretraining/results",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="experiments/pretraining/analysis",
                        help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define the configurations to compare
    configs = {
        "Pretrained + Frozen": "pretrained_frozen",
        "Pretrained + Unfrozen": "pretrained_unfrozen",
        "No Pretraining": "no_pretrained"
    }
    
    # Load metrics for each configuration
    metrics_dict = {}
    for config_name, config_dir in configs.items():
        results_dir = os.path.join(args.results_dir, config_dir)
        metrics = load_metrics(results_dir)
        if metrics:
            metrics_dict[config_name] = metrics
    
    if not metrics_dict:
        print("No metrics found. Make sure the experiments have completed.")
        return
    
    # Create comparison table
    comparison_table = create_comparison_table(metrics_dict)
    
    # Save comparison table as CSV
    comparison_table.to_csv(os.path.join(args.output_dir, "comparison_table.csv"), index=False)
    
    # Save comparison table as Markdown
    with open(os.path.join(args.output_dir, "comparison_table.md"), 'w') as f:
        f.write("# Pretraining Experiment Results Comparison\n\n")
        f.write(comparison_table.to_markdown(index=False))
    
    # Create comparison plots
    plot_comparison(metrics_dict, args.output_dir)
    
    print(f"Analysis completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
