#!/usr/bin/env python3
"""
Script to evaluate masking configurations based on result metrics.
This script reads all the result JSON files in the results/masking directory,
applies weights to different metrics, and ranks the configurations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the weights for different metrics
# Higher weight means more importance
WEIGHTS = {
    'reconstruction_mse': 0.25,  # High importance (lower is better, so we'll invert)
    'ar_accuracy': 0.25,         # High importance
    'ri_accuracy': 0.15,         # Medium importance
    'gc_accuracy': 0.05,         # Lower importance
    'bone_len': 0.05,            # Physical plausibility metrics (lower is better)
    'joint_angle': 0.05,
    'smoothness': 0.05,
    'vel_cons': 0.05,
    'foot_contact': 0.05,
    'fid_score': 0.05            # Lower is better
}

# Define which metrics are "lower is better"
LOWER_IS_BETTER = {
    'reconstruction_mse': True,
    'ar_accuracy': False,
    'ri_accuracy': False,
    'gc_accuracy': False,
    'bone_len': True,
    'joint_angle': True,
    'smoothness': True,
    'vel_cons': True,
    'foot_contact': True,
    'fid_score': True
}

def load_results(results_dir='results/masking'):
    """Load all result JSON files from the specified directory."""
    results = []
    
    for file_path in Path(results_dir).glob('*.json'):
        if 'README' in file_path.name:
            continue
            
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                # Add filename for reference
                data['filename'] = file_path.name
                results.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")
    
    return results

def normalize_metrics(results):
    """Normalize metrics to a 0-1 scale for fair comparison."""
    # Extract all metrics
    metrics = {}
    for result in results:
        for key, value in result.items():
            if key in WEIGHTS and key not in metrics:
                metrics[key] = []
            
            if key in WEIGHTS:
                if key == 'gc_accuracy' and value == -1:
                    # Handle special case for gc_accuracy
                    value = 0
                metrics[key].append(value)
        
        # Add utility metrics
        if 'utility_metrics' in result:
            for key, value in result['utility_metrics'].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    # Normalize each metric
    normalized_metrics = {}
    for key, values in metrics.items():
        if len(values) == 0:
            continue
            
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            # All values are the same, set to 0.5
            normalized_metrics[key] = [0.5] * len(values)
        else:
            # Normalize to 0-1 range
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
            
            # For metrics where lower is better, invert the normalization
            if key in LOWER_IS_BETTER and LOWER_IS_BETTER[key]:
                normalized = [1 - n for n in normalized]
                
            normalized_metrics[key] = normalized
    
    return normalized_metrics

def calculate_weighted_scores(results, normalized_metrics):
    """Calculate weighted scores for each configuration."""
    scores = []
    
    for i, result in enumerate(results):
        score = 0
        
        # Add weighted normalized metrics
        for key, weight in WEIGHTS.items():
            if key in normalized_metrics and i < len(normalized_metrics[key]):
                score += weight * normalized_metrics[key][i]
            elif key in result['utility_metrics'] and key in normalized_metrics and i < len(normalized_metrics[key]):
                score += weight * normalized_metrics[key][i]
        
        scores.append({
            'temporal_masking_ratio': result['temporal_masking_ratio'],
            'spatial_masking_ratio': result['spatial_masking_ratio'],
            'score': score,
            'reconstruction_mse': result['reconstruction_mse'],
            'ar_accuracy': result['ar_accuracy'],
            'ri_accuracy': result['ri_accuracy'],
            'gc_accuracy': result['gc_accuracy'] if result['gc_accuracy'] != -1 else 0,
            'bone_len': result['utility_metrics']['bone_len'],
            'joint_angle': result['utility_metrics']['joint_angle'],
            'smoothness': result['utility_metrics']['smoothness'],
            'vel_cons': result['utility_metrics']['vel_cons'],
            'foot_contact': result['utility_metrics']['foot_contact'],
            'fid_score': result['fid_score']
        })
    
    return scores

def create_heatmap(scores, metric='score'):
    """Create a heatmap of scores for different masking ratios."""
    # Extract unique temporal and spatial masking ratios
    temporal_ratios = sorted(list(set([s['temporal_masking_ratio'] for s in scores])))
    spatial_ratios = sorted(list(set([s['spatial_masking_ratio'] for s in scores])))
    
    # Create a matrix of scores
    matrix = np.zeros((len(temporal_ratios), len(spatial_ratios)))
    
    for score in scores:
        t_idx = temporal_ratios.index(score['temporal_masking_ratio'])
        s_idx = spatial_ratios.index(score['spatial_masking_ratio'])
        matrix[t_idx, s_idx] = score[metric]
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", 
                xticklabels=spatial_ratios, 
                yticklabels=temporal_ratios,
                cmap='viridis')
    plt.xlabel('Spatial Masking Ratio')
    plt.ylabel('Temporal Masking Ratio')
    plt.title(f'Heatmap of {metric.replace("_", " ").title()}')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'results/masking/plots/{metric}_heatmap.png')
    plt.close()

def main():
    """Main function to evaluate masking results."""
    # Create plots directory if it doesn't exist
    os.makedirs('results/masking/plots', exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No result files found.")
        return
    
    # Normalize metrics
    normalized_metrics = normalize_metrics(results)
    
    # Calculate weighted scores
    scores = calculate_weighted_scores(results, normalized_metrics)
    
    # Sort scores by score (descending)
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Print the top configurations
    print("\n===== Top Masking Configurations =====")
    for i, score in enumerate(scores):
        print(f"{i+1}. Temporal: {score['temporal_masking_ratio']}, Spatial: {score['spatial_masking_ratio']}")
        print(f"   Score: {score['score']:.4f}")
        print(f"   MSE: {score['reconstruction_mse']:.6f}")
        print(f"   AR: {score['ar_accuracy']:.4f}, RI: {score['ri_accuracy']:.4f}, GC: {score['gc_accuracy']:.4f}")
        print(f"   Physical Metrics: BL={score['bone_len']:.4f}, JA={score['joint_angle']:.4f}, TS={score['smoothness']:.4f}, VC={score['vel_cons']:.4f}, FC={score['foot_contact']:.4f}")
        print(f"   FID: {score['fid_score']:.6f}")
        print()
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(scores)
    
    # Save the DataFrame to CSV
    df.to_csv('results/masking/masking_evaluation.csv', index=False)
    
    # Create heatmaps for different metrics
    create_heatmap(scores, 'score')
    create_heatmap(scores, 'reconstruction_mse')
    create_heatmap(scores, 'fid_score')
    
    # Create a combined plot for physical plausibility metrics
    physical_metrics = ['bone_len', 'joint_angle', 'smoothness', 'vel_cons', 'foot_contact']
    for metric in physical_metrics:
        create_heatmap(scores, metric)
    
    print("Evaluation complete. Results saved to results/masking/masking_evaluation.csv")
    print("Heatmaps saved to results/masking/plots/")

if __name__ == "__main__":
    main()
