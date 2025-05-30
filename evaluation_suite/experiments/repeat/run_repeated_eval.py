#!/usr/bin/env python3
"""
Run repeated evaluation for better statistics.
This script runs evaluation multiple times and computes mean and standard deviation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

def run_single_evaluation(args):
    """Run a single evaluation and return the results."""
    
    # Build the evaluation command
    cmd = [
        'python', 'eval_model.py',
        '--dataset', args.dataset,
        '--setting', args.setting,
        '--eval_model', args.eval_model,
        '--test_samples', str(args.test_samples),
        '--transformer_model_path', args.transformer_model_path,
        '--hpc'
    ]
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr}")
            return None
            
        # Parse the output to extract metrics
        output_lines = result.stdout.strip().split('\n')
        metrics = {}
        
        for line in output_lines:
            if 'AR:' in line:
                try:
                    ar_value = float(line.split('AR:')[1].strip().split()[0])
                    metrics['AR'] = ar_value
                except:
                    pass
            elif 'RI:' in line:
                try:
                    ri_value = float(line.split('RI:')[1].strip().split()[0])
                    metrics['RI'] = ri_value
                except:
                    pass
            elif 'MSE:' in line:
                try:
                    mse_value = float(line.split('MSE:')[1].strip().split()[0])
                    metrics['MSE'] = mse_value
                except:
                    pass
            elif 'BLC:' in line:
                try:
                    blc_value = float(line.split('BLC:')[1].strip().split()[0])
                    metrics['BLC'] = blc_value
                except:
                    pass
            elif 'JAL:' in line:
                try:
                    jal_value = float(line.split('JAL:')[1].strip().split()[0])
                    metrics['JAL'] = jal_value
                except:
                    pass
            elif 'TS:' in line:
                try:
                    ts_value = float(line.split('TS:')[1].strip().split()[0])
                    metrics['TS'] = ts_value
                except:
                    pass
            elif 'VC:' in line:
                try:
                    vc_value = float(line.split('VC:')[1].strip().split()[0])
                    metrics['VC'] = vc_value
                except:
                    pass
                    
        return metrics
        
    except subprocess.TimeoutExpired:
        print("Evaluation timed out")
        return None
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None

def compute_statistics(results_list):
    """Compute mean and standard deviation for each metric."""
    if not results_list:
        return {}
        
    # Get all metric names
    all_metrics = set()
    for result in results_list:
        if result:
            all_metrics.update(result.keys())
    
    statistics = {}
    
    for metric in all_metrics:
        values = []
        for result in results_list:
            if result and metric in result:
                values.append(result[metric])
        
        if values:
            statistics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
                'count': len(values)
            }
    
    return statistics

def main():
    parser = argparse.ArgumentParser(description='Run repeated evaluation for better statistics')
    parser.add_argument('--dataset', required=True, help='Dataset name (ntu, ntu120, etc.)')
    parser.add_argument('--setting', required=True, help='Setting (cv, cs, etc.)')
    parser.add_argument('--eval_model', required=True, help='Evaluation model (sgn, mixformer)')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of evaluation runs')
    parser.add_argument('--test_samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--transformer_model_path', required=True, help='Path to transformer model')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"Running {args.num_runs} evaluations for better statistics...")
    print(f"Model: {args.transformer_model_path}")
    print(f"Eval model: {args.eval_model}")
    print(f"Test samples: {args.test_samples}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run multiple evaluations
    all_results = []
    successful_runs = 0
    
    for run_idx in range(args.num_runs):
        print(f"\nRun {run_idx + 1}/{args.num_runs}...")
        
        result = run_single_evaluation(args)
        
        if result:
            all_results.append(result)
            successful_runs += 1
            print(f"  Success! Metrics: {result}")
        else:
            all_results.append(None)
            print(f"  Failed!")
    
    print(f"\nCompleted {successful_runs}/{args.num_runs} successful runs")
    
    if successful_runs == 0:
        print("No successful runs! Cannot compute statistics.")
        return 1
    
    # Compute statistics
    statistics = compute_statistics(all_results)
    
    # Print results
    print("\n" + "="*50)
    print("REPEATED EVALUATION RESULTS")
    print("="*50)
    
    for metric, stats in statistics.items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Runs: {stats['count']}/{args.num_runs}")
    
    # Save detailed results
    results_file = output_dir / 'repeated_evaluation_results.json'
    
    detailed_results = {
        'experiment_info': {
            'dataset': args.dataset,
            'setting': args.setting,
            'eval_model': args.eval_model,
            'transformer_model_path': args.transformer_model_path,
            'test_samples': args.test_samples,
            'num_runs': args.num_runs,
            'successful_runs': successful_runs
        },
        'individual_results': all_results,
        'statistics': statistics
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary for easy parsing
    summary_file = output_dir / 'evaluation_summary.json'
    summary = {
        'successful_runs': successful_runs,
        'total_runs': args.num_runs,
        'metrics': {metric: {'mean': stats['mean'], 'std': stats['std']} 
                   for metric, stats in statistics.items()}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
