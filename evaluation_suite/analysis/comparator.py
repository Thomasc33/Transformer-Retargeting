"""
Result comparison utilities for statistical analysis and ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import logging


class ResultComparator:
    """
    Compares results across different experiments and models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def compare_experiments(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results across multiple experiments.
        
        Args:
            results: Dictionary of experiment results
            
        Returns:
            Comparison analysis including rankings and statistical tests
        """
        comparison_data = {
            'rankings': {},
            'significance_tests': {},
            'effect_sizes': {},
            'summary_statistics': {}
        }
        
        # Extract metrics for comparison
        all_metrics = self.extract_comparable_metrics(results)
        
        # Generate rankings
        comparison_data['rankings'] = self.generate_rankings(all_metrics)
        
        # Perform statistical tests
        comparison_data['significance_tests'] = self.perform_significance_tests(all_metrics)
        
        # Calculate effect sizes
        comparison_data['effect_sizes'] = self.calculate_effect_sizes(all_metrics)
        
        # Generate summary statistics
        comparison_data['summary_statistics'] = self.generate_summary_statistics(all_metrics)
        
        return comparison_data
        
    def extract_comparable_metrics(self, results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """Extract metrics that can be compared across experiments."""
        comparable_metrics = {}
        
        for exp_name, exp_results in results.items():
            if 'metrics' not in exp_results:
                continue
                
            metrics = exp_results['metrics']
            
            for model_name, model_metrics in metrics.items():
                if model_name == 'aggregate':
                    continue
                    
                if model_name not in comparable_metrics:
                    comparable_metrics[model_name] = {}
                    
                for data_name, data_metrics in model_metrics.items():
                    # Action Recognition Accuracy
                    if 'action_recognition' in data_metrics:
                        metric_name = 'action_recognition_accuracy'
                        if metric_name not in comparable_metrics[model_name]:
                            comparable_metrics[model_name][metric_name] = []
                        comparable_metrics[model_name][metric_name].append(
                            data_metrics['action_recognition'].get('accuracy', 0)
                        )
                        
                    # Re-identification Accuracy (for anonymization rate)
                    if 'reidentification' in data_metrics:
                        # Identity accuracy (lower is better for privacy)
                        metric_name = 'identity_accuracy'
                        if metric_name not in comparable_metrics[model_name]:
                            comparable_metrics[model_name][metric_name] = []
                        comparable_metrics[model_name][metric_name].append(
                            data_metrics['reidentification'].get('identity_accuracy', 0)
                        )
                        
                        # Anonymization rate (higher is better for privacy)
                        metric_name = 'anonymization_rate'
                        if metric_name not in comparable_metrics[model_name]:
                            comparable_metrics[model_name][metric_name] = []
                        ri_acc = data_metrics['reidentification'].get('identity_accuracy', 0)
                        comparable_metrics[model_name][metric_name].append(100 - ri_acc)
                        
                    # Physical plausibility metrics
                    if 'physical_plausibility' in data_metrics:
                        phys_metrics = data_metrics['physical_plausibility']
                        
                        for phys_metric_name in ['mse', 'bone_length_consistency', 
                                               'temporal_smoothness', 'velocity_consistency']:
                            if phys_metric_name in phys_metrics:
                                metric_name = phys_metric_name
                                if metric_name not in comparable_metrics[model_name]:
                                    comparable_metrics[model_name][metric_name] = []
                                comparable_metrics[model_name][metric_name].append(
                                    phys_metrics[phys_metric_name].get('mean', 0)
                                )
                                
        return comparable_metrics
        
    def generate_rankings(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[Tuple[str, float]]]:
        """Generate rankings for each metric."""
        rankings = {}
        
        # Get all metric names
        all_metric_names = set()
        for model_metrics in metrics_data.values():
            all_metric_names.update(model_metrics.keys())
            
        for metric_name in all_metric_names:
            model_scores = []
            
            for model_name, model_metrics in metrics_data.items():
                if metric_name in model_metrics:
                    # Calculate mean score for this model on this metric
                    scores = model_metrics[metric_name]
                    if scores:
                        mean_score = np.mean(scores)
                        model_scores.append((model_name, mean_score))
                        
            if model_scores:
                # Sort based on metric type (higher is better for most metrics except MSE and identity_accuracy)
                reverse_sort = metric_name not in ['mse', 'identity_accuracy']
                model_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
                rankings[metric_name] = model_scores
                
        return rankings
        
    def perform_significance_tests(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests."""
        significance_tests = {}
        
        # Get all metric names
        all_metric_names = set()
        for model_metrics in metrics_data.values():
            all_metric_names.update(model_metrics.keys())
            
        for metric_name in all_metric_names:
            # Collect data for this metric
            metric_groups = {}
            for model_name, model_metrics in metrics_data.items():
                if metric_name in model_metrics and model_metrics[metric_name]:
                    metric_groups[model_name] = model_metrics[metric_name]
                    
            if len(metric_groups) >= 2:
                significance_tests[metric_name] = self.pairwise_tests(metric_groups)
                
        return significance_tests
        
    def pairwise_tests(self, groups: Dict[str, List[float]]) -> Dict[str, float]:
        """Perform pairwise t-tests between groups."""
        pairwise_results = {}
        
        group_names = list(groups.keys())
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                group1_data = groups[group1_name]
                group2_data = groups[group2_name]
                
                if len(group1_data) > 1 and len(group2_data) > 1:
                    try:
                        # Perform independent t-test
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        comparison_name = f"{group1_name} vs {group2_name}"
                        pairwise_results[comparison_name] = p_value
                    except Exception as e:
                        self.logger.warning(f"Could not perform t-test for {group1_name} vs {group2_name}: {e}")
                        
        return pairwise_results
        
    def calculate_effect_sizes(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes (Cohen's d) for pairwise comparisons."""
        effect_sizes = {}
        
        # Get all metric names
        all_metric_names = set()
        for model_metrics in metrics_data.values():
            all_metric_names.update(model_metrics.keys())
            
        for metric_name in all_metric_names:
            # Collect data for this metric
            metric_groups = {}
            for model_name, model_metrics in metrics_data.items():
                if metric_name in model_metrics and model_metrics[metric_name]:
                    metric_groups[model_name] = model_metrics[metric_name]
                    
            if len(metric_groups) >= 2:
                effect_sizes[metric_name] = self.pairwise_effect_sizes(metric_groups)
                
        return effect_sizes
        
    def pairwise_effect_sizes(self, groups: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate pairwise effect sizes (Cohen's d)."""
        effect_sizes = {}
        
        group_names = list(groups.keys())
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                group1_data = np.array(groups[group1_name])
                group2_data = np.array(groups[group2_name])
                
                if len(group1_data) > 1 and len(group2_data) > 1:
                    try:
                        # Calculate Cohen's d
                        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
                        std1, std2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
                        
                        # Pooled standard deviation
                        n1, n2 = len(group1_data), len(group2_data)
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (mean1 - mean2) / pooled_std
                            comparison_name = f"{group1_name} vs {group2_name}"
                            effect_sizes[comparison_name] = abs(cohens_d)
                    except Exception as e:
                        self.logger.warning(f"Could not calculate effect size for {group1_name} vs {group2_name}: {e}")
                        
        return effect_sizes
        
    def generate_summary_statistics(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Generate summary statistics for each model and metric."""
        summary_stats = {}
        
        for model_name, model_metrics in metrics_data.items():
            summary_stats[model_name] = {}
            
            for metric_name, metric_values in model_metrics.items():
                if metric_values:
                    values = np.array(metric_values)
                    summary_stats[model_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'count': len(values)
                    }
                    
        return summary_stats
        
    def identify_best_models(self, metrics_data: Dict[str, Dict[str, List[float]]], 
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """Identify best models for different criteria."""
        if weights is None:
            # Default weights for different metrics
            weights = {
                'action_recognition_accuracy': 0.3,
                'anonymization_rate': 0.3,
                'mse': -0.2,  # Negative because lower is better
                'bone_length_consistency': 0.1,
                'temporal_smoothness': 0.1
            }
            
        best_models = {}
        
        # Calculate weighted scores for each model
        model_scores = {}
        for model_name, model_metrics in metrics_data.items():
            total_score = 0
            total_weight = 0
            
            for metric_name, weight in weights.items():
                if metric_name in model_metrics and model_metrics[metric_name]:
                    metric_score = np.mean(model_metrics[metric_name])
                    total_score += weight * metric_score
                    total_weight += abs(weight)
                    
            if total_weight > 0:
                model_scores[model_name] = total_score / total_weight
                
        # Find best models for different criteria
        if model_scores:
            # Overall best model
            best_overall = max(model_scores.items(), key=lambda x: x[1])
            best_models['overall'] = best_overall[0]
            
            # Best for privacy (highest anonymization rate)
            privacy_scores = {}
            for model_name, model_metrics in metrics_data.items():
                if 'anonymization_rate' in model_metrics and model_metrics['anonymization_rate']:
                    privacy_scores[model_name] = np.mean(model_metrics['anonymization_rate'])
            if privacy_scores:
                best_privacy = max(privacy_scores.items(), key=lambda x: x[1])
                best_models['privacy'] = best_privacy[0]
                
            # Best for utility (highest action recognition)
            utility_scores = {}
            for model_name, model_metrics in metrics_data.items():
                if 'action_recognition_accuracy' in model_metrics and model_metrics['action_recognition_accuracy']:
                    utility_scores[model_name] = np.mean(model_metrics['action_recognition_accuracy'])
            if utility_scores:
                best_utility = max(utility_scores.items(), key=lambda x: x[1])
                best_models['utility'] = best_utility[0]
                
        return best_models
        
    def generate_comparison_table(self, metrics_data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """Generate a comparison table of all models and metrics."""
        # Prepare data for DataFrame
        table_data = []
        
        for model_name, model_metrics in metrics_data.items():
            row = {'Model': model_name}
            
            for metric_name, metric_values in model_metrics.items():
                if metric_values:
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)
                    row[f"{metric_name}_mean"] = mean_val
                    row[f"{metric_name}_std"] = std_val
                    row[f"{metric_name}_formatted"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    
            table_data.append(row)
            
        return pd.DataFrame(table_data)
