"""
Comprehensive metrics calculator that consolidates all evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import logging

# Import existing metrics from eval_model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from eval_model import (
        calculate_bone_length_consistency,
        calculate_joint_angle_limits,
        calculate_temporal_smoothness,
        calculate_velocity_consistency,
        calculate_foot_contact_consistency,
        calculate_fid_for_skeletons,
        extract_velocity_features
    )
except ImportError:
    logging.warning("Could not import metrics from eval_model.py. Using placeholder implementations.")


class MetricsCalculator:
    """
    Comprehensive metrics calculator for all evaluation tasks.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_metrics(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all metrics for the given results.
        
        Args:
            results: Raw evaluation results
            config: Experiment configuration
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        for model_name, model_results in results.items():
            metrics[model_name] = {}
            
            for data_name, data_results in model_results.items():
                metrics[model_name][data_name] = self.calculate_metrics_for_data(
                    data_results, config
                )
                
        # Calculate aggregate metrics
        metrics['aggregate'] = self.calculate_aggregate_metrics(metrics, config)
        
        return metrics
        
    def calculate_metrics_for_data(self, data_results: Dict[str, Any], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a specific dataset."""
        metrics = {}
        
        # Task Performance Metrics
        if 'action_recognition' in data_results:
            metrics['action_recognition'] = self.calculate_ar_metrics(
                data_results['action_recognition']
            )
            
        if 'reidentification' in data_results:
            metrics['reidentification'] = self.calculate_ri_metrics(
                data_results['reidentification']
            )
            
        if 'gender_classification' in data_results:
            metrics['gender_classification'] = self.calculate_gc_metrics(
                data_results['gender_classification']
            )
            
        # Physical Plausibility Metrics
        if 'skeletons' in data_results:
            metrics['physical_plausibility'] = self.calculate_physical_metrics(
                data_results['skeletons'], config
            )
            
        # Privacy Metrics
        if 'privacy_analysis' in data_results:
            metrics['privacy'] = self.calculate_privacy_metrics(
                data_results['privacy_analysis']
            )
            
        return metrics
        
    def calculate_ar_metrics(self, ar_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Action Recognition metrics."""
        metrics = {}
        
        if 'predictions' in ar_results and 'labels' in ar_results:
            predictions = np.array(ar_results['predictions'])
            labels = np.array(ar_results['labels'])
            
            # Overall accuracy
            metrics['accuracy'] = np.mean(predictions == labels) * 100
            
            # Per-class accuracy
            unique_classes = np.unique(labels)
            per_class_acc = {}
            for cls in unique_classes:
                mask = labels == cls
                if np.sum(mask) > 0:
                    per_class_acc[int(cls)] = np.mean(predictions[mask] == labels[mask]) * 100
            metrics['per_class_accuracy'] = per_class_acc
            
            # Top-k accuracy if available
            if 'top5_predictions' in ar_results:
                top5_preds = np.array(ar_results['top5_predictions'])
                top5_acc = np.mean([label in top5_preds[i] for i, label in enumerate(labels)]) * 100
                metrics['top5_accuracy'] = top5_acc
                
        return metrics
        
    def calculate_ri_metrics(self, ri_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Re-identification metrics."""
        metrics = {}
        
        if 'predictions' in ri_results and 'labels' in ri_results:
            predictions = np.array(ri_results['predictions'])
            labels = np.array(ri_results['labels'])
            
            # Identity accuracy (lower is better for privacy)
            metrics['identity_accuracy'] = np.mean(predictions == labels) * 100
            
            # Anonymization rate (higher is better for privacy)
            metrics['anonymization_rate'] = 100 - metrics['identity_accuracy']
            
            # Per-subject analysis
            unique_subjects = np.unique(labels)
            per_subject_acc = {}
            for subj in unique_subjects:
                mask = labels == subj
                if np.sum(mask) > 0:
                    per_subject_acc[int(subj)] = np.mean(predictions[mask] == labels[mask]) * 100
            metrics['per_subject_accuracy'] = per_subject_acc
            
        return metrics
        
    def calculate_gc_metrics(self, gc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Gender Classification metrics."""
        metrics = {}
        
        if 'predictions' in gc_results and 'labels' in gc_results:
            predictions = np.array(gc_results['predictions'])
            labels = np.array(gc_results['labels'])
            
            # Overall accuracy
            metrics['accuracy'] = np.mean(predictions == labels) * 100
            
            # Gender-specific metrics
            if 'original_predictions' in gc_results:
                orig_preds = np.array(gc_results['original_predictions'])
                
                # GC-Orig: Gender classification on original data
                metrics['gc_orig'] = np.mean(orig_preds == labels) * 100
                
                # GC-Ret: Gender classification on retargeted data
                metrics['gc_ret'] = metrics['accuracy']
                
                # GC-Cross: Cross-gender accuracy
                if 'cross_gender_predictions' in gc_results:
                    cross_preds = np.array(gc_results['cross_gender_predictions'])
                    metrics['gc_cross'] = np.mean(cross_preds == labels) * 100
                    
        return metrics
        
    def calculate_physical_metrics(self, skeleton_data: Dict[str, Any], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate physical plausibility metrics."""
        metrics = {}
        
        if 'generated_skeletons' in skeleton_data and 'reference_skeletons' in skeleton_data:
            gen_skels = skeleton_data['generated_skeletons']
            ref_skels = skeleton_data['reference_skeletons']
            dataset = config.get('dataset', 'ntu')
            
            # Calculate all physical metrics
            blc_values = []
            jal_values = []
            ts_values = []
            vc_values = []
            fcc_values = []
            mse_values = []
            
            for gen_skel, ref_skel in zip(gen_skels, ref_skels):
                # Convert to tensors if needed
                if not isinstance(gen_skel, torch.Tensor):
                    gen_skel = torch.tensor(gen_skel, dtype=torch.float32)
                if not isinstance(ref_skel, torch.Tensor):
                    ref_skel = torch.tensor(ref_skel, dtype=torch.float32)
                    
                # Calculate metrics
                try:
                    blc = calculate_bone_length_consistency(gen_skel, dataset=dataset)
                    blc_values.append(blc)
                except:
                    pass
                    
                try:
                    jal = calculate_joint_angle_limits(gen_skel, dataset=dataset)
                    jal_values.append(jal)
                except:
                    pass
                    
                try:
                    ts = calculate_temporal_smoothness(gen_skel)
                    ts_values.append(ts)
                except:
                    pass
                    
                try:
                    vc = calculate_velocity_consistency(gen_skel, ref_skel)
                    vc_values.append(vc)
                except:
                    pass
                    
                try:
                    fcc = calculate_foot_contact_consistency(gen_skel, ref_skel, dataset=dataset)
                    fcc_values.append(fcc)
                except:
                    pass
                    
                # MSE
                mse = F.mse_loss(gen_skel, ref_skel).item()
                mse_values.append(mse)
                
            # Aggregate metrics
            if blc_values:
                metrics['bone_length_consistency'] = {
                    'mean': np.mean(blc_values),
                    'std': np.std(blc_values),
                    'values': blc_values
                }
                
            if jal_values:
                metrics['joint_angle_limits'] = {
                    'mean': np.mean(jal_values),
                    'std': np.std(jal_values),
                    'values': jal_values
                }
                
            if ts_values:
                metrics['temporal_smoothness'] = {
                    'mean': np.mean(ts_values),
                    'std': np.std(ts_values),
                    'values': ts_values
                }
                
            if vc_values:
                metrics['velocity_consistency'] = {
                    'mean': np.mean(vc_values),
                    'std': np.std(vc_values),
                    'values': vc_values
                }
                
            if fcc_values:
                metrics['foot_contact_consistency'] = {
                    'mean': np.mean(fcc_values),
                    'std': np.std(fcc_values),
                    'values': fcc_values
                }
                
            metrics['mse'] = {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'values': mse_values
            }
            
            # Calculate FID if enough samples
            if len(gen_skels) > 10:
                try:
                    fid_score = self.calculate_fid_score(gen_skels, ref_skels)
                    metrics['fid_score'] = fid_score
                except:
                    pass
                    
        return metrics
        
    def calculate_fid_score(self, generated_skeletons: List, reference_skeletons: List) -> float:
        """Calculate FID score for skeleton data."""
        try:
            # Extract features
            gen_features = []
            ref_features = []
            
            for gen_skel, ref_skel in zip(generated_skeletons, reference_skeletons):
                gen_feat = extract_velocity_features(gen_skel)
                ref_feat = extract_velocity_features(ref_skel)
                gen_features.append(gen_feat)
                ref_features.append(ref_feat)
                
            gen_features = np.array(gen_features)
            ref_features = np.array(ref_features)
            
            # Calculate FID
            fid_score = calculate_fid_for_skeletons(gen_features, ref_features)
            return fid_score
            
        except Exception as e:
            self.logger.warning(f"Could not calculate FID score: {e}")
            return float('inf')
            
    def calculate_privacy_metrics(self, privacy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate privacy-specific metrics."""
        metrics = {}
        
        # Privacy-utility tradeoff
        if 'utility_scores' in privacy_results and 'privacy_scores' in privacy_results:
            utility = np.array(privacy_results['utility_scores'])
            privacy = np.array(privacy_results['privacy_scores'])
            
            # Calculate tradeoff metrics
            metrics['utility_mean'] = np.mean(utility)
            metrics['privacy_mean'] = np.mean(privacy)
            metrics['tradeoff_ratio'] = metrics['privacy_mean'] / (metrics['utility_mean'] + 1e-8)
            
        return metrics
        
    def calculate_aggregate_metrics(self, all_metrics: Dict[str, Any], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all models and datasets."""
        aggregate = {}
        
        # Collect all metric values
        ar_accuracies = []
        ri_accuracies = []
        gc_accuracies = []
        mse_values = []
        
        for model_name, model_metrics in all_metrics.items():
            if model_name == 'aggregate':
                continue
                
            for data_name, data_metrics in model_metrics.items():
                if 'action_recognition' in data_metrics:
                    ar_accuracies.append(data_metrics['action_recognition'].get('accuracy', 0))
                    
                if 'reidentification' in data_metrics:
                    ri_accuracies.append(data_metrics['reidentification'].get('identity_accuracy', 0))
                    
                if 'gender_classification' in data_metrics:
                    gc_accuracies.append(data_metrics['gender_classification'].get('accuracy', 0))
                    
                if 'physical_plausibility' in data_metrics:
                    mse_values.extend(data_metrics['physical_plausibility'].get('mse', {}).get('values', []))
                    
        # Calculate aggregate statistics
        if ar_accuracies:
            aggregate['action_recognition'] = {
                'mean': np.mean(ar_accuracies),
                'std': np.std(ar_accuracies),
                'min': np.min(ar_accuracies),
                'max': np.max(ar_accuracies)
            }
            
        if ri_accuracies:
            aggregate['reidentification'] = {
                'mean': np.mean(ri_accuracies),
                'std': np.std(ri_accuracies),
                'min': np.min(ri_accuracies),
                'max': np.max(ri_accuracies)
            }
            
        if gc_accuracies:
            aggregate['gender_classification'] = {
                'mean': np.mean(gc_accuracies),
                'std': np.std(gc_accuracies),
                'min': np.min(gc_accuracies),
                'max': np.max(gc_accuracies)
            }
            
        if mse_values:
            aggregate['mse'] = {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            }
            
        return aggregate
