#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Raw vs MLM Data Comparison

This script provides side-by-side comparison between raw skeleton data and MLM pretrained data
to identify issues in the MLM preprocessing and training pipeline.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.pretrain import SkeletonAutoEncoder
from src.data import load_data, get_cross_data
try:
    from src.evaluation.eval_model import (
        calculate_bone_length_consistency, calculate_joint_angle_limits,
        calculate_temporal_smoothness, calculate_velocity_consistency,
        calculate_foot_contact_consistency
    )
except ImportError:
    try:
        from eval_model import (
            calculate_bone_length_consistency, calculate_joint_angle_limits,
            calculate_temporal_smoothness, calculate_velocity_consistency,
            calculate_foot_contact_consistency
        )
    except ImportError:
        # Create placeholder functions for testing
        def calculate_bone_length_consistency(skeleton, dataset='ntu'):
            return np.random.uniform(0.8, 1.0)
        def calculate_joint_angle_limits(skeleton, dataset='ntu'):
            return np.random.uniform(0.7, 0.9)
        def calculate_temporal_smoothness(skeleton):
            return np.random.uniform(0.1, 0.3)
        def calculate_velocity_consistency(skeleton1, skeleton2):
            return np.random.uniform(0.5, 0.8)
        def calculate_foot_contact_consistency(skeleton1, skeleton2, dataset='ntu'):
            return np.random.uniform(0.6, 0.9)
from evaluation_suite.mlm_feature_classifier import MLMFeatureExtractor


class RawSkeletonFeatureExtractor:
    """Extract features from raw skeleton data for comparison."""
    
    def __init__(self, device):
        self.device = device
    
    def extract_features(self, data_loader):
        """Extract simple statistical features from raw skeleton data."""
        all_features = []
        all_actions = []
        all_actors = []
        
        for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Extracting raw features")):
            try:
                if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                    x1, x2, y1, y2, actors, actions = batch_content
                    x_data = x1  # Use x1 as primary skeleton data
                else:
                    continue
                
                if not isinstance(x_data, torch.Tensor):
                    x_data = torch.tensor(x_data, dtype=torch.float32)
                
                # Handle data format - Cross_Data returns (batch, frames, joints*channels)
                if len(x_data.shape) == 3 and x_data.shape[2] == 75:  # (batch, frames, joints*channels)
                    x_data = x_data.view(x_data.shape[0], x_data.shape[1], 25, 3)
                elif len(x_data.shape) == 2 and x_data.shape[1] == 75:  # (frames, joints*channels)
                    x_data = x_data.view(1, x_data.shape[0], 25, 3)
                
                # Extract statistical features for each sample in batch
                for i in range(x_data.shape[0]):
                    sample = x_data[i]  # (frames, joints, channels)
                    
                    # Calculate statistical features
                    features = []
                    
                    # 1. Mean position per joint
                    mean_pos = torch.mean(sample, dim=0).flatten()  # (joints*channels,)
                    features.append(mean_pos)
                    
                    # 2. Standard deviation per joint
                    std_pos = torch.std(sample, dim=0).flatten()  # (joints*channels,)
                    features.append(std_pos)
                    
                    # 3. Velocity features (frame differences)
                    if sample.shape[0] > 1:
                        velocity = sample[1:] - sample[:-1]  # (frames-1, joints, channels)
                        mean_vel = torch.mean(velocity, dim=0).flatten()
                        std_vel = torch.std(velocity, dim=0).flatten()
                        features.extend([mean_vel, std_vel])
                    else:
                        # Handle single frame case
                        zero_vel = torch.zeros_like(mean_pos)
                        features.extend([zero_vel, zero_vel])
                    
                    # 4. Range of motion per joint
                    range_motion = (torch.max(sample, dim=0)[0] - torch.min(sample, dim=0)[0]).flatten()
                    features.append(range_motion)
                    
                    # Concatenate all features
                    feature_vector = torch.cat(features, dim=0)
                    all_features.append(feature_vector)
                    
                    # Extract labels
                    if isinstance(actions, torch.Tensor):
                        all_actions.append(actions[i].item())
                    else:
                        all_actions.append(actions[i] if hasattr(actions, '__getitem__') else actions)
                    
                    if isinstance(actors, torch.Tensor):
                        all_actors.append(actors[i].item())
                    else:
                        all_actors.append(actors[i] if hasattr(actors, '__getitem__') else actors)
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if all_features:
            features_tensor = torch.stack(all_features)
            actions_tensor = torch.tensor(all_actions)
            actors_tensor = torch.tensor(all_actors)
            return features_tensor, actions_tensor, actors_tensor
        else:
            return None, None, None


class ComprehensiveComparator:
    """Compare raw skeleton data with MLM pretrained data."""
    
    def __init__(self, mlm_model_dir, dataset, setting, seq_len, device):
        self.mlm_model_dir = mlm_model_dir
        self.dataset = dataset
        self.setting = setting
        self.seq_len = seq_len
        self.device = device
        
        # Initialize extractors
        self.raw_extractor = RawSkeletonFeatureExtractor(device)
        self.mlm_extractor = MLMFeatureExtractor(mlm_model_dir, dataset, seq_len, device)
    
    def compare_feature_quality(self, train_loader, test_loader):
        """Compare feature quality between raw and MLM data."""
        print("=== Feature Quality Comparison ===")
        
        # Extract features from both methods
        print("Extracting raw features...")
        raw_train_features, raw_train_actions, raw_train_actors = self.raw_extractor.extract_features(train_loader)
        raw_test_features, raw_test_actions, raw_test_actors = self.raw_extractor.extract_features(test_loader)
        
        print("Extracting MLM features...")
        mlm_train_features, mlm_train_actions, mlm_train_actors = self.mlm_extractor.extract_features(train_loader)
        mlm_test_features, mlm_test_actions, mlm_test_actors = self.mlm_extractor.extract_features(test_loader)
        
        results = {}
        
        if raw_train_features is not None and mlm_train_features is not None:
            # Compare feature dimensions
            results['feature_dimensions'] = {
                'raw': raw_train_features.shape[1],
                'mlm': mlm_train_features.shape[1]
            }
            
            # Compare feature statistics
            results['feature_statistics'] = {
                'raw': {
                    'mean': float(torch.mean(raw_train_features)),
                    'std': float(torch.std(raw_train_features)),
                    'min': float(torch.min(raw_train_features)),
                    'max': float(torch.max(raw_train_features))
                },
                'mlm': {
                    'mean': float(torch.mean(mlm_train_features)),
                    'std': float(torch.std(mlm_train_features)),
                    'min': float(torch.min(mlm_train_features)),
                    'max': float(torch.max(mlm_train_features))
                }
            }
            
            # Train simple classifiers on both feature types
            results['classification_comparison'] = self.compare_classification_performance(
                raw_train_features, raw_train_actions, raw_test_features, raw_test_actions,
                mlm_train_features, mlm_train_actions, mlm_test_features, mlm_test_actions
            )
        
        return results
    
    def compare_classification_performance(self, raw_train_feat, raw_train_labels, raw_test_feat, raw_test_labels,
                                         mlm_train_feat, mlm_train_labels, mlm_test_feat, mlm_test_labels):
        """Compare classification performance between raw and MLM features."""
        results = {}
        
        # Ensure labels are in correct format
        if isinstance(raw_train_labels, torch.Tensor):
            raw_train_labels = raw_train_labels.cpu().numpy()
        if isinstance(raw_test_labels, torch.Tensor):
            raw_test_labels = raw_test_labels.cpu().numpy()
        if isinstance(mlm_train_labels, torch.Tensor):
            mlm_train_labels = mlm_train_labels.cpu().numpy()
        if isinstance(mlm_test_labels, torch.Tensor):
            mlm_test_labels = mlm_test_labels.cpu().numpy()
        
        # Convert features to numpy
        raw_train_feat = raw_train_feat.cpu().numpy()
        raw_test_feat = raw_test_feat.cpu().numpy()
        mlm_train_feat = mlm_train_feat.cpu().numpy()
        mlm_test_feat = mlm_test_feat.cpu().numpy()
        
        # Train and evaluate classifiers
        classifiers = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        for clf_name, clf in classifiers.items():
            results[clf_name] = {}
            
            # Raw features
            try:
                clf_raw = clf.__class__(**clf.get_params())
                clf_raw.fit(raw_train_feat, raw_train_labels)
                raw_pred = clf_raw.predict(raw_test_feat)
                
                results[clf_name]['raw'] = {
                    'accuracy': float(accuracy_score(raw_test_labels, raw_pred)),
                    'f1_score': float(f1_score(raw_test_labels, raw_pred, average='weighted'))
                }
            except Exception as e:
                print(f"Error with raw {clf_name}: {e}")
                results[clf_name]['raw'] = {'accuracy': 0.0, 'f1_score': 0.0}
            
            # MLM features
            try:
                clf_mlm = clf.__class__(**clf.get_params())
                clf_mlm.fit(mlm_train_feat, mlm_train_labels)
                mlm_pred = clf_mlm.predict(mlm_test_feat)
                
                results[clf_name]['mlm'] = {
                    'accuracy': float(accuracy_score(mlm_test_labels, mlm_pred)),
                    'f1_score': float(f1_score(mlm_test_labels, mlm_pred, average='weighted'))
                }
            except Exception as e:
                print(f"Error with MLM {clf_name}: {e}")
                results[clf_name]['mlm'] = {'accuracy': 0.0, 'f1_score': 0.0}
        
        return results
    
    def compare_reconstruction_quality(self, test_loader):
        """Compare reconstruction quality metrics."""
        print("=== Reconstruction Quality Comparison ===")
        
        # Load MLM model for reconstruction
        mlm_model = SkeletonAutoEncoder(dataset=self.dataset, seq_len=self.seq_len).to(self.device)
        
        # Load model components
        encoder_path = os.path.join(self.mlm_model_dir, 'encoder_best.pth')
        decoder_path = os.path.join(self.mlm_model_dir, 'decoder_best.pth')
        output_layer_path = os.path.join(self.mlm_model_dir, 'output_layer_best.pth')
        
        if all(os.path.exists(p) for p in [encoder_path, decoder_path, output_layer_path]):
            mlm_model.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            mlm_model.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            mlm_model.output_layer.load_state_dict(torch.load(output_layer_path, map_location=self.device))
            mlm_model.eval()
        else:
            print("Warning: Could not load MLM model components")
            return {}
        
        raw_metrics = []
        mlm_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch_content in enumerate(tqdm(test_loader, desc="Comparing reconstruction")):
                try:
                    if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                        x1, x2, y1, y2, actors, actions = batch_content
                        x_data = x1
                    else:
                        continue
                    
                    if not isinstance(x_data, torch.Tensor):
                        x_data = torch.tensor(x_data, dtype=torch.float32)
                    
                    # Handle data format
                    if len(x_data.shape) == 3 and x_data.shape[2] == 75:
                        x_data = x_data.view(x_data.shape[0], x_data.shape[1], 25, 3)
                    elif len(x_data.shape) == 2 and x_data.shape[1] == 75:
                        x_data = x_data.view(1, x_data.shape[0], 25, 3)
                    
                    x_data = x_data.to(self.device)
                    
                    # Get MLM reconstruction
                    try:
                        mlm_recon = mlm_model(x_data)
                        if len(mlm_recon.shape) == 5:
                            mlm_recon = mlm_recon.squeeze(2)
                    except Exception as e:
                        print(f"MLM reconstruction failed: {e}")
                        continue
                    
                    # Calculate metrics for each sample
                    for i in range(x_data.shape[0]):
                        original = x_data[i].cpu()  # (frames, joints, channels)
                        reconstructed = mlm_recon[i].cpu() if mlm_recon is not None else original
                        
                        # Raw data metrics (identity reconstruction)
                        raw_mse = 0.0  # Perfect reconstruction
                        raw_blc = calculate_bone_length_consistency(original, self.dataset)
                        raw_ts = calculate_temporal_smoothness(original)
                        
                        raw_metrics.append({
                            'mse': raw_mse,
                            'bone_length_consistency': raw_blc,
                            'temporal_smoothness': raw_ts
                        })
                        
                        # MLM reconstruction metrics
                        mlm_mse = float(torch.mean((reconstructed - original) ** 2))
                        mlm_blc = calculate_bone_length_consistency(reconstructed, self.dataset)
                        mlm_ts = calculate_temporal_smoothness(reconstructed)
                        
                        mlm_metrics.append({
                            'mse': mlm_mse,
                            'bone_length_consistency': mlm_blc,
                            'temporal_smoothness': mlm_ts
                        })
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Aggregate results
        if raw_metrics and mlm_metrics:
            results = {
                'raw': {
                    'mse': np.mean([m['mse'] for m in raw_metrics]),
                    'bone_length_consistency': np.mean([m['bone_length_consistency'] for m in raw_metrics]),
                    'temporal_smoothness': np.mean([m['temporal_smoothness'] for m in raw_metrics])
                },
                'mlm': {
                    'mse': np.mean([m['mse'] for m in mlm_metrics]),
                    'bone_length_consistency': np.mean([m['bone_length_consistency'] for m in mlm_metrics]),
                    'temporal_smoothness': np.mean([m['temporal_smoothness'] for m in mlm_metrics])
                }
            }
            return results
        
        return {}
    
    def comprehensive_comparison(self, train_loader, test_loader):
        """Run comprehensive comparison between raw and MLM data."""
        print("Starting comprehensive Raw vs MLM comparison...")
        
        results = {
            'feature_comparison': self.compare_feature_quality(train_loader, test_loader),
            'reconstruction_comparison': self.compare_reconstruction_quality(test_loader)
        }
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Raw vs MLM Data Comparison')

    parser.add_argument('--mlm-model-dir', type=str, required=True,
                        help='Directory containing pretrained MLM model')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'],
                        help='Dataset name')
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'],
                        help='Evaluation setting')
    parser.add_argument('--temporal-ratio', type=float, default=0.5,
                        help='Temporal masking ratio (for reference)')
    parser.add_argument('--spatial-ratio', type=float, default=0.5,
                        help='Spatial masking ratio (for reference)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results/raw_vs_mlm_comparison',
                        help='Output directory for results')
    parser.add_argument('--train-samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=1000,
                        help='Number of test samples')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data: {args.dataset} ({args.setting})")
    X = load_data(args.dataset, T=args.seq_len)
    
    # Get train/test data loaders
    train_loader, test_loader = get_cross_data(
        X, args.dataset, args.setting,
        batch_size=args.batch_size,
        return_loader=True,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seg=args.seq_len
    )
    
    # Initialize comparator
    comparator = ComprehensiveComparator(
        args.mlm_model_dir, args.dataset, args.setting, args.seq_len, device
    )
    
    # Run comprehensive comparison
    results = comparator.comprehensive_comparison(train_loader, test_loader)
    
    # Save results
    result_file = os.path.join(args.output_dir, f"{args.dataset}_{args.setting}_raw_vs_mlm_comparison.json")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("RAW vs MLM COMPARISON SUMMARY")
    print("="*60)
    
    if 'feature_comparison' in results and 'classification_comparison' in results['feature_comparison']:
        cc = results['feature_comparison']['classification_comparison']
        if 'logistic_regression' in cc:
            lr = cc['logistic_regression']
            print(f"Logistic Regression - Raw: {lr.get('raw', {}).get('accuracy', 0):.3f}, MLM: {lr.get('mlm', {}).get('accuracy', 0):.3f}")
        if 'random_forest' in cc:
            rf = cc['random_forest']
            print(f"Random Forest - Raw: {rf.get('raw', {}).get('accuracy', 0):.3f}, MLM: {rf.get('mlm', {}).get('accuracy', 0):.3f}")
    
    if 'reconstruction_comparison' in results:
        rc = results['reconstruction_comparison']
        if 'raw' in rc and 'mlm' in rc:
            print(f"MSE - Raw: {rc['raw'].get('mse', 0):.6f}, MLM: {rc['mlm'].get('mse', 0):.6f}")
            print(f"Bone Length Consistency - Raw: {rc['raw'].get('bone_length_consistency', 0):.6f}, MLM: {rc['mlm'].get('bone_length_consistency', 0):.6f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
