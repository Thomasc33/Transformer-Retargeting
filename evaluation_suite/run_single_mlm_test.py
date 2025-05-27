#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run a single MLM evaluation test with comprehensive debugging.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_mlm_test():
    """Run MLM evaluation test."""
    
    # Default parameters
    model_dir = "eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3"
    dataset = "ntu"
    setting = "cv"
    temporal_ratio = 0.3
    spatial_ratio = 0.3
    
    print("=" * 80)
    print("MLM EVALUATION TEST")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Dataset: {dataset}")
    print(f"Setting: {setting}")
    print(f"Temporal ratio: {temporal_ratio}")
    print(f"Spatial ratio: {spatial_ratio}")
    print()
    
    try:
        # Import modules
        print("Importing modules...")
        import torch
        from data import load_data, get_cross_data
        from evaluation_suite.comprehensive_mlm_evaluation import ComprehensiveMLMEvaluator
        print("✓ Modules imported successfully")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        # Load data
        print("\nLoading data...")
        X = load_data(dataset, T=64)
        print(f"✓ Loaded {len(X)} samples")
        
        # Create data loaders with small sample sizes for testing
        print("Creating data loaders...")
        train_loader, test_loader = get_cross_data(
            X, dataset, setting,
            batch_size=4,  # Small batch size
            return_loader=True,
            train_samples=20,  # Small number for testing
            test_samples=10,
            seg=64
        )
        print(f"✓ Created data loaders: train={len(train_loader)}, test={len(test_loader)}")
        
        # Initialize evaluator
        print("\nInitializing evaluator...")
        evaluator = ComprehensiveMLMEvaluator(
            model_dir, dataset, setting, 64, device
        )
        print("✓ Evaluator initialized")
        
        # Run evaluation
        print("\nRunning comprehensive evaluation...")
        results, models = evaluator.comprehensive_evaluation(
            train_loader, test_loader,
            create_visualizations=False,  # Skip visualizations for testing
            temporal_ratio=temporal_ratio,
            spatial_ratio=spatial_ratio
        )
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print results
        classification = results['classification']
        plausibility = results['physical_plausibility']
        
        print(f"Action Recognition Accuracy: {classification['action_recognition']['accuracy']:.4f}")
        print(f"Re-Identification Accuracy: {classification['re_identification']['accuracy']:.4f}")
        print(f"Reconstruction MSE: {plausibility['reconstruction_mse']:.6f}")
        print(f"Bone Length Consistency: {plausibility['bone_length_consistency']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("EVALUATION FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run single MLM evaluation test')
    parser.add_argument('--model-dir', type=str, 
                       default="eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3",
                       help='Model directory')
    parser.add_argument('--dataset', type=str, default='ntu', help='Dataset')
    parser.add_argument('--setting', type=str, default='cv', help='Setting')
    
    args = parser.parse_args()
    
    success = run_mlm_test()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
