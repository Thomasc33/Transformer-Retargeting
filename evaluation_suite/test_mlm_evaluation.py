#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for MLM evaluation to verify the fixes work.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_comprehensive_evaluation():
    """Test the comprehensive MLM evaluation with a small sample."""
    
    # Import here to avoid module loading issues
    from evaluation_suite.comprehensive_mlm_evaluation import main as eval_main
    
    # Mock command line arguments for testing
    import sys
    original_argv = sys.argv
    
    try:
        # Set up test arguments
        sys.argv = [
            'comprehensive_mlm_evaluation.py',
            '--model-dir', 'eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3',
            '--dataset', 'ntu',
            '--setting', 'cv',
            '--temporal-ratio', '0.3',
            '--spatial-ratio', '0.3',
            '--seq-len', '64',
            '--batch-size', '4',  # Small batch for testing
            '--output-dir', 'results/test_mlm_evaluation',
            '--train-samples', '50',  # Very small for testing
            '--test-samples', '20',
            '--debug'
        ]
        
        print("Testing comprehensive MLM evaluation...")
        eval_main()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    return True


def test_feature_extraction():
    """Test just the feature extraction part."""
    
    try:
        import torch
        from data import load_data, get_cross_data
        from evaluation_suite.mlm_feature_classifier import MLMFeatureExtractor
        
        print("Testing feature extraction...")
        
        # Load small amount of data
        X = load_data('ntu', T=64)
        print(f"Loaded data with {len(X)} samples")
        
        # Get small data loaders
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=2,
            return_loader=True,
            train_samples=10,
            test_samples=5,
            seg=64
        )
        
        print(f"Created data loaders: train={len(train_loader)}, test={len(test_loader)}")
        
        # Test data format
        for batch_idx, batch_content in enumerate(test_loader):
            print(f"Batch {batch_idx}: type={type(batch_content)}")
            if isinstance(batch_content, tuple):
                print(f"  Tuple length: {len(batch_content)}")
                for i, item in enumerate(batch_content):
                    print(f"    Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            break
        
        print("Data format test completed!")
        return True
        
    except Exception as e:
        print(f"Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test MLM Evaluation')
    parser.add_argument('--test', choices=['feature', 'comprehensive', 'both'], 
                       default='feature', help='Which test to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLM EVALUATION TEST SUITE")
    print("=" * 60)
    
    success = True
    
    if args.test in ['feature', 'both']:
        print("\n1. Testing feature extraction...")
        success &= test_feature_extraction()
    
    if args.test in ['comprehensive', 'both']:
        print("\n2. Testing comprehensive evaluation...")
        success &= test_comprehensive_evaluation()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
