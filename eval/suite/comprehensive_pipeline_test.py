#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive MLM Pipeline Test

This script performs end-to-end testing of the MLM evaluation pipeline
to ensure all components work together properly.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrain import SkeletonAutoEncoder
from src.data import load_data, get_cross_data
from evaluation_suite.comprehensive_mlm_evaluation import ComprehensiveMLMEvaluator
from evaluation_suite.raw_vs_mlm_comparison import ComprehensiveComparator


def test_data_loading():
    """Test data loading functionality."""
    print("=== Testing Data Loading ===")
    
    try:
        # Load small dataset
        X = load_data('ntu', T=64)
        print(f"✓ Data loaded: {len(X)} samples")
        
        # Test data loaders
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=4,
            return_loader=True,
            train_samples=20,
            test_samples=10,
            seg=64
        )
        
        print(f"✓ Data loaders created: {len(train_loader)} train batches, {len(test_loader)} test batches")
        
        # Test batch format
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 1:  # Only test first batch
                break
            
            if isinstance(batch, (tuple, list)) and len(batch) == 6:
                x1, x2, y1, y2, actors, actions = batch
                print(f"✓ Batch format correct: x1 shape {x1.shape}")
            else:
                print(f"✗ Unexpected batch format: {type(batch)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_mlm_model_loading():
    """Test MLM model loading and forward pass."""
    print("\n=== Testing MLM Model Loading ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find available MLM models
    base_dir = "eval/mixformer/pretrained/ntu"
    if not os.path.exists(base_dir):
        print(f"✗ MLM model directory not found: {base_dir}")
        return False
    
    # Look for comprehensive models
    model_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and 'comprehensive' in d]
    
    if not model_dirs:
        print(f"✗ No comprehensive MLM models found in {base_dir}")
        return False
    
    # Test first available model
    model_dir = os.path.join(base_dir, model_dirs[0])
    print(f"Testing model: {model_dir}")
    
    try:
        # Load MLM model
        model = SkeletonAutoEncoder(dataset='ntu', seq_len=64).to(device)
        
        # Load components
        encoder_path = os.path.join(model_dir, 'encoder_best.pth')
        decoder_path = os.path.join(model_dir, 'decoder_best.pth')
        output_layer_path = os.path.join(model_dir, 'output_layer_best.pth')
        
        if not all(os.path.exists(p) for p in [encoder_path, decoder_path, output_layer_path]):
            print(f"✗ Missing model files in {model_dir}")
            return False
        
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        model.output_layer.load_state_dict(torch.load(output_layer_path, map_location=device))
        
        model.eval()
        print("✓ MLM model loaded successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 64, 75).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful: input {dummy_input.shape} -> output {output.shape}")
        
        return True, model_dir
        
    except Exception as e:
        print(f"✗ MLM model loading failed: {e}")
        return False, None


def test_comprehensive_evaluation():
    """Test comprehensive MLM evaluation."""
    print("\n=== Testing Comprehensive Evaluation ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test model loading first
    success, model_dir = test_mlm_model_loading()
    if not success:
        return False
    
    try:
        # Load small dataset
        X = load_data('ntu', T=64)
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=4,
            return_loader=True,
            train_samples=20,
            test_samples=10,
            seg=64
        )
        
        # Initialize evaluator
        evaluator = ComprehensiveMLMEvaluator(
            model_dir, 'ntu', 'cv', 64, device
        )
        
        print("✓ Comprehensive evaluator initialized")
        
        # Test feature extraction
        print("Testing feature extraction...")
        classification_results, models = evaluator.evaluate_feature_classification(train_loader, test_loader)
        
        print(f"✓ Feature classification completed")
        print(f"  AR Accuracy: {classification_results['action_recognition']['accuracy']:.3f}")
        print(f"  RI Accuracy: {classification_results['re_identification']['accuracy']:.3f}")
        
        # Test reconstruction evaluation
        print("Testing reconstruction evaluation...")
        plausibility_results = evaluator.evaluate_reconstruction_and_plausibility(test_loader)
        
        print(f"✓ Reconstruction evaluation completed")
        print(f"  MSE: {plausibility_results['reconstruction_mse']:.6f}")
        print(f"  Bone Length Consistency: {plausibility_results['bone_length_consistency']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_vs_mlm_comparison():
    """Test raw vs MLM comparison."""
    print("\n=== Testing Raw vs MLM Comparison ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test model loading first
    success, model_dir = test_mlm_model_loading()
    if not success:
        return False
    
    try:
        # Load small dataset
        X = load_data('ntu', T=64)
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=4,
            return_loader=True,
            train_samples=20,
            test_samples=10,
            seg=64
        )
        
        # Initialize comparator
        comparator = ComprehensiveComparator(
            model_dir, 'ntu', 'cv', 64, device
        )
        
        print("✓ Comparator initialized")
        
        # Run comparison
        results = comparator.comprehensive_comparison(train_loader, test_loader)
        
        print("✓ Raw vs MLM comparison completed")
        
        # Print results
        if 'feature_comparison' in results and 'classification_comparison' in results['feature_comparison']:
            cc = results['feature_comparison']['classification_comparison']
            if 'logistic_regression' in cc:
                lr = cc['logistic_regression']
                print(f"  Logistic Regression - Raw: {lr.get('raw', {}).get('accuracy', 0):.3f}, MLM: {lr.get('mlm', {}).get('accuracy', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Raw vs MLM comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization components."""
    print("\n=== Testing Visualization ===")
    
    try:
        from evaluation_suite.mlm_visualizer import create_comparison_gif, create_overlay_gif
        
        # Create dummy skeleton data
        original = np.random.randn(10, 25, 3)
        reconstructed = original + np.random.randn(10, 25, 3) * 0.1
        
        # Test GIF creation (without actually saving)
        temp_dir = "/tmp/test_viz"
        os.makedirs(temp_dir, exist_ok=True)
        
        sample_info = {
            'action': 'test',
            'actor': 'test',
            'dataset': 'ntu',
            'setting': 'cv'
        }
        
        # Test comparison GIF
        comparison_path = os.path.join(temp_dir, "test_comparison.gif")
        create_comparison_gif(original, reconstructed, comparison_path, sample_info, max_frames=5)
        
        if os.path.exists(comparison_path):
            print("✓ Comparison GIF created successfully")
            os.remove(comparison_path)
        
        # Test overlay GIF
        overlay_path = os.path.join(temp_dir, "test_overlay.gif")
        create_overlay_gif(original, reconstructed, overlay_path, sample_info, max_frames=5)
        
        if os.path.exists(overlay_path):
            print("✓ Overlay GIF created successfully")
            os.remove(overlay_path)
        
        # Clean up
        os.rmdir(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and generate report."""
    print("🚀 Comprehensive MLM Pipeline Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("MLM Model Loading", lambda: test_mlm_model_loading()[0]),
        ("Comprehensive Evaluation", test_comprehensive_evaluation),
        ("Raw vs MLM Comparison", test_raw_vs_mlm_comparison),
        ("Visualization", test_visualization),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            result = test_func()
            test_results[test_name] = result
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            test_results[test_name] = False
            print(f"{test_name}: ✗ FAILED - {e}")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25}: {status}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! MLM pipeline is ready for evaluation.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    print("=" * 60)
    
    return test_results


if __name__ == "__main__":
    run_comprehensive_test()
