#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step-by-step debugging script for MLM evaluation.
"""

import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_step_1_data_loading():
    """Test step 1: Basic data loading."""
    print("=" * 60)
    print("STEP 1: Testing basic data loading")
    print("=" * 60)
    
    try:
        from data import load_data
        print("✓ Successfully imported load_data")
        
        X = load_data('ntu', T=64)
        print(f"✓ Successfully loaded data: {len(X)} samples")
        
        # Check a sample
        sample_key = list(X.keys())[0]
        sample_data = X[sample_key]
        print(f"✓ Sample data shape: {sample_data.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_2_cross_data():
    """Test step 2: Cross data generation."""
    print("\n" + "=" * 60)
    print("STEP 2: Testing cross data generation")
    print("=" * 60)
    
    try:
        from data import load_data, get_cross_data
        
        X = load_data('ntu', T=64)
        print("✓ Data loaded")
        
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=2,
            return_loader=True,
            train_samples=10,
            test_samples=5,
            seg=64
        )
        print(f"✓ Data loaders created: train={len(train_loader)}, test={len(test_loader)}")
        
        # Test first batch
        for batch_idx, batch_content in enumerate(test_loader):
            print(f"✓ Batch {batch_idx}: type={type(batch_content)}")
            if isinstance(batch_content, tuple):
                print(f"  Tuple length: {len(batch_content)}")
                for i, item in enumerate(batch_content):
                    print(f"    Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_3_model_loading():
    """Test step 3: Model loading."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing model loading")
    print("=" * 60)
    
    try:
        model_dir = "eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3"
        
        print(f"Checking model directory: {model_dir}")
        if not os.path.exists(model_dir):
            print(f"✗ Model directory does not exist: {model_dir}")
            return False
        
        files = os.listdir(model_dir)
        print(f"✓ Model directory exists with files: {files}")
        
        # Check for required files
        required_files = ['encoder_best.pth', 'decoder_best.pth', 'output_layer_best.pth']
        for file in required_files:
            if file in files:
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
        
        # Try loading the model
        from evaluation_suite.mlm_feature_classifier import MLMFeatureExtractor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        extractor = MLMFeatureExtractor(model_dir, 'ntu', 64, device)
        print("✓ MLM model loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_4_feature_extraction():
    """Test step 4: Feature extraction."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing feature extraction")
    print("=" * 60)
    
    try:
        from data import load_data, get_cross_data
        from evaluation_suite.mlm_feature_classifier import MLMFeatureExtractor
        
        # Load data
        X = load_data('ntu', T=64)
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=2,
            return_loader=True,
            train_samples=10,
            test_samples=5,
            seg=64
        )
        
        # Load model
        model_dir = "eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = MLMFeatureExtractor(model_dir, 'ntu', 64, device)
        
        # Extract features
        print("Attempting feature extraction...")
        train_features, train_action_labels, train_actor_labels = extractor.extract_features(train_loader)
        
        print(f"✓ Feature extraction successful!")
        print(f"  Features shape: {train_features.shape}")
        print(f"  Action labels shape: {train_action_labels.shape}")
        print(f"  Actor labels shape: {train_actor_labels.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all debugging steps."""
    print("MLM EVALUATION STEP-BY-STEP DEBUGGING")
    print("This will help identify exactly where the issue occurs.")
    
    steps = [
        ("Data Loading", test_step_1_data_loading),
        ("Cross Data Generation", test_step_2_cross_data),
        ("Model Loading", test_step_3_model_loading),
        ("Feature Extraction", test_step_4_feature_extraction),
    ]
    
    for step_name, step_func in steps:
        success = step_func()
        if not success:
            print(f"\n{'='*60}")
            print(f"DEBUGGING STOPPED AT: {step_name}")
            print(f"Please fix the issue above before proceeding.")
            print(f"{'='*60}")
            return False
    
    print(f"\n{'='*60}")
    print("ALL STEPS PASSED!")
    print("The MLM evaluation should work correctly now.")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
