#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick test to verify the list/tuple fix works.
"""

import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_list_handling():
    """Test that the feature extractor can handle lists."""
    
    try:
        from data import load_data, get_cross_data
        from evaluation_suite.mlm_feature_classifier import MLMFeatureExtractor
        
        print("Testing list/tuple handling fix...")
        
        # Load small amount of data
        X = load_data('ntu', T=64)
        train_loader, test_loader = get_cross_data(
            X, 'ntu', 'cv',
            batch_size=2,
            return_loader=True,
            train_samples=10,
            test_samples=5,
            seg=64
        )
        
        print("Data loaders created successfully")
        
        # Check what the data loader actually returns
        for batch_idx, batch_content in enumerate(test_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Type: {type(batch_content)}")
            print(f"  Length: {len(batch_content) if hasattr(batch_content, '__len__') else 'N/A'}")
            
            if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                x1, x2, y1, y2, actors, actions = batch_content
                print(f"  x1 shape: {x1.shape}")
                print(f"  actions shape: {actions.shape}")
                print(f"  actors shape: {actors.shape}")
                print("  ✓ Successfully unpacked 6-element batch")
            else:
                print(f"  ✗ Cannot unpack batch")
            
            if batch_idx >= 2:  # Only check first few batches
                break
        
        # Now test feature extraction
        model_dir = "eval/mixformer/pretrained/ntu/epochs_cv_comprehensive_temporal_0.3_spatial_0.3"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\nTesting feature extraction with device: {device}")
        extractor = MLMFeatureExtractor(model_dir, 'ntu', 64, device)
        
        # Extract features from a very small subset
        small_loader_data = []
        for i, batch in enumerate(test_loader):
            small_loader_data.append(batch)
            if i >= 1:  # Only take 2 batches
                break
        
        # Create a minimal data loader
        from torch.utils.data import DataLoader
        
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        mini_loader = DataLoader(SimpleDataset(small_loader_data), batch_size=1, shuffle=False)
        
        print("Attempting feature extraction on mini dataset...")
        features, action_labels, actor_labels = extractor.extract_features(mini_loader)
        
        print(f"✓ Feature extraction successful!")
        print(f"  Features shape: {features.shape}")
        print(f"  Action labels shape: {action_labels.shape}")
        print(f"  Actor labels shape: {actor_labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_list_handling()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
