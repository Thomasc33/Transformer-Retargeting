#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to understand the data format from get_cross_data.
"""

import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data import load_data, get_cross_data


def debug_data_loader():
    """Debug the data loader format."""
    print("Loading NTU data...")
    X = load_data('ntu', T=64)
    print(f"Loaded data shape: {X.shape}")
    
    print("\nGetting cross-view data loaders...")
    train_loader, test_loader = get_cross_data(
        X, 'ntu', 'cv',
        batch_size=4,  # Small batch for debugging
        return_loader=True,
        train_samples=100,  # Small number for debugging
        test_samples=20,
        seg=64
    )
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Test loader length: {len(test_loader)}")
    
    print("\nInspecting first batch from train loader...")
    for batch_idx, batch_content in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Type: {type(batch_content)}")
        
        if isinstance(batch_content, tuple):
            print(f"  Tuple length: {len(batch_content)}")
            for i, item in enumerate(batch_content):
                print(f"    Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
                if hasattr(item, 'dtype'):
                    print(f"             dtype={item.dtype}")
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    print(f"             first element: {item[0]}")
        else:
            print(f"  Shape: {getattr(batch_content, 'shape', 'N/A')}")
            print(f"  Type: {type(batch_content)}")
        
        # Only check first batch
        break
    
    print("\nInspecting first batch from test loader...")
    for batch_idx, batch_content in enumerate(test_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Type: {type(batch_content)}")
        
        if isinstance(batch_content, tuple):
            print(f"  Tuple length: {len(batch_content)}")
            for i, item in enumerate(batch_content):
                print(f"    Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
                if hasattr(item, 'dtype'):
                    print(f"             dtype={item.dtype}")
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    print(f"             first element: {item[0]}")
        else:
            print(f"  Shape: {getattr(batch_content, 'shape', 'N/A')}")
            print(f"  Type: {type(batch_content)}")
        
        # Only check first batch
        break


if __name__ == "__main__":
    debug_data_loader()
