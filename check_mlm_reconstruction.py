#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check MLM Reconstruction Quality

This script evaluates the reconstruction quality of MLM pretrained models
to determine if they are learning meaningful representations.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrain import SkeletonAutoEncoder
from data import load_data, get_cross_data


def check_reconstruction_quality(model_dir, dataset='ntu', setting='cv', seq_len=64):
    """Check the reconstruction quality of an MLM model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Checking reconstruction quality for: {model_dir}")
    
    # Load model
    model = SkeletonAutoEncoder(dataset=dataset, seq_len=seq_len).to(device)
    
    # Load all components
    encoder_path = os.path.join(model_dir, 'encoder_best.pth')
    decoder_path = os.path.join(model_dir, 'decoder_best.pth')
    output_layer_path = os.path.join(model_dir, 'output_layer_best.pth')
    
    if os.path.exists(encoder_path):
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print("✅ Encoder loaded")
    else:
        print("❌ Encoder not found")
        return None
        
    if os.path.exists(decoder_path):
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print("✅ Decoder loaded")
    else:
        print("❌ Decoder not found")
        return None
        
    if os.path.exists(output_layer_path):
        model.output_layer.load_state_dict(torch.load(output_layer_path, map_location=device))
        print("✅ Output layer loaded")
    else:
        print("❌ Output layer not found")
        return None
    
    model.eval()
    
    # Load test data
    X = load_data(dataset, T=seq_len)
    _, test_loader = get_cross_data(
        X, dataset, setting,
        batch_size=8,
        return_loader=True,
        train_samples=50,
        test_samples=50,
        seg=seq_len
    )
    
    reconstruction_errors = []
    original_stats = []
    reconstructed_stats = []
    
    print("Evaluating reconstruction quality...")
    
    with torch.no_grad():
        for batch_idx, batch_content in enumerate(tqdm(test_loader, desc="Reconstruction evaluation")):
            if batch_idx >= 10:  # Only check first 10 batches
                break
                
            try:
                # Parse batch content
                if isinstance(batch_content, (tuple, list)) and len(batch_content) == 6:
                    x1, x2, y1, y2, actors, actions = batch_content
                    x_data = x1
                else:
                    continue
                
                # Process data format
                if not isinstance(x_data, torch.Tensor):
                    x_data = torch.tensor(x_data, dtype=torch.float32)
                
                # Reshape to (batch, frames, joints, channels)
                if len(x_data.shape) == 3 and x_data.shape[2] == 75:
                    x_data = x_data.view(x_data.shape[0], x_data.shape[1], 25, 3)
                elif len(x_data.shape) == 2 and x_data.shape[1] == 75:
                    x_data = x_data.view(1, x_data.shape[0], 25, 3)
                
                x_data = x_data.to(device)
                batch_size = x_data.shape[0]
                
                # Store original data statistics
                for i in range(batch_size):
                    sample = x_data[i].cpu().numpy()
                    original_stats.append({
                        'mean': np.mean(sample),
                        'std': np.std(sample),
                        'min': np.min(sample),
                        'max': np.max(sample)
                    })
                
                # Prepare for encoder: (batch, channels, frames, joints, persons)
                x_encoder = x_data.permute(0, 3, 1, 2).unsqueeze(-1)
                
                # Forward pass through full autoencoder
                try:
                    reconstructed = model(x_encoder)
                    
                    print(f"Batch {batch_idx}:")
                    print(f"  Input shape: {x_encoder.shape}")
                    print(f"  Output shape: {reconstructed.shape}")
                    
                    # Calculate reconstruction error
                    if reconstructed.shape != x_encoder.shape:
                        print(f"  ⚠️ Shape mismatch: input {x_encoder.shape}, output {reconstructed.shape}")
                        
                        # Try to match shapes for error calculation
                        if len(reconstructed.shape) == 5:
                            # Reshape reconstructed to match input
                            if reconstructed.shape[0] == x_encoder.shape[0]:  # Batch size matches
                                # Try to interpolate or crop to match
                                target_shape = x_encoder.shape
                                if reconstructed.shape[2] != target_shape[2]:  # Time dimension
                                    # Interpolate time dimension
                                    reconstructed = torch.nn.functional.interpolate(
                                        reconstructed.squeeze(-1), 
                                        size=(target_shape[2], target_shape[3]), 
                                        mode='bilinear', 
                                        align_corners=False
                                    ).unsqueeze(-1)
                                
                                if reconstructed.shape == target_shape:
                                    mse = torch.mean((reconstructed - x_encoder) ** 2).item()
                                    reconstruction_errors.append(mse)
                                    print(f"  MSE: {mse:.6f}")
                                else:
                                    print(f"  ⚠️ Could not match shapes for MSE calculation")
                            else:
                                print(f"  ⚠️ Batch size mismatch")
                        else:
                            print(f"  ⚠️ Unexpected output dimensions")
                    else:
                        # Shapes match, calculate MSE directly
                        mse = torch.mean((reconstructed - x_encoder) ** 2).item()
                        reconstruction_errors.append(mse)
                        print(f"  MSE: {mse:.6f}")
                    
                    # Store reconstructed data statistics
                    if len(reconstructed.shape) == 5:
                        # Convert back to (batch, frames, joints, channels)
                        reconstructed_data = reconstructed.squeeze(-1).permute(0, 2, 3, 1)
                        for i in range(min(batch_size, reconstructed_data.shape[0])):
                            sample = reconstructed_data[i].cpu().numpy()
                            reconstructed_stats.append({
                                'mean': np.mean(sample),
                                'std': np.std(sample),
                                'min': np.min(sample),
                                'max': np.max(sample)
                            })
                    
                except Exception as e:
                    print(f"  ❌ Forward pass error: {e}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                continue
    
    # Analyze results
    print(f"\n=== Reconstruction Quality Analysis ===")
    
    if reconstruction_errors:
        print(f"Reconstruction MSE:")
        print(f"  Mean: {np.mean(reconstruction_errors):.6f}")
        print(f"  Std: {np.std(reconstruction_errors):.6f}")
        print(f"  Min: {np.min(reconstruction_errors):.6f}")
        print(f"  Max: {np.max(reconstruction_errors):.6f}")
    else:
        print("❌ No reconstruction errors calculated")
    
    if original_stats and reconstructed_stats:
        print(f"\nOriginal Data Statistics:")
        orig_means = [s['mean'] for s in original_stats]
        orig_stds = [s['std'] for s in original_stats]
        print(f"  Mean: {np.mean(orig_means):.4f} ± {np.std(orig_means):.4f}")
        print(f"  Std: {np.mean(orig_stds):.4f} ± {np.std(orig_stds):.4f}")
        
        print(f"\nReconstructed Data Statistics:")
        recon_means = [s['mean'] for s in reconstructed_stats]
        recon_stds = [s['std'] for s in reconstructed_stats]
        print(f"  Mean: {np.mean(recon_means):.4f} ± {np.std(recon_means):.4f}")
        print(f"  Std: {np.mean(recon_stds):.4f} ± {np.std(recon_stds):.4f}")
        
        # Check if reconstructed data has reasonable statistics
        mean_diff = abs(np.mean(orig_means) - np.mean(recon_means))
        std_diff = abs(np.mean(orig_stds) - np.mean(recon_stds))
        
        print(f"\nStatistical Differences:")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Std difference: {std_diff:.4f}")
        
        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ Reconstructed data has similar statistics to original")
        else:
            print("⚠️ Reconstructed data statistics differ significantly from original")
    
    return {
        'reconstruction_errors': reconstruction_errors,
        'original_stats': original_stats,
        'reconstructed_stats': reconstructed_stats
    }


def main():
    """Check reconstruction quality for all available MLM models."""
    
    print("🔍 Checking MLM Reconstruction Quality")
    print("=" * 60)
    
    dataset = 'ntu'
    setting = 'cv'
    
    # Find all available MLM models
    import glob
    pattern = f"eval/mixformer/pretrained/{dataset}/epochs_{setting}_comprehensive_temporal_*_spatial_*"
    available_dirs = glob.glob(pattern)
    
    if not available_dirs:
        print("❌ No MLM models found!")
        return
    
    print(f"Found {len(available_dirs)} MLM models")
    
    results = {}
    
    # Check first few models
    for i, model_dir in enumerate(available_dirs[:3]):  # Check first 3 models
        print(f"\n{'='*60}")
        print(f"Model {i+1}/{min(3, len(available_dirs))}: {os.path.basename(model_dir)}")
        print(f"{'='*60}")
        
        result = check_reconstruction_quality(model_dir, dataset, setting)
        if result:
            results[model_dir] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for model_dir, result in results.items():
        model_name = os.path.basename(model_dir)
        if result['reconstruction_errors']:
            avg_mse = np.mean(result['reconstruction_errors'])
            print(f"{model_name}: Average MSE = {avg_mse:.6f}")
        else:
            print(f"{model_name}: No reconstruction errors calculated")


if __name__ == "__main__":
    main()
