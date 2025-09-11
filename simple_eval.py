#!/usr/bin/env python3
"""
Simple evaluation script to test current model performance.
This is a minimal working version to get baseline metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def load_model(model_path):
    """Load the transformer model."""
    try:
        from src.model.autoencoder import Model

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle different checkpoint formats - this is a direct state_dict from DataParallel
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            # Remove 'module.' prefix from DataParallel model
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create model with proper parameters
        model = Model(
            num_class=60,
            num_point=25,
            num_person=1,
            graph='src.graph.ntu_rgb_d.Graph',
            in_channels=3,
            dataset='ntu',
            device='cpu'
        )
        model.load_state_dict(state_dict)
        model.eval()

        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_data(dataset='ntu', setting='cv', samples=1000):
    """Load test data."""
    try:
        data_path = f"data/{dataset}_{setting}_paired_10000_2000.pt"
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return None, None
            
        data = torch.load(data_path, map_location='cpu')
        
        # Get test data
        test_data = data.get('test', data)
        
        # Limit samples
        if len(test_data) > samples:
            indices = np.random.choice(len(test_data), samples, replace=False)
            test_data = [test_data[i] for i in indices]
        
        print(f"✅ Loaded {len(test_data)} test samples from {data_path}")
        return test_data, data_path
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None, None

def evaluate_model(model, test_data, device='cpu'):
    """Simple evaluation - compute reconstruction loss."""
    if model is None or test_data is None:
        return {"status": "failed", "error": "Model or data not available"}
    
    model.to(device)
    total_loss = 0
    num_samples = 0
    
    print(f"🧪 Running evaluation on {len(test_data)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(test_data[:100]):  # Limit to 100 for speed
            try:
                # Extract data (format: x1, x2, y1, y2, actors, actions)
                if isinstance(sample, (list, tuple)) and len(sample) >= 6:
                    x1, x2, y1, y2, actors, actions = sample
                else:
                    continue
                    
                # Reshape from (T, V*C) to (N, C, T, V, M) format
                # (64, 75) -> (1, 3, 64, 25, 1)
                def reshape_data(data):
                    T, VC = data.shape  # (64, 75)
                    V, C = 25, 3
                    data = data.view(T, V, C)  # (64, 25, 3)
                    data = data.permute(2, 0, 1)  # (3, 64, 25) -> (C, T, V)
                    data = data.unsqueeze(0).unsqueeze(-1)  # (1, 3, 64, 25, 1) -> (N, C, T, V, M)
                    return data.to(device)

                x1 = reshape_data(x1)
                x2 = reshape_data(x2)
                y1 = reshape_data(y1)
                y2 = reshape_data(y2)
                
                # Forward pass
                output = model(x1, x2, target_motion=y1, teacher_forcing_ratio=0.0)
                
                # Compute MSE loss (handle sequence length mismatch)
                # Model might output shorter sequence due to autoregressive nature
                min_len = min(output.shape[2], y1.shape[2])
                output_trimmed = output[:, :, :min_len, :, :]
                y1_trimmed = y1[:, :, :min_len, :, :]
                loss = torch.nn.functional.mse_loss(output_trimmed, y1_trimmed)
                total_loss += loss.item()
                num_samples += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/100 samples...")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing sample {i}: {e}")
                continue
    
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        print(f"✅ Evaluation complete!")
        print(f"   Average MSE Loss: {avg_loss:.6f}")
        print(f"   Samples processed: {num_samples}")
        
        return {
            "status": "success",
            "metrics": {
                "mse_loss": avg_loss,
                "samples_processed": num_samples,
                "total_samples": len(test_data)
            }
        }
    else:
        return {"status": "failed", "error": "No samples processed successfully"}

def main():
    parser = argparse.ArgumentParser(description='Simple model evaluation')
    parser.add_argument('--model-path', default='data/models_output/model.pth', help='Path to model')
    parser.add_argument('--dataset', default='ntu', choices=['ntu', 'ntu120', 'etri'], help='Dataset')
    parser.add_argument('--setting', default='cv', choices=['cv', 'cs'], help='Setting')
    parser.add_argument('--samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--device', default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print("🚀 SIMPLE MODEL EVALUATION")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} ({args.setting})")
    print(f"Samples: {args.samples}")
    print(f"Device: {args.device}")
    print()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load data
    test_data, data_path = load_data(args.dataset, args.setting, args.samples)
    
    # Run evaluation
    results = evaluate_model(model, test_data, args.device)
    
    print("\n📊 RESULTS:")
    print("=" * 50)
    if results["status"] == "success":
        metrics = results["metrics"]
        print(f"✅ Status: SUCCESS")
        print(f"📈 MSE Loss: {metrics['mse_loss']:.6f}")
        print(f"📊 Samples: {metrics['samples_processed']}/{metrics['total_samples']}")
    else:
        print(f"❌ Status: FAILED")
        print(f"💥 Error: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    main()
