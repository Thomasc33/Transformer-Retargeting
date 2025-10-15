#!/usr/bin/env python3
"""
Same-Action Evaluation Script

This script evaluates TMR on pairs where both actors perform the SAME action.
This is a more controlled test since the action is consistent across both inputs.

Expected behavior:
- AR (Action Recognition): Should be higher since action is preserved
- RI (Re-Identification): Should still be high (privacy preserved)

Usage:
    python scripts/eval_same_action.py --dataset ntu_cv --num_pairs 100 --device cuda
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import data utilities
from src.data import load_data

# Import models
from src.model.autoencoder import Model as TMRModel
from src.model.ske_mixf import Model as MixformerModel
from src.model.sgn import SGN

# Import preprocessing
from eval.preprocess import mixformer_preprocess_single_skeleton, sgn_preprocess_single_skeleton


def load_same_action_pairs(dataset_name, num_pairs=100):
    """
    Load pairs where both actors perform the same action.
    
    Args:
        dataset_name: Name of dataset (e.g., 'ntu_cv')
        num_pairs: Number of pairs to sample
        
    Returns:
        List of samples where actions[0] == actions[1]
    """
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*80}\n")
    
    # Load full dataset
    data_path = PROJECT_ROOT / "data" / f"{dataset_name}_paired_comprehensive.pt"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / f"{dataset_name}_paired_10000_2000.pt"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading from: {data_path}")
    data = torch.load(data_path)
    
    # Extract test data
    if isinstance(data, dict):
        test_data = data.get('test', data.get('test_data', None))
    else:
        test_data = data
    
    print(f"Total test samples: {len(test_data)}")
    
    # Filter for same-action pairs
    same_action_samples = []
    for sample in tqdm(test_data, desc="Filtering same-action pairs"):
        # sample[5] = actions = [a1, a2]
        actions = sample[5]
        if actions[0] == actions[1]:
            same_action_samples.append(sample)
    
    print(f"\nFound {len(same_action_samples)} same-action pairs")
    
    # Sample requested number
    if len(same_action_samples) < num_pairs:
        print(f"⚠️  Warning: Only {len(same_action_samples)} pairs available, using all")
        return same_action_samples
    
    # Random sample
    indices = np.random.choice(len(same_action_samples), num_pairs, replace=False)
    sampled = [same_action_samples[i] for i in indices]
    
    print(f"Sampled {len(sampled)} pairs for evaluation")
    
    return sampled


def load_state_dict_with_module_fix(model_path, device='cuda'):
    """Load state dict and handle 'module.' prefix from DataParallel."""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state dict if checkpoint is a dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def evaluate_tmr_same_action(samples, dataset='ntu', setting='cv', device='cuda'):
    """
    Evaluate TMR on same-action pairs.

    Args:
        samples: List of same-action samples
        dataset: Dataset name (ntu, ntu120)
        setting: Setting (cv, cs)
        device: Device to run on

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating TMR on Same-Action Pairs")
    print(f"{'='*80}\n")

    # Load TMR model
    model_path = PROJECT_ROOT / "data" / "models" / "tmr" / "model.pth"
    if not model_path.exists():
        # Try alternative path
        model_path = PROJECT_ROOT / "models" / "tmr" / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"TMR model not found: {model_path}")

    print(f"Loading TMR from: {model_path}")
    num_class = 60 if dataset == 'ntu' else 120
    tmr = TMRModel(
        num_class=num_class,
        num_point=25,
        num_person=1,
        graph='src.graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
        dataset=dataset,
        device=device
    ).to(device)

    state_dict = load_state_dict_with_module_fix(model_path, device)
    tmr.load_state_dict(state_dict, strict=False)
    tmr.eval()
    print("✓ TMR model loaded")

    # Load AR model (Mixformer)
    print("\nLoading AR model (Mixformer)...")
    setting_suffix = 'cview' if setting == 'cv' else 'csub'
    ar_model_path = PROJECT_ROOT / "data" / "models_output" / "output" / f"{dataset}_mixformer_ar_{setting_suffix}" / f"NTU_mixformer_ar_{setting_suffix}" / "model_best.pth.tar"

    if not ar_model_path.exists():
        raise FileNotFoundError(f"AR model not found: {ar_model_path}")

    ar_model = MixformerModel(
        num_class=num_class,
        num_point=25,
        num_person=2,
        graph='src.graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3
    ).to(device)

    state_dict = load_state_dict_with_module_fix(ar_model_path, device)
    ar_model.load_state_dict(state_dict, strict=False)
    ar_model.eval()
    print(f"✓ AR model loaded from {ar_model_path}")

    # Load RI model (SGN)
    print("\nLoading RI model (SGN)...")
    ri_model_path = PROJECT_ROOT / "data" / "models_output" / "output" / f"{dataset}_sgn_ri_{setting_suffix}" / f"NTU_sgn_ri_{setting_suffix}" / "model_best.pth.tar"

    if not ri_model_path.exists():
        raise FileNotFoundError(f"RI model not found: {ri_model_path}")

    num_subjects = 40 if dataset == 'ntu' else 106
    ri_model = SGN(
        num_classes=num_subjects,
        seg=20,
        bias=True,
        dataset=dataset
    ).to(device)

    state_dict = load_state_dict_with_module_fix(ri_model_path, device)
    ri_model.load_state_dict(state_dict, strict=False)
    ri_model.eval()
    print(f"✓ RI model loaded from {ri_model_path}")
    
    # Evaluation
    ar_correct = 0
    ri_correct = 0
    total = len(samples)

    all_ar_preds = []
    all_ar_labels = []
    all_ri_preds = []
    all_ri_labels = []

    print(f"\nEvaluating {total} same-action pairs...")

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            # Extract data
            x1 = sample[0]  # P1 doing A1 (C, T, V, M)
            x2 = sample[1]  # P2 doing A1 (same action!)
            actors = sample[4]  # [p1, p2]
            actions = sample[5]  # [a1, a1] (same!)

            # Prepare inputs for TMR (needs batch dimension and proper shape)
            x1_batch = x1.unsqueeze(0).to(device)  # (1, C, T, V, M)
            x2_batch = x2.unsqueeze(0).to(device)  # (1, C, T, V, M)

            # Generate retargeted skeleton: P1 → P2 with action A1
            # TMR forward: (source_motion, dummy_skeleton, target_motion, teacher_forcing_ratio)
            retargeted = tmr(x1_batch, x2_batch, target_motion=None, teacher_forcing_ratio=0.0)
            # retargeted shape: (1, C, T-1, V, M)

            # AR Evaluation: Does retargeted skeleton show action A1?
            # Preprocess for Mixformer (needs 2 persons)
            ar_input = mixformer_preprocess_single_skeleton(retargeted.squeeze(0))  # (C, T, V, M) -> (C, T, V, 2)
            ar_input = ar_input.unsqueeze(0).to(device)  # (1, C, T, V, 2)

            ar_logits = ar_model(ar_input)
            ar_pred = ar_logits.argmax(dim=1).item()
            ar_label = actions[0]  # Use the shared action label
            all_ar_preds.append(ar_pred)
            all_ar_labels.append(ar_label)
            if ar_pred == ar_label:
                ar_correct += 1

            # RI Evaluation: Does retargeted skeleton look like P2?
            # Preprocess for SGN
            ri_input = sgn_preprocess_single_skeleton(retargeted.squeeze(0))  # (C, T, V, M) -> (C, T, V, M)
            ri_input = ri_input.unsqueeze(0).to(device)  # (1, C, T, V, M)

            ri_logits = ri_model(ri_input)
            ri_pred = ri_logits.argmax(dim=1).item()
            ri_label = actors[1]  # Target person is P2
            all_ri_preds.append(ri_pred)
            all_ri_labels.append(ri_label)
            if ri_pred == ri_label:
                ri_correct += 1
    
    # Calculate metrics
    ar_accuracy = (ar_correct / total) * 100
    ri_accuracy = (ri_correct / total) * 100
    
    results = {
        'total_samples': total,
        'ar_accuracy': ar_accuracy,
        'ri_accuracy': ri_accuracy,
        'ar_correct': ar_correct,
        'ri_correct': ri_correct,
        'ar_predictions': all_ar_preds,
        'ar_labels': all_ar_labels,
        'ri_predictions': all_ri_preds,
        'ri_labels': all_ri_labels,
    }
    
    return results


def print_results(results):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"RESULTS: Same-Action Evaluation")
    print(f"{'='*80}\n")
    
    print(f"Total Samples:     {results['total_samples']}")
    print(f"\nAction Recognition (AR):")
    print(f"  Accuracy:        {results['ar_accuracy']:.2f}%")
    print(f"  Correct:         {results['ar_correct']}/{results['total_samples']}")
    
    print(f"\nRe-Identification (RI):")
    print(f"  Accuracy:        {results['ri_accuracy']:.2f}%")
    print(f"  Correct:         {results['ri_correct']}/{results['total_samples']}")
    
    print(f"\n{'='*80}")
    print(f"INTERPRETATION:")
    print(f"{'='*80}\n")
    
    if results['ar_accuracy'] > 50:
        print("✅ AR > 50%: TMR is preserving action information!")
    elif results['ar_accuracy'] > 10:
        print("⚠️  AR 10-50%: TMR is partially preserving action information")
    else:
        print("❌ AR < 10%: TMR is NOT preserving action information")
    
    if results['ri_accuracy'] > 80:
        print("✅ RI > 80%: TMR is achieving good privacy (identity confusion)")
    elif results['ri_accuracy'] > 50:
        print("⚠️  RI 50-80%: TMR is achieving moderate privacy")
    else:
        print("❌ RI < 50%: TMR is NOT achieving good privacy")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TMR on same-action pairs')
    parser.add_argument('--dataset', type=str, default='ntu_cv',
                        help='Dataset name (default: ntu_cv)')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='Number of pairs to evaluate (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Parse dataset name to extract base dataset and setting
    if '_' in args.dataset:
        dataset_parts = args.dataset.split('_')
        if dataset_parts[-1] in ['cv', 'cs']:
            dataset = '_'.join(dataset_parts[:-1])
            setting = dataset_parts[-1]
        else:
            dataset = args.dataset
            setting = 'cv'
    else:
        dataset = args.dataset
        setting = 'cv'

    print(f"\n{'='*80}")
    print(f"Same-Action Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Setting: {setting}")
    print(f"Num Pairs: {args.num_pairs}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load same-action pairs
    samples = load_same_action_pairs(args.dataset, args.num_pairs)

    if len(samples) == 0:
        print("❌ No same-action pairs found!")
        return 1

    # Evaluate
    results = evaluate_tmr_same_action(samples, dataset=dataset, setting=setting, device=args.device)
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = PROJECT_ROOT / "results" / "same_action_evaluation.json"
    output_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'num_pairs': args.num_pairs,
            'ar_accuracy': results['ar_accuracy'],
            'ri_accuracy': results['ri_accuracy'],
            'total_samples': results['total_samples'],
        }, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

