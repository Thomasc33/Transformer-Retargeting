#!/usr/bin/env python3
"""
Compute reconstruction quality metrics on retargeted data.
Compares retargeted output against the source motion (same action, different identity).

Metrics:
- MPJPE: Mean Per-Joint Position Error (mm) between retargeted and source
- Bone Length Error: Absolute deviation of bone lengths from target skeleton
- Temporal Smoothness: Mean acceleration magnitude of retargeted motion
- Velocity MSE: MSE of joint velocities between retargeted and source
"""

import argparse
import pickle
import numpy as np
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# NTU skeleton bone connections (parent, child) - 0-indexed
NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),      # spine
    (20, 4), (4, 5), (5, 6), (6, 7),        # left arm
    (20, 8), (8, 9), (9, 10), (10, 11),     # right arm
    (0, 12), (12, 13), (13, 14), (14, 15),  # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
    (7, 21), (7, 22),                        # left hand
    (11, 23), (11, 24),                      # right hand
]


def compute_bone_lengths(skeleton, bones):
    """Compute bone lengths for a skeleton sequence.
    skeleton: (T, V, C) or (T, V*C)
    Returns: (T, num_bones)
    """
    if skeleton.ndim == 2:
        T, VC = skeleton.shape
        V = 25
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    T, V, C = skeleton.shape
    bone_lengths = np.zeros((T, len(bones)))
    for i, (p, c) in enumerate(bones):
        diff = skeleton[:, c, :] - skeleton[:, p, :]
        bone_lengths[:, i] = np.linalg.norm(diff, axis=-1)
    return bone_lengths


def compute_metrics(retargeted_data, raw_data, dataset='ntu'):
    """Compute reconstruction quality metrics."""
    from src.data.datasets import parse_file_name

    metrics = {
        'mpjpe_values': [],
        'bone_length_errors': [],
        'smoothness_values': [],
        'velocity_mse_values': [],
    }

    retargeted_keys = list(retargeted_data.keys())
    raw_keys = list(raw_data.keys())

    # Build identity-to-sequences mapping for target bone lengths
    identity_sequences = {}
    for fname, seq in raw_data.items():
        info = parse_file_name(fname, dataset)
        pid = info['P']
        if pid not in identity_sequences:
            identity_sequences[pid] = []
        identity_sequences[pid].append(seq)

    count = 0
    for fname in retargeted_keys:
        if fname not in raw_data:
            continue

        src_seq = raw_data[fname]  # Original source motion
        ret_seq = retargeted_data[fname]  # Retargeted output

        # Ensure same shape
        if isinstance(src_seq, np.ndarray):
            src = src_seq.copy()
        else:
            src = np.array(src_seq)

        if isinstance(ret_seq, np.ndarray):
            ret = ret_seq.copy()
        else:
            ret = np.array(ret_seq)

        # Handle shape: expect (T, V*C) where V=25, C=3
        if src.ndim == 2 and ret.ndim == 2:
            T_src, VC = src.shape
            T_ret, _ = ret.shape
            T = min(T_src, T_ret)
            V = 25
            C = VC // V

            src = src[:T].reshape(T, V, C)
            ret = ret[:T].reshape(T, V, C)
        elif src.ndim == 3 and ret.ndim == 3:
            T = min(src.shape[0], ret.shape[0])
            src = src[:T]
            ret = ret[:T]
        else:
            continue

        # 1. MPJPE: mean per-joint position error
        # Compare retargeted action dynamics against source action
        # Note: retargeted has different identity, so absolute positions differ
        # We center each frame at the spine joint (joint 0) before comparison
        src_centered = src - src[:, 0:1, :]
        ret_centered = ret - ret[:, 0:1, :]
        mpjpe = np.mean(np.sqrt(np.sum((src_centered - ret_centered) ** 2, axis=-1)))
        metrics['mpjpe_values'].append(mpjpe)

        # 2. Bone length consistency: std of bone lengths across frames
        ret_bone_lengths = compute_bone_lengths(ret.reshape(T, V, C), NTU_BONES)
        # Bone length error = mean std across frames (should be low = consistent skeleton)
        bone_std = np.mean(np.std(ret_bone_lengths, axis=0))
        metrics['bone_length_errors'].append(bone_std)

        # 3. Temporal smoothness: mean acceleration magnitude
        if T >= 3:
            velocity = np.diff(ret_centered, axis=0)  # (T-1, V, C)
            acceleration = np.diff(velocity, axis=0)    # (T-2, V, C)
            accel_mag = np.mean(np.sqrt(np.sum(acceleration ** 2, axis=-1)))
            metrics['smoothness_values'].append(accel_mag)

        # 4. Velocity MSE: how well motion dynamics are preserved
        if T >= 2:
            src_vel = np.diff(src_centered, axis=0)
            ret_vel = np.diff(ret_centered, axis=0)
            vel_mse = np.mean((src_vel - ret_vel) ** 2)
            metrics['velocity_mse_values'].append(vel_mse)

        count += 1

    results = {
        'num_samples': count,
        'mpjpe_mean': float(np.mean(metrics['mpjpe_values'])),
        'mpjpe_std': float(np.std(metrics['mpjpe_values'])),
        'bone_length_consistency': float(np.mean(metrics['bone_length_errors'])),
        'temporal_smoothness': float(np.mean(metrics['smoothness_values'])),
        'velocity_mse': float(np.mean(metrics['velocity_mse_values'])),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute reconstruction quality metrics')
    parser.add_argument('--retargeted_path', type=str, required=True,
                        help='Path to retargeted .pkl file')
    parser.add_argument('--dataset', type=str, default='ntu',
                        choices=['ntu', 'ntu120', 'etri'])
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save metrics JSON')
    args = parser.parse_args()

    print(f"Loading retargeted data from {args.retargeted_path}...")
    with open(args.retargeted_path, 'rb') as f:
        retargeted_data = pickle.load(f)
    print(f"  Loaded {len(retargeted_data)} retargeted samples")

    print(f"Loading raw data for dataset '{args.dataset}'...")
    from src.data.datasets import load_data
    raw_data = load_data(args.dataset)
    print(f"  Loaded {len(raw_data)} raw samples")

    print("Computing metrics...")
    results = compute_metrics(retargeted_data, raw_data, args.dataset)

    print("\n" + "=" * 50)
    print("RECONSTRUCTION QUALITY METRICS")
    print("=" * 50)
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"MPJPE (centered):     {results['mpjpe_mean']:.4f} ± {results['mpjpe_std']:.4f}")
    print(f"Bone Length Consistency (lower=better): {results['bone_length_consistency']:.6f}")
    print(f"Temporal Smoothness (accel, lower=better): {results['temporal_smoothness']:.6f}")
    print(f"Velocity MSE (lower=better): {results['velocity_mse']:.6f}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_path}")


if __name__ == '__main__':
    main()
