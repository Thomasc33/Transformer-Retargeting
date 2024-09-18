#!/usr/bin/env python3
"""
Bone-Length Normalization for Retargeted Skeletons (Post-Processing).

Rescales bone lengths in retargeted skeletons to match the population-average
bone lengths from raw NTU data. This preserves joint angles and motion dynamics
while eliminating the body-proportion mismatch that causes frozen evaluators
to fail on retargeted data.

Algorithm per frame:
  1. Build bone tree from root (joint 0)
  2. For each bone (parent->child), compute current direction + length
  3. Replace length with population-average length
  4. Reconstruct joint positions from root outward

Usage:
    python scripts/normalize_bone_lengths.py \
        --retargeted_path output/mirage_enhanced/abl_output_act/retargeted_ntu.pkl \
        --output_path output/bone_normalized/retargeted_ntu_normalized.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets

# NTU skeleton bone pairs: (parent_joint, child_joint)
BONE_PAIRS = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NUM_JOINTS = 25
NUM_COORDS = 3


def build_bone_tree():
    """Build a traversal order from root (joint 0) outward.

    Returns list of (parent, child) in BFS order from root.
    """
    children_map = {}
    for parent, child in BONE_PAIRS:
        children_map.setdefault(parent, []).append(child)

    order = []
    queue = [0]
    visited = {0}
    while queue:
        node = queue.pop(0)
        for child in children_map.get(node, []):
            if child not in visited:
                order.append((node, child))
                visited.add(child)
                queue.append(child)
    return order


def compute_population_bone_lengths(raw_data):
    """Compute average bone length for each bone pair across the population.

    Args:
        raw_data: dict {sample_name: ndarray(T, 75)}

    Returns:
        dict mapping (parent, child) -> average bone length (float)
    """
    bone_lengths = {pair: [] for pair in BONE_PAIRS}

    for name, seq in tqdm(raw_data.items(), desc="Computing population bone lengths"):
        joints = seq.reshape(-1, NUM_JOINTS, NUM_COORDS)

        for parent, child in BONE_PAIRS:
            bone_vec = joints[:, child, :] - joints[:, parent, :]
            lengths = np.linalg.norm(bone_vec, axis=1)
            valid = lengths > 1e-6
            if valid.any():
                bone_lengths[(parent, child)].append(np.mean(lengths[valid]))

    avg_lengths = {}
    for pair, length_list in bone_lengths.items():
        if length_list:
            avg_lengths[pair] = float(np.mean(length_list))
        else:
            avg_lengths[pair] = 0.1
    return avg_lengths


def normalize_skeleton_frame(joints_frame, target_lengths, bone_order):
    """Normalize bone lengths for a single frame.

    Args:
        joints_frame: (25, 3) joint positions
        target_lengths: dict (parent, child) -> target length
        bone_order: list of (parent, child) in BFS traversal order

    Returns:
        new (25, 3) joint positions with normalized bone lengths
    """
    result = joints_frame.copy()

    for parent, child in bone_order:
        bone_vec = result[child] - result[parent]
        current_length = np.linalg.norm(bone_vec)

        if current_length < 1e-8:
            continue

        target_len = target_lengths.get((parent, child), current_length)
        direction = bone_vec / current_length
        result[child] = result[parent] + direction * target_len

    return result


def normalize_sequence(seq, target_lengths, bone_order):
    """Normalize bone lengths for an entire sequence.

    Args:
        seq: (T, 75) skeleton sequence
        target_lengths: dict (parent, child) -> target length
        bone_order: list of (parent, child) in BFS order

    Returns:
        (T, 75) normalized sequence
    """
    T = seq.shape[0]
    joints = seq.reshape(T, NUM_JOINTS, NUM_COORDS)
    result = np.zeros_like(joints)

    for t in range(T):
        result[t] = normalize_skeleton_frame(joints[t], target_lengths, bone_order)

    return result.reshape(T, NUM_JOINTS * NUM_COORDS)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize bone lengths in retargeted skeletons to population average"
    )
    parser.add_argument(
        "--retargeted_path", type=str, required=True,
        help="Path to retargeted pkl file"
    )
    parser.add_argument(
        "--raw_data_path", type=str, default=None,
        help="Path to raw NTU pkl (default: auto from datasets config)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save normalized pkl"
    )
    parser.add_argument(
        "--dataset", type=str, default="ntu",
        choices=["ntu", "ntu120", "etri"],
        help="Dataset name for auto-loading raw data path"
    )
    args = parser.parse_args()

    if args.raw_data_path is None:
        raw_path = datasets[args.dataset]['path']
    else:
        raw_path = args.raw_data_path

    print(f"Loading raw data from {raw_path}...")
    with open(raw_path, "rb") as f:
        raw_data = pickle.load(f)
    print(f"  {len(raw_data)} raw samples loaded")

    target_lengths = compute_population_bone_lengths(raw_data)
    print("\nPopulation-average bone lengths:")
    for (p, c), length in sorted(target_lengths.items()):
        print(f"  Joint {p:2d} -> {c:2d}: {length:.4f}")

    bone_order = build_bone_tree()

    print(f"\nLoading retargeted data from {args.retargeted_path}...")
    with open(args.retargeted_path, "rb") as f:
        retargeted_data = pickle.load(f)
    print(f"  {len(retargeted_data)} retargeted samples loaded")

    normalized_data = {}
    for name, seq in tqdm(retargeted_data.items(), desc="Normalizing bone lengths"):
        normalized_data[name] = normalize_sequence(seq, target_lengths, bone_order)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(normalized_data, f)
    print(f"\nSaved normalized data to {args.output_path}")

    # Verification
    print("\nVerification - bone length stats (sample of first 100):")
    sample_keys = list(retargeted_data.keys())[:100]
    for pair in [(0, 1), (1, 20), (20, 4), (4, 5)]:
        before_lengths = []
        after_lengths = []
        for k in sample_keys:
            orig = retargeted_data[k].reshape(-1, NUM_JOINTS, NUM_COORDS)
            norm = normalized_data[k].reshape(-1, NUM_JOINTS, NUM_COORDS)
            p, c = pair
            before_lengths.extend(np.linalg.norm(orig[:, c] - orig[:, p], axis=1).tolist())
            after_lengths.extend(np.linalg.norm(norm[:, c] - norm[:, p], axis=1).tolist())
        print(f"  Bone ({pair[0]:2d},{pair[1]:2d}): "
              f"before={np.mean(before_lengths):.4f}+/-{np.std(before_lengths):.4f} "
              f"-> after={np.mean(after_lengths):.4f}+/-{np.std(after_lengths):.4f} "
              f"(target={target_lengths[pair]:.4f})")


if __name__ == "__main__":
    main()
