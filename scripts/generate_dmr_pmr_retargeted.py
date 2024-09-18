#!/usr/bin/env python3
"""
Generate retargeted datasets from pretrained DMR and PMR baseline models.

Loads a pretrained DMR or PMR checkpoint, retargets each sample in the NTU
dataset to a random different identity, and saves the result as a .pkl file
compatible with scripts/train_downstream_models.py.

Key details:
  - DMR/PMR models operate at T=75 frames with input shape (B, T, 25, 3)
  - Downstream classifiers expect T=64 with shape (T, V*C) = (64, 75)
  - This script resamples from T=75 to T=64 using linear interpolation

Usage:
    python scripts/generate_dmr_pmr_retargeted.py \
        --model_type pmr \
        --checkpoint data/models/trained_models/pmr_ntu_cv_best.pth \
        --output_path output/retargeted_pmr_ntu_cv.pkl \
        --device cuda

    python scripts/generate_dmr_pmr_retargeted.py \
        --model_type dmr \
        --checkpoint data/models/trained_models/dmr_ntu_cv_best.pth \
        --output_path output/retargeted_dmr_ntu_cv.pkl \
        --device cuda
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import datasets as DATASETS_CONFIG, parse_file_name

# Model T (frame count used by DMR/PMR)
MODEL_T = 75
# Downstream T (frame count expected by downstream classifiers)
DOWNSTREAM_T = 64


def load_raw_data(dataset='ntu', T=75):
    """
    Load raw skeleton data from the dataset pickle, padded/trimmed to T frames.

    Returns:
        dict: {filename_str: numpy_array(T, V*C=75)}
    """
    data_path = DATASETS_CONFIG[dataset]['path']
    joints = DATASETS_CONFIG[dataset]['joints']
    channels = DATASETS_CONFIG[dataset]['channels']
    max_actors = DATASETS_CONFIG[dataset]['max_actors']

    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Two-person actions to exclude (same as in datasets.py)
    two_person_actions = set([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                              106, 107, 108, 109, 110, 111, 112, 113, 114,
                              115, 116, 117, 118, 119, 120])

    processed = {}
    for fname, seq in data.items():
        # Skip two-person actions
        if dataset in ['ntu', 'ntu120']:
            info = parse_file_name(fname, dataset)
            if info['A'] in two_person_actions:
                continue

        # Keep only first actor's joints
        if max_actors == 1:
            seq = seq[:, :joints * channels]

        # Remove zero-padded frames
        non_zero = seq[~np.all(seq == 0, axis=1)]
        num_frames = len(non_zero)

        # Pad or trim to T
        if num_frames == 0:
            padded = np.zeros((T, joints * channels), dtype=np.float32)
        elif num_frames < T:
            last = non_zero[-1:]
            pad_count = T - num_frames
            padded = np.vstack([non_zero] + [last] * pad_count)
        else:
            padded = non_zero[:T]

        processed[fname] = padded.astype(np.float32)

    print(f"Loaded {len(processed)} samples (T={T})")
    return processed


def reshape_for_model(seq_flat, T=75):
    """
    Reshape from (T, V*C=75) flat format to (T, 25, 3) for model input.
    """
    return seq_flat.reshape(T, 25, 3)


def reshape_from_model(seq_3d, T=75):
    """
    Reshape from (T, 25, 3) or (T, 75) model output back to (T, V*C=75) flat format.
    """
    if seq_3d.ndim == 3:
        return seq_3d.reshape(T, -1)
    return seq_3d


def resample_frames(seq, src_T, dst_T):
    """
    Resample a sequence from src_T frames to dst_T frames using linear interpolation.

    Args:
        seq: numpy array (src_T, features)
        src_T: source frame count
        dst_T: target frame count

    Returns:
        numpy array (dst_T, features)
    """
    if src_T == dst_T:
        return seq

    # Use torch interpolate for clean resampling
    # (1, features, src_T) -> interpolate -> (1, features, dst_T)
    tensor = torch.from_numpy(seq).float().T.unsqueeze(0)  # (1, features, src_T)
    resampled = F.interpolate(tensor, size=dst_T, mode='linear', align_corners=True)
    return resampled.squeeze(0).T.numpy()  # (dst_T, features)


def load_model(model_type, checkpoint_path, device):
    """
    Load a pretrained DMR or PMR model.

    Args:
        model_type: 'dmr' or 'pmr'
        checkpoint_path: path to .pth checkpoint
        device: torch device

    Returns:
        loaded model in eval mode
    """
    print(f"Loading {model_type.upper()} model from {checkpoint_path}...")

    if model_type == 'dmr':
        from eval.dmr.dmr import DMR
        model = DMR(use_adv=False)
    elif model_type == 'pmr':
        from eval.pmr.pmr import AutoEncoder
        model = AutoEncoder(use_adv=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint -- handle multiple formats
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Raw state_dict or OrderedDict
        state_dict = checkpoint

    # Filter out adversary/discriminator/loss keys that may be in _full.pth checkpoints
    # Keep only core model keys: static_encoder.*, dynamic_encoder.*, decoder.*
    core_prefixes = ('static_encoder.', 'dynamic_encoder.', 'decoder.')
    filtered = {k: v for k, v in state_dict.items()
                if any(k.startswith(p) for p in core_prefixes)}

    if len(filtered) == 0:
        # If no keys match our core prefixes, maybe the state_dict uses different naming
        print(f"  WARNING: No keys matched core prefixes. State dict keys: {list(state_dict.keys())[:10]}")
        print(f"  Falling back to filtering out known adversary prefixes...")
        adversary_prefixes = ('priv_adv.', 'priv_coop.', 'util_adv.', 'util_coop.',
                              'discriminator.', 'triplet_loss.', 'bce_loss.',
                              'cross_entropy.', 'end_effectors', 'chain_lengths')
        filtered = {k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in adversary_prefixes)}

    print(f"  Filtered state_dict: {len(filtered)} keys (from {len(state_dict)} total)")

    # Try loading with strict=False first to see what keys are missing/unexpected
    try:
        model.load_state_dict(filtered, strict=True)
        print(f"  Loaded all {len(filtered)} parameters (strict)")
    except RuntimeError as e:
        print(f"  Strict loading failed: {e}")
        print(f"  Trying strict=False...")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  WARNING: Missing keys: {missing}")
        if unexpected:
            print(f"  INFO: Unexpected keys (ignored): {unexpected[:5]}...")

    model = model.to(device)
    model.eval()

    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    return model


def retarget_dataset(model, data_dict, device, dataset='ntu', batch_size=32, seed=42):
    """
    Retarget every sample in the dataset to a random different identity.

    For each sample (action source), we pick a random sample from a different
    person as the identity target. The model takes:
        forward(x_action_source, x_identity_target)
    and produces retargeted motion with the action of x_action_source and the
    skeletal structure of x_identity_target.

    Args:
        model: loaded DMR/PMR model
        data_dict: {filename: numpy(T=75, 75)}
        device: torch device
        dataset: dataset name for filename parsing ('ntu', 'ntu120', 'etri')
        batch_size: how many samples to process at once
        seed: random seed for reproducibility

    Returns:
        dict: {filename: numpy(T=75, 75)} retargeted sequences
    """
    np.random.seed(seed)

    filenames = list(data_dict.keys())
    n = len(filenames)

    # Build a mapping from person_id -> list of filenames
    person_to_files = {}
    for fname in filenames:
        info = parse_file_name(fname, dataset)
        pid = info['P']
        if pid not in person_to_files:
            person_to_files[pid] = []
        person_to_files[pid].append(fname)

    # For each sample, pre-select a target with different identity
    all_persons = list(person_to_files.keys())
    targets = {}
    for fname in filenames:
        info = parse_file_name(fname, dataset)
        src_pid = info['P']
        # Pick a random different person
        other_persons = [p for p in all_persons if p != src_pid]
        tgt_pid = np.random.choice(other_persons)
        tgt_fname = np.random.choice(person_to_files[tgt_pid])
        targets[fname] = tgt_fname

    # Process in batches
    retargeted = {}
    print(f"Retargeting {n} samples (batch_size={batch_size})...")

    batch_fnames = []
    batch_src = []
    batch_tgt = []

    for i, fname in enumerate(tqdm(filenames)):
        src_seq = data_dict[fname]       # (T, 75)
        tgt_seq = data_dict[targets[fname]]  # (T, 75)

        # Reshape to model input: (T, 25, 3)
        src_3d = reshape_for_model(src_seq, MODEL_T)
        tgt_3d = reshape_for_model(tgt_seq, MODEL_T)

        batch_fnames.append(fname)
        batch_src.append(src_3d)
        batch_tgt.append(tgt_3d)

        if len(batch_fnames) == batch_size or i == n - 1:
            # Stack into batch tensors: (B, T, 25, 3)
            x_src = torch.from_numpy(np.stack(batch_src)).float().to(device)
            x_tgt = torch.from_numpy(np.stack(batch_tgt)).float().to(device)

            with torch.no_grad():
                # forward(action_source, identity_target)
                output = model(x_src, x_tgt)  # (B, T, 75)

            output_np = output.cpu().numpy()  # (B, T, 75)

            for j, fn in enumerate(batch_fnames):
                retargeted[fn] = output_np[j]  # (T, 75) -- still at T=75

            batch_fnames = []
            batch_src = []
            batch_tgt = []

    return retargeted


def main():
    parser = argparse.ArgumentParser(
        description="Generate retargeted dataset from pretrained DMR/PMR model")
    parser.add_argument("--model_type", required=True, choices=["dmr", "pmr"],
                        help="Model type: dmr or pmr")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pretrained model checkpoint (.pth)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save retargeted .pkl file")
    parser.add_argument("--dataset", default="ntu",
                        help="Dataset name (default: ntu)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for retargeting (default: 32)")
    parser.add_argument("--downstream_T", type=int, default=64,
                        help="Output frame count for downstream models (default: 64)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load raw data at T=75 (model's native frame count)
    raw_data = load_raw_data(args.dataset, T=MODEL_T)

    # 2. Load model
    model = load_model(args.model_type, args.checkpoint, device)

    # 3. Retarget dataset
    retargeted_75 = retarget_dataset(model, raw_data, device,
                                     dataset=args.dataset,
                                     batch_size=args.batch_size, seed=args.seed)

    # 4. Resample from T=75 to downstream T (64) for classifier compatibility
    print(f"Resampling {len(retargeted_75)} sequences from T={MODEL_T} to T={args.downstream_T}...")
    retargeted_final = {}
    for fname, seq in tqdm(retargeted_75.items()):
        retargeted_final[fname] = resample_frames(seq, MODEL_T, args.downstream_T)

    # 5. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"Saving retargeted dataset to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(retargeted_final, f)

    # Print summary
    sample_fname = list(retargeted_final.keys())[0]
    sample_shape = retargeted_final[sample_fname].shape
    print(f"\nSummary:")
    print(f"  Model: {args.model_type.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {len(retargeted_final)}")
    print(f"  Output shape per sample: {sample_shape}")
    print(f"  Saved to: {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
