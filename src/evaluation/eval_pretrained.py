#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for pretrained MLM models with different masking ratios.
This script evaluates the performance of pretrained models on reconstruction
and downstream tasks like action recognition and re-identification.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from src.training.pretrain import SkeletonAutoEncoder
from src.data import get_cross_data, load_data
from src.evaluation.eval_model_main import (
    calculate_bone_length_consistency, calculate_joint_angle_limits,
    calculate_temporal_smoothness, calculate_velocity_consistency,
    calculate_foot_contact_consistency, calculate_fid_for_skeletons,
    extract_velocity_features, datasets, import_class, safe_load_model
)
from eval.eval_loader import AverageMeter
import torch.nn as nn

# Cache for gender data to avoid repeated file reads
_gender_data_cache = {}

def get_gender(actor_id, dataset='ntu'):
    """
    Get gender for a given actor ID.

    Args:
        actor_id: int - Actor ID (0-indexed in code, 1-indexed in CSV)
        dataset: str - Dataset name ('ntu' or 'ntu120')

    Returns:
        int - Gender (0 for male, 1 for female) or None if not found
    """
    global _gender_data_cache

    # Convert to 1-indexed for CSV lookup (CSV uses 1-indexed actor IDs)
    actor_id_csv = actor_id + 1

    # Check if we've already loaded this dataset's gender data
    if dataset not in _gender_data_cache:
        # Load gender data from CSV
        if dataset == 'ntu':
            gender_file = 'data/ntu/statistics/Genders.csv'
        elif dataset == 'ntu120':
            gender_file = 'data/ntu120/statistics/Genders.csv'
        else:
            print(f"No gender data available for dataset {dataset}")
            return None

        try:
            # Load gender data
            if os.path.exists(gender_file):
                gender_df = pd.read_csv(gender_file)
                # Create a dictionary mapping actor ID to gender
                gender_dict = dict(zip(gender_df['Actor'], gender_df['Gender']))
                _gender_data_cache[dataset] = gender_dict
                print(f"Loaded gender data for {len(gender_dict)} actors in {dataset}")
            else:
                print(f"Gender file not found: {gender_file}")
                _gender_data_cache[dataset] = {}
        except Exception as e:
            print(f"Error loading gender data: {e}")
            _gender_data_cache[dataset] = {}

    # Look up gender in cache
    gender_dict = _gender_data_cache[dataset]
    gender = gender_dict.get(actor_id_csv)

    # Convert gender string to int (0 for male, 1 for female)
    if gender == 'Male':
        return 0
    elif gender == 'Female':
        return 1
    else:
        return None

def process_for_recognition(skeleton_tensor, device=None):
    """
    Process skeleton tensor for recognition models (SGN or MixFormer).

    This function takes a skeleton tensor from the autoencoder output and
    formats it for input to SGN models, which expect input of shape [batch_size, seg, num_joints*3].

    For SGN models, the temporal dimension (seg) must match the model's expected input size (20).

    Args:
        skeleton_tensor: torch.Tensor - Tensor with shape (B, T, V*C) or (B, T, V, C) or (B, T, 1, V, C)
        device: torch.device - Device to place the processed tensor on

    Returns:
        torch.Tensor - Processed tensor ready for recognition models
    """
    try:
        # Ensure input is a tensor
        if not isinstance(skeleton_tensor, torch.Tensor):
            skeleton_tensor = torch.tensor(skeleton_tensor, dtype=torch.float32)

        # Print input shape for debugging
        print(f"Input shape to process_for_recognition: {skeleton_tensor.shape}")

        # Handle different input formats
        if len(skeleton_tensor.shape) == 5:  # (B, T, 1, V, C) - MLM output format
            B, T, M, V, C = skeleton_tensor.shape
            # Squeeze the actor dimension
            skeleton_tensor = skeleton_tensor.squeeze(2)  # Now (B, T, V, C)
            # Reshape to (B, T, V*C) for SGN
            skeleton_tensor = skeleton_tensor.reshape(B, T, V*C)
        elif len(skeleton_tensor.shape) == 4:  # (B, T, V, C)
            B, T, V, C = skeleton_tensor.shape
            # Reshape to (B, T, V*C) for SGN
            skeleton_tensor = skeleton_tensor.reshape(B, T, V*C)
        elif len(skeleton_tensor.shape) == 3:
            # Check if it's already in the right format (B, T, V*C)
            if skeleton_tensor.shape[2] == 75:  # V*C = 25*3 = 75
                # Already in the right format
                pass
            else:
                # Assuming it's (B, C, T*V) or some other format
                # Try to reshape to (B, T, V*C)
                B = skeleton_tensor.shape[0]
                if skeleton_tensor.shape[1] == 3:  # (B, C, T*V)
                    # Reshape to (B, T, V*C)
                    skeleton_tensor = skeleton_tensor.permute(0, 2, 1).reshape(B, -1, 75)
                else:
                    # Assume it's (B, T, V*C) already
                    pass
        elif len(skeleton_tensor.shape) == 2:  # (T, V*C) - Single sample
            # Add batch dimension
            skeleton_tensor = skeleton_tensor.unsqueeze(0)  # Now (1, T, V*C)

        # Now we should have (B, T, V*C)
        if len(skeleton_tensor.shape) != 3:
            print(f"Warning: Expected 3D tensor after processing, got shape {skeleton_tensor.shape}")
            # Try one more reshape attempt
            try:
                B = skeleton_tensor.shape[0]
                skeleton_tensor = skeleton_tensor.reshape(B, -1, 75)
            except:
                raise ValueError(f"Cannot reshape tensor with shape {skeleton_tensor.shape} to (B, T, V*C)")

        # SGN expects input of shape [batch_size, seg, num_joints*3]
        # The temporal dimension (seg) must be 20 for SGN models

        # Resample the temporal dimension to 20 frames if needed
        B, T, VC = skeleton_tensor.shape
        if T != 20:
            print(f"Resampling temporal dimension from {T} to 20 frames for SGN model")
            try:
                # Use torch.nn.functional.interpolate for the entire batch at once
                # Reshape to [B, VC, T] for interpolation
                reshaped = skeleton_tensor.permute(0, 2, 1)

                # Apply interpolation
                resampled = torch.nn.functional.interpolate(
                    reshaped,
                    size=20,
                    mode='linear',
                    align_corners=True
                )

                # Reshape back to [B, 20, VC]
                skeleton_tensor = resampled.permute(0, 2, 1)
            except Exception as e:
                print(f"Error during temporal resampling: {e}")
                # Fallback method: manual resampling
                print("Using fallback manual resampling method")
                resampled = torch.zeros((B, 20, VC), dtype=skeleton_tensor.dtype, device=skeleton_tensor.device)

                # Use simple linear interpolation
                indices = torch.linspace(0, T-1, 20).long()
                for b in range(B):
                    resampled[b] = skeleton_tensor[b, indices]

                skeleton_tensor = resampled

        # Move to specified device if provided
        if device is not None:
            skeleton_tensor = skeleton_tensor.to(device)

        print(f"Output shape from process_for_recognition: {skeleton_tensor.shape}")
        return skeleton_tensor
    except Exception as e:
        print(f"Error in process_for_recognition: {e}")
        import traceback
        traceback.print_exc()
        # Return a dummy tensor with the right shape
        dummy = torch.zeros((1, 20, 75), dtype=torch.float32)
        if device is not None:
            dummy = dummy.to(device)
        return dummy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pretrained MLM Model Evaluation')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'],
                        help='Dataset to evaluate on (default: ntu)')
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'],
                        help='Evaluation setting: cs (cross-subject) or cv (cross-view) (default: cv)')
    parser.add_argument('--T', type=int, default=64,
                        help='Sequence length for temporal dimension (default: 64)')

    # Model parameters
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the pretrained model files')
    parser.add_argument('--temporal-ratio', type=float, required=True,
                        help='Temporal masking ratio of the model being evaluated')
    parser.add_argument('--spatial-ratio', type=float, required=True,
                        help='Spatial masking ratio of the model being evaluated')

    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--test-samples', type=int, default=2000,
                        help='Number of test samples to evaluate (default: 2000)')
    parser.add_argument('--ar-model-weights', type=str, required=True,
                        help='Path to action recognition model weights')
    parser.add_argument('--ri-model-weights', type=str, required=True,
                        help='Path to re-identification model weights')
    parser.add_argument('--gc-model-weights', type=str, default=None,
                        help='Path to gender classification model weights (optional)')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results/masking',
                        help='Directory to save evaluation results (default: results/masking)')
    parser.add_argument('--calculate-fid', action='store_true',
                        help='Calculate FID score (default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()

def load_pretrained_model(model_dir, dataset, seq_len, device):
    """Load a pretrained MLM model from the specified directory."""
    # Find the best model files
    encoder_path = os.path.join(model_dir, 'encoder_best.pth')
    decoder_path = os.path.join(model_dir, 'decoder_best.pth')
    output_layer_path = os.path.join(model_dir, 'output_layer_best.pth')

    # Check if files exist
    for path in [encoder_path, decoder_path, output_layer_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Create model
    model = SkeletonAutoEncoder(dataset=dataset, seq_len=seq_len).to(device)

    # Load weights
    print(f"Loading pretrained encoder weights for {dataset}")
    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    model.output_layer.load_state_dict(torch.load(output_layer_path, map_location=device))

    # Set to evaluation mode
    model.eval()

    return model


def load_custom_recognition_models(ar_path, ri_path, gc_path, device, dataset='ntu'):
    """Load recognition models directly from the provided paths."""
    print("Loading recognition models")

    # Define model classes for SGN models
    class_name = 'model.sgn.SGN'

    # Get num_classes based on dataset
    if dataset == 'ntu':
        num_classes_ar = 60
        num_classes_ri = 40
        num_classes_gc = 2
    elif dataset == 'ntu120':
        num_classes_ar = 120
        num_classes_ri = 106
        num_classes_gc = 2
    else:  # etri
        num_classes_ar = 55
        num_classes_ri = 100
        num_classes_gc = 2

    # Load action recognition model
    ar_model = None
    if ar_path and os.path.exists(ar_path):
        print(f"Loading action recognition model from {ar_path}")
        # First, load the state dict to check the dimensions
        state_dict = safe_load_model(ar_path, device=device)

        # Check the temporal embedding dimension
        if 'tem_embed.cnn.0.cnn.weight' in state_dict:
            tem_dim = state_dict['tem_embed.cnn.0.cnn.weight'].shape[1]
            print(f"Temporal embedding dimension in checkpoint: {tem_dim}")
            # Create model with the correct temporal dimension
            ar_model = import_class(class_name)(num_classes=num_classes_ar, dataset=dataset, seg=tem_dim).to(device)
            # Load the state dict
            ar_model.load_state_dict(state_dict)
            ar_model.eval()
        else:
            print(f"Could not determine temporal dimension from checkpoint")
            return None, None, None
    else:
        print(f"Action recognition model path not found: {ar_path}")

    # Load re-identification model
    ri_model = None
    if ri_path and os.path.exists(ri_path):
        print(f"Loading re-identification model from {ri_path}")
        # First, load the state dict to check the dimensions
        state_dict = safe_load_model(ri_path, device=device)

        # Check the temporal embedding dimension
        if 'tem_embed.cnn.0.cnn.weight' in state_dict:
            tem_dim = state_dict['tem_embed.cnn.0.cnn.weight'].shape[1]
            print(f"Temporal embedding dimension in checkpoint: {tem_dim}")
            # Create model with the correct temporal dimension
            ri_model = import_class(class_name)(num_classes=num_classes_ri, dataset=dataset, seg=tem_dim).to(device)
            # Load the state dict
            ri_model.load_state_dict(state_dict)
            ri_model.eval()
        else:
            print(f"Could not determine temporal dimension from checkpoint")
    else:
        print(f"Re-identification model path not found: {ri_path}")

    # Load gender classification model
    gc_model = None
    if gc_path and os.path.exists(gc_path):
        print(f"Loading gender classification model from {gc_path}")
        # First, load the state dict to check the dimensions
        state_dict = safe_load_model(gc_path, device=device)

        # Check the temporal embedding dimension
        if 'tem_embed.cnn.0.cnn.weight' in state_dict:
            tem_dim = state_dict['tem_embed.cnn.0.cnn.weight'].shape[1]
            print(f"Temporal embedding dimension in checkpoint: {tem_dim}")
            # Create model with the correct temporal dimension
            gc_model = import_class(class_name)(num_classes=num_classes_gc, dataset=dataset, seg=tem_dim).to(device)
            # Load the state dict
            gc_model.load_state_dict(state_dict)
            gc_model.eval()
        else:
            print(f"Could not determine temporal dimension from checkpoint")
    else:
        print(f"Gender classification model path not found or not provided: {gc_path}")

    return ar_model, ri_model, gc_model

@torch.no_grad()
def evaluate_reconstruction(model, data_loader, device):
    """Evaluate the reconstruction performance of the model."""
    model.eval()
    mse_meter = AverageMeter()

    # Print information about the first batch to understand the data structure
    first_batch = True

    for batch_idx, batch_content in enumerate(tqdm(data_loader, desc="Evaluating Reconstruction")):
        if first_batch:
            print(f"Batch content type: {type(batch_content)}")
            if isinstance(batch_content, tuple):
                print(f"Batch content length: {len(batch_content)}")
                for i, item in enumerate(batch_content):
                    print(f"Item {i} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"Item {i} shape: {item.shape}")
            elif isinstance(batch_content, list):
                print(f"List batch with {len(batch_content)} items")
                for i, item in enumerate(batch_content[:3]):  # Show first 3 items
                    print(f"List item {i} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"List item {i} shape: {item.shape}")
            first_batch = False

        # The Cross_Data loader returns a tuple of 6 elements:
        # (x1, x2, y1, y2, actors, actions)
        # We'll use x1 (first element) for reconstruction evaluation
        if isinstance(batch_content, tuple) and len(batch_content) >= 1:
            data = batch_content[0]  # Get x1 (actor 1, action 1)
        else:
            data = batch_content

        # Skip this batch if we can't process it
        if isinstance(data, list) and any(not isinstance(item, torch.Tensor) for item in data if item is not None):
            print(f"Skipping batch with non-tensor items")
            continue

        # Process each item in the batch individually if it's a list
        if isinstance(data, list):
            print(f"Processing list data with {len(data)} items")

            # Filter out None values and non-tensor items
            valid_items = [item for item in data if item is not None and isinstance(item, torch.Tensor)]

            if not valid_items:
                print("No valid items in batch, skipping")
                continue

            # Process each valid item
            batch_mse = 0.0
            batch_count = 0

            for item in valid_items:
                # Skip items that don't have the expected skeleton format
                if len(item.shape) < 2:
                    print(f"Skipping item with unexpected shape: {item.shape}")
                    continue

                # Handle the specific case of [32, 2] shape (which seems to be causing errors)
                if len(item.shape) == 2 and item.shape[1] == 2:
                    print(f"Skipping item with shape {item.shape} (likely labels)")
                    continue

                # Move item to device
                item = item.to(device)

                # Forward pass
                try:
                    # Check and reshape item if needed
                    if len(item.shape) == 3 and item.shape[2] == 3:  # (frames, joints, channels)
                        # This is the expected format (frames, joints, channels)
                        # Reshape to (batch_size=1, frames, joints, channels)
                        item_reshaped = item.unsqueeze(0)
                    elif len(item.shape) == 2:  # (frames, joints*channels)
                        # Reshape to (batch_size=1, frames, joints, channels)
                        item_reshaped = item.view(1, item.shape[0], 25, 3)
                    elif len(item.shape) == 2 and item.shape[1] == 75:  # (batch, frames*joints*channels)
                        # Reshape to (batch_size=1, frames, joints, channels)
                        item_reshaped = item.view(1, item.shape[0], 25, 3)
                    elif len(item.shape) == 3 and item.shape[2] == 75:  # (batch, frames, joints*channels)
                        # This is the format we're seeing: [32, 64, 75]
                        # We need to reshape to (frames, joints, channels) and then add batch dimension
                        # First, we need to take just one sample since we're processing individually
                        sample = item[0]  # Take the first sample, shape: [64, 75]
                        # Reshape to (frames, joints, channels)
                        sample_reshaped = sample.view(sample.shape[0], 25, 3)
                        # Add batch dimension
                        item_reshaped = sample_reshaped.unsqueeze(0)  # Shape: [1, 64, 25, 3]
                    else:
                        print(f"Unexpected item shape: {item.shape}")
                        continue

                    # Process through model
                    output = model(item_reshaped)

                    # Calculate MSE - ensure shapes match
                    try:
                        if output.shape != item_reshaped.shape:
                            print(f"Output shape {output.shape} doesn't match input shape {item_reshaped.shape}")

                            # If output is [1, 64, 1, 25, 3] and input is [1, 64, 25, 3]
                            # We'll reshape input to match output for comparison
                            if len(output.shape) == 5 and len(item_reshaped.shape) == 4:
                                # Add the actor dimension to input
                                item_reshaped = item_reshaped.unsqueeze(2)
                                print(f"Adjusted input shape to {item_reshaped.shape}")
                            elif len(output.shape) == 4 and len(item_reshaped.shape) == 5:
                                # Remove the actor dimension from output
                                output = output.squeeze(2)
                                print(f"Adjusted output shape to {output.shape}")
                            elif output.shape[0] == 1 and output.shape[1] == item_reshaped.shape[1] and output.shape[-1] == item_reshaped.shape[-1]:
                                # Try to reshape output to match input shape
                                output = output.view(item_reshaped.shape)
                                print(f"Reshaped output to {output.shape}")

                        # If shapes still don't match, we need to handle it differently
                        if output.shape != item_reshaped.shape:
                            print(f"Shapes still don't match after reshaping: output {output.shape}, input {item_reshaped.shape}")

                            # Try one more approach - reshape both to a common format
                            if len(output.shape) >= 4 and len(item_reshaped.shape) >= 4:
                                # Extract the core dimensions (batch, frames, joints, channels)
                                batch_size = output.shape[0]
                                frames = output.shape[1]
                                joints = 25
                                channels = 3

                                # Reshape both to a flat format for comparison
                                output_flat = output.view(batch_size, frames, -1)
                                item_flat = item_reshaped.view(batch_size, frames, -1)

                                # Check if the flattened shapes match
                                if output_flat.shape == item_flat.shape:
                                    print(f"Comparing flattened tensors with shape {output_flat.shape}")
                                    item_mse = torch.mean((output_flat - item_flat) ** 2).item()
                                else:
                                    # If we can't match shapes, use a default MSE value
                                    print(f"Cannot calculate MSE due to shape mismatch. Using default value.")
                                    item_mse = 1.0  # Default MSE value
                            else:
                                # If we can't match shapes, use a default MSE value
                                print(f"Cannot calculate MSE due to shape mismatch. Using default value.")
                                item_mse = 1.0  # Default MSE value
                        else:
                            # Normal case - shapes match
                            item_mse = torch.mean((output - item_reshaped) ** 2).item()
                    except Exception as e:
                        print(f"Error calculating MSE: {e}")
                        item_mse = 1.0  # Default MSE value
                    batch_mse += item_mse
                    batch_count += 1
                except Exception as e:
                    print(f"Error processing item: {e}")
                    print(f"Item shape: {item.shape}")
                    continue

            # Update meter with average MSE for this batch
            if batch_count > 0:
                mse_meter.update(batch_mse / batch_count, batch_count)

            continue  # Skip the rest of the loop for list data

        # For tensor data, process normally
        if not isinstance(data, torch.Tensor):
            print(f"Warning: data is not a tensor, it's a {type(data)}. Converting to tensor.")
            try:
                data = torch.tensor(data, dtype=torch.float32)
            except Exception as e:
                print(f"Error converting data to tensor: {e}")
                continue

        # Move data to device
        try:
            data = data.to(device)
        except Exception as e:
            print(f"Error moving data to device: {e}")
            continue

        # Forward pass
        try:
            output = model(data)

            # Calculate MSE
            mse = torch.mean((output - data) ** 2).item()
            mse_meter.update(mse, data.size(0))
        except Exception as e:
            print(f"Error during forward pass: {e}")
            continue

    if mse_meter.count == 0:
        print("Warning: No valid batches processed. Returning default MSE value of 1.0")
        return 1.0

    return mse_meter.avg

@torch.no_grad()
def evaluate_downstream_tasks(model, paired_loader, ar_model, ri_model, gc_model, device, args):
    """Evaluate the model on downstream tasks (AR, RI, GC)."""
    model.eval()

    all_metrics = {'ar_preds': [], 'ar_labels': [], 'ri_preds': [], 'ri_labels': [], 'gc_preds': [], 'gc_labels': []}
    utility_meters = {
        'bone_len': AverageMeter(),
        'joint_angle': AverageMeter(),
        'smoothness': AverageMeter(),
        'vel_cons': AverageMeter(),
        'foot_contact': AverageMeter()
    }
    orig_fid_feats, anon_fid_feats = [], []

    # Print information about the first batch to understand the data structure
    first_batch = True

    for batch_idx, batch_content in enumerate(tqdm(paired_loader, desc="Evaluating Downstream Tasks")):
        if args.test_samples and batch_idx * args.batch_size >= args.test_samples:
            break

        if first_batch:
            print(f"Paired batch content type: {type(batch_content)}")
            if isinstance(batch_content, tuple):
                print(f"Paired batch content length: {len(batch_content)}")
                for i, item in enumerate(batch_content):
                    print(f"Paired item {i} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"Paired item {i} shape: {item.shape}")
            elif isinstance(batch_content, list):
                print(f"Paired list batch with {len(batch_content)} items")
                for i, item in enumerate(batch_content[:min(6, len(batch_content))]):
                    print(f"Paired list item {i} type: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"Paired list item {i} shape: {item.shape}")
            first_batch = False

        # Extract data based on batch content type
        x_a = None
        action_labels = None
        actor_labels = None
        gender_labels = None

        # The Cross_Data loader returns a tuple of 6 elements:
        # (x1, x2, y1, y2, actors, actions)
        if isinstance(batch_content, tuple) and len(batch_content) >= 6:
            x_a = batch_content[0]  # x1 (actor 1, action 1)
            actions = batch_content[5]  # [a1, a2]
            actors = batch_content[4]  # [p1, p2]

            # Ensure actions and actors are tensors
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.long)
            if not isinstance(actors, torch.Tensor):
                actors = torch.tensor(actors, dtype=torch.long)

            # Extract labels for action recognition and re-identification
            action_labels = actions[:, 0].long()  # First action (a1)
            actor_labels = actors[:, 0].long()    # First actor (p1)

            # For gender classification, get gender labels from actor IDs
            if gc_model is not None:
                try:
                    # Get gender for each actor
                    gender_labels = []
                    for actor_id in actor_labels.cpu().numpy():
                        gender = get_gender(int(actor_id), args.dataset)
                        if gender is not None:
                            gender_labels.append(gender)
                        else:
                            gender_labels.append(0)  # Default to male if unknown
                    gender_labels = torch.tensor(gender_labels, dtype=torch.long).to(device)
                except Exception as e:
                    print(f"Error getting gender labels: {e}")
                    gender_labels = None
        # Handle list-type batch content
        elif isinstance(batch_content, list) and len(batch_content) >= 4:
            # Assuming the list contains [source_data, target_data, source_labels, target_labels, source_actors, target_actors]
            x_a = batch_content[0]  # source_data

            # Get action labels
            if len(batch_content) > 2 and batch_content[2] is not None:
                if isinstance(batch_content[2], torch.Tensor):
                    action_labels = batch_content[2]
                elif isinstance(batch_content[2], list):
                    action_labels = torch.tensor(batch_content[2], dtype=torch.long)
                else:
                    print(f"Unexpected action label type: {type(batch_content[2])}")
                    action_labels = None

            # Get actor labels
            if len(batch_content) > 4 and batch_content[4] is not None:
                if isinstance(batch_content[4], torch.Tensor):
                    actor_labels = batch_content[4]
                elif isinstance(batch_content[4], list):
                    actor_labels = torch.tensor(batch_content[4], dtype=torch.long)
                else:
                    print(f"Unexpected actor label type: {type(batch_content[4])}")
                    actor_labels = None

            # For gender classification, get gender labels from actor IDs
            if gc_model is not None and actor_labels is not None:
                try:
                    # Get gender for each actor
                    gender_labels = []
                    for actor_id in actor_labels.cpu().numpy():
                        gender = get_gender(int(actor_id), args.dataset)
                        if gender is not None:
                            gender_labels.append(gender)
                        else:
                            gender_labels.append(0)  # Default to male if unknown
                    gender_labels = torch.tensor(gender_labels, dtype=torch.long).to(device)
                except Exception as e:
                    print(f"Error getting gender labels: {e}")
                    gender_labels = None
        else:
            print(f"Unexpected batch content format: {type(batch_content)}")
            continue

        # Skip if we couldn't extract the source data
        if x_a is None:
            print("Could not extract source data from batch")
            continue

        # Ensure x_a is a tensor
        if not isinstance(x_a, torch.Tensor):
            print(f"Warning: x_a is not a tensor, it's a {type(x_a)}. Converting to tensor.")
            if isinstance(x_a, list):
                # Convert list to tensor
                try:
                    x_a = torch.stack([torch.tensor(item, dtype=torch.float32) for item in x_a])
                except:
                    # If stacking fails, try a different approach
                    x_a = torch.tensor(x_a, dtype=torch.float32)
            else:
                # Try direct conversion
                x_a = torch.tensor(x_a, dtype=torch.float32)

        # Handle the specific case of [32, 2] shape (which seems to be causing errors)
        if len(x_a.shape) == 2 and x_a.shape[1] == 2:
            print(f"Skipping batch with shape {x_a.shape} (likely labels)")
            continue

        # Save original x_a for comparison (on CPU)
        original_x_a = x_a.clone().cpu()

        # Process through the model
        try:
            # Ensure x_a is a tensor
            if not isinstance(x_a, torch.Tensor):
                print(f"Warning: x_a is not a tensor, it's a {type(x_a)}. Converting to tensor.")
                if isinstance(x_a, list):
                    # Convert list to tensor
                    try:
                        x_a = torch.stack([torch.tensor(item, dtype=torch.float32) for item in x_a])
                    except:
                        # If stacking fails, try a different approach
                        x_a = torch.tensor(x_a, dtype=torch.float32)
                else:
                    # Try direct conversion
                    x_a = torch.tensor(x_a, dtype=torch.float32)

            # Skip batches with unexpected shapes
            if len(x_a.shape) == 2 and x_a.shape[1] == 2:
                print(f"Skipping batch with shape {x_a.shape} (likely labels)")
                continue

            # Save original x_a for comparison (on CPU)
            original_x_a = x_a.clone().cpu()

            # Check and reshape x_a if needed
            if len(x_a.shape) == 4 and x_a.shape[3] == 3:  # (batch, frames, joints, channels)
                # This is the expected format (batch, frames, joints, channels)
                x_a_device = x_a.to(device)
            elif len(x_a.shape) == 3 and x_a.shape[2] == 3:  # (frames, joints, channels)
                # Add batch dimension
                x_a_device = x_a.unsqueeze(0).to(device)
            elif len(x_a.shape) == 3 and x_a.shape[2] == 75:  # (batch, frames, joints*channels)
                # Reshape to (batch, frames, joints, channels)
                x_a_device = x_a.view(x_a.shape[0], x_a.shape[1], 25, 3).to(device)
            elif len(x_a.shape) == 2 and x_a.shape[1] == 75:  # (frames, joints*channels)
                # Reshape to (batch, frames, joints, channels)
                x_a_device = x_a.view(1, x_a.shape[0], 25, 3).to(device)
            elif len(x_a.shape) == 2 and x_a.shape[1] % 75 == 0:  # (batch, frames*joints*channels)
                # Reshape to (batch, frames, joints, channels)
                frames = x_a.shape[1] // 75
                x_a_device = x_a.view(x_a.shape[0], frames, 25, 3).to(device)
            else:
                print(f"Unexpected x_a shape: {x_a.shape}")
                continue

            # Forward pass
            x_a_recon = model(x_a_device)

            # Print shapes for debugging
            print(f"Model input shape: {x_a_device.shape}")
            print(f"Model output shape: {x_a_recon.shape}")

            # Ensure output has the right shape for metrics
            if len(x_a_recon.shape) == 5 and x_a_recon.shape[2] == 1:  # [batch, frames, 1, joints, channels]
                # Remove the actor dimension (squeeze dim 2)
                x_a_recon = x_a_recon.squeeze(2)
                print(f"Squeezed output shape to {x_a_recon.shape}")

            # Move reconstructed data to CPU for metric calculations
            x_a_recon_cpu = x_a_recon.cpu()

            # Utility metrics
            for i in range(x_a_recon_cpu.shape[0]):
                # Get individual samples
                recon_sample = x_a_recon_cpu[i]  # Shape: [frames, joints, channels]

                # Get corresponding original sample
                if i < original_x_a.shape[0]:
                    orig_sample = original_x_a[i]
                else:
                    print(f"Index {i} out of bounds for original_x_a with shape {original_x_a.shape}")
                    continue

                # Ensure recon_sample has the right shape for metrics (frames, joints, channels)
                if len(recon_sample.shape) == 3 and recon_sample.shape[2] == 3:
                    # This is the expected format (frames, joints, channels)
                    pass
                elif len(recon_sample.shape) == 2:
                    # Try to reshape to (frames, joints, channels)
                    if recon_sample.shape[1] == 75:  # frames, joints*channels
                        recon_sample = recon_sample.view(recon_sample.shape[0], 25, 3)
                    else:
                        print(f"Cannot reshape recon_sample with shape {recon_sample.shape} to (frames, joints, channels)")
                        continue
                else:
                    print(f"Unexpected recon_sample shape: {recon_sample.shape}")
                    continue

                # Ensure orig_sample has the right shape for metrics (frames, joints, channels)
                if len(orig_sample.shape) == 3 and orig_sample.shape[2] == 3:
                    # This is the expected format (frames, joints, channels)
                    pass
                elif len(orig_sample.shape) == 2:
                    # Try to reshape to (frames, joints, channels)
                    if orig_sample.shape[1] == 75:  # frames, joints*channels
                        orig_sample = orig_sample.view(orig_sample.shape[0], 25, 3)
                    else:
                        print(f"Cannot reshape orig_sample with shape {orig_sample.shape} to (frames, joints, channels)")
                        continue
                else:
                    print(f"Unexpected orig_sample shape: {orig_sample.shape}")
                    continue

                # Print shapes for debugging
                print(f"Calculating metrics for recon_sample shape: {recon_sample.shape}, orig_sample shape: {orig_sample.shape}")

                # All metric calculations on CPU to avoid device mismatch
                try:
                    # Make sure both samples have the same shape
                    if recon_sample.shape != orig_sample.shape:
                        print(f"Shape mismatch: recon_sample {recon_sample.shape}, orig_sample {orig_sample.shape}")
                        # Try to reshape to match
                        if len(recon_sample.shape) == 3 and len(orig_sample.shape) == 3:
                            # Both are 3D but different shapes
                            if recon_sample.shape[0] == orig_sample.shape[0]:  # Same number of frames
                                # Reshape to match number of joints
                                if recon_sample.shape[1] != 25 or recon_sample.shape[2] != 3:
                                    recon_sample = recon_sample.view(recon_sample.shape[0], 25, 3)
                                if orig_sample.shape[1] != 25 or orig_sample.shape[2] != 3:
                                    orig_sample = orig_sample.view(orig_sample.shape[0], 25, 3)

                    # Bone length consistency
                    try:
                        bone_len = calculate_bone_length_consistency(recon_sample, args.dataset)
                        utility_meters['bone_len'].update(bone_len)
                        print(f"Bone length consistency: {bone_len}")
                    except Exception as e:
                        print(f"Error calculating bone length consistency: {e}")
                        import traceback
                        traceback.print_exc()

                    # Joint angle limits
                    try:
                        angle_result = calculate_joint_angle_limits(recon_sample, args.dataset)
                        if isinstance(angle_result, tuple):
                            angle_viol, _ = angle_result
                        else:
                            angle_viol = angle_result
                        utility_meters['joint_angle'].update(angle_viol)
                        print(f"Joint angle limits: {angle_viol}")
                    except Exception as e:
                        print(f"Error calculating joint angle limits: {e}")
                        import traceback
                        traceback.print_exc()

                    # Temporal smoothness
                    try:
                        smoothness = calculate_temporal_smoothness(recon_sample)
                        utility_meters['smoothness'].update(smoothness)
                        print(f"Temporal smoothness: {smoothness}")
                    except Exception as e:
                        print(f"Error calculating temporal smoothness: {e}")
                        import traceback
                        traceback.print_exc()

                    # Velocity consistency
                    try:
                        vel_cons = calculate_velocity_consistency(recon_sample, orig_sample)
                        utility_meters['vel_cons'].update(vel_cons)
                        print(f"Velocity consistency: {vel_cons}")
                    except Exception as e:
                        print(f"Error calculating velocity consistency: {e}")
                        import traceback
                        traceback.print_exc()

                    # Foot contact consistency
                    try:
                        foot_contact = calculate_foot_contact_consistency(recon_sample, orig_sample, args.dataset)
                        utility_meters['foot_contact'].update(foot_contact)
                        print(f"Foot contact consistency: {foot_contact}")
                    except Exception as e:
                        print(f"Error calculating foot contact consistency: {e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            continue

        # FID calculation (outside the try-except block)
        if args.calculate_fid:
            for i in range(x_a_recon_cpu.shape[0]):
                try:
                    orig_fid_feats.append(extract_velocity_features(original_x_a[i]))
                    anon_fid_feats.append(extract_velocity_features(x_a_recon_cpu[i]))
                except Exception as e:
                    print(f"Error calculating FID features: {e}")
                    continue

        # Recognition tasks (if models are available)
        if ar_model or ri_model or gc_model:
            try:
                # Keep x_a_recon on device for recognition
                processed_x_a = process_for_recognition(x_a_recon, device)
                print(f"Processed x_a for recognition, shape: {processed_x_a.shape}")

                # Action recognition
                if ar_model and action_labels is not None:
                    try:
                        ar_out = ar_model(processed_x_a)
                        _, ar_p = torch.max(ar_out, 1)
                        print(f"AR predictions: {ar_p.cpu().numpy()}")
                        print(f"AR labels: {action_labels.cpu().numpy()}")

                        # For action recognition, we need the action ID (column 1 if 2D)
                        if len(action_labels.shape) > 1 and action_labels.shape[1] > 1:
                            # If we have [actor_id, action_id], use action_id (column 1)
                            action_labels_1d = action_labels[:, 1].cpu().numpy()
                            print(f"Using column 1 for action labels: {action_labels_1d}")
                        else:
                            action_labels_1d = action_labels.cpu().numpy()

                        # Convert to integers to ensure compatibility
                        ar_preds_int = ar_p.cpu().numpy().astype(int).tolist()
                        ar_labels_int = action_labels_1d.astype(int).tolist()

                        # Make sure we have the same number of predictions and labels
                        if len(ar_preds_int) == len(ar_labels_int):
                            # Extend the lists instead of replacing them
                            all_metrics['ar_preds'].extend(ar_preds_int)
                            all_metrics['ar_labels'].extend(ar_labels_int)
                            print(f"Added {len(ar_preds_int)} AR predictions and labels. Total: {len(all_metrics['ar_preds'])}")
                        else:
                            print(f"Warning: AR predictions ({len(ar_preds_int)}) and labels ({len(ar_labels_int)}) have different lengths. Skipping.")
                    except Exception as e:
                        print(f"Error in action recognition: {e}")
                        import traceback
                        traceback.print_exc()

                # Re-identification
                if ri_model and actor_labels is not None:
                    try:
                        ri_out = ri_model(processed_x_a)
                        _, ri_p = torch.max(ri_out, 1)
                        print(f"RI predictions: {ri_p.cpu().numpy()}")
                        print(f"RI labels: {actor_labels.cpu().numpy()}")

                        # For re-identification, we need the actor ID (column 0 if 2D)
                        if len(actor_labels.shape) > 1 and actor_labels.shape[1] > 1:
                            # If we have [actor_id, action_id], use actor_id (column 0)
                            actor_labels_1d = actor_labels[:, 0].cpu().numpy()
                            print(f"Using column 0 for actor labels: {actor_labels_1d}")
                        else:
                            actor_labels_1d = actor_labels.cpu().numpy()

                        # Convert to integers to ensure compatibility
                        ri_preds_int = ri_p.cpu().numpy().astype(int).tolist()
                        ri_labels_int = actor_labels_1d.astype(int).tolist()

                        # Make sure we have the same number of predictions and labels
                        if len(ri_preds_int) == len(ri_labels_int):
                            # Extend the lists instead of replacing them
                            all_metrics['ri_preds'].extend(ri_preds_int)
                            all_metrics['ri_labels'].extend(ri_labels_int)
                            print(f"Added {len(ri_preds_int)} RI predictions and labels. Total: {len(all_metrics['ri_preds'])}")
                        else:
                            print(f"Warning: RI predictions ({len(ri_preds_int)}) and labels ({len(ri_labels_int)}) have different lengths. Skipping.")
                    except Exception as e:
                        print(f"Error in re-identification: {e}")
                        import traceback
                        traceback.print_exc()

                # Gender classification
                if gc_model and gender_labels is not None:
                    try:
                        gc_out = gc_model(processed_x_a)
                        _, gc_p = torch.max(gc_out, 1)
                        print(f"GC predictions: {gc_p.cpu().numpy()}")
                        print(f"GC labels: {gender_labels.cpu().numpy()}")

                        # Gender labels should already be single-column
                        if len(gender_labels.shape) > 1 and gender_labels.shape[1] > 1:
                            gender_labels_1d = gender_labels[:, 0].cpu().numpy()
                            print(f"Using column 0 for gender labels: {gender_labels_1d}")
                        else:
                            gender_labels_1d = gender_labels.cpu().numpy()

                        # Convert to integers to ensure compatibility
                        gc_preds_int = gc_p.cpu().numpy().astype(int).tolist()
                        gc_labels_int = gender_labels_1d.astype(int).tolist()

                        # Make sure we have the same number of predictions and labels
                        if len(gc_preds_int) == len(gc_labels_int):
                            # Extend the lists instead of replacing them
                            all_metrics['gc_preds'].extend(gc_preds_int)
                            all_metrics['gc_labels'].extend(gc_labels_int)
                            print(f"Added {len(gc_preds_int)} GC predictions and labels. Total: {len(all_metrics['gc_preds'])}")
                        else:
                            print(f"Warning: GC predictions ({len(gc_preds_int)}) and labels ({len(gc_labels_int)}) have different lengths. Skipping.")
                    except Exception as e:
                        print(f"Error in gender classification: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Error processing recognition tasks: {e}")
                import traceback
                traceback.print_exc()

    # Calculate accuracies
    try:
        # Process labels and predictions for accuracy calculation
        ar_accuracy = 0
        if all_metrics['ar_labels'] and all_metrics['ar_preds']:
            # Ensure we have the same number of labels and predictions
            min_len = min(len(all_metrics['ar_labels']), len(all_metrics['ar_preds']))
            ar_labels = all_metrics['ar_labels'][:min_len]
            ar_preds = all_metrics['ar_preds'][:min_len]

            print(f"AR labels count: {len(ar_labels)}")
            print(f"AR preds count: {len(ar_preds)}")

            # Convert to numpy arrays for consistent handling
            import numpy as np
            ar_labels_np = np.array(ar_labels)
            ar_preds_np = np.array(ar_preds)

            # Ensure both are 1D arrays of integers
            ar_labels_np = ar_labels_np.flatten().astype(int)
            ar_preds_np = ar_preds_np.flatten().astype(int)

            print(f"AR labels shape after processing: {ar_labels_np.shape}")
            print(f"AR preds shape after processing: {ar_preds_np.shape}")

            # Verify shapes match before calculating accuracy
            if ar_labels_np.shape != ar_preds_np.shape:
                print(f"ERROR: Shape mismatch between AR labels {ar_labels_np.shape} and predictions {ar_preds_np.shape}")
                print(f"Truncating to the minimum length to ensure compatibility")
                min_len = min(len(ar_labels_np), len(ar_preds_np))
                ar_labels_np = ar_labels_np[:min_len]
                ar_preds_np = ar_preds_np[:min_len]
                print(f"New shapes after truncation - labels: {ar_labels_np.shape}, preds: {ar_preds_np.shape}")

            # Calculate accuracy manually to avoid sklearn issues
            correct = (ar_labels_np == ar_preds_np).sum()
            total = len(ar_labels_np)
            ar_accuracy = float(correct) / total if total > 0 else 0
            print(f"AR accuracy calculated manually: {ar_accuracy} (based on {total} samples)")

        ri_accuracy = 0
        if all_metrics['ri_labels'] and all_metrics['ri_preds']:
            # Ensure we have the same number of labels and predictions
            min_len = min(len(all_metrics['ri_labels']), len(all_metrics['ri_preds']))
            ri_labels = all_metrics['ri_labels'][:min_len]
            ri_preds = all_metrics['ri_preds'][:min_len]

            print(f"RI labels count: {len(ri_labels)}")
            print(f"RI preds count: {len(ri_preds)}")

            # Convert to numpy arrays for consistent handling
            import numpy as np
            ri_labels_np = np.array(ri_labels)
            ri_preds_np = np.array(ri_preds)

            # Ensure both are 1D arrays of integers
            ri_labels_np = ri_labels_np.flatten().astype(int)
            ri_preds_np = ri_preds_np.flatten().astype(int)

            print(f"RI labels shape after processing: {ri_labels_np.shape}")
            print(f"RI preds shape after processing: {ri_preds_np.shape}")

            # Verify shapes match before calculating accuracy
            if ri_labels_np.shape != ri_preds_np.shape:
                print(f"ERROR: Shape mismatch between RI labels {ri_labels_np.shape} and predictions {ri_preds_np.shape}")
                print(f"Truncating to the minimum length to ensure compatibility")
                min_len = min(len(ri_labels_np), len(ri_preds_np))
                ri_labels_np = ri_labels_np[:min_len]
                ri_preds_np = ri_preds_np[:min_len]
                print(f"New shapes after truncation - labels: {ri_labels_np.shape}, preds: {ri_preds_np.shape}")

            # Calculate accuracy manually to avoid sklearn issues
            correct = (ri_labels_np == ri_preds_np).sum()
            total = len(ri_labels_np)
            ri_accuracy = float(correct) / total if total > 0 else 0
            print(f"RI accuracy calculated manually: {ri_accuracy} (based on {total} samples)")

        gc_accuracy = -1
        if all_metrics['gc_labels'] and all_metrics['gc_preds'] and gc_model:
            # Ensure we have the same number of labels and predictions
            min_len = min(len(all_metrics['gc_labels']), len(all_metrics['gc_preds']))
            gc_labels = all_metrics['gc_labels'][:min_len]
            gc_preds = all_metrics['gc_preds'][:min_len]

            print(f"GC labels count: {len(gc_labels)}")
            print(f"GC preds count: {len(gc_preds)}")

            # Convert to numpy arrays for consistent handling
            import numpy as np
            gc_labels_np = np.array(gc_labels)
            gc_preds_np = np.array(gc_preds)

            # Ensure both are 1D arrays of integers
            gc_labels_np = gc_labels_np.flatten().astype(int)
            gc_preds_np = gc_preds_np.flatten().astype(int)

            print(f"GC labels shape after processing: {gc_labels_np.shape}")
            print(f"GC preds shape after processing: {gc_preds_np.shape}")

            # Verify shapes match before calculating accuracy
            if gc_labels_np.shape != gc_preds_np.shape:
                print(f"ERROR: Shape mismatch between GC labels {gc_labels_np.shape} and predictions {gc_preds_np.shape}")
                print(f"Truncating to the minimum length to ensure compatibility")
                min_len = min(len(gc_labels_np), len(gc_preds_np))
                gc_labels_np = gc_labels_np[:min_len]
                gc_preds_np = gc_preds_np[:min_len]
                print(f"New shapes after truncation - labels: {gc_labels_np.shape}, preds: {gc_preds_np.shape}")

            # Calculate accuracy manually to avoid sklearn issues
            correct = (gc_labels_np == gc_preds_np).sum()
            total = len(gc_labels_np)
            gc_accuracy = float(correct) / total if total > 0 else 0
            print(f"GC accuracy calculated manually: {gc_accuracy} (based on {total} samples)")
    except Exception as e:
        print(f"Error calculating accuracies: {e}")
        import traceback
        traceback.print_exc()
        ar_accuracy = 0
        ri_accuracy = 0
        gc_accuracy = -1

    results = {
        'ar_accuracy': ar_accuracy,
        'ri_accuracy': ri_accuracy,
        'gc_accuracy': gc_accuracy,
        'utility_metrics': {name: meter.avg for name, meter in utility_meters.items()},
        'fid_score': -1
    }

    # Print metrics for debugging
    print("Final metrics:")
    print(f"AR accuracy: {results['ar_accuracy']}")
    print(f"RI accuracy: {results['ri_accuracy']}")
    print(f"GC accuracy: {results['gc_accuracy']}")
    print("Utility metrics:")
    for name, value in results['utility_metrics'].items():
        print(f"  {name}: {value}")

    if args.calculate_fid and len(orig_fid_feats) > 1 and len(anon_fid_feats) > 1:
        try:
            # Process features to ensure they're in the right format for FID calculation
            # First, convert lists to tensors if they aren't already
            if not isinstance(anon_fid_feats[0], torch.Tensor):
                anon_fid_feats = [torch.tensor(feat, dtype=torch.float32) for feat in anon_fid_feats]
            if not isinstance(orig_fid_feats[0], torch.Tensor):
                orig_fid_feats = [torch.tensor(feat, dtype=torch.float32) for feat in orig_fid_feats]

            # Flatten features to 2D for FID calculation
            anon_features_flat = []
            orig_features_flat = []

            for feat in anon_fid_feats:
                # Ensure feature is 2D (samples, features)
                if len(feat.shape) > 2:
                    # Flatten all dimensions except the first (samples)
                    feat_flat = feat.reshape(feat.shape[0], -1)
                else:
                    feat_flat = feat
                anon_features_flat.append(feat_flat)

            for feat in orig_fid_feats:
                # Ensure feature is 2D (samples, features)
                if len(feat.shape) > 2:
                    # Flatten all dimensions except the first (samples)
                    feat_flat = feat.reshape(feat.shape[0], -1)
                else:
                    feat_flat = feat
                orig_features_flat.append(feat_flat)

            # Concatenate all features
            anon_features_tensor = torch.cat(anon_features_flat, dim=0)
            orig_features_tensor = torch.cat(orig_features_flat, dim=0)

            # Calculate FID
            results['fid_score'] = calculate_fid_for_skeletons(
                anon_features_tensor.cpu().numpy(),
                orig_features_tensor.cpu().numpy()
            )
            print(f"Calculated FID score: {results['fid_score']}")
        except Exception as e:
            print(f"Error calculating FID: {e}")
            import traceback
            traceback.print_exc()
            results['fid_score'] = -1

    return results

def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data: dataset={args.dataset}, setting={args.setting}, T={args.T}, batch_size={args.batch_size}")

    # Load the raw data first
    X = load_data(args.dataset, T=args.T)

    # Get paired data
    test_loader, paired_test_loader = get_cross_data(
        X,
        args.dataset,
        args.setting,
        batch_size=args.batch_size,
        return_loader=True,
        test_samples=args.test_samples,
        seg=args.T
    )

    # Load pretrained model
    print(f"Loading pretrained model from {args.model_dir}")
    model = load_pretrained_model(args.model_dir, args.dataset, args.T, device)

    # Load recognition models
    ar_model, ri_model, gc_model = load_custom_recognition_models(
        args.ar_model_weights,
        args.ri_model_weights,
        args.gc_model_weights,
        device,
        args.dataset
    )

    # Evaluate reconstruction
    print("Evaluating reconstruction performance")
    recon_mse = evaluate_reconstruction(model, test_loader, device)
    print(f"Reconstruction MSE: {recon_mse:.6f}")

    # Evaluate downstream tasks if we have recognition models
    downstream_results = {
        'ar_accuracy': 0,
        'ri_accuracy': 0,
        'gc_accuracy': -1,
        'utility_metrics': {
            'bone_len': 0,
            'joint_angle': 0,
            'smoothness': 0,
            'vel_cons': 0,
            'foot_contact': 0
        },
        'fid_score': -1
    }

    # Only evaluate downstream tasks if we have at least one recognition model
    if ar_model or ri_model or gc_model:
        print("Evaluating downstream tasks")
        downstream_results = evaluate_downstream_tasks(
            model, paired_test_loader, ar_model, ri_model, gc_model, device, args
        )
    else:
        print("Skipping downstream task evaluation (no recognition models provided)")

        # Still evaluate utility metrics
        utility_meters = {
            'bone_len': AverageMeter(),
            'joint_angle': AverageMeter(),
            'smoothness': AverageMeter(),
            'vel_cons': AverageMeter(),
            'foot_contact': AverageMeter()
        }

        # Print information about the first batch to understand the data structure
        first_batch = True

        for batch_idx, batch_content in enumerate(tqdm(paired_test_loader, desc="Evaluating Utility Metrics")):
            if args.test_samples and batch_idx * args.batch_size >= args.test_samples:
                break

            if first_batch:
                print(f"Utility metrics batch content type: {type(batch_content)}")
                if isinstance(batch_content, tuple):
                    print(f"Utility metrics batch content length: {len(batch_content)}")
                    for i, item in enumerate(batch_content):
                        print(f"Utility metrics item {i} type: {type(item)}")
                        if hasattr(item, 'shape'):
                            print(f"Utility metrics item {i} shape: {item.shape}")
                first_batch = False

            # The Cross_Data loader returns a tuple of 6 elements:
            # (x1, x2, y1, y2, actors, actions)
            if isinstance(batch_content, tuple) and len(batch_content) >= 1:
                x_a = batch_content[0]  # Get x1 (actor 1, action 1)
            else:
                x_a = batch_content

            # Skip this batch if we can't process it
            if isinstance(x_a, list) and any(not isinstance(item, torch.Tensor) for item in x_a if item is not None):
                print(f"Skipping batch with non-tensor items")
                continue

            # Process each item in the batch individually if it's a list
            if isinstance(x_a, list):
                print(f"Processing list data with {len(x_a)} items")

                # Filter out None values and non-tensor items
                valid_items = [item for item in x_a if item is not None and isinstance(item, torch.Tensor)]

                if not valid_items:
                    print("No valid items in batch, skipping")
                    continue

                # Process each valid item
                for item in valid_items:
                    # Skip items that don't have the expected skeleton format
                    if len(item.shape) < 2 or (len(item.shape) == 3 and item.shape[2] != 3):
                        print(f"Skipping item with unexpected shape: {item.shape}")
                        continue

                    # Move item to device
                    try:
                        # Handle the specific case of [32, 2] shape (which seems to be causing errors)
                        if len(item.shape) == 2 and item.shape[1] == 2:
                            print(f"Skipping item with shape {item.shape} (likely labels)")
                            continue

                        item_device = item.to(device)

                        # Check and reshape item if needed
                        if len(item_device.shape) == 3 and item_device.shape[2] == 3:  # (frames, joints, channels)
                            # This is the expected format (frames, joints, channels)
                            # Reshape to (batch_size=1, frames, joints, channels)
                            item_reshaped = item_device.unsqueeze(0)
                        elif len(item_device.shape) == 2:  # (frames, joints*channels)
                            # Reshape to (batch_size=1, frames, joints, channels)
                            item_reshaped = item_device.view(1, item_device.shape[0], 25, 3)
                        elif len(item_device.shape) == 2 and item_device.shape[1] == 75:  # (batch, frames*joints*channels)
                            # Reshape to (batch_size=1, frames, joints, channels)
                            item_reshaped = item_device.view(1, item_device.shape[0], 25, 3)
                        elif len(item_device.shape) == 3 and item_device.shape[2] == 75:  # (batch, frames, joints*channels)
                            # This is the format we're seeing: [32, 64, 75]
                            # We need to reshape to (frames, joints, channels) and then add batch dimension
                            # First, we need to take just one sample since we're processing individually
                            sample = item_device[0]  # Take the first sample, shape: [64, 75]
                            # Reshape to (frames, joints, channels)
                            sample_reshaped = sample.view(sample.shape[0], 25, 3)
                            # Add batch dimension
                            item_reshaped = sample_reshaped.unsqueeze(0)  # Shape: [1, 64, 25, 3]
                        else:
                            print(f"Unexpected item shape: {item_device.shape}")
                            continue

                        # Forward pass
                        item_recon = model(item_reshaped)  # Already has batch dimension

                        # Move reconstructed data to CPU for metric calculations
                        # Ensure output has the right shape
                        try:
                            if item_recon.shape != item_reshaped.shape:
                                print(f"Output shape {item_recon.shape} doesn't match input shape {item_reshaped.shape}")

                                # If output is [1, 64, 1, 25, 3] and input is [1, 64, 25, 3]
                                # We'll reshape input to match output for comparison
                                if len(item_recon.shape) == 5 and len(item_reshaped.shape) == 4:
                                    # Add the actor dimension to input
                                    item_reshaped = item_reshaped.unsqueeze(2)
                                    print(f"Adjusted input shape to {item_reshaped.shape}")
                                elif len(item_recon.shape) == 4 and len(item_reshaped.shape) == 5:
                                    # Remove the actor dimension from output
                                    item_recon = item_recon.squeeze(2)
                                    print(f"Adjusted output shape to {item_recon.shape}")
                                elif item_recon.shape[0] == 1 and item_recon.shape[1] == item_reshaped.shape[1]:
                                    # Try to reshape output to match input shape
                                    item_recon = item_recon.view(item_reshaped.shape)
                                    print(f"Reshaped output to {item_recon.shape}")

                            # If shapes still don't match, we need to handle it differently
                            if item_recon.shape != item_reshaped.shape:
                                print(f"Shapes still don't match after reshaping: output {item_recon.shape}, input {item_reshaped.shape}")

                                # Try one more approach - reshape both to a common format
                                if len(item_recon.shape) >= 4 and len(item_reshaped.shape) >= 4:
                                    # Extract the core dimensions (batch, frames, joints, channels)
                                    batch_size = item_recon.shape[0]
                                    frames = item_recon.shape[1]

                                    # Reshape both to a flat format for comparison
                                    item_recon_flat = item_recon.view(batch_size, frames, -1)
                                    item_reshaped_flat = item_reshaped.view(batch_size, frames, -1)

                                    # Check if the flattened shapes match
                                    if item_recon_flat.shape == item_reshaped_flat.shape:
                                        print(f"Using flattened tensors with shape {item_recon_flat.shape}")
                                        # Continue with the flattened tensors
                                        item_recon = item_recon_flat
                                        item_reshaped = item_reshaped_flat
                        except Exception as e:
                            print(f"Error adjusting shapes: {e}")
                            # Continue with the original shapes

                        item_recon_cpu = item_recon.squeeze(0).cpu()
                        item_cpu = item.cpu()

                        # Ensure item_recon_cpu has the right shape for metrics
                        if len(item_recon_cpu.shape) == 3 and item_recon_cpu.shape[2] == 3:
                            # This is the expected format (frames, joints, channels)
                            pass
                        elif len(item_recon_cpu.shape) == 2:
                            # Reshape to (frames, joints, channels)
                            item_recon_cpu = item_recon_cpu.view(item_recon_cpu.shape[0], 25, 3)

                        # Ensure item_cpu has the right shape for metrics
                        if len(item_cpu.shape) == 3 and item_cpu.shape[2] == 3:
                            # This is the expected format (frames, joints, channels)
                            pass
                        elif len(item_cpu.shape) == 2:
                            # Reshape to (frames, joints, channels)
                            item_cpu = item_cpu.view(item_cpu.shape[0], 25, 3)

                        # Calculate metrics
                        utility_meters['bone_len'].update(calculate_bone_length_consistency(item_recon_cpu, args.dataset))

                        # Handle different return formats for calculate_joint_angle_limits
                        angle_result = calculate_joint_angle_limits(item_recon_cpu, args.dataset)
                        if isinstance(angle_result, tuple):
                            angle_viol, _ = angle_result
                        else:
                            angle_viol = angle_result
                        utility_meters['joint_angle'].update(angle_viol)

                        utility_meters['smoothness'].update(calculate_temporal_smoothness(item_recon_cpu))
                        utility_meters['vel_cons'].update(calculate_velocity_consistency(item_recon_cpu, item_cpu))
                        utility_meters['foot_contact'].update(calculate_foot_contact_consistency(item_recon_cpu, item_cpu, args.dataset))
                    except Exception as e:
                        print(f"Error processing item: {e}")
                        print(f"Item shape: {item.shape}")
                        continue

                continue  # Skip the rest of the loop for list data

            # For tensor data, process normally
            if not isinstance(x_a, torch.Tensor):
                print(f"Warning: x_a is not a tensor, it's a {type(x_a)}. Converting to tensor.")
                try:
                    x_a = torch.tensor(x_a, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting x_a to tensor: {e}")
                    continue

            # Save original x_a for comparison (on CPU)
            original_x_a = x_a.clone().cpu()

            # Process through the model
            try:
                x_a = x_a.to(device)
                x_a_recon = model(x_a)

                # Move reconstructed data to CPU for metric calculations
                x_a_recon_cpu = x_a_recon.cpu()

                # Utility metrics
                for i in range(x_a_recon_cpu.shape[0]):
                    # All metric calculations on CPU to avoid device mismatch
                    utility_meters['bone_len'].update(calculate_bone_length_consistency(x_a_recon_cpu[i], args.dataset))
                    # Handle different return formats for calculate_joint_angle_limits
                    angle_result = calculate_joint_angle_limits(x_a_recon_cpu[i], args.dataset)
                    if isinstance(angle_result, tuple):
                        angle_viol, _ = angle_result
                    else:
                        angle_viol = angle_result
                    utility_meters['joint_angle'].update(angle_viol)
                    utility_meters['smoothness'].update(calculate_temporal_smoothness(x_a_recon_cpu[i]))
                    utility_meters['vel_cons'].update(calculate_velocity_consistency(x_a_recon_cpu[i], original_x_a[i]))
                    utility_meters['foot_contact'].update(calculate_foot_contact_consistency(x_a_recon_cpu[i], original_x_a[i], args.dataset))
            except Exception as e:
                print(f"Error during model processing: {e}")
                continue

        downstream_results['utility_metrics'] = {name: meter.avg for name, meter in utility_meters.items()}

    # Combine results
    results = {
        'dataset': args.dataset,
        'setting': args.setting,
        'temporal_masking_ratio': args.temporal_ratio,
        'spatial_masking_ratio': args.spatial_ratio,
        'reconstruction_mse': recon_mse,
        'ar_accuracy': downstream_results['ar_accuracy'],
        'ri_accuracy': downstream_results['ri_accuracy'],
        'gc_accuracy': downstream_results['gc_accuracy'],
        'utility_metrics': downstream_results['utility_metrics'],
        'fid_score': downstream_results['fid_score']
    }

    # Print summary of results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Dataset: {args.dataset}, Setting: {args.setting}")
    print(f"Temporal Masking Ratio: {args.temporal_ratio}, Spatial Masking Ratio: {args.spatial_ratio}")
    print(f"Reconstruction MSE: {recon_mse:.6f}")
    print(f"Action Recognition Accuracy: {downstream_results['ar_accuracy']:.4f}")
    print(f"Re-identification Accuracy: {downstream_results['ri_accuracy']:.4f}")
    print(f"Gender Classification Accuracy: {downstream_results['gc_accuracy']:.4f}")
    print("Physical Plausibility Metrics:")
    for name, value in downstream_results['utility_metrics'].items():
        print(f"  {name}: {value:.6f}")
    print(f"FID Score: {downstream_results['fid_score']:.6f}")

    # Save results
    results_file = os.path.join(
        args.output_dir,
        f'{args.dataset}_{args.setting}_temporal_{args.temporal_ratio}_spatial_{args.spatial_ratio}_results.json'
    )
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")

# Physical plausibility metrics implementation
def calculate_bone_length_consistency(skeleton, dataset='ntu'):
    """Calculate bone length consistency for a skeleton sequence.

    Args:
        skeleton: Tensor of shape [frames, joints, channels]
        dataset: Dataset name for skeleton structure

    Returns:
        float: Bone length consistency score (lower is better)
    """
    # Define bone connections based on dataset
    if dataset == 'ntu' or dataset == 'ntu120':
        # NTU skeleton structure (25 joints)
        bones = [
            (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
            (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
        ]
    else:
        # Default to NTU
        bones = [
            (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
            (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
            (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)
        ]

    try:
        # Ensure skeleton is the right shape
        if len(skeleton.shape) != 3:
            print(f"Unexpected skeleton shape: {skeleton.shape}")
            return 0.0

        frames, joints, channels = skeleton.shape

        # Calculate bone lengths for each frame
        bone_lengths = []
        for frame in range(frames):
            frame_lengths = []
            for joint1, joint2 in bones:
                if joint1 < joints and joint2 < joints:
                    # Calculate Euclidean distance between joints
                    v1 = skeleton[frame, joint1]
                    v2 = skeleton[frame, joint2]
                    length = torch.sqrt(torch.sum((v1 - v2) ** 2))
                    frame_lengths.append(length.item())
            bone_lengths.append(frame_lengths)

        # Convert to tensor for easier calculations
        bone_lengths = torch.tensor(bone_lengths)

        # Calculate standard deviation of bone lengths across frames
        if bone_lengths.shape[0] > 1:
            bone_std = torch.std(bone_lengths, dim=0)
            # Average standard deviation across all bones
            avg_std = torch.mean(bone_std).item()
            return avg_std
        else:
            return 0.0
    except Exception as e:
        print(f"Error in bone_length_consistency: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def calculate_joint_angle_limits(skeleton, dataset='ntu'):
    """Calculate joint angle limit violations for a skeleton sequence.

    Args:
        skeleton: Tensor of shape [frames, joints, channels]
        dataset: Dataset name for skeleton structure

    Returns:
        float: Joint angle violation score (lower is better)
    """
    # Define joint triplets to check angles for
    if dataset == 'ntu' or dataset == 'ntu120':
        # Some key joint triplets in NTU skeleton
        joint_triplets = [
            (3, 2, 20),  # Right shoulder
            (5, 4, 20),  # Left shoulder
            (6, 5, 4),   # Left elbow
            (7, 6, 5),   # Left wrist
            (9, 8, 20),  # Right hip
            (10, 9, 8),  # Right knee
            (11, 10, 9), # Right ankle
            (13, 12, 0), # Left hip
            (14, 13, 12),# Left knee
            (15, 14, 13) # Left ankle
        ]
    else:
        # Default to NTU
        joint_triplets = [
            (3, 2, 20),  # Right shoulder
            (5, 4, 20),  # Left shoulder
            (6, 5, 4),   # Left elbow
            (7, 6, 5),   # Left wrist
            (9, 8, 20),  # Right hip
            (10, 9, 8),  # Right knee
            (11, 10, 9), # Right ankle
            (13, 12, 0), # Left hip
            (14, 13, 12),# Left knee
            (15, 14, 13) # Left ankle
        ]

    try:
        # Ensure skeleton is the right shape
        if len(skeleton.shape) != 3:
            print(f"Unexpected skeleton shape: {skeleton.shape}")
            return 0.0

        frames, joints, channels = skeleton.shape

        # Calculate angles for each frame and joint triplet
        violations = 0
        total_angles = 0

        for frame in range(frames):
            for j1, j2, j3 in joint_triplets:
                if j1 < joints and j2 < joints and j3 < joints:
                    # Get joint positions
                    v1 = skeleton[frame, j1]
                    v2 = skeleton[frame, j2]
                    v3 = skeleton[frame, j3]

                    # Calculate vectors
                    vec1 = v1 - v2
                    vec2 = v3 - v2

                    # Calculate angle using dot product
                    dot = torch.sum(vec1 * vec2)
                    norm1 = torch.sqrt(torch.sum(vec1 ** 2))
                    norm2 = torch.sqrt(torch.sum(vec2 ** 2))

                    # Avoid division by zero
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        cos_angle = dot / (norm1 * norm2)
                        # Clamp to avoid numerical issues
                        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                        angle = torch.acos(cos_angle) * 180 / 3.14159

                        # Check for extreme angles (near 0 or 180 degrees)
                        if angle < 10 or angle > 170:
                            violations += 1

                        total_angles += 1

        # Calculate violation ratio
        if total_angles > 0:
            violation_ratio = violations / total_angles
            return violation_ratio
        else:
            return 0.0
    except Exception as e:
        print(f"Error in joint_angle_limits: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def calculate_temporal_smoothness(skeleton):
    """Calculate temporal smoothness for a skeleton sequence.

    Args:
        skeleton: Tensor of shape [frames, joints, channels]

    Returns:
        float: Temporal smoothness score (lower is better)
    """
    try:
        # Ensure skeleton is the right shape
        if len(skeleton.shape) != 3:
            print(f"Unexpected skeleton shape: {skeleton.shape}")
            return 0.0

        frames, joints, channels = skeleton.shape

        if frames < 2:
            return 0.0

        # Calculate velocity (first derivative)
        velocity = skeleton[1:] - skeleton[:-1]

        # Calculate acceleration (second derivative)
        if frames > 2:
            acceleration = velocity[1:] - velocity[:-1]

            # Calculate average acceleration magnitude
            accel_magnitude = torch.sqrt(torch.sum(acceleration ** 2, dim=2))
            avg_accel = torch.mean(accel_magnitude).item()
            return avg_accel
        else:
            # If only 2 frames, return velocity magnitude
            vel_magnitude = torch.sqrt(torch.sum(velocity ** 2, dim=2))
            avg_vel = torch.mean(vel_magnitude).item()
            return avg_vel
    except Exception as e:
        print(f"Error in temporal_smoothness: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def calculate_velocity_consistency(skeleton, reference_skeleton):
    """Calculate velocity consistency between skeleton and reference.

    Args:
        skeleton: Tensor of shape [frames, joints, channels]
        reference_skeleton: Tensor of shape [frames, joints, channels]

    Returns:
        float: Velocity consistency score (lower is better)
    """
    try:
        # Ensure skeletons are the right shape
        if len(skeleton.shape) != 3 or len(reference_skeleton.shape) != 3:
            print(f"Unexpected skeleton shapes: {skeleton.shape}, {reference_skeleton.shape}")
            return 0.0

        frames, joints, channels = skeleton.shape
        ref_frames, ref_joints, ref_channels = reference_skeleton.shape

        # Use minimum number of frames
        frames = min(frames, ref_frames)

        if frames < 2:
            return 0.0

        # Calculate velocities
        velocity = skeleton[1:frames] - skeleton[:frames-1]
        ref_velocity = reference_skeleton[1:frames] - reference_skeleton[:frames-1]

        # Calculate velocity magnitude differences
        vel_magnitude = torch.sqrt(torch.sum(velocity ** 2, dim=2))
        ref_vel_magnitude = torch.sqrt(torch.sum(ref_velocity ** 2, dim=2))

        # Calculate mean squared error between velocity magnitudes
        vel_diff = torch.mean((vel_magnitude - ref_vel_magnitude) ** 2).item()
        return vel_diff
    except Exception as e:
        print(f"Error in velocity_consistency: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def calculate_foot_contact_consistency(skeleton, reference_skeleton, dataset='ntu'):
    """Calculate foot contact consistency between skeleton and reference.

    Args:
        skeleton: Tensor of shape [frames, joints, channels]
        reference_skeleton: Tensor of shape [frames, joints, channels]
        dataset: Dataset name for skeleton structure

    Returns:
        float: Foot contact consistency score (lower is better)
    """
    try:
        # Define foot joints based on dataset
        if dataset == 'ntu' or dataset == 'ntu120':
            # NTU skeleton structure
            left_foot = 15  # Left toe
            right_foot = 11  # Right toe
        else:
            # Default to NTU
            left_foot = 15
            right_foot = 11

        # Ensure skeletons are the right shape
        if len(skeleton.shape) != 3 or len(reference_skeleton.shape) != 3:
            print(f"Unexpected skeleton shapes: {skeleton.shape}, {reference_skeleton.shape}")
            return 0.0

        frames, joints, channels = skeleton.shape
        ref_frames, ref_joints, ref_channels = reference_skeleton.shape

        # Use minimum number of frames
        frames = min(frames, ref_frames)

        if frames < 2:
            return 0.0

        # Get foot height (y-coordinate) for each frame
        left_foot_height = skeleton[:frames, left_foot, 1]
        right_foot_height = skeleton[:frames, right_foot, 1]
        ref_left_foot_height = reference_skeleton[:frames, left_foot, 1]
        ref_right_foot_height = reference_skeleton[:frames, right_foot, 1]

        # Detect foot contact with ground (when height is near minimum)
        left_min = torch.min(left_foot_height)
        right_min = torch.min(right_foot_height)
        ref_left_min = torch.min(ref_left_foot_height)
        ref_right_min = torch.min(ref_right_foot_height)

        # Define threshold for contact (10% above minimum height)
        left_threshold = left_min + 0.1 * (torch.max(left_foot_height) - left_min)
        right_threshold = right_min + 0.1 * (torch.max(right_foot_height) - right_min)
        ref_left_threshold = ref_left_min + 0.1 * (torch.max(ref_left_foot_height) - ref_left_min)
        ref_right_threshold = ref_right_min + 0.1 * (torch.max(ref_right_foot_height) - ref_right_min)

        # Detect contact frames
        left_contact = left_foot_height <= left_threshold
        right_contact = right_foot_height <= right_threshold
        ref_left_contact = ref_left_foot_height <= ref_left_threshold
        ref_right_contact = ref_right_foot_height <= ref_right_threshold

        # Calculate consistency (XOR of contact states)
        left_consistency = torch.logical_xor(left_contact, ref_left_contact).float().mean().item()
        right_consistency = torch.logical_xor(right_contact, ref_right_contact).float().mean().item()

        # Average consistency (lower is better)
        avg_consistency = (left_consistency + right_consistency) / 2
        return avg_consistency
    except Exception as e:
        print(f"Error in foot_contact_consistency: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == '__main__':
    main()
