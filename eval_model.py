#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import os
import pickle
import numpy as np
import sys
import traceback
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.linalg import sqrtm
import logging

# Import data handling code
from data import datasets, load_data, get_cross_data
import csv

# Import safe model loading utilities
from safe_model_loading import safe_load_model as safe_load_model_util
from safe_model_loading import fix_missing_buffers

# Conditionally import evaluation modules
original_sys_path = sys.path.copy()
try:
    sys.path.append(os.path.abspath('eval'))
    from preprocess import (
        sgn_preprocess_single_skeleton,
        mixformer_preprocess_single_skeleton
    )
    from eval_loader import AverageMeter

except ImportError as e:
    print(f"Warning: Could not import some evaluation modules: {e}")
finally:
    sys.path = original_sys_path

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

def load_gender_data(dataset):
    """
    Load gender data from CSV file.

    Args:
        dataset: str - Dataset name ('ntu', 'ntu120')

    Returns:
        dict - Mapping of actor ID (1-indexed) to gender ('M' or 'F')
    """
    gender_map = {}
    gender_file = f"data/{dataset}/statistics/Genders.csv"

    try:
        with open(gender_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV has 1-indexed actor IDs
                actor_id = int(row['P'])
                gender = row['Gender']
                gender_map[actor_id] = gender
        print(f"Loaded gender data for {len(gender_map)} subjects from {gender_file}")
    except Exception as e:
        print(f"Error loading gender data: {e}")
        print(f"Gender classification will not be available")

    return gender_map

def import_class(import_str):
    """Dynamically import a class from a string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exc()})')

def safe_load_model(model_path, device='cpu'):
    """
    Safely load model weights with proper device mapping.
    This is a wrapper around the safe_load_model_util function for backward compatibility.

    Args:
        model_path: Path to the model weights
        device: Device to load the model onto

    Returns:
        The loaded state dictionary
    """
    print(f"Loading model from {model_path} (device: {device})")
    try:
        # Check if the model path contains .tar extension
        is_tar_file = '.tar' in model_path.lower()

        # Load the model with proper device mapping
        checkpoint = torch.load(model_path, map_location=device)

        # If it's a .tar file, it likely has the structure:
        # dict_keys(['epoch', 'state_dict', 'best_acc', 'optimizer', 'scheduler'])
        if is_tar_file:
            print(f"Detected .tar checkpoint file with keys: {checkpoint.keys()}")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                print(f"Warning: .tar file doesn't contain expected keys. Available keys: {checkpoint.keys()}")
                # Try to use the checkpoint directly if it doesn't have the expected structure
                state_dict = checkpoint
        else:
            # For .pt or .pth files, check if it's already a dict with model_state_dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume it's already a state dict
                    state_dict = checkpoint
            else:
                # If it's not a dict, assume it's already a state dict
                state_dict = checkpoint

        # Clean up state dict keys
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        return state_dict
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("This could be due to a corrupted model file or incompatible format.")
        print("Please check that the model file exists and is in the correct format.")
        print("If you're using a PMR or DMR model, make sure it's been properly trained.")
        sys.exit(1)

def prep_data(x):
    """Prepare data for the Transformer model."""
    N, T, D = x.shape
    M = 1
    V = 25
    C_in = 3
    assert D == (V*C_in), f"Unexpected D={D}. Should be 75 for single-person with 25 joints."
    x = x.view(N, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
    return x

def anonymizer_to_sgn(t, max_frames=300):
    """Convert anonymizer output format to SGN format."""
    # Pad t to max_frames and add duplicate actor
    T = t.shape[0]
    if T < max_frames:
        pad_size = max_frames - T
        padding = torch.zeros(pad_size, t.shape[1])
        t = torch.cat([t, padding], dim=0)
    if t.shape[1] == 75:
        buffer = torch.zeros_like(t)
        t = torch.cat([t, buffer], dim=1)
    return t

#------------------------------------------------------------------------------
# Model Loading Functions
#------------------------------------------------------------------------------

def load_anonymizer(model_type, model_path, device, args, ds=None):
    """
    Dynamically load and return an anonymization model or None (for raw).

    Args:
        model_type: str - 'transformer', 'pmr', 'dmr', or 'raw'
        model_path: str - Path to the model weights (if not raw)
        device: torch.device - The device to load the model onto
        args: argparse.Namespace - Arguments including dataset info
        ds: dict - The dataset config, used only for transformer

    Returns:
        The loaded model or None for 'raw'
    """
    if model_type == 'raw':
        return None

    elif model_type == 'transformer':
        from model.autoencoder import Model
        # Set a flag in the args to indicate we're using transformer
        args.loading_transformer = True

        autoenc_model = Model(
            num_class=ds['num_class'],
            num_point=ds['joints'],
            num_person=ds['max_actors'],
            graph=ds['graph'],
            graph_args=ds['graph_args'],
            device=device,
            dataset=args.dataset,
            debug=False
        ).to(device)

        # Reset the flag
        args.loading_transformer = False

        # Fix missing buffers if any
        autoenc_model = fix_missing_buffers(autoenc_model)

        # Load model weights
        state_dict = safe_load_model(model_path, device)
        autoenc_model.load_state_dict(state_dict, strict=False)
        autoenc_model.eval()
        return autoenc_model

    elif model_type == 'pmr':
        sys.path.append('./eval/pmr')
        from pmr import PMR
        model = PMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)

        # Fix missing buffers if any
        model = fix_missing_buffers(model)

        # Load model weights
        state_dict = safe_load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.set_eval(True)
        return model

    elif model_type == 'dmr':
        sys.path.append('./eval/dmr')
        from dmr import DMR
        model = DMR(dataset=args.dataset, datasets=datasets, batch_size=args.batch_size).to(device)

        # Fix missing buffers if any
        model = fix_missing_buffers(model)

        # Load model weights
        state_dict = safe_load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.set_eval(True)
        return model

    else:
        raise ValueError(f"Unrecognized model_type={model_type}")

#------------------------------------------------------------------------------
# Anonymization Functions
#------------------------------------------------------------------------------

@torch.no_grad()
def get_anonymized_paired_raw(batch, gender_map=None):
    """Process raw (unanonymized) skeleton data."""
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    out = []

    for i in range(N):
        # Ensure actions are 0-indexed for models but 1-indexed in data
        action1 = int(actions[i, 0]) - 1
        action2 = int(actions[i, 1]) - 1
        actor1 = int(actors[i, 0]) - 1
        actor2 = int(actors[i, 1]) - 1

        # Add gender information (1-indexed in CSV, so add 1 to actor ID)
        gender1 = None
        gender2 = None
        if gender_map:
            # Convert 0-indexed actor ID to 1-indexed for gender map lookup
            gender1 = 1 if gender_map.get(actor1 + 1, 'M') == 'M' else 0
            gender2 = 1 if gender_map.get(actor2 + 1, 'M') == 'M' else 0

        # Output four combinations of data
        item1 = {
            'skeleton': x1[i].cpu().to(torch.float32),  # p1 a1
            'gt_skeleton': y2[i].cpu().to(torch.float32),  # p2 a1
            'reference_skeleton': x1[i].cpu().to(torch.float32),  # p1 a1 (reference = skeleton for raw)
            'retargeted_actor': actor2,  # p2
            'original_actor': actor1,  # p1
            'action': action1  # a1
        }
        if gender1 is not None:
            item1['gender'] = gender1
        out.append(item1)

        item2 = {
            'skeleton': x2[i].cpu().to(torch.float32),  # p2 a2
            'gt_skeleton': y1[i].cpu().to(torch.float32),  # p1 a2
            'reference_skeleton': x2[i].cpu().to(torch.float32),  # p2 a2 (reference = skeleton for raw)
            'retargeted_actor': actor1,  # p1
            'original_actor': actor2,  # p2
            'action': action2  # a2
        }
        if gender2 is not None:
            item2['gender'] = gender2
        out.append(item2)

        item3 = {
            'skeleton': y1[i].cpu().to(torch.float32),  # p1 a2
            'gt_skeleton': x2[i].cpu().to(torch.float32),  # p2 a2
            'reference_skeleton': y1[i].cpu().to(torch.float32),  # p1 a2 (reference = skeleton for raw)
            'retargeted_actor': actor2,  # p2
            'original_actor': actor1,  # p1
            'action': action2
        }
        if gender1 is not None:
            item3['gender'] = gender1
        out.append(item3)

        item4 = {
            'skeleton': y2[i].cpu().to(torch.float32),  # p2 a1
            'gt_skeleton': x1[i].cpu().to(torch.float32),  # p1 a1
            'reference_skeleton': y2[i].cpu().to(torch.float32),  # p2 a1 (reference = skeleton for raw)
            'retargeted_actor': actor1,  # p1
            'original_actor': actor2,  # p2
            'action': action1
        }
        if gender2 is not None:
            item4['gender'] = gender2
        out.append(item4)
    return out

@torch.no_grad()
def get_anonymized_paired_transformer(batch, model, prep_data_fn, gender_map=None):
    """Process skeleton data through transformer anonymizer."""
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    T = x1.shape[1]

    # Prepare data for transformer model
    x1_ = prep_data_fn(x1).cuda()
    x2_ = prep_data_fn(x2).cuda()
    y1_ = prep_data_fn(y1).cuda()
    y2_ = prep_data_fn(y2).cuda()

    # Process all data combinations through model
    inputs  = torch.cat([x1_, x2_, y1_, y2_], dim=0)
    dummy   = torch.cat([x2_, x1_, y2_, y1_], dim=0)
    targets = torch.cat([y2_, y1_, x2_, x1_], dim=0)

    outputs = model(inputs, dummy, teacher_forcing_ratio=0.0)
    first_frames = inputs[:, :, 0:1, :, :]
    full_out = torch.cat([first_frames, outputs], dim=2)

    # Separate results and reshape to original format
    x1_hat, x2_hat, y1_hat, y2_hat = torch.split(full_out, N, dim=0)
    D = x1.shape[2]

    def unreshape(x_):
        return x_.permute(0, 2, 4, 3, 1).contiguous().view(N, T, D)

    x1_hat = unreshape(x1_hat)
    x2_hat = unreshape(x2_hat)
    y1_hat = unreshape(y1_hat)
    y2_hat = unreshape(y2_hat)

    # Format results
    out = []
    for i in range(N):
        # Explicitly convert labels to integers
        action1 = int(actions[i, 0]) - 1
        action2 = int(actions[i, 1]) - 1
        actor1 = int(actors[i, 0]) - 1
        actor2 = int(actors[i, 1]) - 1

        # Add gender information (1-indexed in CSV, so add 1 to actor ID)
        gender1 = None
        gender2 = None
        if gender_map:
            # Convert 0-indexed actor ID to 1-indexed for gender map lookup
            gender1 = 1 if gender_map.get(actor1 + 1, 'M') == 'M' else 0
            gender2 = 1 if gender_map.get(actor2 + 1, 'M') == 'M' else 0

        item1 = {
            'skeleton': x1_hat[i].cpu(),
            'gt_skeleton': y2[i].cpu(),
            'reference_skeleton': x1[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor2,
            'original_actor': actor1,
            'action': action1
        }
        if gender1 is not None:
            item1['gender'] = gender1
        out.append(item1)

        item2 = {
            'skeleton': x2_hat[i].cpu(),
            'gt_skeleton': y1[i].cpu(),
            'reference_skeleton': x2[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor1,
            'original_actor': actor2,
            'action': action2
        }
        if gender2 is not None:
            item2['gender'] = gender2
        out.append(item2)

        item3 = {
            'skeleton': y1_hat[i].cpu(),
            'gt_skeleton': x2[i].cpu(),
            'reference_skeleton': y1[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor2,
            'original_actor': actor1,
            'action': action2
        }
        if gender1 is not None:
            item3['gender'] = gender1
        out.append(item3)

        item4 = {
            'skeleton': y2_hat[i].cpu(),
            'gt_skeleton': x1[i].cpu(),
            'reference_skeleton': y2[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor1,
            'original_actor': actor2,
            'action': action1
        }
        if gender2 is not None:
            item4['gender'] = gender2
        out.append(item4)
    return out

@torch.no_grad()
def get_anonymized_paired_dmr_pmr(batch, model, T=75, mixformer_mode=False, gender_map=None):
    """
    Process skeleton data through DMR/PMR anonymizer matching original preprocessing.

    Args:
        batch: Tuple of (x1, x2, y1, y2, actors, actions)
        model: The DMR/PMR model to use
        T: Number of frames to use for the model (default=75 for DMR/PMR)
        mixformer_mode: If True, trim output to 64 frames for Mixformer compatibility
        gender_map: Optional dictionary mapping actor IDs to gender labels
    """
    x1, x2, y1, y2, actors, actions = batch
    N = x1.shape[0]
    old_T = x1.shape[1]

    # DMR/PMR models always require 75 frames as input
    # Regardless of the T parameter passed in
    dmr_pmr_T = 75

    # Store the original T for later use
    target_T = T

    # Override T for DMR/PMR processing
    T = dmr_pmr_T

    # When padding, repeat the last frame instead of using zeros
    # This matches the original preprocessing method
    if old_T < T:
        pad_sizes = [T - old_T] * N
        padded_x1, padded_x2 = [], []
        padded_y1, padded_y2 = [], []

        for i in range(N):
            # Get last frames
            x1_last = x1[i, -1:].repeat(pad_sizes[i], 1)
            x2_last = x2[i, -1:].repeat(pad_sizes[i], 1)
            y1_last = y1[i, -1:].repeat(pad_sizes[i], 1)
            y2_last = y2[i, -1:].repeat(pad_sizes[i], 1)

            # Concat with original
            padded_x1.append(torch.cat([x1[i], x1_last], dim=0))
            padded_x2.append(torch.cat([x2[i], x2_last], dim=0))
            padded_y1.append(torch.cat([y1[i], y1_last], dim=0))
            padded_y2.append(torch.cat([y2[i], y2_last], dim=0))

        # Stack back to tensors
        x1 = torch.stack(padded_x1)
        x2 = torch.stack(padded_x2)
        y1 = torch.stack(padded_y1)
        y2 = torch.stack(padded_y2)

    out = []
    for i in range(N):
        # Prepare data for model using original format
        # with explicit reshaping to 25 joints x 3 dimensions
        # Handle both CPU and GPU devices
        device = next(model.parameters()).device
        x1_in = x1[i].unsqueeze(0).float().to(device).view(1, T, 25, 3)
        x2_in = x2[i].unsqueeze(0).float().to(device).view(1, T, 25, 3)
        y1_in = y1[i].unsqueeze(0).float().to(device).view(1, T, 25, 3)
        y2_in = y2[i].unsqueeze(0).float().to(device).view(1, T, 25, 3)

        # Run model on all combinations
        x1_hat = model.eval(x1_in, x2_in).squeeze(0).cpu()
        x2_hat = model.eval(x2_in, x1_in).squeeze(0).cpu()
        y1_hat = model.eval(y1_in, y2_in).squeeze(0).cpu()
        y2_hat = model.eval(y2_in, y1_in).squeeze(0).cpu()

        # If in mixformer mode, trim to 64 frames (Mixformer requirement)
        if mixformer_mode:
            print(f"Trimming DMR/PMR output from {x1_hat.shape[0]} to 64 frames for Mixformer compatibility")
            x1_hat = x1_hat[:64]
            x2_hat = x2_hat[:64]
            y1_hat = y1_hat[:64]
            y2_hat = y2_hat[:64]
        # Otherwise, trim to target_T if different from DMR/PMR's T
        elif target_T != T:
            x1_hat = x1_hat[:target_T]
            x2_hat = x2_hat[:target_T]
            y1_hat = y1_hat[:target_T]
            y2_hat = y2_hat[:target_T]
        # Otherwise, trim back to original length if needed
        elif old_T < T:
            x1_hat = x1_hat[:old_T]
            x2_hat = x2_hat[:old_T]
            y1_hat = y1_hat[:old_T]
            y2_hat = y2_hat[:old_T]

        # Explicitly convert labels to integers
        action1 = int(actions[i, 0]) - 1
        action2 = int(actions[i, 1]) - 1
        actor1 = int(actors[i, 0]) - 1
        actor2 = int(actors[i, 1]) - 1

        # Add gender information (1-indexed in CSV, so add 1 to actor ID)
        gender1 = None
        gender2 = None
        if gender_map:
            # Convert 0-indexed actor ID to 1-indexed for gender map lookup
            gender1 = 1 if gender_map.get(actor1 + 1, 'M') == 'M' else 0
            gender2 = 1 if gender_map.get(actor2 + 1, 'M') == 'M' else 0

        # Store results
        item1 = {
            'skeleton': x1_hat,
            'gt_skeleton': y2[i].cpu(),
            'reference_skeleton': x1[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor2,
            'original_actor': actor1,
            'action': action1
        }
        if gender1 is not None:
            item1['gender'] = gender1
        out.append(item1)

        item2 = {
            'skeleton': x2_hat,
            'gt_skeleton': y1[i].cpu(),
            'reference_skeleton': x2[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor1,
            'original_actor': actor2,
            'action': action2
        }
        if gender2 is not None:
            item2['gender'] = gender2
        out.append(item2)

        item3 = {
            'skeleton': y1_hat,
            'gt_skeleton': x2[i].cpu(),
            'reference_skeleton': y1[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor2,
            'original_actor': actor1,
            'action': action2
        }
        if gender1 is not None:
            item3['gender'] = gender1
        out.append(item3)

        item4 = {
            'skeleton': y2_hat,
            'gt_skeleton': x1[i].cpu(),
            'reference_skeleton': y2[i].cpu(),  # Add reference skeleton
            'retargeted_actor': actor1,
            'original_actor': actor2,
            'action': action1
        }
        if gender2 is not None:
            item4['gender'] = gender2
        out.append(item4)
    return out

#------------------------------------------------------------------------------
# Recognition Processing Functions
#------------------------------------------------------------------------------

def process_for_recognition(skeleton_tensor, device=None):
    """
    Process skeleton tensor for recognition models (SGN or MixFormer).

    This function takes a skeleton tensor from the autoencoder output and
    formats it for input to recognition models.

    Args:
        skeleton_tensor: torch.Tensor - Tensor with shape (B, T, V*C) or (B, T, V, C)
        device: torch.device - Device to place the processed tensor on

    Returns:
        torch.Tensor - Processed tensor ready for recognition models
    """
    # Ensure input is a tensor
    if not isinstance(skeleton_tensor, torch.Tensor):
        skeleton_tensor = torch.tensor(skeleton_tensor, dtype=torch.float32)

    # Get batch size and sequence length
    B = skeleton_tensor.size(0)
    T = skeleton_tensor.size(1)

    # Handle different input formats
    if len(skeleton_tensor.shape) == 3:  # (B, T, V*C)
        # Assuming V=25, C=3 for standard skeleton
        V = 25
        C = 3
        # Reshape to (B, T, V, C)
        skeleton_tensor = skeleton_tensor.view(B, T, V, C)

    # Now we should have (B, T, V, C)
    if len(skeleton_tensor.shape) != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {skeleton_tensor.shape}")

    # Reshape to (B, C, T, V) for recognition models
    processed = skeleton_tensor.permute(0, 3, 1, 2).contiguous()

    # Add M dimension for models that expect it (B, C, T, V, M)
    processed = processed.unsqueeze(-1)

    # Move to specified device if provided
    if device is not None:
        processed = processed.to(device)

    return processed

#------------------------------------------------------------------------------
# SGN Evaluation Helpers
#------------------------------------------------------------------------------

def accuracy_snippet(output, target):
    """Calculate top-1 accuracy for SGN model."""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)

def top_k_accuracy_snippet(output, target, k=3):
    """Calculate top-k accuracy for SGN model."""
    batch_size = target.size(0)
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

def test_snippet(test_loader, model, k=3):
    """Test SGN model with test-time snippet evaluation logic."""
    acces = AverageMeter()
    topk_acces = AverageMeter()
    model.eval()
    label_output = []
    pred_output = []

    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1))
            )
            output = output.mean(1)

        label_output.append(target.numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy_snippet(output.cpu(), target)
        acces.update(acc[0], inputs.size(0))

        topk_acc = top_k_accuracy_snippet(output.cpu(), target, k=k)
        topk_acces.update(topk_acc[0], inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    pred_output = np.concatenate(pred_output, axis=0)

    label_index = np.argmax(label_output, axis=1)
    pred_index = np.argmax(pred_output, axis=1)

    f1 = f1_score(label_index, pred_index, average='macro', zero_division=0)
    precision = precision_score(label_index, pred_index, average='macro', zero_division=0)
    recall = recall_score(label_index, pred_index, average='macro', zero_division=0)

    return acces.avg, f1, precision, recall, topk_acces.avg

#------------------------------------------------------------------------------
# Advanced Utility Metrics
#------------------------------------------------------------------------------

# Define bone pairs for NTU-type skeleton
bone_pairs_dict = {
    'ntu': [
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ],
    'ntu120': [
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ],
    'etri': [
        (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (20, 16), (16, 17), (17, 18), (18, 19),
        (1, 5), (5, 6), (6, 7), (1, 12), (12, 13), (13, 14), (14, 15)
    ]
}

# Define foot indices for foot contact detection
foot_indices_dict = {
    'ntu': [15, 19],  # Right ankle, Left ankle
    'ntu120': [15, 19],
    'etri': [15, 19]
}

# Define joint angle ranges for anatomical constraints
joint_angle_ranges = {
    'ntu': {
        (0, 1, 20):  (-95,  95),   # HipCenter -> Spine -> SpineShoulder
        (1, 20, 2):  (-120, 120),   # Spine -> SpineShoulder -> Neck
        (20, 2, 3):  (-45,  45),    # SpineShoulder -> Neck -> Head
        (20, 8, 9):  (-120, 120),   # SpineShoulder -> RightShoulder -> RightElbow
        (8, 9, 10):  (-120, 120),   # RightShoulder -> RightElbow -> RightWrist
        (20, 4, 5):  (-120, 120),   # SpineShoulder -> LeftShoulder -> LeftElbow
        (4, 5, 6):   (-120, 120),   # LeftShoulder -> LeftElbow -> LeftWrist
        (1, 12, 13): (-120, 120),   # Spine -> RightHip -> RightKnee
        (12, 13, 14):(-120, 120),   # RightHip -> RightKnee -> RightAnkle
        (1, 16, 17): (-120, 120),   # Spine -> LeftHip -> LeftKnee
        (16, 17, 18):(-120, 120)    # LeftHip -> LeftKnee -> LeftAnkle
    }
}

def calculate_bone_length_consistency(skeleton, dataset='ntu'):
    """
    Calculate how consistent bone lengths are throughout a sequence.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C) where T=frames, V=joints, C=channels
        dataset: Dataset name to determine bone pairs

    Returns:
        float: Standard deviation of bone lengths across frames (lower is better)
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    # Get bone pairs for this dataset
    bone_pairs = bone_pairs_dict.get(dataset, bone_pairs_dict['ntu'])

    # Calculate bone lengths for each frame
    bone_lengths = []
    for j1, j2 in bone_pairs:
        # Extract joint positions
        j1_pos = skeleton[:, j1, :]  # (T, C)
        j2_pos = skeleton[:, j2, :]  # (T, C)

        # Calculate bone length for each frame
        bone_length = torch.norm(j1_pos - j2_pos, dim=1)  # (T,)

        # Calculate std dev across frames for this bone
        if len(bone_length) > 1:
            bone_std = torch.std(bone_length).item()
            bone_lengths.append(bone_std)

    # Average std dev across all bones
    if bone_lengths:
        return sum(bone_lengths) / len(bone_lengths)
    return 0.0

def calculate_joint_angle_limits(skeleton, dataset='ntu'):
    """
    Calculate percentage of frames with anatomically valid joint angles.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C) where T=frames, V=joints, C=channels
        dataset: Dataset name to determine joint angle ranges

    Returns:
        tuple: (percentage, angle_data) where:
            - percentage: float - Percentage of frames with valid joint angles (higher is better)
            - angle_data: dict - Additional angle data (for future use)
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    # Get joint angle ranges for this dataset
    angle_ranges = joint_angle_ranges.get(dataset, joint_angle_ranges.get('ntu', {}))
    if not angle_ranges:
        return 100.0, {}  # No constraints defined, all frames are valid

    total_angles = 0
    valid_angles = 0
    angle_data = {}  # Store additional data for future use

    for (p, j, c), (min_deg, max_deg) in angle_ranges.items():
        # Calculate vectors from parent to joint and joint to child
        pj_vec = skeleton[:, j, :] - skeleton[:, p, :]  # (T, C)
        jc_vec = skeleton[:, c, :] - skeleton[:, j, :]  # (T, C)

        # Calculate angles between vectors
        dot_product = torch.sum(pj_vec * jc_vec, dim=1)  # (T,)
        pj_norm = torch.norm(pj_vec, dim=1)  # (T,)
        jc_norm = torch.norm(jc_vec, dim=1)  # (T,)

        # Handle zero-length vectors
        valid_mask = (pj_norm > 1e-6) & (jc_norm > 1e-6)
        if not valid_mask.any():
            continue

        # Calculate angle in degrees
        cos_angle = dot_product[valid_mask] / (pj_norm[valid_mask] * jc_norm[valid_mask])
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Numerical stability
        angles = torch.acos(cos_angle) * (180.0 / math.pi)  # Convert to degrees

        # Check if angles are within limits
        valid = ((angles >= min_deg) & (angles <= max_deg)).float().sum().item()
        total = valid_mask.sum().item()

        valid_angles += valid
        total_angles += total

        # Store angle data for this joint triplet
        angle_key = f"{p}_{j}_{c}"
        angle_data[angle_key] = {
            'valid': valid,
            'total': total,
            'min_deg': min_deg,
            'max_deg': max_deg
        }

    # Calculate percentage of valid angles
    if total_angles > 0:
        percentage = 100.0 * valid_angles / total_angles
    else:
        percentage = 100.0  # No angles checked

    return percentage, angle_data

def calculate_temporal_smoothness(skeleton):
    """
    Calculate temporal smoothness by measuring average acceleration magnitude.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C) where T=frames, V=joints, C=channels

    Returns:
        float: Average acceleration magnitude (lower is better)
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    # Need at least 3 frames to calculate acceleration
    if skeleton.shape[0] < 3:
        return 0.0

    # Calculate velocities (first derivative)
    velocities = skeleton[1:] - skeleton[:-1]  # (T-1, V, C)

    # Calculate accelerations (second derivative)
    accelerations = velocities[1:] - velocities[:-1]  # (T-2, V, C)

    # Calculate magnitude of acceleration for each joint
    accel_magnitudes = torch.norm(accelerations, dim=2)  # (T-2, V)

    # Average across all frames and joints
    avg_accel = torch.mean(accel_magnitudes).item()

    return avg_accel

def calculate_velocity_consistency(skeleton, reference_skeleton):
    """
    Calculate cosine similarity between velocity vectors of original and generated motion.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C) - generated skeleton
        reference_skeleton: Tensor of shape (T, V, C) or (T, V*C) - original skeleton

    Returns:
        float: Average cosine similarity between velocity vectors (higher is better)
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    if len(reference_skeleton.shape) == 2:
        T_ref, VC_ref = reference_skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC_ref // V
        reference_skeleton = reference_skeleton.reshape(T_ref, V, C)

    # Need at least 2 frames to calculate velocity
    if skeleton.shape[0] < 2 or reference_skeleton.shape[0] < 2:
        return 1.0  # Perfect consistency for single frame

    # Find the minimum number of frames between the two sequences
    T_min = min(skeleton.shape[0], reference_skeleton.shape[0])

    # Truncate both sequences to the same length
    skeleton_trunc = skeleton[:T_min]
    reference_skeleton_trunc = reference_skeleton[:T_min]

    # Calculate velocities
    velocities = skeleton_trunc[1:] - skeleton_trunc[:-1]  # (T_min-1, V, C)
    ref_velocities = reference_skeleton_trunc[1:] - reference_skeleton_trunc[:-1]  # (T_min-1, V, C)

    # Calculate cosine similarity for each joint and frame
    similarities = []
    for j in range(skeleton_trunc.shape[1]):  # For each joint
        v1 = velocities[:, j, :]  # (T_min-1, C)
        v2 = ref_velocities[:, j, :]  # (T_min-1, C)

        # Calculate norms
        v1_norm = torch.norm(v1, dim=1)  # (T_min-1,)
        v2_norm = torch.norm(v2, dim=1)  # (T_min-1,)

        # Find frames where both velocities are non-zero
        valid_mask = (v1_norm > 1e-6) & (v2_norm > 1e-6)
        if not valid_mask.any():
            continue

        # Calculate dot product
        dot_product = torch.sum(v1[valid_mask] * v2[valid_mask], dim=1)  # (valid_frames,)

        # Calculate cosine similarity
        similarity = dot_product / (v1_norm[valid_mask] * v2_norm[valid_mask])  # (valid_frames,)
        similarity = torch.clamp(similarity, -1.0, 1.0)  # Numerical stability

        # Average similarity for this joint
        if len(similarity) > 0:
            similarities.append(torch.mean(similarity).item())

    # Average across all joints
    if similarities:
        return sum(similarities) / len(similarities)
    return 0.0

def calculate_foot_contact_consistency(skeleton, reference_skeleton, dataset='ntu'):
    """
    Calculate how well foot contact with the ground is preserved.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C) - generated skeleton
        reference_skeleton: Tensor of shape (T, V, C) or (T, V*C) - original skeleton
        dataset: Dataset name to determine foot indices

    Returns:
        float: Percentage of frames where foot contact is preserved (higher is better)
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    if len(reference_skeleton.shape) == 2:
        T_ref, VC_ref = reference_skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC_ref // V
        reference_skeleton = reference_skeleton.reshape(T_ref, V, C)

    # Need at least 2 frames to calculate velocity
    if skeleton.shape[0] < 2 or reference_skeleton.shape[0] < 2:
        return 100.0  # Perfect consistency for single frame

    # Find the minimum number of frames between the two sequences
    T_min = min(skeleton.shape[0], reference_skeleton.shape[0])

    # Truncate both sequences to the same length
    skeleton_trunc = skeleton[:T_min]
    reference_skeleton_trunc = reference_skeleton[:T_min]

    # Get foot indices for this dataset
    foot_indices = foot_indices_dict.get(dataset, foot_indices_dict['ntu'])

    # Calculate velocities
    velocities = skeleton_trunc[1:] - skeleton_trunc[:-1]  # (T_min-1, V, C)
    ref_velocities = reference_skeleton_trunc[1:] - reference_skeleton_trunc[:-1]  # (T_min-1, V, C)

    # Extract foot velocities
    foot_velocities = velocities[:, foot_indices, :]  # (T_min-1, num_feet, C)
    ref_foot_velocities = ref_velocities[:, foot_indices, :]  # (T_min-1, num_feet, C)

    # Calculate velocity magnitudes
    foot_speed = torch.norm(foot_velocities, dim=2)  # (T_min-1, num_feet)
    ref_foot_speed = torch.norm(ref_foot_velocities, dim=2)  # (T_min-1, num_feet)

    # Define contact threshold
    threshold = 0.05

    # Determine contact frames in reference
    ref_contact = ref_foot_speed < threshold  # (T_min-1, num_feet)

    # Determine contact frames in generated
    gen_contact = foot_speed < threshold  # (T_min-1, num_feet)

    # Calculate consistency
    consistent = (ref_contact == gen_contact).float()  # (T_min-1, num_feet)
    consistency = torch.mean(consistent).item() * 100.0  # Percentage

    return consistency

def calculate_fid_for_skeletons(skeleton_features, reference_features):
    """
    Calculate Frechet Inception Distance (FID) for skeleton features.

    Args:
        skeleton_features: Tensor of shape (N, D) - features from generated skeletons
        reference_features: Tensor of shape (N, D) - features from reference skeletons

    Returns:
        float: FID score (lower is better)
    """
    # Convert to numpy for calculation
    if isinstance(skeleton_features, torch.Tensor):
        skeleton_features = skeleton_features.cpu().numpy()
    if isinstance(reference_features, torch.Tensor):
        reference_features = reference_features.cpu().numpy()

    # Calculate mean and covariance for both distributions
    mu1 = np.mean(skeleton_features, axis=0)
    sigma1 = np.cov(skeleton_features, rowvar=False)

    mu2 = np.mean(reference_features, axis=0)
    sigma2 = np.cov(reference_features, rowvar=False)

    # Calculate FID
    diff = mu1 - mu2

    # Product of covariances might not be positive definite, so we need to handle this
    covmean_sqrt = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean_sqrt):
        covmean_sqrt = covmean_sqrt.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean_sqrt)

    return float(fid)

def extract_velocity_features(skeleton):
    """
    Extract velocity features from skeleton for FID calculation.

    Args:
        skeleton: Tensor of shape (T, V, C) or (T, V*C)

    Returns:
        Tensor of shape (T-1, V*C) containing velocity features
    """
    # Reshape if needed
    if len(skeleton.shape) == 2:
        T, VC = skeleton.shape
        V = 25  # Assuming 25 joints
        C = VC // V
        skeleton = skeleton.reshape(T, V, C)

    # Need at least 2 frames to calculate velocity
    if skeleton.shape[0] < 2:
        return torch.zeros(1, skeleton.shape[1] * skeleton.shape[2])

    # Calculate velocities
    velocities = skeleton[1:] - skeleton[:-1]  # (T-1, V, C)

    # Flatten to feature vectors
    features = velocities.reshape(velocities.shape[0], -1)  # (T-1, V*C)

    return features

#------------------------------------------------------------------------------
# Evaluation Functions
#------------------------------------------------------------------------------

def eval_skeleton_mixformer(
    paired_test, model, dataset_name,
    ar_model_weights, ri_model_weights, gc_model_weights=None, args=None
):
    """
    Evaluate anonymized skeleton data using MixFormer models for AR, RI, and GC tasks.

    Args:
        paired_test: DataLoader - Test data loader
        model: torch.nn.Module - The anonymizer model (or None for raw)
        dataset_name: str - Name of the dataset (ntu, ntu120, etc.)
        ar_model_weights: str - Path to action recognition model weights
        ri_model_weights: str - Path to re-identification model weights
        gc_model_weights: str - Path to gender classification model weights (optional)
        args: argparse.Namespace - Arguments including model type
    """
    try:
        from eval.preprocess import mixformer_preprocess_single_skeleton
    except ImportError:
        print("Could not import 'mixformer_preprocess_single_skeleton'.")
        def mixformer_preprocess_single_skeleton(x):
            return x

    # Load Action Recognition model
    AR_Model = import_class('model.ske_mixf.Model')
    ar_model = AR_Model(
        num_class=datasets[dataset_name]['num_class'],
        num_point=25,
        num_person=2,
        graph=datasets[dataset_name]['graph']
    )

    # Fix missing buffers in the model before loading weights
    ar_model = fix_missing_buffers(ar_model)
    ar_model = ar_model.to(args.device)

    # Load model weights using the safe_load_model function
    print(f"Loading AR model weights from: {ar_model_weights}")
    ar_state_dict = safe_load_model(ar_model_weights, args.device)

    # Load state dict with strict=False to ignore missing keys
    ar_model.load_state_dict(ar_state_dict, strict=False)
    ar_model.eval()

    # Load Re-identification model
    RI_Model = import_class('model.ske_mixf.Model')
    ri_model = RI_Model(
        num_class=datasets[dataset_name]['num_actor'],  # Use num_actor for RI model
        num_point=25,
        num_person=2,
        graph=datasets[dataset_name]['graph']
    )

    # Fix missing buffers in the model before loading weights
    ri_model = fix_missing_buffers(ri_model)
    ri_model = ri_model.to(args.device)

    # Load model weights using the safe_load_model function
    print(f"Loading RI model weights from: {ri_model_weights}")
    ri_state_dict = safe_load_model(ri_model_weights, args.device)

    # Load state dict with strict=False to ignore missing keys
    ri_model.load_state_dict(ri_state_dict, strict=False)
    ri_model.eval()

    # Load Gender Classification model if provided
    gc_model = None
    if gc_model_weights and os.path.exists(gc_model_weights):
        print(f"Loading Gender Classification model from: {gc_model_weights}")
        GC_Model = import_class('model.ske_mixf.Model')
        gc_model = GC_Model(
            num_class=2,  # Binary classification: Male/Female
            num_point=25,
            num_person=2,
            graph=datasets[dataset_name]['graph']
        )

        # Fix missing buffers in the model before loading weights
        gc_model = fix_missing_buffers(gc_model)
        gc_model = gc_model.to(args.device)

        # Load model weights using the safe_load_model function
        gc_state_dict = safe_load_model(gc_model_weights, args.device)

        # Load state dict with strict=False to ignore missing keys
        gc_model.load_state_dict(gc_state_dict, strict=False)
        gc_model.eval()

    # Initialize evaluation counters
    total_samples = 0
    correct_action = 0
    correct_actor_ret = 0
    correct_actor_orig = 0

    # Gender classification counters
    correct_gender_orig = 0  # Correctly predicted original gender
    correct_gender_ret = 0   # Correctly predicted retargeted gender
    correct_gender_cross = 0  # Correctly predicted original gender in cross-gender cases
    gender_orig_samples = 0
    gender_ret_samples = 0
    gender_cross_samples = 0

    mse_sum = 0.0
    mse_count = 0
    ref_mse_sum = 0.0
    ref_mse_count = 0

    # Initialize advanced metrics
    bone_length_consistency_values = []
    joint_angle_limits_values = []
    temporal_smoothness_values = []
    velocity_consistency_values = []
    foot_contact_consistency_values = []

    # For FID calculation
    all_skeleton_features = []
    all_reference_features = []

    # Function to route batch to appropriate anonymizer
    def anonymize_batch(batch):
        # Get gender_map from args
        gender_map = args.gender_map if hasattr(args, 'gender_map') else None

        if args.model_type == 'raw':
            return get_anonymized_paired_raw(batch, gender_map=gender_map)
        elif args.model_type == 'transformer':
            return get_anonymized_paired_transformer(batch, model, prep_data, gender_map=gender_map)
        else:
            # When using DMR/PMR with Mixformer, we need to trim to 64 frames
            return get_anonymized_paired_dmr_pmr(batch, model, T=args.T, mixformer_mode=True, gender_map=gender_map)

    # Evaluate each batch
    for batch in tqdm(paired_test, desc='Evaluating with MixFormer', mininterval=1.0, miniters=max(1, len(paired_test)//10)):
        anonymized_data = anonymize_batch(batch)

        for item in anonymized_data:
            anonymized_skel = item['skeleton']
            gt_skel = item['gt_skeleton']
            act_label = item['action']
            ret_actor = item['retargeted_actor']
            orig_actor = item['original_actor']

            # Calculate MSE between anonymized and ground truth
            t_out = anonymized_skel.shape[0]
            t_in = gt_skel.shape[0]
            Tm = min(t_out, t_in)
            mse_ = F.mse_loss(anonymized_skel[:Tm], gt_skel[:Tm])
            mse_sum += mse_.item()
            mse_count += 1

            # Calculate MSE between reference skeleton and generated skeleton
            ref_skel = item.get('reference_skeleton', None)
            if ref_skel is not None:
                t_ref = ref_skel.shape[0]
                Tm_ref = min(t_out, t_ref)
                ref_mse_ = F.mse_loss(anonymized_skel[:Tm_ref], ref_skel[:Tm_ref])
                if 'ref_mse_sum' not in locals():
                    ref_mse_sum = 0.0
                    ref_mse_count = 0
                ref_mse_sum += ref_mse_.item()
                ref_mse_count += 1

                # Calculate advanced metrics
                # 1. Bone Length Consistency
                blc = calculate_bone_length_consistency(anonymized_skel, dataset=dataset_name)
                bone_length_consistency_values.append(blc)

                # 2. Joint Angle Limits
                jal = calculate_joint_angle_limits(anonymized_skel, dataset=dataset_name)
                joint_angle_limits_values.append(jal)

                # 3. Temporal Smoothness
                ts = calculate_temporal_smoothness(anonymized_skel)
                temporal_smoothness_values.append(ts)

                # 4. Velocity Consistency
                vc = calculate_velocity_consistency(anonymized_skel, ref_skel)
                velocity_consistency_values.append(vc)

                # 5. Foot Contact Consistency
                fcc = calculate_foot_contact_consistency(anonymized_skel, ref_skel, dataset=dataset_name)
                foot_contact_consistency_values.append(fcc)

                # 6. Extract features for FID calculation
                skel_features = extract_velocity_features(anonymized_skel)
                ref_features = extract_velocity_features(ref_skel)
                all_skeleton_features.append(skel_features)
                all_reference_features.append(ref_features)

            # Preprocess for MixFormer
            skel_np = anonymized_skel.detach().cpu().numpy()
            prepped = mixformer_preprocess_single_skeleton(skel_np)

            # Add second empty person for MixFormer model
            zeros_ = np.zeros_like(prepped)
            prepped_2p = np.concatenate([prepped, zeros_], axis=3)
            ar_input = torch.tensor(prepped_2p, dtype=torch.float32).unsqueeze(0).to(args.device)

            # Run action recognition model
            with torch.no_grad():
                ar_out = ar_model(ar_input)
                _, ar_pred = torch.max(ar_out, 1)
            correct_action += (ar_pred.item() == act_label)

            # Run re-identification model
            with torch.no_grad():
                ri_out = ri_model(ar_input)
                _, ri_pred = torch.max(ri_out, 1)
            if ri_pred.item() == ret_actor:
                correct_actor_ret += 1
            if ri_pred.item() == orig_actor:
                correct_actor_orig += 1

            # Run gender classification model if available
            if gc_model is not None:
                with torch.no_grad():
                    gc_out = gc_model(ar_input)
                    _, gc_pred = torch.max(gc_out, 1)

                # Get original gender from the item
                if 'gender' in item:
                    orig_gender = item['gender']
                    gender_orig_samples += 1
                    if gc_pred.item() == orig_gender:
                        correct_gender_orig += 1

                # Get retargeted gender from the gender map
                gender_map = args.gender_map if hasattr(args, 'gender_map') else None
                if gender_map and item['retargeted_actor'] + 1 in gender_map:
                    ret_gender = 1 if gender_map[item['retargeted_actor'] + 1] == 'M' else 0
                    gender_ret_samples += 1
                    if gc_pred.item() == ret_gender:
                        correct_gender_ret += 1

                    # Cross-gender transfer evaluation (when original and retargeted genders differ)
                    if 'gender' in item and orig_gender != ret_gender:
                        gender_cross_samples += 1
                        if gc_pred.item() == orig_gender:
                            correct_gender_cross += 1

            total_samples += 1

    # Calculate metrics
    act_acc = 100.0 * correct_action / total_samples if total_samples else 0
    ret_acc = 100.0 * correct_actor_ret / total_samples if total_samples else 0
    orig_acc = 100.0 * correct_actor_orig / total_samples if total_samples else 0
    neither = total_samples - correct_actor_ret - correct_actor_orig
    neither_acc = 100.0 * neither / total_samples if total_samples else 0
    avg_mse = mse_sum / mse_count if mse_count else 0
    avg_ref_mse = ref_mse_sum / ref_mse_count if ref_mse_count else 0

    # Calculate gender classification metrics
    gc_orig_acc = 100.0 * correct_gender_orig / gender_orig_samples if gender_orig_samples else 0
    gc_ret_acc = 100.0 * correct_gender_ret / gender_ret_samples if gender_ret_samples else 0
    gc_cross_acc = 100.0 * correct_gender_cross / gender_cross_samples if gender_cross_samples else 0

    # Calculate advanced metrics averages
    avg_bone_length_consistency = sum(bone_length_consistency_values) / len(bone_length_consistency_values) if bone_length_consistency_values else 0
    avg_joint_angle_limits = sum(joint_angle_limits_values) / len(joint_angle_limits_values) if joint_angle_limits_values else 0
    avg_temporal_smoothness = sum(temporal_smoothness_values) / len(temporal_smoothness_values) if temporal_smoothness_values else 0
    avg_velocity_consistency = sum(velocity_consistency_values) / len(velocity_consistency_values) if velocity_consistency_values else 0
    avg_foot_contact_consistency = sum(foot_contact_consistency_values) / len(foot_contact_consistency_values) if foot_contact_consistency_values else 0

    # Calculate FID if we have enough samples
    fid_score = 0.0
    if len(all_skeleton_features) > 10 and len(all_reference_features) > 10:
        # Concatenate all features
        try:
            skeleton_features = torch.cat(all_skeleton_features, dim=0)
            reference_features = torch.cat(all_reference_features, dim=0)
            fid_score = calculate_fid_for_skeletons(skeleton_features, reference_features)
        except Exception as e:
            print(f"Error calculating FID: {e}")
            fid_score = -1.0

    # Print results
    print(f'[MixFormer] Action Recognition Accuracy: {act_acc:.2f}%')
    print('[MixFormer] Re-identification Results:')
    print(f'  Retargeted Actor: {correct_actor_ret}/{total_samples} ({ret_acc:.2f}%)')
    print(f'  Original  Actor: {correct_actor_orig}/{total_samples} ({orig_acc:.2f}%)')
    print(f'  Neither:         {neither}/{total_samples} ({neither_acc:.2f}%)')

    # Print gender classification metrics if available
    if gender_orig_samples > 0 or gender_ret_samples > 0 or gender_cross_samples > 0:
        print(f'\n[MixFormer] Gender Classification Results:')
        if gender_orig_samples > 0:
            print(f'  Original Gender: {correct_gender_orig}/{gender_orig_samples} ({gc_orig_acc:.2f}%)')
        if gender_ret_samples > 0:
            print(f'  Retargeted Gender: {correct_gender_ret}/{gender_ret_samples} ({gc_ret_acc:.2f}%)')
        if gender_cross_samples > 0:
            print(f'  Cross-Gender Transfer: {correct_gender_cross}/{gender_cross_samples} ({gc_cross_acc:.2f}%)')

    print(f'[MixFormer] Average MSE (GT): {avg_mse:.6f}')
    print(f'[MixFormer] Average MSE (Reference): {avg_ref_mse:.6f}')

    # Print advanced metrics
    print('\n[MixFormer] Advanced Utility Metrics:')
    print(f'  Bone Length Consistency: {avg_bone_length_consistency:.6f} (lower is better)')
    print(f'  Joint Angle Limits: {avg_joint_angle_limits:.2f}% (higher is better)')
    print(f'  Temporal Smoothness: {avg_temporal_smoothness:.6f} (lower is better)')
    print(f'  Velocity Consistency: {avg_velocity_consistency:.6f} (higher is better)')
    print(f'  Foot Contact Consistency: {avg_foot_contact_consistency:.2f}% (higher is better)')
    if fid_score >= 0:
        print(f'  FID Score: {fid_score:.6f} (lower is better)')

    # Return metrics dictionary for potential saving
    metrics = {
        'action_recognition_accuracy': act_acc,
        'reidentification_accuracy': ret_acc,
        'original_actor_accuracy': orig_acc,
        'mse_gt': avg_mse,
        'mse_reference': avg_ref_mse,
        'bone_length_consistency': avg_bone_length_consistency,
        'joint_angle_limits': avg_joint_angle_limits,
        'temporal_smoothness': avg_temporal_smoothness,
        'velocity_consistency': avg_velocity_consistency,
        'foot_contact_consistency': avg_foot_contact_consistency,
        'fid_score': fid_score if fid_score >= 0 else None
    }

    # Add gender classification metrics to the results
    if gender_orig_samples > 0:
        metrics['gender_classification_orig'] = gc_orig_acc
    if gender_ret_samples > 0:
        metrics['gender_classification_ret'] = gc_ret_acc
    if gender_cross_samples > 0:
        metrics['gender_classification_cross'] = gc_cross_acc

    return metrics

def eval_skeleton_sgn(paired_test, anonymizer_model, dataset_name, ar_model_weights, ri_model_weights, gc_model_weights=None, args=None):
    """
    Evaluate anonymized skeleton data using SGN models for AR, RI, and GC tasks.

    Args:
        paired_test: DataLoader - Test data loader
        anonymizer_model: torch.nn.Module - The anonymizer model (or None for raw)
        dataset_name: str - Name of the dataset (ntu, ntu120, etc.)
        ar_model_weights: str - Path to action recognition model weights
        ri_model_weights: str - Path to re-identification model weights
        gc_model_weights: str - Path to gender classification model weights (optional)
        args: argparse.Namespace - Arguments including model type
    """
    from model.sgn import SGN
    from data import get_num_classes

    # Debug logging removed

    # Initialize models with correct number of classes for dataset
    num_classes_ar = get_num_classes(dataset_name, 'ar')
    num_classes_ri = get_num_classes(dataset_name, 'ri')

    print(f"Loading Action Recognition model with {num_classes_ar} classes")

    # Create models and ensure training=False (important for consistent behavior)
    sgn_ar = SGN(num_classes=num_classes_ar, dataset=dataset_name.upper(), seg=20).to(args.device)
    sgn_ar.train(False)  # Explicitly set to evaluation mode

    print(f"Loading Re-identification model with {num_classes_ri} classes")
    sgn_ri = SGN(num_classes=num_classes_ri, dataset=dataset_name.upper(), seg=20).to(args.device)
    sgn_ri.train(False)  # Explicitly set to evaluation mode

    # Load Gender Classification model if provided
    sgn_gc = None
    if gc_model_weights and os.path.exists(gc_model_weights):
        print(f"Loading Gender Classification model from: {gc_model_weights}")
        sgn_gc = SGN(num_classes=2, dataset=dataset_name.upper(), seg=20).to(args.device)  # Binary classification
        sgn_gc.train(False)  # Explicitly set to evaluation mode

        # Fix missing buffers if any
        sgn_gc = fix_missing_buffers(sgn_gc)

        # Load model weights
        print(f"Loading GC model weights from: {gc_model_weights}")
        gc_state_dict = safe_load_model(gc_model_weights, args.device)
        sgn_gc.load_state_dict(gc_state_dict, strict=False)
        sgn_gc.eval()

    # Fix missing buffers if any
    sgn_ar = fix_missing_buffers(sgn_ar)
    sgn_ri = fix_missing_buffers(sgn_ri)

    # Load model weights with proper handling of state_dict
    print(f"Loading AR model weights from: {ar_model_weights}")
    ar_state_dict = safe_load_model(ar_model_weights, args.device)
    sgn_ar.load_state_dict(ar_state_dict, strict=False)

    print(f"Loading RI model weights from: {ri_model_weights}")
    ri_state_dict = safe_load_model(ri_model_weights, args.device)
    sgn_ri.load_state_dict(ri_state_dict, strict=False)

    # Ensure models are in evaluation mode
    sgn_ar.eval()
    sgn_ri.eval()

    # Function to process a sample using SGN's proper test-time evaluation method
    def process_sample_sgn_style(model, crops):
        """
        Process sample using SGN's test-time evaluation method.

        Args:
            model: The SGN model
            crops: Tensor of shape (5, seg, 75) containing the 5 test-time crops

        Returns:
            The predicted class index (zero-indexed)
        """
        # Make sure crops is a tensor
        if isinstance(crops, np.ndarray):
            crops = torch.from_numpy(crops).float()

        # Move to device if needed
        if next(model.parameters()).is_cuda and not crops.is_cuda:
            crops = crops.cuda()

        # Run the model on all 5 crops
        outputs = model(crops)

        # Reshape to get outputs in the right format
        if outputs.dim() == 2 and outputs.size(0) == 5:
            # Handle case where model returns (5, num_classes)
            outputs = outputs.unsqueeze(0)  # Add batch dimension

        # Average over the 5 crops, as done in SGN test_snippet
        outputs = outputs.mean(1)

        # Get the prediction
        _, pred = torch.max(outputs, 1)

        return pred.item()

    # Process data through anonymizer
    anonymized_data = []
    for batch in tqdm(paired_test, desc='Processing through anonymizer', mininterval=1.0, miniters=max(1, len(paired_test)//10)):
        # Get gender_map from args
        gender_map = args.gender_map if hasattr(args, 'gender_map') else None

        # Get anonymized skeletons based on model type
        if args.model_type == 'raw':
            batch_data = get_anonymized_paired_raw(batch, gender_map=gender_map)
        elif args.model_type == 'transformer':
            batch_data = get_anonymized_paired_transformer(batch, anonymizer_model, prep_data, gender_map=gender_map)
        else:
            # When using DMR/PMR with Mixformer, we need to trim to 64 frames
            mixformer_mode = args.eval_model == 'mixformer'
            batch_data = get_anonymized_paired_dmr_pmr(batch, anonymizer_model, T=args.T, mixformer_mode=mixformer_mode, gender_map=gender_map)
        anonymized_data.extend(batch_data)

    # Evaluation variables
    test_size = len(anonymized_data)
    ar_correct = 0
    ar_predicted_dummy = 0  # Count when model predicts the "dummy" action
    ri_ret_correct = 0
    ri_orig_correct = 0

    # Gender classification counters
    correct_gender_orig = 0  # Correctly predicted original gender
    correct_gender_ret = 0   # Correctly predicted retargeted gender
    correct_gender_cross = 0  # Correctly predicted original gender in cross-gender cases
    gender_orig_samples = 0
    gender_ret_samples = 0
    gender_cross_samples = 0

    mse_values = []
    ref_mse_values = []  # MSE between reference and generated skeleton

    all_ar_preds = []
    all_ar_labels = []
    all_ri_preds = []
    all_ri_labels = []
    all_gc_preds = []
    all_gc_labels = []

    # Advanced metrics
    bone_length_consistency_values = []
    joint_angle_limits_values = []
    temporal_smoothness_values = []
    velocity_consistency_values = []
    foot_contact_consistency_values = []

    # For FID calculation
    all_skeleton_features = []
    all_reference_features = []

    # Debug info collection removed

    # Organize items by their batch and within-batch index to track dummy actions
    # This helps us know which items are paired together
    action_pairs = {}  # Maps (batch_idx, item_idx//2) -> [action1, action2]

    # First pass to organize the action pairs
    for item_idx, item in enumerate(anonymized_data):
        batch_idx = item_idx // 4  # Each batch produces 4 outputs
        pair_idx = (item_idx % 4) // 2  # 0 for first pair (0,1), 1 for second pair (2,3)
        group_key = (batch_idx, pair_idx)

        if group_key not in action_pairs:
            action_pairs[group_key] = [None, None]

        # Store the action for this item
        position = item_idx % 2  # 0 for items 0,2 and 1 for items 1,3
        action_pairs[group_key][position] = item['action']

    # Process each sample
    for item_idx, item in enumerate(tqdm(anonymized_data, desc="Evaluating SGN", mininterval=1.0, miniters=max(1, len(anonymized_data)//10))):
        # Extract data from item
        skel = item['skeleton']
        gt_skel = item['gt_skeleton']
        action_label = item['action']  # Already 0-indexed
        ret_actor = item['retargeted_actor']
        orig_actor = item['original_actor']

        # Find the "dummy" action for this sample
        batch_idx = item_idx // 4
        pair_idx = (item_idx % 4) // 2
        group_key = (batch_idx, pair_idx)

        # Get the paired actions
        if group_key in action_pairs:
            paired_actions = action_pairs[group_key]

            if args.same_action:
                # In same-action mode, the dummy action is the same as the original
                dummy_action = action_label
            else:
                # In cross-action mode, the dummy is the other action in the pair
                dummy_action = paired_actions[1] if action_label == paired_actions[0] else paired_actions[0]
        else:
            dummy_action = None

        # Calculate MSE between generated and ground truth
        t_out = skel.shape[0]
        t_in = gt_skel.shape[0]
        Tm = min(t_out, t_in)
        mse_ = F.mse_loss(skel[:Tm], gt_skel[:Tm])
        mse_values.append(mse_.item())

        # Calculate MSE between reference and generated skeleton
        ref_skel = item.get('reference_skeleton', None)
        if ref_skel is not None:
            t_ref = ref_skel.shape[0]
            Tm_ref = min(t_out, t_ref)
            ref_mse_ = F.mse_loss(skel[:Tm_ref], ref_skel[:Tm_ref])
            ref_mse_values.append(ref_mse_.item())

            # Calculate advanced metrics
            # 1. Bone Length Consistency
            blc = calculate_bone_length_consistency(skel, dataset=dataset_name)
            bone_length_consistency_values.append(blc)

            # 2. Joint Angle Limits
            jal = calculate_joint_angle_limits(skel, dataset=dataset_name)
            joint_angle_limits_values.append(jal)

            # 3. Temporal Smoothness
            ts = calculate_temporal_smoothness(skel)
            temporal_smoothness_values.append(ts)

            # 4. Velocity Consistency
            vc = calculate_velocity_consistency(skel, ref_skel)
            velocity_consistency_values.append(vc)

            # 5. Foot Contact Consistency
            fcc = calculate_foot_contact_consistency(skel, ref_skel, dataset=dataset_name)
            foot_contact_consistency_values.append(fcc)

            # 6. Extract features for FID calculation
            skel_features = extract_velocity_features(skel)
            ref_features = extract_velocity_features(ref_skel)
            all_skeleton_features.append(skel_features)
            all_reference_features.append(ref_features)

        try:
            # Convert to numpy for preprocessing - fix data types
            skel_np = skel.cpu().numpy().astype(np.float32)

            # Handle NaN values which can cause issues
            skel_np = np.nan_to_num(skel_np)

            # Create preprocessed crops for testing using the same function as used during training
            processed_crops = sgn_preprocess_single_skeleton(skel_np, seg=20, dataset=dataset_name.upper())

            # Run action recognition model
            ar_pred = process_sample_sgn_style(sgn_ar, processed_crops)

            # Run re-identification model
            ri_pred = process_sample_sgn_style(sgn_ri, processed_crops)

            # Run gender classification model if available
            if sgn_gc is not None:
                gc_pred = process_sample_sgn_style(sgn_gc, processed_crops)
                # Get original gender from the item
                if 'gender' in item:
                    orig_gender = item['gender']
                    gender_orig_samples += 1
                    if gc_pred == orig_gender:
                        correct_gender_orig += 1
                    all_gc_preds.append(gc_pred)
                    all_gc_labels.append(orig_gender)

                # Get retargeted gender from the gender map
                gender_map = args.gender_map if hasattr(args, 'gender_map') else None
                if gender_map and item['retargeted_actor'] + 1 in gender_map:
                    ret_gender = 1 if gender_map[item['retargeted_actor'] + 1] == 'M' else 0
                    gender_ret_samples += 1
                    if gc_pred == ret_gender:
                        correct_gender_ret += 1

                    # Cross-gender transfer evaluation (when original and retargeted genders differ)
                    if 'gender' in item and orig_gender != ret_gender:
                        gender_cross_samples += 1
                        if gc_pred == orig_gender:
                            correct_gender_cross += 1

            # Store predictions and labels for metrics
            all_ar_preds.append(ar_pred)
            all_ar_labels.append(action_label)
            all_ri_preds.append(ri_pred)
            all_ri_labels.append(ret_actor)

            # Skip collecting detailed prediction info to reduce memory usage

            # Skip error tracking for debugging

            # Check correctness
            if ar_pred == action_label:
                ar_correct += 1
            # Check if the predicted action is the dummy action (paired action)
            elif dummy_action is not None and ar_pred == dummy_action:
                ar_predicted_dummy += 1

            if ri_pred == ret_actor:
                ri_ret_correct += 1
            elif ri_pred == orig_actor:
                ri_orig_correct += 1

        except Exception as e:
            # Silently continue on errors
            continue

    # Detailed analysis removed to reduce debug output

    # Calculate metrics
    ar_acc = 100 * ar_correct / test_size if test_size > 0 else 0
    ar_dummy_acc = 100 * ar_predicted_dummy / test_size if test_size > 0 else 0
    ar_other_acc = 100 - ar_acc - ar_dummy_acc
    ri_ret_acc = 100 * ri_ret_correct / test_size if test_size > 0 else 0
    ri_orig_acc = 100 * ri_orig_correct / test_size if test_size > 0 else 0
    ri_neither = test_size - ri_ret_correct - ri_orig_correct
    ri_neither_acc = 100 * ri_neither / test_size if test_size > 0 else 0
    avg_mse = sum(mse_values) / len(mse_values) if mse_values else 0
    avg_ref_mse = sum(ref_mse_values) / len(ref_mse_values) if ref_mse_values else 0

    # Calculate gender classification metrics
    gc_orig_acc = 100.0 * correct_gender_orig / gender_orig_samples if gender_orig_samples > 0 else 0
    gc_ret_acc = 100.0 * correct_gender_ret / gender_ret_samples if gender_ret_samples > 0 else 0
    gc_cross_acc = 100.0 * correct_gender_cross / gender_cross_samples if gender_cross_samples > 0 else 0

    # Calculate advanced metrics averages
    avg_bone_length_consistency = sum(bone_length_consistency_values) / len(bone_length_consistency_values) if bone_length_consistency_values else 0
    avg_joint_angle_limits = sum(joint_angle_limits_values) / len(joint_angle_limits_values) if joint_angle_limits_values else 0
    avg_temporal_smoothness = sum(temporal_smoothness_values) / len(temporal_smoothness_values) if temporal_smoothness_values else 0
    avg_velocity_consistency = sum(velocity_consistency_values) / len(velocity_consistency_values) if velocity_consistency_values else 0
    avg_foot_contact_consistency = sum(foot_contact_consistency_values) / len(foot_contact_consistency_values) if foot_contact_consistency_values else 0

    # Calculate FID if we have enough samples
    fid_score = 0.0
    if len(all_skeleton_features) > 10 and len(all_reference_features) > 10:
        # Concatenate all features
        try:
            skeleton_features = torch.cat(all_skeleton_features, dim=0)
            reference_features = torch.cat(all_reference_features, dim=0)
            fid_score = calculate_fid_for_skeletons(skeleton_features, reference_features)
        except Exception as e:
            print(f"Error calculating FID: {e}")
            fid_score = -1.0

    # Calculate F1, precision, recall
    if len(all_ar_preds) > 0:
        ar_f1 = f1_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
        ar_prec = precision_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
        ar_rec = recall_score(all_ar_labels, all_ar_preds, average='macro', zero_division=0)
    else:
        ar_f1 = ar_prec = ar_rec = 0

    # Calculate gender classification metrics if available
    if len(all_gc_preds) > 0:
        gc_f1 = f1_score(all_gc_labels, all_gc_preds, average='macro', zero_division=0)
    else:
        gc_f1 = 0

    # Output results
    print(f"\n[SGN] Action Recognition Results:")
    print(f"  Correct Action: {ar_correct}/{test_size} ({ar_acc:.2f}%)")
    if not args.same_action:
        print(f"  Dummy Action:   {ar_predicted_dummy}/{test_size} ({ar_dummy_acc:.2f}%)")
    print(f"  Other Action:   {test_size-ar_correct-ar_predicted_dummy}/{test_size} ({ar_other_acc:.2f}%)")
    print(f"  F1: {ar_f1*100:.2f}%   Prec: {ar_prec*100:.2f}%   Rec: {ar_rec*100:.2f}%")

    print(f"\n[SGN] Re-identification Results:")
    print(f"  Retargeted Actor: {ri_ret_correct}/{test_size} ({ri_ret_acc:.2f}%)")
    print(f"  Original  Actor: {ri_orig_correct}/{test_size} ({ri_orig_acc:.2f}%)")
    print(f"  Neither:         {ri_neither}/{test_size} ({ri_neither_acc:.2f}%)")
    print(f"  Accuracy: {ri_ret_acc:.2f}%   F1: {f1_score(all_ri_labels, all_ri_preds, average='macro', zero_division=0)*100:.2f}%")

    # Print gender classification metrics if available
    if gender_orig_samples > 0 or gender_ret_samples > 0 or gender_cross_samples > 0:
        print(f"\n[SGN] Gender Classification Results:")
        if gender_orig_samples > 0:
            print(f"  Original Gender: {correct_gender_orig}/{gender_orig_samples} ({gc_orig_acc:.2f}%)")
        if gender_ret_samples > 0:
            print(f"  Retargeted Gender: {correct_gender_ret}/{gender_ret_samples} ({gc_ret_acc:.2f}%)")
        if gender_cross_samples > 0:
            print(f"  Cross-Gender Transfer: {correct_gender_cross}/{gender_cross_samples} ({gc_cross_acc:.2f}%)")
        if all_gc_preds and all_gc_labels:
            gc_f1 = f1_score(all_gc_labels, all_gc_preds, average='macro', zero_division=0)
            print(f"  F1: {gc_f1*100:.2f}%")

    print(f"\n[SGN] Average MSE (GT): {avg_mse:.6f}")
    print(f"[SGN] Average MSE (Reference): {avg_ref_mse:.6f}")

    # Print advanced metrics
    print('\n[SGN] Advanced Utility Metrics:')
    print(f'  Bone Length Consistency: {avg_bone_length_consistency:.6f} (lower is better)')
    print(f'  Joint Angle Limits: {avg_joint_angle_limits:.2f}% (higher is better)')
    print(f'  Temporal Smoothness: {avg_temporal_smoothness:.6f} (lower is better)')
    print(f'  Velocity Consistency: {avg_velocity_consistency:.6f} (higher is better)')
    print(f'  Foot Contact Consistency: {avg_foot_contact_consistency:.2f}% (higher is better)')
    if fid_score >= 0:
        print(f'  FID Score: {fid_score:.6f} (lower is better)')

    # Skip detailed debug output

    # Return metrics dictionary for potential saving
    metrics = {
        'action_recognition_accuracy': ar_acc,
        'reidentification_accuracy': ri_ret_acc,
        'original_actor_accuracy': ri_orig_acc,
        'mse_gt': avg_mse,
        'mse_reference': avg_ref_mse,
        'bone_length_consistency': avg_bone_length_consistency,
        'joint_angle_limits': avg_joint_angle_limits,
        'temporal_smoothness': avg_temporal_smoothness,
        'velocity_consistency': avg_velocity_consistency,
        'foot_contact_consistency': avg_foot_contact_consistency,
        'fid_score': fid_score if fid_score >= 0 else None
    }

    # Add gender classification metrics to the results
    if gender_orig_samples > 0:
        metrics['gender_classification_orig'] = gc_orig_acc
    if gender_ret_samples > 0:
        metrics['gender_classification_ret'] = gc_ret_acc
    if gender_cross_samples > 0:
        metrics['gender_classification_cross'] = gc_cross_acc

    return metrics

#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------

def main():
    """Main function to parse arguments and run evaluation."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    parser = argparse.ArgumentParser(description="Evaluate anonymized skeleton data with AR/RI models.")
    parser.add_argument('--dataset', default='ntu120', choices=['ntu120','ntu','etri'],
                      help='Dataset to evaluate on')
    parser.add_argument('--setting', default='cs', choices=['cs','cv'],
                      help='Cross-subject or cross-view evaluation')
    parser.add_argument('--model_type', default='transformer', choices=['raw','transformer','pmr','dmr'],
                      help="Type of anonymizer model to use")
    parser.add_argument('--transformer_model_path', default='model.pth', type=str,
                      help='Path to transformer model weights')
    parser.add_argument('--ar_model_weights', default='', type=str,
                      help='Path to action recognition model weights')
    parser.add_argument('--ri_model_weights', default='', type=str,
                      help='Path to re-identification model weights')
    parser.add_argument('--gc_model_weights', default='', type=str,
                      help='Path to gender classification model weights')
    parser.add_argument('--eval_model', default='mixformer', choices=['mixformer','sgn'],
                      help='Model to use for evaluation (MixFormer or SGN)')

    # Data / sampling parameters
    parser.add_argument('--batch_size', default=32, type=int,
                      help='Batch size for data loading')
    parser.add_argument('--paired_batch_size', default=8, type=int,
                      help='Batch size for paired data')
    parser.add_argument('--test_samples', default=5000, type=int,
                      help='Number of test samples to use')
    parser.add_argument('--use_cache', action='store_true',
                      help='Use cached paired data if available')
    parser.add_argument('--save_cache', action='store_true',
                      help='Save paired data to cache')
    parser.add_argument('--T', default=64, type=int,
                      help='Number of frames (Transformer=64, PMR/DMR=75)')
    # Add flag for same action mode
    parser.add_argument('--same_action', action='store_true', default=True,
                      help='Set if data contains same action for both actors (default: True)')
    parser.add_argument('--output_dir', type=str, default='',
                      help='Directory to save evaluation results')
    args = parser.parse_args()

    # Configure device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is NOT available. Using CPU.")

    args.device = device
    args.loading_transformer = False  # Flag to control encoder behavior

    print("Arguments:", args)

    # Load data
    X = load_data(args.dataset)

    # Load gender data
    gender_map = load_gender_data(args.dataset)

    # Load paired data, potentially from cache
    paired_path = f'data/{args.dataset}_{args.setting}_paired.pkl'
    if args.use_cache and os.path.exists(paired_path):
        with open(paired_path, 'rb') as f:
            paired_data = pickle.load(f)
            paired_train = paired_data['train']
            paired_test  = paired_data['test']
        print("Loaded paired data from:", paired_path)
    else:
        paired_train, paired_test = get_cross_data(
            X, args.dataset, args.setting,
            batch_size=args.paired_batch_size,
            return_loader=True,
            train_samples=args.batch_size,
            test_samples=args.test_samples
        )
        if args.save_cache:
            with open(paired_path, 'wb') as f:
                pickle.dump({'train': paired_train, 'test': paired_test}, f)
            print("Saved paired data to:", paired_path)

    # Load anonymizer model
    ds = datasets[args.dataset]
    anonymizer_model = None
    if args.model_type != 'raw':
        if args.model_type == 'transformer':
            # Check if default path is provided
            if args.transformer_model_path == 'model.pth':
                # Try to find model in output directory
                output_path = f'output/transformer_{args.dataset}_{args.setting}/{args.dataset.upper()}_transformer_{args.setting}/model_best.pth.tar'
                if os.path.exists(output_path):
                    anonymizer_path = output_path
                    print(f"Loading Transformer model from output directory: {anonymizer_path}")
                else:
                    anonymizer_path = args.transformer_model_path
                    print(f"Using specified Transformer model path: {anonymizer_path}")
            else:
                anonymizer_path = args.transformer_model_path
                print(f"Using specified Transformer model path: {anonymizer_path}")

            # Check if the file exists
            if not os.path.exists(anonymizer_path):
                print(f"ERROR: Transformer model file not found: {anonymizer_path}")
                print(f"Please make sure you have specified the correct path with --transformer_model_path")
                sys.exit(1)
        else:
            # For PMR and DMR models, check trained_models directory
            # Based on the file search, these models are in trained_models/

            # First try the full model path
            full_model_path = f'trained_models/{args.model_type}_{args.dataset}_{args.setting}_final_full.pth'

            # Then try the regular model path
            regular_model_path = f'trained_models/{args.model_type}_{args.dataset}_{args.setting}_final.pth'

            # Try to find the model in one of these locations
            if os.path.exists(full_model_path):
                anonymizer_path = full_model_path
                print(f"Loading {args.model_type.upper()} model from: {anonymizer_path}")
            elif os.path.exists(regular_model_path):
                anonymizer_path = regular_model_path
                print(f"Loading {args.model_type.upper()} model from: {anonymizer_path}")
            else:
                print(f"ERROR: {args.model_type.upper()} model file not found in either location:")
                print(f"  - {full_model_path}")
                print(f"  - {regular_model_path}")
                print(f"Please make sure you have trained the {args.model_type.upper()} model for {args.dataset}_{args.setting} dataset.")
                print(f"You can train it using: bash train_{args.model_type}_{args.dataset}_{args.setting}.bash")
                sys.exit(1)
        anonymizer_model = load_anonymizer(
            args.model_type, anonymizer_path, args.device, args, ds=ds
        )

    # Set default model weights paths if not provided
    if args.ar_model_weights == '':
        if args.eval_model == 'mixformer':
            default_ar_path = f'output/{args.dataset}_mixformer_ar_{args.setting}/{args.dataset.upper()}_mixformer_ar_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_ar_path):
                args.ar_model_weights = default_ar_path
                print(f"Using default AR model weights: {default_ar_path}")
            else:
                args.ar_model_weights = f'eval/mixformer/pretrained/{args.dataset}/{args.setting}_ar.pth'
                print(f"Using fallback AR model weights: {args.ar_model_weights}")
        else:  # sgn
            # SGN models don't have the 'sgn' in the path
            default_ar_path = f'output/{args.dataset}_ar_{args.setting}/{args.dataset.upper()}_ar_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_ar_path):
                args.ar_model_weights = default_ar_path
                print(f"Using default AR model weights: {default_ar_path}")
            else:
                # Check in old directory
                old_ar_path = f'output/old/{args.dataset.upper()}_ar_{args.setting}/model_best.pth.tar'
                if os.path.exists(old_ar_path):
                    args.ar_model_weights = old_ar_path
                    print(f"Using old AR model weights: {old_ar_path}")
                else:
                    args.ar_model_weights = f'eval/sgn/pretrained/{args.dataset}/{args.setting}_ar.pth'
                    print(f"Using fallback AR model weights: {args.ar_model_weights}")

    if args.ri_model_weights == '':
        if args.eval_model == 'mixformer':
            default_ri_path = f'output/{args.dataset}_mixformer_ri_{args.setting}/{args.dataset.upper()}_mixformer_ri_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_ri_path):
                args.ri_model_weights = default_ri_path
                print(f"Using default RI model weights: {default_ri_path}")
            else:
                args.ri_model_weights = f'eval/mixformer/pretrained/{args.dataset}/{args.setting}_ri.pth'
                print(f"Using fallback RI model weights: {args.ri_model_weights}")
        else:  # sgn
            # SGN models don't have the 'sgn' in the path
            default_ri_path = f'output/{args.dataset}_ri_{args.setting}/{args.dataset.upper()}_ri_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_ri_path):
                args.ri_model_weights = default_ri_path
                print(f"Using default RI model weights: {default_ri_path}")
            else:
                # Check in old directory
                old_ri_path = f'output/old/{args.dataset.upper()}_ri_{args.setting}/model_best.pth.tar'
                if os.path.exists(old_ri_path):
                    args.ri_model_weights = old_ri_path
                    print(f"Using old RI model weights: {old_ri_path}")
                else:
                    args.ri_model_weights = f'eval/sgn/pretrained/{args.dataset}/{args.setting}_ri.pth'
                    print(f"Using fallback RI model weights: {args.ri_model_weights}")

    if args.gc_model_weights == '':
        if args.eval_model == 'mixformer':
            default_gc_path = f'output/{args.dataset}_mixformer_gc_{args.setting}/{args.dataset.upper()}_mixformer_gc_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_gc_path):
                args.gc_model_weights = default_gc_path
                print(f"Using default GC model weights: {default_gc_path}")
            else:
                args.gc_model_weights = f'eval/mixformer/pretrained/{args.dataset}/{args.setting}_gc.pth'
                print(f"Using fallback GC model weights: {args.gc_model_weights}")
        else:  # sgn
            # SGN models don't have the 'sgn' in the path
            default_gc_path = f'output/{args.dataset}_gc_{args.setting}/{args.dataset.upper()}_gc_{args.setting}/model_best.pth.tar'
            if os.path.exists(default_gc_path):
                args.gc_model_weights = default_gc_path
                print(f"Using default GC model weights: {default_gc_path}")
            else:
                args.gc_model_weights = f'eval/sgn/pretrained/{args.dataset}/{args.setting}_gc.pth'
                print(f"Using fallback GC model weights: {args.gc_model_weights}")

    # Run evaluation with appropriate model
    if args.eval_model == 'mixformer':
        print("[MAIN] Evaluate with MixFormer AR/RI/GC...")
        # Add gender_map to args for use in evaluation functions
        args.gender_map = gender_map
        metrics = eval_skeleton_mixformer(
            paired_test,
            anonymizer_model,
            dataset_name=args.dataset,
            ar_model_weights=args.ar_model_weights,
            ri_model_weights=args.ri_model_weights,
            gc_model_weights=args.gc_model_weights,
            args=args
        )
    else:
        print("[MAIN] Evaluate with SGN-based approach...")
        # Add gender_map to args for use in evaluation functions
        args.gender_map = gender_map
        metrics = eval_skeleton_sgn(
            paired_test,
            anonymizer_model,
            dataset_name=args.dataset,
            ar_model_weights=args.ar_model_weights,
            ri_model_weights=args.ri_model_weights,
            gc_model_weights=args.gc_model_weights,
            args=args
        )

    # Save metrics to file if output_dir is specified
    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{args.model_type}_metrics.json")
        with open(output_file, 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {output_file}")

if __name__ == '__main__':
    main()