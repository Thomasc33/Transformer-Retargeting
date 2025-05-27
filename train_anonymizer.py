#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import wandb
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import data handling functions
from data import datasets, load_data

# Import models and evaluation utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
from dmr.dmr import DMR
from pmr.pmr import PMR
from eval_loader import AverageMeter, Dataloaders

# Define training stages
training_stages = [
    # Pre-Train Cross to separate embeddings
    {'epochs': 5, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'save': False},
#5
    # Pre-Train AE
    {'epochs': 20, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'save': False},
#20
    # Pre-Train Adversaries (Paired)
    {'epochs': 20, 'paired': True, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'save': False},
#20
    # Pre-Train Adversaries
    {'epochs': 50, 'paired': False, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'save': False},
#50
    # Train AE and adversaries with adversary loss
    {'epochs': 100, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'save': True},
#100
    # Paired Training (Crossing)
    {'epochs': 100, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'save': True},
#100
]

def preprocess_skeleton(skeleton, args):
    """
    Preprocess skeleton data to match the format expected by the models.
    For DMR/PMR, ensures the data is in [B, T, J, C] format where T=75.
    """
    # Get the sequence length parameter from args
    T = args.T

    # Convert to tensor if needed
    if not isinstance(skeleton, torch.Tensor):
        skeleton = torch.tensor(skeleton, dtype=torch.float32)

    # Get batch size
    B = skeleton.shape[0]

    # For DMR/PMR, we need to ensure data is in [B, T, J, C] format
    if len(skeleton.shape) == 2:  # [B, flattened_features]
        # Reshape to [B, T, 75] (75 = 25 joints * 3 channels)
        skeleton = skeleton.reshape(B, -1, 75)

        # Handle sequence length - pad or truncate
        seq_len = skeleton.shape[1]
        if seq_len < T:
            # Repeat last frame for padding (matches evaluation code)
            padding = skeleton[:, -1:].repeat(1, T - seq_len, 1)
            skeleton = torch.cat([skeleton, padding], dim=1)
        elif seq_len > T:
            # Truncate to T frames
            skeleton = skeleton[:, :T]

        # Reshape for DMR/PMR: [B, T, J, C]
        skeleton = skeleton.reshape(B, T, 25, 3)

    elif len(skeleton.shape) == 3:  # [B, seq_len, features]
        # If features dimension is 75, reshape to [B, seq_len, 25, 3]
        if skeleton.shape[2] == 75:
            # Handle sequence length
            seq_len = skeleton.shape[1]
            if seq_len < T:
                padding = skeleton[:, -1:].repeat(1, T - seq_len, 1)
                skeleton = torch.cat([skeleton, padding], dim=1)
            elif seq_len > T:
                skeleton = skeleton[:, :T]

            # Reshape to [B, T, J, C]
            skeleton = skeleton.reshape(B, T, 25, 3)
        else:
            # Handle other possible formats - customize as needed
            raise ValueError(f"Unexpected feature dimension: {skeleton.shape[2]}")

    elif len(skeleton.shape) == 4:  # [B, C, J, T] or [B, T, J, C]
        # Determine format and convert to [B, T, J, C]
        if skeleton.shape[1] == 3 and skeleton.shape[3] > 3:  # Likely [B, C, J, T]
            skeleton = skeleton.permute(0, 3, 2, 1)  # Convert to [B, T, J, C]

        # Handle sequence length
        seq_len = skeleton.shape[1]
        if seq_len < T:
            padding = skeleton[:, -1:].repeat(1, T - seq_len, 1, 1)
            skeleton = torch.cat([skeleton, padding], dim=1)
        elif seq_len > T:
            skeleton = skeleton[:, :T]

    else:
        raise ValueError(f"Unexpected input shape: {skeleton.shape}")

    return skeleton

def train_one_epoch_paired(model, data_loader, optimizer, args, model_type='dmr', stage_params=None, cur_epoch=0):
    """Train model for one epoch using paired data"""

    model = model.cuda()

    # Verify all model parameters are on CUDA (only log once if issues found)
    cuda_issues = False
    for name, param in model.named_parameters():
        if not param.is_cuda:
            cuda_issues = True
            param.data = param.data.cuda()

    if cuda_issues:
        logger.info("Some parameters were moved to CUDA")

    model.set_eval(False)
    losses = AverageMeter()
    metrics = {
        'rec_loss': AverageMeter(),
        'cross_loss': AverageMeter(),
        'end_effector_loss': AverageMeter(),
        'triplet_loss': AverageMeter(),
        'latent_consistency_loss': AverageMeter(),
        'smoothing_loss': AverageMeter(),
    }

    # Add adversarial metrics if using PMR
    if model_type == 'pmr':
        adv_metrics = {
            'privacy_loss': AverageMeter(),
            'privacy_acc_adv': AverageMeter(),
            'privacy_acc_coop': AverageMeter(),
            'utility_loss': AverageMeter(),
            'utility_acc_adv': AverageMeter(),
            'utility_acc_coop': AverageMeter(),
            'discriminator_loss': AverageMeter(),
            'discriminator_acc': AverageMeter(),
            'privacy_loss_adv': AverageMeter(),
            'privacy_loss_coop': AverageMeter(),
            'utility_loss_adv': AverageMeter(),
            'utility_loss_coop': AverageMeter(),
            'priv_training_loss': AverageMeter(),
            'priv_coop_training_loss': AverageMeter(),
            'util_training_loss': AverageMeter(),
            'util_coop_training_loss': AverageMeter(),
            'discriminator_train_loss': AverageMeter(),
            'priv_training_acc': AverageMeter(),
            'priv_coop_training_acc': AverageMeter(),
            'util_training_acc': AverageMeter(),
            'util_coop_training_acc': AverageMeter(),
            'discriminator_train_acc': AverageMeter(),
        }
        metrics.update(adv_metrics)

    # Unpack stage parameters
    train_emb_adv = stage_params.get('train_emb_adv', False)
    train_discrim_adv = stage_params.get('train_discrim_adv', False)
    use_emb_adv = stage_params.get('emb_adv', False)
    use_discrim_adv = stage_params.get('discrim_adv', False)
    use_cross = stage_params.get('cross', True)
    use_ae = stage_params.get('ae', True)
    use_ee = stage_params.get('ee', True)
    use_triplet = stage_params.get('triplet', True)

    # Determine if we should train adversaries this epoch (update frequency)
    train_emb_this_epoch = True
    emb_clf_update_per_epoch_paired = args.emb_clf_update_per_epoch_paired

    if emb_clf_update_per_epoch_paired < 1 and (train_emb_adv or train_discrim_adv):
        if cur_epoch % int(1 / emb_clf_update_per_epoch_paired) != 0:
            train_emb_this_epoch = False

    # Process each batch with reduced tqdm updates
    for batch_idx, (x1, x2) in enumerate(tqdm(data_loader, desc="Training paired",
                                              mininterval=5.0, leave=False)):
        # x1 and x2 contain [data, label] pairs
        # Extract data and labels
        x1_data, x1_label = x1[0].float(), x1[1].long()
        x2_data, x2_label = x2[0].float(), x2[1].long()

        # Create two different combinations:
        # (actor1, action1) -> (actor2, action1)
        # (actor1, action2) -> (actor2, action2)
        batch_size = x1_data.size(0)

        # Randomly select pairs for cross-subject training
        idx_perm = torch.randperm(batch_size)
        x2_data_perm = x2_data[idx_perm]
        x2_label_perm = x2_label[idx_perm]

        x1 = x1_data
        x2 = x2_data_perm
        y1 = x1_data  # Same motion, different actor
        y2 = x2_data_perm  # Same motion, different actor

        # Create actors and actions tensors
        actors = torch.stack([x1_label, x2_label_perm], dim=1)  # Different actors
        actions = torch.stack([x1_label, x1_label], dim=1)  # Same actions

        # Move data to CUDA
        x1, x2 = x1.cuda(), x2.cuda()
        y1, y2 = y1.cuda(), y2.cuda()
        actors, actions = actors.cuda(), actions.cuda()

        # Preprocess data to match the format expected by the model
        x1_rot = preprocess_skeleton(x1, args)
        x2_rot = preprocess_skeleton(x2, args)
        y1_rot = preprocess_skeleton(y1, args)
        y2_rot = preprocess_skeleton(y2, args)

        # Train adversaries first if this stage uses them (PMR only)
        if model_type == 'pmr' and (train_emb_adv or train_discrim_adv) and train_emb_this_epoch:
            # Allow multiple iterations of adversary training per batch
            iterations = max(1, int(args.emb_clf_update_per_epoch_paired)) if args.emb_clf_update_per_epoch_paired > 1 else 1

            # Ensure batch size is consistent
            if batch_size < 2:
                logger.warning(f"Batch size {batch_size} is too small for paired adversarial training, skipping")
                continue

            # Only log once per epoch instead of per batch
            if batch_idx == 0:
                logger.info(f"Running {iterations} iterations of paired adversarial training")

            for iter_idx in range(iterations):
                try:
                    priv_loss, priv_coop_loss, util_loss, util_coop_loss, discrim_loss, priv_acc, util_acc, priv_coop_acc, util_coop_acc, discrim_acc = model.train_adv_paired(
                        x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot,
                        actors, actions,
                        train_emb=train_emb_adv,
                        train_discrim=train_discrim_adv
                    )

                    # Update metrics for adversarial training
                    metrics['priv_training_loss'].update(priv_loss)
                    metrics['priv_coop_training_loss'].update(priv_coop_loss)
                    metrics['util_training_loss'].update(util_loss)
                    metrics['util_coop_training_loss'].update(util_coop_loss)
                    metrics['discriminator_train_loss'].update(discrim_loss)
                    metrics['priv_training_acc'].update(priv_acc)
                    metrics['priv_coop_training_acc'].update(priv_coop_acc)
                    metrics['util_training_acc'].update(util_acc)
                    metrics['util_coop_training_acc'].update(util_coop_acc)
                    metrics['discriminator_train_acc'].update(discrim_acc)

                except RuntimeError as e:
                    logger.error(f"Error during adversarial training: {e}")
                    continue

        # Skip autoencoder training if not needed for this stage
        if not use_ae and not use_cross:
            continue

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with full loss computation
        try:
            loss, x1_hat, x2_hat, y1_hat, y2_hat, batch_losses = model.loss_paired(
                x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot,
                actors, actions,
                cross=use_cross,
                reconstruction=use_ae,
                emb_adv=use_emb_adv and model_type=='pmr',
                discrim_adv=use_discrim_adv and model_type=='pmr',
                verbose=False
            )
        except RuntimeError as e:
            logger.error(f"Error during loss computation: {e}")
            # Skip this batch
            continue

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item())
        for k, v in batch_losses.items():
            if k in metrics:
                metrics[k].update(v)

    # Prepare results
    results = {'loss': losses.avg}
    for k, meter in metrics.items():
        results[k] = meter.avg

    return results

def train_one_epoch_unpaired(model, data_loader, optimizer, args, model_type='dmr', stage_params=None, cur_epoch=0):
    """Train model for one epoch using unpaired data"""
    # Force model to CUDA
    model = model.cuda()

    model.set_eval(False)
    losses = AverageMeter()
    metrics = {
        'rec_loss': AverageMeter(),
        'end_effector_loss': AverageMeter(),
        'triplet_loss': AverageMeter(),
        'smoothing_loss': AverageMeter(),
    }

    # Add adversarial metrics if using PMR
    if model_type == 'pmr':
        adv_metrics = {
            'privacy_loss': AverageMeter(),
            'privacy_acc_adv': AverageMeter(),
            'privacy_acc_coop': AverageMeter(),
            'utility_loss': AverageMeter(),
            'utility_acc_adv': AverageMeter(),
            'utility_acc_coop': AverageMeter(),
            'discriminator_loss': AverageMeter(),
            'discriminator_acc': AverageMeter(),
            'privacy_loss_adv': AverageMeter(),
            'privacy_loss_coop': AverageMeter(),
            'utility_loss_adv': AverageMeter(),
            'utility_loss_coop': AverageMeter(),
            'priv_training_loss': AverageMeter(),
            'priv_coop_training_loss': AverageMeter(),
            'util_training_loss': AverageMeter(),
            'util_coop_training_loss': AverageMeter(),
            'discriminator_train_loss': AverageMeter(),
            'priv_training_acc': AverageMeter(),
            'priv_coop_training_acc': AverageMeter(),
            'util_training_acc': AverageMeter(),
            'util_coop_training_acc': AverageMeter(),
            'discriminator_train_acc': AverageMeter(),
        }
        metrics.update(adv_metrics)

    # Unpack stage parameters
    train_emb_adv = stage_params.get('train_emb_adv', False)
    train_discrim_adv = stage_params.get('train_discrim_adv', False)
    use_emb_adv = stage_params.get('emb_adv', False)
    use_discrim_adv = stage_params.get('discrim_adv', False)
    use_ae = stage_params.get('ae', True)
    use_ee = stage_params.get('ee', True)
    use_triplet = stage_params.get('triplet', True)

    # Determine if we should train adversaries this epoch
    train_emb_this_epoch = True
    emb_clf_update_per_epoch_unpaired = args.emb_clf_update_per_epoch_unpaired

    if emb_clf_update_per_epoch_unpaired < 1 and (train_emb_adv or train_discrim_adv):
        if cur_epoch % int(1 / emb_clf_update_per_epoch_unpaired) != 0:
            train_emb_this_epoch = False

    # Process each batch with reduced tqdm updates
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training unpaired",
                                         mininterval=5.0, leave=False)):
        # Extract features and labels - data is a list with [tensor_data, tensor_label]
        skeleton = data[0].float()
        label = data[1].long()

        # For action recognition task, the label is the action
        action = label + 1  # Convert 0-indexed to 1-indexed

        # For actor recognition in NTU, the actor is derived from features of the data
        # In this simplified implementation, we use a deterministic mapping based on batch_idx
        actor_ids = torch.full((skeleton.size(0),), (batch_idx % 40) + 1, device=skeleton.device)

        # Move data to CUDA
        skeleton = skeleton.cuda()
        actor = actor_ids.cuda()
        action = action.cuda()

        # IMPORTANT: Skip if batch size is too small
        if skeleton.size(0) < 2 and model_type == 'pmr' and (train_emb_adv or train_discrim_adv):
            continue

        # Preprocess data to match the format expected by the model
        skeleton_rot = preprocess_skeleton(skeleton, args)

        # Train adversaries first if this stage uses them (PMR only)
        if model_type == 'pmr' and (train_emb_adv or train_discrim_adv) and train_emb_this_epoch:
            # Allow multiple iterations of adversary training per batch
            iterations = max(1, int(args.emb_clf_update_per_epoch_unpaired)) if args.emb_clf_update_per_epoch_unpaired > 1 else 1

            # Log only once per epoch instead of per batch
            if batch_idx == 0:
                logger.info(f"Running {iterations} iterations of adversarial training")

            for iter_idx in range(iterations):
                try:
                    priv_loss, priv_coop_loss, util_loss, util_coop_loss, discrim_loss, priv_acc, util_acc, priv_coop_acc, util_coop_acc, discrim_acc = model.train_adv_unpaired(
                        skeleton, skeleton_rot, actor, action,
                        train_emb=train_emb_adv,
                        train_discrim=train_discrim_adv
                    )

                    # Update metrics for adversarial training
                    metrics['priv_training_loss'].update(priv_loss)
                    metrics['priv_coop_training_loss'].update(priv_coop_loss)
                    metrics['util_training_loss'].update(util_loss)
                    metrics['util_coop_training_loss'].update(util_coop_loss)
                    metrics['discriminator_train_loss'].update(discrim_loss)
                    metrics['priv_training_acc'].update(priv_acc)
                    metrics['priv_coop_training_acc'].update(priv_coop_acc)
                    metrics['util_training_acc'].update(util_acc)
                    metrics['util_coop_training_acc'].update(util_coop_acc)
                    metrics['discriminator_train_acc'].update(discrim_acc)

                except RuntimeError as e:
                    logger.error(f"Error during training: {e}")
                    # If this is a device error, try to recover by ensuring everything is on CUDA
                    if "device" in str(e).lower() or "cuda" in str(e).lower():
                        logger.warning("Device error detected. Attempting to move all tensors to CUDA.")
                        # Just break and try the next batch
                        break
                    continue

        # Skip autoencoder training if not needed for this stage
        if not use_ae:
            continue

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with loss computation
        try:
            loss, skeleton_hat, batch_losses = model.loss_unpaired(
                skeleton, skeleton_rot, actor, action,
                reconstruction=use_ae,
                emb_adv=use_emb_adv and model_type=='pmr',
                discrim_adv=use_discrim_adv and model_type=='pmr',
                ee=use_ee,
                triplet=use_triplet,
                verbose=False
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            losses.update(loss.item())
            for k, v in batch_losses.items():
                if k in metrics:
                    metrics[k].update(v)

        except RuntimeError as e:
            logger.error(f"Error during forward/backward pass: {e}")
            continue

    # Prepare results
    results = {'loss': losses.avg}
    for k, meter in metrics.items():
        results[k] = meter.avg

    return results

def validate(model, data_loader, args, model_type='dmr', paired=False, stage_params=None):
    """Validate model on validation data"""
    # Force model to CUDA
    model = model.cuda()
    model.set_eval()
    losses = AverageMeter()
    metrics = {
        'rec_loss': AverageMeter(),
        'cross_loss': AverageMeter() if paired else None,
        'end_effector_loss': AverageMeter(),
        'triplet_loss': AverageMeter(),
        'latent_consistency_loss': AverageMeter() if paired else None,
        'smoothing_loss': AverageMeter(),
    }

    # Filter out None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    # Add adversarial metrics if using PMR
    if model_type == 'pmr':
        adv_metrics = {
            'privacy_loss': AverageMeter(),
            'privacy_acc_adv': AverageMeter(),
            'privacy_acc_coop': AverageMeter(),
            'utility_loss': AverageMeter(),
            'utility_acc_adv': AverageMeter(),
            'utility_acc_coop': AverageMeter(),
            'discriminator_loss': AverageMeter(),
            'discriminator_acc': AverageMeter(),
            'privacy_loss_adv': AverageMeter(),
            'privacy_loss_coop': AverageMeter(),
            'utility_loss_adv': AverageMeter(),
            'utility_loss_coop': AverageMeter(),
        }
        metrics.update(adv_metrics)

    # Unpack stage parameters
    use_emb_adv = stage_params.get('emb_adv', False)
    use_discrim_adv = stage_params.get('discrim_adv', False)
    use_cross = stage_params.get('cross', True) if paired else False
    use_ae = stage_params.get('ae', True)
    use_ee = stage_params.get('ee', True)
    use_triplet = stage_params.get('triplet', True)

    with torch.no_grad():
        if paired:
            # Paired validation with reduced tqdm updates
            for batch_idx, (x1, x2) in enumerate(tqdm(data_loader, desc="Validating",
                                                     mininterval=10.0, leave=False)):
                # Extract data and labels
                x1_data, x1_label = x1[0].float(), x1[1].long()
                x2_data, x2_label = x2[0].float(), x2[1].long()

                batch_size = x1_data.size(0)

                # Use same pairing strategy as in training
                idx_perm = torch.randperm(batch_size)
                x2_data_perm = x2_data[idx_perm]
                x2_label_perm = x2_label[idx_perm]

                x1 = x1_data
                x2 = x2_data_perm
                y1 = x1_data
                y2 = x2_data_perm

                # Create actors and actions tensors for each sample
                actors = torch.stack([x1_label, x2_label_perm], dim=1)
                actions = torch.stack([x1_label, x1_label], dim=1)

                # Move data to CUDA
                x1, x2 = x1.cuda(), x2.cuda()
                y1, y2 = y1.cuda(), y2.cuda()
                actors, actions = actors.cuda(), actions.cuda()

                # Preprocess data to match the format expected by the model
                x1_rot = preprocess_skeleton(x1, args)
                x2_rot = preprocess_skeleton(x2, args)
                y1_rot = preprocess_skeleton(y1, args)
                y2_rot = preprocess_skeleton(y2, args)

                # Forward pass
                try:
                    loss, x1_hat, x2_hat, y1_hat, y2_hat, batch_losses = model.loss_paired(
                        x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot,
                        actors, actions,
                        cross=use_cross,
                        reconstruction=use_ae,
                        emb_adv=use_emb_adv and model_type=='pmr',
                        discrim_adv=use_discrim_adv and model_type=='pmr',
                        verbose=False
                    )
                except RuntimeError as e:
                    logger.error(f"Error during validation loss computation: {e}")
                    # Skip this batch
                    continue

                # Update metrics
                losses.update(loss.item())
                for k, v in batch_losses.items():
                    if k in metrics:
                        metrics[k].update(v)
        else:
            # Unpaired validation with reduced tqdm updates
            for batch_idx, data in enumerate(tqdm(data_loader, desc="Validating",
                                                 mininterval=10.0, leave=False)):
                # Extract features and labels
                skeleton = data[0].float()
                label = data[1].long()

                # For action recognition task, the label is the action
                action = label + 1  # Convert 0-indexed to 1-indexed

                # For actor recognition, create deterministic actor IDs
                actor_ids = torch.full((skeleton.size(0),), (batch_idx % 40) + 1, device=skeleton.device)

                # Move data to CUDA
                skeleton = skeleton.cuda()
                actor = actor_ids.cuda()
                action = action.cuda()

                # Preprocess data to match the format expected by the model
                skeleton_rot = preprocess_skeleton(skeleton, args)

                # Forward pass
                try:
                    loss, skeleton_hat, batch_losses = model.loss_unpaired(
                        skeleton, skeleton_rot, actor, action,
                        reconstruction=use_ae,
                        emb_adv=use_emb_adv and model_type=='pmr',
                        discrim_adv=use_discrim_adv and model_type=='pmr',
                        ee=use_ee,
                        triplet=use_triplet,
                        verbose=False
                    )
                except RuntimeError as e:
                    logger.error(f"Error during unpaired validation loss computation: {e}")
                    # Skip this batch
                    continue

                # Update metrics
                losses.update(loss.item())
                for k, v in batch_losses.items():
                    if k in metrics:
                        metrics[k].update(v)

    # Prepare results
    results = {'val_loss': losses.avg}
    for k, meter in metrics.items():
        results['val_' + k] = meter.avg

    return results

def log_metrics_to_wandb(results, val_results=None, stage_info=None, epoch=None, train_emb_this_epoch=True):
    """Log metrics to wandb with proper formatting"""
    metrics_to_log = {}

    # Add stage info to metrics if provided
    if stage_info:
        metrics_to_log.update({
            "stage": stage_info["stage_num"],
            "stage_epoch": stage_info["stage_epoch"],
            "paired": stage_info["paired"]
        })

    if epoch is not None:
        metrics_to_log["epoch"] = epoch

    # Add main training metrics
    metrics_to_log["train/loss"] = results["loss"]

    # Log training component losses
    for key in ['rec_loss', 'cross_loss', 'end_effector_loss', 'triplet_loss',
                'latent_consistency_loss', 'smoothing_loss']:
        if key in results:
            metrics_to_log[f"train/{key}"] = results[key]

    # Log adversarial metrics if present
    adv_prefixes = {
        "privacy": "priv",
        "utility": "util",
        "discriminator": "discrim"
    }

    for prefix, short in adv_prefixes.items():
        # Main adversarial losses
        if f"{prefix}_loss" in results:
            metrics_to_log[f"train/adv/{prefix}_loss"] = results[f"{prefix}_loss"]

        # Adversarial accuracy metrics
        if f"{prefix}_acc_adv" in results:
            metrics_to_log[f"train/adv/{prefix}_acc_adv"] = results[f"{prefix}_acc_adv"]
        if f"{prefix}_acc_coop" in results:
            metrics_to_log[f"train/adv/{prefix}_acc_coop"] = results[f"{prefix}_acc_coop"]
        if f"{prefix}_acc" in results:
            metrics_to_log[f"train/adv/{prefix}_acc"] = results[f"{prefix}_acc"]

        # Dynamic vs static losses
        if f"{prefix}_loss_adv" in results:
            metrics_to_log[f"train/adv/{prefix}_loss_adv"] = results[f"{prefix}_loss_adv"]
        if f"{prefix}_loss_coop" in results:
            metrics_to_log[f"train/adv/{prefix}_loss_coop"] = results[f"{prefix}_loss_coop"]

    # Log adversarial training metrics if they exist and were trained this epoch
    if train_emb_this_epoch:
        for prefix, short in adv_prefixes.items():
            if f"{short}_training_loss" in results:
                metrics_to_log[f"train/adv_training/{prefix}_loss"] = results[f"{short}_training_loss"]
            if f"{short}_coop_training_loss" in results:
                metrics_to_log[f"train/adv_training/{prefix}_coop_loss"] = results[f"{short}_coop_training_loss"]
            if f"{short}_training_acc" in results:
                metrics_to_log[f"train/adv_training/{prefix}_acc"] = results[f"{short}_training_acc"]
            if f"{short}_coop_training_acc" in results:
                metrics_to_log[f"train/adv_training/{prefix}_coop_acc"] = results[f"{short}_coop_training_acc"]

    # Add validation metrics if available
    if val_results:
        metrics_to_log["val/loss"] = val_results["val_loss"]

        # Component validation losses
        for key in ['rec_loss', 'cross_loss', 'end_effector_loss', 'triplet_loss',
                   'latent_consistency_loss', 'smoothing_loss']:
            val_key = f"val_{key}"
            if val_key in val_results:
                metrics_to_log[f"val/{key}"] = val_results[val_key]

        # Validation adversarial metrics
        for prefix, short in adv_prefixes.items():
            val_key = f"val_{prefix}_loss"
            if val_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_loss"] = val_results[val_key]

            val_acc_key = f"val_{prefix}_acc_adv"
            if val_acc_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_acc_adv"] = val_results[val_acc_key]

            val_acc_coop_key = f"val_{prefix}_acc_coop"
            if val_acc_coop_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_acc_coop"] = val_results[val_acc_coop_key]

            val_acc_key = f"val_{prefix}_acc"
            if val_acc_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_acc"] = val_results[val_acc_key]

            # Dynamic vs static losses
            val_loss_adv_key = f"val_{prefix}_loss_adv"
            if val_loss_adv_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_loss_adv"] = val_results[val_loss_adv_key]

            val_loss_coop_key = f"val_{prefix}_loss_coop"
            if val_loss_coop_key in val_results:
                metrics_to_log[f"val/adv/{prefix}_loss_coop"] = val_results[val_loss_coop_key]

    # Log metrics to wandb
    wandb.log(metrics_to_log)

def print_stage_summary(stage_idx, total_stages, stage, epoch, total_epochs):
    """Print a concise stage summary"""
    stage_type = "paired" if stage['paired'] else "unpaired"

    components = []
    if stage['ae']: components.append("AE")
    if stage['ee']: components.append("EE")
    if stage['cross']: components.append("Cross")
    if stage['triplet']: components.append("Triplet")
    if stage['train_emb_adv']: components.append("EmbAdv")
    if stage['train_discrim_adv']: components.append("DiscrAdv")
    if stage['emb_adv']: components.append("UseEmbAdv")
    if stage['discrim_adv']: components.append("UseDiscrAdv")

    stage_desc = ", ".join(components)

    logger.info(f"Stage {stage_idx+1}/{total_stages} | Epoch {epoch}/{total_epochs} | {stage_type.upper()} | {stage_desc}")

def print_epoch_summary(results, val_results=None, train_emb_this_epoch=True):
    """Print a concise epoch summary of key metrics"""
    # Build a formatted summary string
    summary = [f"Loss: {results['loss']:.4f}"]

    # Add validation loss if available
    if val_results and 'val_loss' in val_results:
        summary.append(f"Val Loss: {val_results['val_loss']:.4f}")

    # Add reconstruction losses if available
    if 'rec_loss' in results:
        summary.append(f"Rec: {results['rec_loss']:.4f}")
    if 'cross_loss' in results:
        summary.append(f"Cross: {results['cross_loss']:.4f}")

    # Add key adversarial metrics if available
    if 'privacy_acc_adv' in results:
        summary.append(f"Priv Acc: {results['privacy_acc_adv']:.4f}")
    if 'utility_acc_adv' in results:
        summary.append(f"Util Acc: {results['utility_acc_adv']:.4f}")

    # Log the summary
    logger.info(" | ".join(summary))

    # Optionally log detailed adversarial metrics if trained this epoch
    if train_emb_this_epoch and ('priv_training_loss' in results or 'util_training_loss' in results):
        adv_summary = []
        if 'priv_training_acc' in results:
            adv_summary.append(f"Priv Adv Acc: {results['priv_training_acc']:.4f}")
        if 'util_training_acc' in results:
            adv_summary.append(f"Util Adv Acc: {results['util_training_acc']:.4f}")
        if 'discriminator_train_acc' in results:
            adv_summary.append(f"Discrim Acc: {results['discriminator_train_acc']:.4f}")

        if adv_summary:
            logger.info("Adversaries | " + " | ".join(adv_summary))

def save_checkpoint(model, optimizer, epoch, save_path, metrics=None):
    """Save model checkpoint and log to wandb"""
    try:
        # Move model to CPU before saving to avoid CUDA errors
        model_cpu = model
        if next(model.parameters()).is_cuda:
            # Create a CPU copy of the model
            try:
                # First try to get state dict while still on GPU
                state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                # Save only the model's state_dict for compatibility with evaluation script
                torch.save(state_dict, save_path)

                # Optionally save the full training state in a separate file
                full_state_path = save_path.replace('.pth', '_full.pth')

                # Get optimizer state dict
                optimizer_state = optimizer.state_dict()

                # Create full state with CPU tensors
                full_state = {
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer_state,
                    'metrics': metrics
                }

                torch.save(full_state, full_state_path)

                # Log checkpoints to wandb
                if not os.environ.get('WANDB_DISABLED', False):
                    try:
                        wandb.save(save_path)
                        wandb.save(full_state_path)
                    except Exception as e:
                        logger.warning(f"Failed to log checkpoints to wandb: {e}")

                logger.info(f"Model saved to {Path(save_path).name}")
                return True

            except RuntimeError as e:
                logger.error(f"Error saving model state dict: {e}")
                # If we get here, there was an error getting the state dict while on GPU
                # Try a different approach
                return False
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False
        logger.warning(f"Failed to log checkpoint to W&B: {e}")
        logger.info(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train anonymizer model')
    parser.add_argument('--model_type', type=str, default='dmr', choices=['dmr', 'pmr'],
                      help='Model type: dmr or pmr')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'etri'],
                      help='Dataset to use')
    parser.add_argument('--setting', type=str, default='cv', choices=['cs', 'cv'],
                      help='Cross-subject (cs) or cross-view (cv) evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for unpaired training')
    parser.add_argument('--paired_batch_size', type=int, default=8,
                      help='Batch size for paired training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs for the final stage')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--adv_lr', type=float, default=1e-4,
                      help='Learning rate for adversarial networks (PMR only)')
    parser.add_argument('--train_samples', type=int, default=10000,
                      help='Number of paired samples for training')
    parser.add_argument('--test_samples', type=int, default=1000,
                      help='Number of paired samples for testing')
    parser.add_argument('--T', type=int, default=75,
                      help='Sequence length')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                      help='Directory to save models')
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='Evaluate every N epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Save model every N epochs')
    parser.add_argument('--emb_clf_update_per_epoch_paired', type=float, default=1.0,
                      help='Update frequency for paired embedding classifiers (adversaries). Values < 1 skip updates.')
    parser.add_argument('--emb_clf_update_per_epoch_unpaired', type=float, default=1.0,
                      help='Update frequency for unpaired embedding classifiers (adversaries). Values < 1 skip updates.')
    parser.add_argument('--metric', type=str, default='val_loss',
                      help='Metric to use for model selection')
    parser.add_argument('--metric_minimize', action='store_true', default=True,
                      help='Whether to minimize or maximize the metric for model selection')
    parser.add_argument('--wandb_project', type=str, default='PMR',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                      help='Disable wandb logging')

    args = parser.parse_args()

    # Set device - always use cuda
    device = torch.device('cuda')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Initialize wandb if enabled
    if not args.no_wandb:
        run_name = args.wandb_name or f"{args.model_type}_{args.dataset}_{args.setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(start_method="thread")
        )
        logger.info(f"W&B initialized: {run_name}")

    # Map the setting parameter to a case value for Dataloaders
    # 'cs' (cross-subject) = 0, 'cv' (cross-view) = 1
    case = 0 if args.setting == 'cs' else 1

    # Create dataloaders for Action Recognition (ar)
    logger.info(f"Creating action recognition dataloader ({args.dataset}, {args.setting})")
    ar_dataloaders = Dataloaders(
        dataset=args.dataset.upper(),
        case=case,
        seg=args.T,
        tag='ar'
    )

    # Create dataloaders for Re-Identification (ri)
    # For ri, we can only use cross-view (cv) setting since we need
    # the same actors in train and test
    if args.setting == 'cs' and args.model_type == 'pmr':
        logger.warning("Using cross-view for re-identification task even though cross-subject was specified")
        ri_case = 1  # Always use cross-view for re-identification
    else:
        ri_case = case

    logger.info(f"Creating re-identification dataloader ({args.dataset}, {'cv' if ri_case == 1 else 'cs'})")
    ri_dataloaders = Dataloaders(
        dataset=args.dataset.upper(),
        case=ri_case,
        seg=args.T,
        tag='ri'
    )

    # Get train and validation loaders
    train_loader_ar = ar_dataloaders.get_train_loader(args.batch_size, args.workers)
    val_loader_ar = ar_dataloaders.get_val_loader(args.batch_size, args.workers)

    train_loader_ri = ri_dataloaders.get_train_loader(args.batch_size, args.workers)
    val_loader_ri = ri_dataloaders.get_val_loader(args.batch_size, args.workers)

    # For paired data, we'll use two loaders simultaneously with zip
    train_loader_paired = zip(train_loader_ar, train_loader_ri)
    val_loader_paired = zip(val_loader_ar, val_loader_ri)

    # For unpaired data, we'll use the action recognition loader
    train_loader_unpaired = train_loader_ar
    val_loader_unpaired = val_loader_ar

    logger.info(f"AR Train size: {ar_dataloaders.get_train_size()}, Val size: {ar_dataloaders.get_val_size()}")
    logger.info(f"RI Train size: {ri_dataloaders.get_train_size()}, Val size: {ri_dataloaders.get_val_size()}")

    # Initialize model - always use cuda
    logger.info(f"Initializing {args.model_type.upper()} model")
    if args.model_type == 'dmr':
        model = DMR(
            batch_size=args.batch_size,
            dataset=args.dataset.lower(),
            datasets=datasets
        ).cuda()
    else:  # pmr
        # For PMR, ensure batch size is at least 2 for paired training
        if args.paired_batch_size < 2:
            logger.warning(f"Paired batch size {args.paired_batch_size} is too small, setting to 2")
            args.paired_batch_size = 2

        model = PMR(
            adv_lr=args.adv_lr,
            use_adv=True,
            batch_size=args.batch_size,
            dataset=args.dataset.lower(),
            datasets=datasets
        ).cuda()

    # Initialize optimizer with CUDA model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup logging
    log_file = os.path.join(args.output_dir, f'{args.model_type}_{args.dataset}_{args.setting}_training_log.csv')
    with open(log_file, 'w') as f:
        # Create header with all possible metrics
        header = ['stage', 'epoch', 'paired', 'loss', 'val_loss']
        if args.model_type == 'pmr':
            header.extend([
                'rec_loss', 'cross_loss', 'end_effector_loss', 'triplet_loss', 'latent_consistency_loss',
                'smoothing_loss', 'privacy_loss', 'privacy_loss_adv', 'privacy_loss_coop',
                'privacy_acc_adv', 'privacy_acc_coop', 'utility_loss', 'utility_loss_adv',
                'utility_loss_coop', 'utility_acc_adv', 'utility_acc_coop', 'discriminator_loss',
                'discriminator_acc', 'priv_training_loss', 'priv_coop_training_loss',
                'util_training_loss', 'util_coop_training_loss', 'discriminator_train_loss',
                'priv_training_acc', 'priv_coop_training_acc', 'util_training_acc',
                'util_coop_training_acc', 'discriminator_train_acc'
            ])
        f.write(','.join(header) + '\n')

    # Adjust training stages based on provided epochs for final stages
    # Only modify the last two stages
    if args.epochs != 100:
        training_stages[-2]['epochs'] = args.epochs
        training_stages[-1]['epochs'] = args.epochs

    # Store T value in args for consistent use in preprocessing
    # Force T=75 for DMR/PMR because that's what the model expects
    args.T = 75 if args.model_type in ['dmr', 'pmr'] else 64

    # Main training loop - run through each stage
    logger.info("Starting multi-stage training...")

    current_epoch = 0
    total_epochs = sum([stage['epochs'] for stage in training_stages])

    best_val_loss = float('inf') if args.metric_minimize else float('-inf')
    best_model_path = os.path.join(args.output_dir, f'{args.model_type}_{args.dataset}_{args.setting}_best.pth')

    try:
        for stage_idx, stage in enumerate(training_stages):
            # Skip adversarial stages for DMR
            if args.model_type == 'dmr' and (stage.get('train_emb_adv', False) or
                                           stage.get('emb_adv', False) or
                                           stage.get('train_discrim_adv', False) or
                                           stage.get('discrim_adv', False)):
                if not stage.get('ae', False) and not stage.get('cross', False):
                    logger.info(f"Skipping stage {stage_idx+1} (adversarial only) for DMR")
                    continue

            # Train for this stage's epochs
            for epoch in range(stage['epochs']):
                # Determine if we should train adversaries this epoch
                train_emb_this_epoch = True
                if args.model_type == 'pmr':
                    if stage['paired'] and args.emb_clf_update_per_epoch_paired < 1:
                        if epoch % int(1 / args.emb_clf_update_per_epoch_paired) != 0:
                            train_emb_this_epoch = False
                    elif not stage['paired'] and args.emb_clf_update_per_epoch_unpaired < 1:
                        if epoch % int(1 / args.emb_clf_update_per_epoch_unpaired) != 0:
                            train_emb_this_epoch = False

                current_epoch += 1

                # Print current stage info with fewer details
                print_stage_summary(stage_idx, len(training_stages), stage, current_epoch, total_epochs)

                if stage['paired']:
                    # Paired training
                    # Need to recreate the zip iterator for each epoch
                    train_paired_data = zip(train_loader_ar, train_loader_ri)
                    try:
                        train_results = train_one_epoch_paired(
                            model, train_paired_data, optimizer, args,
                            model_type=args.model_type,
                            stage_params=stage,
                            cur_epoch=epoch
                        )
                    except RuntimeError as e:
                        logger.error(f"Error during training: {e}")
                        # Provide default results to continue training
                        train_results = {'loss': float('nan')}

                    # Validation if needed
                    if stage['eval'] and current_epoch % args.eval_interval == 0:
                        # Need to recreate the zip iterator for validation
                        val_paired_data = zip(val_loader_ar, val_loader_ri)
                        try:
                            val_results = validate(
                                model, val_paired_data, args,
                                model_type=args.model_type,
                                paired=True,
                                stage_params=stage
                            )
                        except RuntimeError as e:
                            logger.error(f"Error during paired validation: {e}")
                            val_results = {'val_loss': float('nan')}
                    else:
                        val_results = {'val_loss': float('nan')}
                else:
                    # Unpaired training
                    try:
                        train_results = train_one_epoch_unpaired(
                            model, train_loader_unpaired, optimizer, args,
                            model_type=args.model_type,
                            stage_params=stage,
                            cur_epoch=epoch
                        )
                    except RuntimeError as e:
                        logger.error(f"Error during unpaired training: {e}")
                        # Provide default results to continue training
                        train_results = {'loss': float('nan')}

                    # Validation if needed
                    if stage['eval'] and current_epoch % args.eval_interval == 0:
                        try:
                            val_results = validate(
                                model, val_loader_unpaired, args,
                                model_type=args.model_type,
                                paired=False,
                                stage_params=stage
                            )
                        except RuntimeError as e:
                            logger.error(f"Error during unpaired validation: {e}")
                            val_results = {'val_loss': float('nan')}
                    else:
                        val_results = {'val_loss': float('nan')}

                # Print concise epoch summary
                print_epoch_summary(train_results, val_results, train_emb_this_epoch)

                # Log metrics to wandb
                if not args.no_wandb:
                    stage_info = {
                        "stage_num": stage_idx + 1,
                        "stage_epoch": epoch + 1,
                        "paired": stage['paired']
                    }
                    log_metrics_to_wandb(train_results, val_results, stage_info, current_epoch, train_emb_this_epoch)

                # Log results to CSV
                with open(log_file, 'a') as f:
                    # Create a row with all metrics
                    row = [
                        str(stage_idx+1),
                        str(current_epoch),
                        str(stage['paired']),
                        str(train_results['loss']),
                        str(val_results['val_loss']) if 'val_loss' in val_results else 'nan'
                    ]

                    # Add all other metrics if PMR
                    if args.model_type == 'pmr':
                        metrics_to_log = [
                            'rec_loss', 'cross_loss', 'end_effector_loss', 'triplet_loss', 'latent_consistency_loss',
                            'smoothing_loss', 'privacy_loss', 'privacy_loss_adv', 'privacy_loss_coop',
                            'privacy_acc_adv', 'privacy_acc_coop', 'utility_loss', 'utility_loss_adv',
                            'utility_loss_coop', 'utility_acc_adv', 'utility_acc_coop', 'discriminator_loss',
                            'discriminator_acc', 'priv_training_loss', 'priv_coop_training_loss',
                            'util_training_loss', 'util_coop_training_loss', 'discriminator_train_loss',
                            'priv_training_acc', 'priv_coop_training_acc', 'util_training_acc',
                            'util_coop_training_acc', 'discriminator_train_acc'
                        ]

                        for metric in metrics_to_log:
                            if metric in train_results:
                                row.append(str(train_results[metric]))
                            else:
                                row.append('nan')

                    f.write(','.join(row) + '\n')

                # Save model if needed
                if stage['save'] and current_epoch % args.save_interval == 0:
                    checkpoint_path = os.path.join(
                        args.output_dir,
                        f"{args.model_type}_{args.dataset}_{args.setting}_stage{stage_idx+1}_epoch{current_epoch}.pth"
                    )

                    # Combine metrics
                    metrics = {**train_results, **val_results}
                    save_checkpoint(model, optimizer, current_epoch, checkpoint_path, metrics)

                    # Save best model
                    if stage['eval'] and not np.isnan(val_results.get('val_' + args.metric.replace('val_', ''), float('nan'))):
                        metric_val = val_results.get('val_' + args.metric.replace('val_', ''), None)

                        if metric_val is not None:
                            is_best = (args.metric_minimize and metric_val < best_val_loss) or \
                                    (not args.metric_minimize and metric_val > best_val_loss)

                            if is_best:
                                best_val_loss = metric_val
                                save_checkpoint(model, optimizer, current_epoch, best_model_path, metrics)
                                logger.info(f"New best model: {args.metric}={best_val_loss:.6f}")
                                if not args.no_wandb:
                                    wandb.run.summary["best_epoch"] = current_epoch
                                    wandb.run.summary["best_" + args.metric] = best_val_loss

        # Save final model
        final_model_path = os.path.join(args.output_dir, f'{args.model_type}_{args.dataset}_{args.setting}_final.pth')
        save_checkpoint(model, optimizer, current_epoch, final_model_path)
        logger.info(f"Training complete. Final model saved")
        logger.info(f"Best model: {args.metric}={best_val_loss:.6f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
    finally:
        # Close wandb properly
        if not args.no_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
