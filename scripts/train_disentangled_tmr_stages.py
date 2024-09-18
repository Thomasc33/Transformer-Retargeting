"""
Training stages for Disentangled TMR with Enhanced Physical Losses
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import json
import time
import psutil
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.datasets import NTU120_ACTION_REMAP

# Pre-built lookup tensor for NTU120 action label remapping (0-indexed orig -> contiguous)
_NTU120_REMAP_TABLE = None

def remap_action_labels(action_labels, dataset):
    """Remap non-contiguous action labels to contiguous 0-based indices.

    NTU120 has actions 1-49 and 61-105 (single-person only), giving labels
    0-48 and 60-104 after subtracting 1.  This remaps them to 0-93.
    NTU60 (actions 1-49) already has contiguous labels 0-48, so no-op.
    """
    if dataset != 'ntu120':
        return action_labels
    global _NTU120_REMAP_TABLE
    if _NTU120_REMAP_TABLE is None or _NTU120_REMAP_TABLE.device != action_labels.device:
        table = torch.zeros(120, dtype=torch.long, device=action_labels.device)
        for orig, new in NTU120_ACTION_REMAP.items():
            table[orig] = new
        _NTU120_REMAP_TABLE = table
    return _NTU120_REMAP_TABLE[action_labels]

# Import schedulers
# from src.training.schedulers import get_scheduler
def get_scheduler(optimizer, args):
    """
    Get learning rate scheduler
    """
    if args.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_step_size, 
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.lr_min
        )
    elif args.lr_scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.lr_gamma, 
            patience=args.lr_patience, 
            verbose=True
        )
    else:
        return None

from src.training.action_preservation_loss import ActionPreservationLoss, FeatureConsistencyLoss, MotionDynamicsLoss

# Import enhanced physical loss module
try:
    from src.losses.enhanced_physical import EnhancedPhysicalLoss
    ENHANCED_PHYSICAL_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced physical loss module not available. Using basic physical losses.")
    ENHANCED_PHYSICAL_AVAILABLE = False

# Import multi-model auxiliary loss for downstream evaluation
try:
    from src.losses.multi_model_auxiliary import MultiModelAuxiliaryLoss
    MULTI_MODEL_AUX_AVAILABLE = True
except ImportError:
    print("Warning: Multi-model auxiliary loss not available. Downstream AR early stopping disabled.")
    MULTI_MODEL_AUX_AVAILABLE = False


def evaluate_downstream_ar(model, auxiliary_models, val_loader, args, max_samples=1000):
    """
    Evaluate downstream AR using frozen auxiliary models.
    
    Args:
        model: TMR model to evaluate
        auxiliary_models: MultiModelAuxiliaryLoss instance with frozen models
        val_loader: Validation data loader
        args: Training arguments
        max_samples: Maximum samples to evaluate (for speed)
    
    Returns:
        dict: Downstream AR accuracies for each auxiliary model
    """
    if not MULTI_MODEL_AUX_AVAILABLE or auxiliary_models is None:
        return {}
    
    model.eval()
    auxiliary_models.eval()
    
    # Track accuracies for each auxiliary model
    model_accuracies = {config['name']: {'correct': 0, 'total': 0} 
                       for config in auxiliary_models.model_configs}
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if samples_processed >= max_samples:
                break
                
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y2 = y2.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)

            # Generate motion using TMR (inference-time setup)
            output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)
            
            # Pad first frame to make 64 frames for evaluation
            first_frame = x2[:, :, 0:1, :, :]
            output_padded = torch.cat([first_frame, output], dim=2)
            
            # Evaluate on each auxiliary model
            for model_name, aux_model in auxiliary_models.auxiliary_models.items():
                try:
                    logits = aux_model(output_padded)
                    preds = logits.argmax(dim=1)
                    correct = (preds == action_labels).sum().item()
                    
                    model_accuracies[model_name]['correct'] += correct
                    model_accuracies[model_name]['total'] += x1.size(0)
                except Exception as e:
                    print(f"Warning: Error evaluating {model_name}: {e}")
                    continue
            
            samples_processed += x1.size(0)
    
    # Compute accuracies
    downstream_ar = {}
    for model_name, stats in model_accuracies.items():
        if stats['total'] > 0:
            downstream_ar[model_name] = stats['correct'] / stats['total']
        else:
            downstream_ar[model_name] = 0.0
    
    return downstream_ar


def get_vq_loss(model, args):
    """
    Fetch the most recent VQ/codebook loss from the action encoder (if enabled).

    Returns:
        loss_vq: Tensor or 0.0
        vq_info: dict or None
    """
    if not hasattr(model, "action_encoder") or not hasattr(model.action_encoder, "get_vq_info"):
        return 0.0, None
    vq_info = model.action_encoder.get_vq_info()
    if not vq_info or vq_info.get("total") is None:
        return 0.0, vq_info
    weight_vq = getattr(args, "weight_vq", 0.0)
    if weight_vq == 0.0:
        return 0.0, vq_info
    return weight_vq * vq_info["total"], vq_info


def teacher_forcing_schedule(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """
    Teacher forcing schedule used throughout this project (legacy style):
      ratio = max(end, start - epoch / total_epochs)
    """
    total_epochs = max(1, int(total_epochs))
    ratio = float(start) - float(epoch) / float(total_epochs)
    ratio = max(float(end), ratio)
    return float(max(0.0, min(1.0, ratio)))


def train_stage1(model, ar_classifier, ri_classifier, disentangle_losses, 
                train_loader, val_loader, optimizers, args, auxiliary_models=None, logger=None, schedulers=None):
    """
    Stage 1: Encoder Pretraining with Disentanglement Losses
    
    Goal: Learn disentangled action and identity representations
    
    What's trained:
    - Action encoder (with AR loss)
    - Identity encoder (with RI loss)
    - Disentanglement losses (contrastive, adversarial, orthogonality, MI)
    
    What's frozen:
    - Decoder (not used yet)
    
    Args:
        auxiliary_models: MultiModelAuxiliaryLoss instance for downstream AR evaluation
        logger: EnhancedLogger instance for structured logging
    """
    model.freeze_decoder()
    model.unfreeze_action_encoder()
    model.unfreeze_identity_encoder()

    # Initialize scaler for mixed precision
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=not args.no_amp and args.device == 'cuda')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and args.device == 'cuda')
    
    # Gradient accumulation
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    effective_batch_size = args.batch_size * accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (base: {args.batch_size} x {accumulation_steps} steps)")
    
    best_ar_accuracy = 0.0
    best_downstream_ar = 0.0
    epochs_without_improvement = 0
    patience = getattr(args, 'early_stop_patience', 10)
    downstream_eval_freq = getattr(args, 'downstream_eval_freq', 5)
    use_downstream_early_stop = getattr(args, 'use_downstream_early_stop', False) and auxiliary_models is not None
    
    # Initialize stage logging
    if logger:
        logger.start_stage(1, args.stage1_epochs, "Stage 1: Encoder Pretraining")

    for epoch in range(args.stage1_epochs):
        print(f"\n--- Stage 1, Epoch {epoch+1}/{args.stage1_epochs} ---")
        
        # Start epoch logging
        if logger:
            logger.start_epoch(epoch, len(train_loader))
        
        # Training
        model.train()
        ar_classifier.train()
        ri_classifier.train()
        
        total_loss = 0.0
        total_ar_loss = 0.0
        total_ri_loss = 0.0
        total_contrastive = 0.0
        total_adversarial = 0.0
        total_orthogonality = 0.0
        total_mutual_info = 0.0
        total_vq_loss = 0.0
        total_vq_embed = 0.0
        total_vq_commit = 0.0
        total_vq_perplexity = 0.0
        total_vq_usage = 0.0
        vq_steps = 0
        ar_correct = 0
        ri_correct = 0
        total_samples = 0
        
        # Use enhanced logger progress bar or fallback to tqdm
        batch_iterator = enumerate(train_loader)
        if not logger or not logger.enable_progress_bars:
            batch_iterator = enumerate(tqdm(train_loader, desc="Training", mininterval=30.0))
        
        for batch_idx, batch in batch_iterator:
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device, non_blocking=True)
            x2 = x2.to(args.device, non_blocking=True)
            y1 = y1.to(args.device, non_blocking=True)
            y2 = y2.to(args.device, non_blocking=True)
            actors = actors.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)

            # Add M dimension: (B, C, T, V) -> (B, C, T, V, M=1)
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            # Extract labels (convert from 1-indexed to 0-indexed)
            action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)
            identity_labels = (actors[:, 1] - 1).long()  # P2 (1-indexed -> 0-indexed)

            autocast_ctx = torch.amp.autocast('cuda', enabled=not args.no_amp and args.device == 'cuda') if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=not args.no_amp and args.device == 'cuda')

            with autocast_ctx:
                # === ENCODE ONCE ===
                action_features = model.encode_action(x1)  # (T, B, D_action) from (P1, A1)
                identity_features = model.encode_identity(x2)  # pooled if temporal identity is enabled
                
                # === 2. Train AR classifier on action features ===
                ar_logits = ar_classifier(action_features)  # Handles temporal averaging internally
                loss_ar = F.cross_entropy(ar_logits, action_labels) * args.weight_ar
                loss_vq, vq_info = get_vq_loss(model, args)
                loss_ar_total = loss_ar + loss_vq

                # === 3. Train RI classifier on identity features ===
                ri_logits = ri_classifier(identity_features)
                loss_ri = F.cross_entropy(ri_logits, identity_labels) * args.weight_ri
                
                # === 4. Train discriminator ===
                disentangle_dict = disentangle_losses.compute_all(
                    action_features, identity_features, action_labels, identity_labels,
                    train_discriminator=True
                )
                loss_disc = disentangle_dict['adversarial']
                
                # === 5. Train encoders with disentanglement losses ===
                disentangle_dict_enc = disentangle_losses.compute_all(
                    action_features, identity_features, action_labels, identity_labels,
                    train_discriminator=False
                )
                
                loss_contrastive = disentangle_dict_enc['contrastive']
                loss_adversarial = disentangle_dict_enc['adversarial']
                loss_orthogonality = disentangle_dict_enc['orthogonality']
                loss_mutual_info = disentangle_dict_enc['mutual_info']
                
                loss_disentangle = (
                    args.weight_contrastive * loss_contrastive +
                    args.weight_adversarial * loss_adversarial +
                    args.weight_orthogonality * loss_orthogonality +
                    args.weight_mutual_info * loss_mutual_info
                )
                
                loss_disentangle_total = loss_disentangle + loss_vq
            
            if torch.isnan(loss_disentangle_total) or torch.isnan(loss_ar_total) or torch.isnan(loss_ri) or torch.isnan(loss_disc):
                 print(f"NaN loss detected at Stage 1, batch {batch_idx}!")
                 print(f"  AR: {loss_ar_total}")
                 print(f"  RI: {loss_ri}")
                 print(f"  Disc: {loss_disc}")
                 print(f"  Disentangle: {loss_disentangle_total}")
                 print(f"    Contrastive: {loss_contrastive}")
                 print(f"    Adversarial: {loss_adversarial}")
                 print(f"    Orthogonality: {loss_orthogonality}")
                 print(f"    MI: {loss_mutual_info}")
                 print(f"    VQ: {loss_vq}")

            # === OPTIMIZER STEPS WITH GRADIENT ACCUMULATION ===
            # Scale loss for gradient accumulation
            scaled_loss_ar_total = loss_ar_total / accumulation_steps
            scaled_loss_ri = loss_ri / accumulation_steps
            scaled_loss_disc = loss_disc / accumulation_steps
            scaled_loss_disentangle_total = loss_disentangle_total / accumulation_steps
            
            # Step 1: Update AR classifier and action encoder
            optimizers['ar_classifier'].zero_grad(set_to_none=True)
            optimizers['action_encoder'].zero_grad(set_to_none=True)
            scaler.scale(scaled_loss_ar_total).backward(retain_graph=True)
            
            # Step 2: Update RI classifier and identity encoder
            optimizers['ri_classifier'].zero_grad(set_to_none=True)
            optimizers['identity_encoder'].zero_grad(set_to_none=True)
            scaler.scale(scaled_loss_ri).backward(retain_graph=True)
            
            # Step 3: Update discriminator
            optimizers['discriminator'].zero_grad(set_to_none=True)
            scaler.scale(scaled_loss_disc).backward(retain_graph=True)
            
            # Step 4: Update encoders with disentanglement losses
            optimizers['action_encoder'].zero_grad(set_to_none=True)
            optimizers['identity_encoder'].zero_grad(set_to_none=True)
            scaler.scale(scaled_loss_disentangle_total).backward()
            
            # Update weights at accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if args.use_gradient_clip:
                    # Clip gradients for all optimizers
                    all_params = []
                    all_params.extend(list(ar_classifier.parameters()))
                    all_params.extend(list(model.action_encoder.parameters()))
                    all_params.extend(list(ri_classifier.parameters()))
                    all_params.extend(list(model.identity_encoder.parameters()))
                    all_params.extend(list(disentangle_losses.get_discriminator_params()))
                    
                    # Unscale gradients for all optimizers before clipping
                    scaler.unscale_(optimizers['ar_classifier'])
                    scaler.unscale_(optimizers['action_encoder'])
                    scaler.unscale_(optimizers['ri_classifier'])
                    scaler.unscale_(optimizers['identity_encoder'])
                    scaler.unscale_(optimizers['discriminator'])
                    torch.nn.utils.clip_grad_norm_(all_params, args.gradient_clip_value)
                
                # Step all optimizers
                scaler.step(optimizers['ar_classifier'])
                scaler.step(optimizers['action_encoder'])
                scaler.step(optimizers['ri_classifier'])
                scaler.step(optimizers['identity_encoder'])
                scaler.step(optimizers['discriminator'])
                scaler.update()

            # Accumulate losses
            total_loss += loss_disentangle_total.item()
            total_ar_loss += loss_ar.item()
            total_ri_loss += loss_ri.item()
            total_contrastive += loss_contrastive.item()
            total_adversarial += loss_adversarial.item()
            total_orthogonality += loss_orthogonality.item()
            total_mutual_info += loss_mutual_info.item()
            total_samples += x1.size(0)
            
            # Compute accuracies
            ar_preds = ar_logits.argmax(dim=1)
            ar_correct += (ar_preds == action_labels).sum().item()
            ri_preds = ri_logits.argmax(dim=1)
            ri_correct += (ri_preds == identity_labels).sum().item()

            if isinstance(loss_vq, torch.Tensor):
                total_vq_loss += loss_vq.item()
                if vq_info and vq_info.get("embed") is not None:
                    total_vq_embed += vq_info["embed"].item()
                if vq_info and vq_info.get("commit") is not None:
                    total_vq_commit += vq_info["commit"].item()
                if vq_info and vq_info.get("perplexity") is not None:
                    total_vq_perplexity += float(vq_info["perplexity"].item())
                if vq_info and vq_info.get("usage") is not None:
                    total_vq_usage += float(vq_info["usage"].item())
                vq_steps += 1
            
            # Enhanced logging
            if logger:
                loss_dict = {
                    'total_loss': loss_disentangle_total.item(),
                    'ar_loss': loss_ar.item(),
                    'ri_loss': loss_ri.item(),
                    'contrastive_loss': loss_contrastive.item(),
                    'adversarial_loss': loss_adversarial.item(),
                    'orthogonality_loss': loss_orthogonality.item(),
                    'mutual_info_loss': loss_mutual_info.item(),
                }
                if isinstance(loss_vq, torch.Tensor):
                    loss_dict['vq_loss'] = loss_vq.item()
                metrics_dict = {
                    'ar_accuracy': ar_correct / total_samples if total_samples > 0 else 0.0,
                    'ri_accuracy': ri_correct / total_samples if total_samples > 0 else 0.0,
                    'disc_accuracy': disentangle_dict['disc_accuracy'],
                }
                if vq_info and vq_info.get("perplexity") is not None:
                    metrics_dict['vq_perplexity'] = float(vq_info["perplexity"].item())
                if vq_info and vq_info.get("usage") is not None:
                    metrics_dict['vq_usage'] = float(vq_info["usage"].item())
                logger.log_batch(batch_idx, x1.size(0), loss_dict, metrics_dict)
            
            # Fallback logging
            elif (batch_idx + 1) % args.log_freq == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss_disentangle_total.item():.4f}, "
                      f"AR={loss_ar.item():.4f}, "
                      f"RI={loss_ri.item():.4f}")
        
        # Epoch summary
        ar_accuracy = ar_correct / total_samples
        ri_accuracy = ri_correct / total_samples

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  AR Accuracy: {ar_accuracy:.4f}")
        print(f"  RI Accuracy: {ri_accuracy:.4f}")
        print(f"  AR Loss: {total_ar_loss/len(train_loader):.4f}")
        print(f"  RI Loss: {total_ri_loss/len(train_loader):.4f}")
        print(f"  Contrastive Loss: {total_contrastive/len(train_loader):.4f}")
        print(f"  Adversarial Loss: {total_adversarial/len(train_loader):.4f}")
        print(f"  Orthogonality Loss: {total_orthogonality/len(train_loader):.4f}")
        print(f"  Mutual Info Loss: {total_mutual_info/len(train_loader):.4f}")
        print(f"  Discriminator Accuracy: {disentangle_dict['disc_accuracy']:.4f}")
        if vq_steps > 0:
            print(f"  VQ Loss: {total_vq_loss/vq_steps:.4f}")
            print(f"  VQ Embed Loss: {total_vq_embed/vq_steps:.4f}")
            print(f"  VQ Commit Loss: {total_vq_commit/vq_steps:.4f}")
            print(f"  VQ Perplexity: {total_vq_perplexity/vq_steps:.2f}")
            print(f"  VQ Code Usage: {total_vq_usage/vq_steps:.2f}")

        # Validation
        val_metrics = validate_stage1(model, ar_classifier, ri_classifier, val_loader, args)
        print(f"  Val AR Accuracy: {val_metrics['ar_accuracy']:.4f}")
        print(f"  Val RI Accuracy: {val_metrics['ri_accuracy']:.4f}")

        # Downstream AR evaluation (every N epochs)
        downstream_ar_metrics = {}
        if use_downstream_early_stop and (epoch + 1) % downstream_eval_freq == 0:
            print("  Evaluating downstream AR...")
            downstream_ar_metrics = evaluate_downstream_ar(model, auxiliary_models, val_loader, args)
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics) if downstream_ar_metrics else 0.0
            print(f"  Downstream AR (avg): {avg_downstream_ar:.4f}")
            for model_name, acc in downstream_ar_metrics.items():
                print(f"    {model_name}: {acc:.4f}")

        # Log to wandb
        wandb_log = {
            'stage1/epoch': epoch + 1,
            'stage1/train_ar_accuracy': ar_accuracy,
            'stage1/train_ri_accuracy': ri_accuracy,
            'stage1/train_ar_loss': total_ar_loss/len(train_loader),
            'stage1/train_ri_loss': total_ri_loss/len(train_loader),
            'stage1/train_contrastive_loss': total_contrastive/len(train_loader),
            'stage1/train_adversarial_loss': total_adversarial/len(train_loader),
            'stage1/train_orthogonality_loss': total_orthogonality/len(train_loader),
            'stage1/train_mutual_info_loss': total_mutual_info/len(train_loader),
            'stage1/train_disc_accuracy': disentangle_dict['disc_accuracy'],
            'stage1/val_ar_accuracy': val_metrics['ar_accuracy'],
            'stage1/val_ri_accuracy': val_metrics['ri_accuracy'],
        }
        if vq_steps > 0:
            wandb_log.update(
                {
                    "stage1/train_vq_loss": total_vq_loss / vq_steps,
                    "stage1/train_vq_embed": total_vq_embed / vq_steps,
                    "stage1/train_vq_commit": total_vq_commit / vq_steps,
                    "stage1/train_vq_perplexity": total_vq_perplexity / vq_steps,
                    "stage1/train_vq_usage": total_vq_usage / vq_steps,
                }
            )
        
        # Add downstream AR metrics to wandb
        if downstream_ar_metrics:
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            wandb_log['stage1/downstream_ar_avg'] = avg_downstream_ar
            for model_name, acc in downstream_ar_metrics.items():
                wandb_log[f'stage1/downstream_ar_{model_name}'] = acc
        
        if not args.no_wandb:
            wandb.log(wandb_log)
        
        # Enhanced epoch logging
        if logger:
            epoch_metrics = {
                'ar_accuracy': ar_accuracy,
                'ri_accuracy': ri_accuracy,
                'ar_loss': total_ar_loss/len(train_loader),
                'ri_loss': total_ri_loss/len(train_loader),
                'contrastive_loss': total_contrastive/len(train_loader),
                'adversarial_loss': total_adversarial/len(train_loader),
                'orthogonality_loss': total_orthogonality/len(train_loader),
                'mutual_info_loss': total_mutual_info/len(train_loader),
                'disc_accuracy': disentangle_dict['disc_accuracy'],
            }
            if vq_steps > 0:
                epoch_metrics.update(
                    {
                        "vq_loss": total_vq_loss / vq_steps,
                        "vq_embed": total_vq_embed / vq_steps,
                        "vq_commit": total_vq_commit / vq_steps,
                        "vq_perplexity": total_vq_perplexity / vq_steps,
                        "vq_usage": total_vq_usage / vq_steps,
                    }
                )
            validation_metrics = {
                'ar_accuracy': val_metrics['ar_accuracy'],
                'ri_accuracy': val_metrics['ri_accuracy'],
            }
            if downstream_ar_metrics:
                validation_metrics['downstream_ar_avg'] = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
                validation_metrics.update({f'downstream_ar_{k}': v for k, v in downstream_ar_metrics.items()})

        # Early stopping logic
        current_metric = val_metrics['ar_accuracy']
        is_best = False

        if use_downstream_early_stop and downstream_ar_metrics:
            # Use downstream AR for early stopping
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            if avg_downstream_ar > best_downstream_ar:
                best_downstream_ar = avg_downstream_ar
                is_best = True
                epochs_without_improvement = 0
                print(f"  New best downstream AR: {best_downstream_ar:.4f}")
            else:
                epochs_without_improvement += 1
        else:
            # Use validation AR for early stopping
            if current_metric > best_ar_accuracy:
                best_ar_accuracy = current_metric
                is_best = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Log epoch results
        if logger:
            logger.end_epoch(epoch, epoch_metrics, validation_metrics, is_best)

        # Step schedulers if provided
        if schedulers:
            for scheduler in schedulers.values():
                scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            from train_disentangled_tmr import save_checkpoint
            
            # Prepare best metrics and training state for checkpoint
            best_metrics = {
                'best_ar_accuracy': best_ar_accuracy,
                'best_downstream_ar': best_downstream_ar if use_downstream_early_stop else 0.0,
            }
            training_state = {
                'epochs_without_improvement': epochs_without_improvement,
                'patience': patience,
                'use_downstream_early_stop': use_downstream_early_stop,
                'downstream_eval_freq': downstream_eval_freq,
            }
            
            save_checkpoint(model, ar_classifier, ri_classifier, optimizers, epoch, stage=1, 
                          args=args, is_best=is_best, best_metrics=best_metrics, 
                          training_state=training_state)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best AR accuracy: {best_ar_accuracy:.4f}")
            break

    print(f"\nStage 1 complete! Best AR accuracy: {best_ar_accuracy:.4f}")
    return best_ar_accuracy


def validate_stage1(model, ar_classifier, ri_classifier, val_loader, args):
    """Validate Stage 1"""
    model.eval()
    ar_classifier.eval()
    ri_classifier.eval()
    
    ar_correct = 0
    ri_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y1 = y1.to(args.device)
            y2 = y2.to(args.device)
            actors = actors.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)
            identity_labels = (actors[:, 1] - 1).long()

            # Encode
            action_features = model.encode_action(x1)
            identity_features = model.encode_identity(x2)
            
            # AR classification
            ar_logits = ar_classifier(action_features)
            ar_preds = ar_logits.argmax(dim=1)
            ar_correct += (ar_preds == action_labels).sum().item()

            # RI classification
            ri_logits = ri_classifier(identity_features)
            ri_preds = ri_logits.argmax(dim=1)
            ri_correct += (ri_preds == identity_labels).sum().item()
            
            total_samples += x1.size(0)
    
    return {
        'ar_accuracy': ar_correct / total_samples,
        'ri_accuracy': ri_correct / total_samples
    }


def train_stage2(model, recon_loss, physical_loss, physical_weights, frozen_sgn_loss,
                 train_loader, val_loader, optimizers, args, auxiliary_models=None, logger=None,
                 action_encoder=None, ar_classifier=None):
    """
    Stage 2: Decoder Training with Reconstruction Losses + Physical Plausibility + Frozen SGN

    Goal: Learn to reconstruct motion from disentangled features
          Generate physically plausible and recognizable motion

    What's trained:
    - Decoder

    What's frozen (or fine-tuned with small LR):
    - Action encoder
    - Identity encoder
    - Frozen SGN (if enabled)
    """
    model.unfreeze_decoder()
    # Encoders can be fine-tuned with small LR (already set in optimizers)

    best_recon_loss = float('inf')
    best_downstream_ar = 0.0
    epochs_without_improvement = 0
    patience = getattr(args, 'early_stop_patience', 10)
    downstream_eval_freq = getattr(args, 'downstream_eval_freq', 5)
    use_downstream_early_stop = getattr(args, 'use_downstream_early_stop', False) and auxiliary_models is not None
    
    # Initialize stage logging
    if logger:
        logger.start_stage(2, args.stage2_epochs, "Stage 2: Decoder Training")
    
    # Initialize action preservation losses if encoders are provided
    action_preservation_loss = None
    feature_consistency_loss = None
    motion_dynamics_loss = None
    if action_encoder is not None and ar_classifier is not None:
        action_preservation_loss = ActionPreservationLoss(action_encoder, ar_classifier)
        feature_consistency_loss = FeatureConsistencyLoss(action_encoder)
        motion_dynamics_loss = MotionDynamicsLoss()
        print("  Enabled action preservation losses for decoder training")

    for epoch in range(args.stage2_epochs):
        print(f"\n--- Stage 2, Epoch {epoch+1}/{args.stage2_epochs} ---")

        # Training
        model.train()

        total_loss = 0.0
        total_recon_mse = 0.0
        total_bone_loss = 0.0
        total_physical_loss = 0.0
        total_frozen_sgn_loss = 0.0
        total_frozen_sgn_acc = 0.0
        total_vq_loss = 0.0
        total_vq_embed = 0.0
        total_vq_commit = 0.0
        total_vq_perplexity = 0.0
        total_vq_usage = 0.0
        total_action_preservation = 0.0
        total_feature_consistency = 0.0
        total_motion_dynamics = 0.0
        total_action_preservation_acc = 0.0
        vq_steps = 0
        ap_steps = 0

        # Gradient accumulation
        accum_steps = getattr(args, 'gradient_accumulation_steps', 1)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", mininterval=30.0)):
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y1 = y1.to(args.device)
            y2 = y2.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            # Forward pass with teacher forcing
            # FIXED: Use x2 as target_skeleton (P2, A2) and y2 as target_motion (P2, A1)
            teacher_forcing_ratio = teacher_forcing_schedule(
                epoch,
                args.stage2_epochs,
                getattr(args, "stage2_teacher_forcing_start", 1.0),
                getattr(args, "stage2_teacher_forcing_end", 0.5),
            )
            output, action_features, identity_features = model(x1, x2, y2, teacher_forcing_ratio)

            # Compute reconstruction losses
            # Note: output is (B, C, T-1, V, M), so slice target to match
            y2_target = y2[:, :, 1:, :, :]  # Remove first frame to match output shape
            recon_loss_val, loss_dict = recon_loss.loss(output, y2_target, x1)

            # Compute physical plausibility losses
            phys_loss_val, phys_dict = physical_loss(output, y2_target, physical_weights)

            # Compute frozen SGN auxiliary loss (if enabled)
            frozen_sgn_loss_val = 0.0
            frozen_sgn_acc = 0.0
            if frozen_sgn_loss is not None:
                # Pad first frame to make 64 frames for SGN
                first_frame = x2[:, :, 0:1, :, :]
                output_padded = torch.cat([first_frame, output], dim=2)

                # Get action labels (A1 from x1)
                action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)

                frozen_sgn_loss_val, frozen_sgn_acc = frozen_sgn_loss(output_padded, action_labels)
                frozen_sgn_loss_val = args.weight_frozen_sgn * frozen_sgn_loss_val

            # Compute action preservation losses (if enabled)
            ap_loss_val = 0.0
            fc_loss_val = 0.0
            md_loss_val = 0.0
            ap_acc_val = 0.0

            if action_preservation_loss is not None:
                # Get action labels (A1 from x1)
                action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)

                # Pad decoded output once for both preservation losses
                output_for_enc = output
                if output_for_enc.shape[2] < 64:
                    pad_len = 64 - output_for_enc.shape[2]
                    last_frame = output_for_enc[:, :, -1:, :, :]
                    output_for_enc = torch.cat([output_for_enc, last_frame.repeat(1, 1, pad_len, 1, 1)], dim=2)

                # Encode decoded skeletons ONCE (shared between AP and FC losses)
                decoded_features = action_preservation_loss.action_encoder(output_for_enc)
                decoded_features_for_fc = decoded_features  # share

                # Action preservation loss (AR on decoded skeletons)
                ar_logits = action_preservation_loss.ar_classifier(decoded_features)
                ap_loss_val = F.cross_entropy(ar_logits, action_labels)
                ap_acc_val = (ar_logits.argmax(dim=1) == action_labels).float().mean()
                ap_loss_val = getattr(args, 'weight_action_preservation', 1.0) * ap_loss_val

                # Feature consistency loss - reuse decoded_features instead of re-encoding
                orig_feats = action_features.detach()
                if decoded_features_for_fc.dim() == 3:
                    decoded_features_for_fc = decoded_features_for_fc.mean(dim=0)
                if orig_feats.dim() == 3:
                    orig_feats = orig_feats.mean(dim=0)
                orig_norm = F.normalize(orig_feats, dim=1)
                decoded_norm = F.normalize(decoded_features_for_fc, dim=1)
                fc_loss_val = 1 - F.cosine_similarity(orig_norm, decoded_norm, dim=1).mean()
                fc_loss_val = getattr(args, 'weight_feature_consistency', 0.5) * fc_loss_val

                # Motion dynamics loss
                md_loss_val = motion_dynamics_loss(output, y2_target)
                md_loss_val = getattr(args, 'weight_motion_dynamics', 0.1) * md_loss_val

                ap_steps += 1

            # Total loss (scaled for gradient accumulation)
            loss_vq, vq_info = get_vq_loss(model, args)
            loss = recon_loss_val + phys_loss_val + frozen_sgn_loss_val + loss_vq
            loss = loss + ap_loss_val + fc_loss_val + md_loss_val

            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}!")
                print(f"  recon: {recon_loss_val}")
                print(f"  phys: {phys_loss_val}")
                print(f"  sgn: {frozen_sgn_loss_val}")
                print(f"  vq: {loss_vq}")
                print(f"  ap: {ap_loss_val}")
                print(f"  fc: {fc_loss_val}")
                print(f"  md: {md_loss_val}")

            # Backward (scale loss for gradient accumulation)
            scaled_loss = loss / accum_steps
            if (batch_idx % accum_steps == 0):
                for opt in optimizers.values():
                    opt.zero_grad()
            scaled_loss.backward()

            # Step optimizers at accumulation boundary
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if args.use_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_value)
                for opt in optimizers.values():
                    opt.step()

            # Accumulate losses
            total_loss += loss.item()
            mse_val = loss_dict.get('mse', 0.0)
            bone_val = loss_dict.get('bone', 0.0)
            total_recon_mse += mse_val.item() if isinstance(mse_val, torch.Tensor) else mse_val
            total_bone_loss += bone_val.item() if isinstance(bone_val, torch.Tensor) else bone_val
            total_physical_loss += phys_dict['total_physical']
            if frozen_sgn_loss is not None:
                total_frozen_sgn_loss += frozen_sgn_loss_val.item() if isinstance(frozen_sgn_loss_val, torch.Tensor) else frozen_sgn_loss_val
                total_frozen_sgn_acc += frozen_sgn_acc
            if action_preservation_loss is not None:
                total_action_preservation += ap_loss_val.item() if isinstance(ap_loss_val, torch.Tensor) else ap_loss_val
                total_feature_consistency += fc_loss_val.item() if isinstance(fc_loss_val, torch.Tensor) else fc_loss_val
                total_motion_dynamics += md_loss_val.item() if isinstance(md_loss_val, torch.Tensor) else md_loss_val
                total_action_preservation_acc += ap_acc_val.item() if isinstance(ap_acc_val, torch.Tensor) else ap_acc_val
            if isinstance(loss_vq, torch.Tensor):
                total_vq_loss += loss_vq.item()
                if vq_info and vq_info.get("embed") is not None:
                    total_vq_embed += vq_info["embed"].item()
                if vq_info and vq_info.get("commit") is not None:
                    total_vq_commit += vq_info["commit"].item()
                if vq_info and vq_info.get("perplexity") is not None:
                    total_vq_perplexity += float(vq_info["perplexity"].item())
                if vq_info and vq_info.get("usage") is not None:
                    total_vq_usage += float(vq_info["usage"].item())
                vq_steps += 1

            # Log
            if (batch_idx + 1) % args.log_freq == 0:
                mse_log = mse_val.item() if isinstance(mse_val, torch.Tensor) else mse_val
                bone_log = bone_val.item() if isinstance(bone_val, torch.Tensor) else bone_val
                log_str = (f"  Batch {batch_idx+1}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"MSE={mse_log:.4f}, "
                          f"Bone={bone_log:.4f}, "
                          f"Physical={phys_dict['total_physical']:.4f}")
                if frozen_sgn_loss is not None:
                    sgn_loss_log = frozen_sgn_loss_val.item() if isinstance(frozen_sgn_loss_val, torch.Tensor) else frozen_sgn_loss_val
                    log_str += f", FrozenSGN={sgn_loss_log:.4f} (Acc={frozen_sgn_acc:.3f})"
                if action_preservation_loss is not None:
                    ap_log = ap_loss_val.item() if isinstance(ap_loss_val, torch.Tensor) else ap_loss_val
                    fc_log = fc_loss_val.item() if isinstance(fc_loss_val, torch.Tensor) else fc_loss_val
                    md_log = md_loss_val.item() if isinstance(md_loss_val, torch.Tensor) else md_loss_val
                    ap_acc_log = ap_acc_val.item() if isinstance(ap_acc_val, torch.Tensor) else ap_acc_val
                    log_str += f", AP={ap_log:.4f} (Acc={ap_acc_log:.3f}), FC={fc_log:.4f}, MD={md_log:.4f}"
                if isinstance(loss_vq, torch.Tensor):
                    log_str += f", VQ={loss_vq.item():.4f}"
                print(log_str)

        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_recon_mse / len(train_loader)
        avg_bone = total_bone_loss / len(train_loader)
        avg_physical = total_physical_loss / len(train_loader)
        avg_frozen_sgn = total_frozen_sgn_loss / len(train_loader) if frozen_sgn_loss is not None else 0.0
        avg_frozen_sgn_acc = total_frozen_sgn_acc / len(train_loader) if frozen_sgn_loss is not None else 0.0
        avg_action_preservation = total_action_preservation / ap_steps if ap_steps > 0 else 0.0
        avg_feature_consistency = total_feature_consistency / ap_steps if ap_steps > 0 else 0.0
        avg_motion_dynamics = total_motion_dynamics / ap_steps if ap_steps > 0 else 0.0
        avg_action_preservation_acc = total_action_preservation_acc / ap_steps if ap_steps > 0 else 0.0
        avg_vq_loss = total_vq_loss / vq_steps if vq_steps > 0 else 0.0
        avg_vq_embed = total_vq_embed / vq_steps if vq_steps > 0 else 0.0
        avg_vq_commit = total_vq_commit / vq_steps if vq_steps > 0 else 0.0
        avg_vq_perplexity = total_vq_perplexity / vq_steps if vq_steps > 0 else 0.0
        avg_vq_usage = total_vq_usage / vq_steps if vq_steps > 0 else 0.0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  MSE Loss: {avg_mse:.4f}")
        print(f"  Bone Loss: {avg_bone:.4f}")
        print(f"  Physical Loss: {avg_physical:.4f}")
        if vq_steps > 0:
            print(f"  VQ Loss: {avg_vq_loss:.4f}")
            print(f"  VQ Embed Loss: {avg_vq_embed:.4f}")
            print(f"  VQ Commit Loss: {avg_vq_commit:.4f}")
            print(f"  VQ Perplexity: {avg_vq_perplexity:.2f}")
            print(f"  VQ Code Usage: {avg_vq_usage:.2f}")
        if frozen_sgn_loss is not None:
            print(f"  Frozen SGN Loss: {avg_frozen_sgn:.4f} (Acc: {avg_frozen_sgn_acc:.3f})")
        print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")

        # Validation
        val_metrics = validate_stage2(model, recon_loss, physical_loss, physical_weights, frozen_sgn_loss, val_loader, args)
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  Val Bone Loss: {val_metrics['bone']:.4f}")
        print(f"  Val Physical Loss: {val_metrics['physical']:.4f}")
        if frozen_sgn_loss is not None:
            print(f"  Val Frozen SGN Acc: {val_metrics['frozen_sgn_acc']:.3f}")

        # Downstream AR evaluation (every N epochs)
        downstream_ar_metrics = {}
        if use_downstream_early_stop and (epoch + 1) % downstream_eval_freq == 0:
            print("  Evaluating downstream AR...")
            downstream_ar_metrics = evaluate_downstream_ar(model, auxiliary_models, val_loader, args)
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics) if downstream_ar_metrics else 0.0
            print(f"  Downstream AR (avg): {avg_downstream_ar:.4f}")
            for model_name, acc in downstream_ar_metrics.items():
                print(f"    {model_name}: {acc:.4f}")

        # Log to wandb
        log_dict = {
            'stage2/epoch': epoch + 1,
            'stage2/train_loss': avg_loss,
            'stage2/train_mse': avg_mse,
            'stage2/train_bone': avg_bone,
            'stage2/train_physical': avg_physical,
            'stage2/teacher_forcing_ratio': teacher_forcing_ratio,
            'stage2/val_mse': val_metrics['mse'],
            'stage2/val_bone': val_metrics['bone'],
            'stage2/val_physical': val_metrics['physical'],
        }
        if vq_steps > 0:
            log_dict.update(
                {
                    "stage2/train_vq_loss": avg_vq_loss,
                    "stage2/train_vq_embed": avg_vq_embed,
                    "stage2/train_vq_commit": avg_vq_commit,
                    "stage2/train_vq_perplexity": avg_vq_perplexity,
                    "stage2/train_vq_usage": avg_vq_usage,
                }
            )
        if frozen_sgn_loss is not None:
            log_dict['stage2/train_frozen_sgn'] = avg_frozen_sgn
            log_dict['stage2/train_frozen_sgn_acc'] = avg_frozen_sgn_acc
            log_dict['stage2/val_frozen_sgn_acc'] = val_metrics['frozen_sgn_acc']
        
        # Add downstream AR metrics to wandb
        if downstream_ar_metrics:
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            log_dict['stage2/downstream_ar_avg'] = avg_downstream_ar
            for model_name, acc in downstream_ar_metrics.items():
                log_dict[f'stage2/downstream_ar_{model_name}'] = acc
        
        if not args.no_wandb:
            wandb.log(log_dict)

        # Early stopping logic
        is_best = False
        
        if use_downstream_early_stop and downstream_ar_metrics:
            # Use downstream AR for early stopping
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            if avg_downstream_ar > best_downstream_ar:
                best_downstream_ar = avg_downstream_ar
                is_best = True
                epochs_without_improvement = 0
                print(f"  New best downstream AR: {best_downstream_ar:.4f}")
            else:
                epochs_without_improvement += 1
        else:
            # Use reconstruction loss for early stopping
            if val_metrics['mse'] < best_recon_loss:
                best_recon_loss = val_metrics['mse']
                is_best = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if (epoch + 1) % args.save_freq == 0 or is_best:
            from train_disentangled_tmr import save_checkpoint
            
            # Prepare best metrics and training state for checkpoint
            best_metrics = {
                'best_recon_loss': best_recon_loss,
                'best_downstream_ar': best_downstream_ar if use_downstream_early_stop else 0.0,
            }
            training_state = {
                'epochs_without_improvement': epochs_without_improvement,
                'patience': patience,
                'use_downstream_early_stop': use_downstream_early_stop,
                'downstream_eval_freq': downstream_eval_freq,
            }
            
            save_checkpoint(model, None, None, optimizers, epoch, stage=2, 
                          args=args, is_best=is_best, best_metrics=best_metrics, 
                          training_state=training_state)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best reconstruction MSE: {best_recon_loss:.4f}")
            break

    print(f"\nStage 2 complete! Best reconstruction MSE: {best_recon_loss:.4f}")


def validate_stage2(model, recon_loss, physical_loss, physical_weights, frozen_sgn_loss, val_loader, args):
    """Validate Stage 2"""
    model.eval()

    total_mse = 0.0
    total_bone = 0.0
    total_physical = 0.0
    total_frozen_sgn_acc = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y1 = y1.to(args.device)
            y2 = y2.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            # Forward pass without teacher forcing (inference-time setup)
            output, _, _ = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

            # Compute losses
            # Note: output is (B, C, T-1, V, M), so slice target to match
            y2_target = y2[:, :, 1:, :, :]  # Remove first frame to match output shape
            loss, loss_dict = recon_loss.loss(output, y2_target, x1)
            mse_val = loss_dict.get('mse', 0.0)
            bone_val = loss_dict.get('bone', 0.0)
            total_mse += mse_val.item() if isinstance(mse_val, torch.Tensor) else mse_val
            total_bone += bone_val.item() if isinstance(bone_val, torch.Tensor) else bone_val

            # Physical plausibility
            _, phys_dict = physical_loss(output, y2_target, physical_weights)
            total_physical += phys_dict['total_physical']

            # Frozen SGN accuracy (if enabled)
            if frozen_sgn_loss is not None:
                first_frame = x2[:, :, 0:1, :, :]
                output_padded = torch.cat([first_frame, output], dim=2)
                action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)
                _, frozen_sgn_acc = frozen_sgn_loss(output_padded, action_labels)
                total_frozen_sgn_acc += frozen_sgn_acc

    return {
        'mse': total_mse / len(val_loader),
        'bone': total_bone / len(val_loader),
        'physical': total_physical / len(val_loader),
        'frozen_sgn_acc': total_frozen_sgn_acc / len(val_loader) if frozen_sgn_loss is not None else 0.0
    }


def train_stage3(model, ar_classifier, disentangle_losses, recon_loss,
                physical_loss, physical_weights, frozen_sgn_loss,
                train_loader, val_loader, optimizers, args, auxiliary_models=None, logger=None,
                ri_classifier=None, mirage_losses=None):
    """
    Stage 3: End-to-End Fine-Tuning

    Goal: Optimize entire pipeline for action preservation and privacy
          Generate physically plausible and recognizable motion

    What's trained:
    - Everything (action encoder, identity encoder, decoder)

    What's frozen:
    - Frozen SGN (if enabled)
    """
    model.unfreeze_all()

    best_ar_accuracy = 0.0
    best_ri_accuracy = 1.0  # Lower is better, start at worst
    best_downstream_ar = 0.0
    epochs_without_improvement = 0
    patience = getattr(args, 'early_stop_patience', 10)
    downstream_eval_freq = getattr(args, 'downstream_eval_freq', 5)
    use_downstream_early_stop = getattr(args, 'use_downstream_early_stop', False) and auxiliary_models is not None

    # Initialize stage logging
    if logger:
        logger.start_stage(3, args.stage3_epochs, "Stage 3: End-to-End Fine-Tuning")

    for epoch in range(args.stage3_epochs):
        print(f"\n--- Stage 3, Epoch {epoch+1}/{args.stage3_epochs} ---")

        # Training
        model.train()
        ar_classifier.train()
        if mirage_losses is not None:
            mirage_losses.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_ar_loss = 0.0
        total_disentangle_loss = 0.0
        total_physical_loss = 0.0
        total_frozen_sgn_loss = 0.0
        total_frozen_sgn_acc = 0.0
        total_vq_loss = 0.0
        total_vq_embed = 0.0
        total_vq_commit = 0.0
        total_vq_perplexity = 0.0
        total_vq_usage = 0.0
        vq_steps = 0
        ar_correct = 0
        total_samples = 0

        # Gradient accumulation
        accum_steps = getattr(args, 'gradient_accumulation_steps', 1)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", mininterval=30.0)):
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y1 = y1.to(args.device)
            y2 = y2.to(args.device)
            actors = actors.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)
            identity_labels = (actors[:, 1] - 1).long()

            # === 1. Forward pass ===
            # FIXED: Use x2 as target_skeleton (P2, A2) and y2 as target_motion (P2, A1)
            teacher_forcing_ratio = teacher_forcing_schedule(
                epoch,
                args.stage3_epochs,
                getattr(args, "stage3_teacher_forcing_start", 0.5),
                getattr(args, "stage3_teacher_forcing_end", 0.3),
            )
            output, action_features, identity_features = model(x1, x2, y2, teacher_forcing_ratio)

            # === 1b. Post-hoc beta blending: output = beta*retargeted + (1-beta)*source ===
            beta = getattr(args, 'beta', 1.0)
            if beta is not None and beta < 1.0:
                x1_sliced = x1[:, :, 1:, :, :]  # match output temporal dim
                output = beta * output + (1.0 - beta) * x1_sliced

            # === 2. Reconstruction losses ===
            # Note: output is (B, C, T-1, V, M), so slice target to match
            y2_target = y2[:, :, 1:, :, :]  # Remove first frame to match output shape
            loss_recon, recon_loss_dict = recon_loss.loss(output, y2_target, x1)

            # === 3. AR utility loss (on generated motion) ===
            # Evaluate AR on generated motion to ensure action is preserved
            ar_logits = ar_classifier(action_features)
            loss_ar = F.cross_entropy(ar_logits, action_labels) * args.weight_ar

            ar_preds = ar_logits.argmax(dim=1)
            ar_correct += (ar_preds == action_labels).sum().item()

            # === 4. Disentanglement losses ===
            disentangle_dict = disentangle_losses.compute_all(
                action_features, identity_features, action_labels, identity_labels,
                train_discriminator=False
            )

            loss_contrastive = disentangle_dict['contrastive']
            loss_adversarial = disentangle_dict['adversarial']
            loss_orthogonality = disentangle_dict['orthogonality']
            loss_mutual_info = disentangle_dict['mutual_info']

            loss_disentangle = (
                args.weight_contrastive * loss_contrastive +
                args.weight_adversarial * loss_adversarial +
                args.weight_orthogonality * loss_orthogonality +
                args.weight_mutual_info * loss_mutual_info
            )

            # === 5. Physical plausibility losses ===
            loss_physical, phys_dict = physical_loss(output, y2_target, physical_weights)

            # === 6. Frozen SGN auxiliary loss (if enabled) ===
            loss_frozen_sgn = 0.0
            frozen_sgn_acc = 0.0
            if frozen_sgn_loss is not None:
                # Pad first frame to make 64 frames for SGN
                first_frame = x2[:, :, 0:1, :, :]
                output_padded = torch.cat([first_frame, output], dim=2)

                loss_frozen_sgn, frozen_sgn_acc = frozen_sgn_loss(output_padded, action_labels)
                loss_frozen_sgn = args.weight_frozen_sgn * loss_frozen_sgn

            # === 7. Total loss ===
            loss_vq, vq_info = get_vq_loss(model, args)

            # === 8. MIRAGE-inspired losses (if enabled) ===
            loss_mirage_gen = 0.0
            loss_mirage_disc = 0.0
            mirage_log = {}
            if mirage_losses is not None:
                mirage_results = mirage_losses.compute_all(
                    output, y2_target, action_labels, identity_labels
                )
                loss_mirage_gen = mirage_results["gen_total"]
                loss_mirage_disc = mirage_results["disc_total"]
                mirage_log = {
                    "dist_disc_acc": mirage_results["disc_acc"],
                    "output_act_acc": mirage_results["output_act_acc"],
                    "output_id_acc": mirage_results["output_id_acc"],
                    "gen_disc_loss": mirage_results["gen_disc_loss"].item(),
                    "output_act_loss": mirage_results["output_act_loss"].item(),
                    "output_id_adv_loss": mirage_results["output_id_adv_loss"].item(),
                    "output_contrastive_loss": mirage_results["output_contrastive_loss"].item(),
                    "ee_enhanced_loss": mirage_results["ee_enhanced_loss"].item(),
                    "motion_disc_acc": mirage_results.get("motion_disc_acc", 0.0),
                    "gen_motion_disc_loss": mirage_results["gen_motion_disc_loss"].item() if isinstance(mirage_results.get("gen_motion_disc_loss"), torch.Tensor) else 0.0,
                    "coord_std_loss": mirage_results["coord_std_loss"].item() if isinstance(mirage_results.get("coord_std_loss"), torch.Tensor) else 0.0,
                }
            loss = loss_recon + loss_ar + loss_disentangle + loss_physical + loss_frozen_sgn + loss_vq + loss_mirage_gen


            # Generator backward FIRST (needs full graph)
            _mirage_opt_keys = {'mirage_disc', 'mirage_act_cls', 'mirage_contrastive'}
            scaled_loss = loss / accum_steps
            if (batch_idx % accum_steps == 0):
                for k, opt in optimizers.items():
                    if k not in _mirage_opt_keys:
                        opt.zero_grad()
            scaled_loss.backward()

            # MIRAGE discriminator step AFTER generator (uses detached output, no graph conflict)
            if mirage_losses is not None and isinstance(loss_mirage_disc, torch.Tensor):
                scaled_disc_loss = loss_mirage_disc / accum_steps
                if "mirage_disc" in optimizers:
                    optimizers["mirage_disc"].zero_grad()
                if "mirage_act_cls" in optimizers:
                    optimizers["mirage_act_cls"].zero_grad()
                if "mirage_contrastive" in optimizers:
                    optimizers["mirage_contrastive"].zero_grad()
                scaled_disc_loss.backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if "mirage_disc" in optimizers:
                        optimizers["mirage_disc"].step()
                    if "mirage_act_cls" in optimizers:
                        optimizers["mirage_act_cls"].step()
                    if "mirage_contrastive" in optimizers:
                        optimizers["mirage_contrastive"].step()

            # Step optimizers at accumulation boundary
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if args.use_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_value)
                for k, opt in optimizers.items():
                    if k not in _mirage_opt_keys:
                        opt.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_ar_loss += loss_ar.item()
            total_disentangle_loss += loss_disentangle.item()
            total_physical_loss += phys_dict['total_physical']
            if frozen_sgn_loss is not None:
                total_frozen_sgn_loss += loss_frozen_sgn.item() if isinstance(loss_frozen_sgn, torch.Tensor) else loss_frozen_sgn
                total_frozen_sgn_acc += frozen_sgn_acc
            if isinstance(loss_vq, torch.Tensor):
                total_vq_loss += loss_vq.item()
                if vq_info and vq_info.get("embed") is not None:
                    total_vq_embed += vq_info["embed"].item()
                if vq_info and vq_info.get("commit") is not None:
                    total_vq_commit += vq_info["commit"].item()
                if vq_info and vq_info.get("perplexity") is not None:
                    total_vq_perplexity += float(vq_info["perplexity"].item())
                if vq_info and vq_info.get("usage") is not None:
                    total_vq_usage += float(vq_info["usage"].item())
                vq_steps += 1
            total_samples += x1.size(0)

            # Log
            if (batch_idx + 1) % args.log_freq == 0:
                log_str = (f"  Batch {batch_idx+1}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"Recon={loss_recon.item():.4f}, "
                          f"AR={loss_ar.item():.4f}, "
                          f"Disentangle={loss_disentangle.item():.4f}, "
                          f"Physical={phys_dict['total_physical']:.4f}")
                if frozen_sgn_loss is not None:
                    sgn_loss_log = loss_frozen_sgn.item() if isinstance(loss_frozen_sgn, torch.Tensor) else loss_frozen_sgn
                    log_str += f", FrozenSGN={sgn_loss_log:.4f} (Acc={frozen_sgn_acc:.3f})"
                if isinstance(loss_vq, torch.Tensor):
                    log_str += f", VQ={loss_vq.item():.4f}"
                print(log_str)
                if mirage_losses is not None and mirage_log:
                    _mg = loss_mirage_gen.item() if isinstance(loss_mirage_gen, torch.Tensor) else loss_mirage_gen
                    _dd = mirage_log.get('dist_disc_acc', 0)
                    _oa = mirage_log.get('output_act_acc', 0)
                    _oi = mirage_log.get('output_id_acc', 0)
                    print(f'  MIRAGE: gen={_mg:.4f}, dist_disc_acc={_dd:.3f}, out_act_acc={_oa:.3f}, out_id_acc={_oi:.3f}')

        # Epoch summary
        ar_accuracy = ar_correct / total_samples
        avg_physical = total_physical_loss / len(train_loader)
        avg_frozen_sgn = total_frozen_sgn_loss / len(train_loader) if frozen_sgn_loss is not None else 0.0
        avg_frozen_sgn_acc = total_frozen_sgn_acc / len(train_loader) if frozen_sgn_loss is not None else 0.0
        # avg_action_preservation = total_action_preservation / ap_steps if ap_steps > 0 else 0.0
        # avg_feature_consistency = total_feature_consistency / ap_steps if ap_steps > 0 else 0.0
        # avg_motion_dynamics = total_motion_dynamics / ap_steps if ap_steps > 0 else 0.0
        # avg_action_preservation_acc = total_action_preservation_acc / ap_steps if ap_steps > 0 else 0.0
        avg_vq_loss = total_vq_loss / vq_steps if vq_steps > 0 else 0.0
        avg_vq_embed = total_vq_embed / vq_steps if vq_steps > 0 else 0.0
        avg_vq_commit = total_vq_commit / vq_steps if vq_steps > 0 else 0.0
        avg_vq_perplexity = total_vq_perplexity / vq_steps if vq_steps > 0 else 0.0
        avg_vq_usage = total_vq_usage / vq_steps if vq_steps > 0 else 0.0

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Recon Loss: {total_recon_loss/len(train_loader):.4f}")
        print(f"  AR Loss: {total_ar_loss/len(train_loader):.4f}")
        print(f"  Disentangle Loss: {total_disentangle_loss/len(train_loader):.4f}")
        print(f"  Physical Loss: {avg_physical:.4f}")
        if vq_steps > 0:
            print(f"  VQ Loss: {avg_vq_loss:.4f}")
            print(f"  VQ Embed Loss: {avg_vq_embed:.4f}")
            print(f"  VQ Commit Loss: {avg_vq_commit:.4f}")
            print(f"  VQ Perplexity: {avg_vq_perplexity:.2f}")
            print(f"  VQ Code Usage: {avg_vq_usage:.2f}")
        if frozen_sgn_loss is not None:
            print(f"  Frozen SGN Loss: {avg_frozen_sgn:.4f} (Acc: {avg_frozen_sgn_acc:.3f})")
        print(f"  AR Accuracy: {ar_accuracy:.4f}")
        print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")

        # Validation
        val_metrics = validate_stage3(model, ar_classifier, recon_loss, physical_loss, physical_weights, frozen_sgn_loss, val_loader, args, ri_classifier=ri_classifier)
        print(f"  Val AR Accuracy: {val_metrics['ar_accuracy']:.4f}")
        if ri_classifier is not None:
            print(f"  Val RI Accuracy: {val_metrics['ri_accuracy']:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  Val Physical Loss: {val_metrics['physical']:.4f}")
        if frozen_sgn_loss is not None:
            print(f"  Val Frozen SGN Acc: {val_metrics['frozen_sgn_acc']:.3f}")

        # Downstream AR evaluation (every N epochs)
        downstream_ar_metrics = {}
        if use_downstream_early_stop and (epoch + 1) % downstream_eval_freq == 0:
            print("  Evaluating downstream AR...")
            downstream_ar_metrics = evaluate_downstream_ar(model, auxiliary_models, val_loader, args)
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics) if downstream_ar_metrics else 0.0
            print(f"  Downstream AR (avg): {avg_downstream_ar:.4f}")
            for model_name, acc in downstream_ar_metrics.items():
                print(f"    {model_name}: {acc:.4f}")

        # Log to wandb
        wandb_log = {
            'stage3/epoch': epoch + 1,
            'stage3/train_loss': total_loss/len(train_loader),
            'stage3/train_recon_loss': total_recon_loss/len(train_loader),
            'stage3/train_ar_loss': total_ar_loss/len(train_loader),
            'stage3/train_disentangle_loss': total_disentangle_loss/len(train_loader),
            'stage3/train_ar_accuracy': ar_accuracy,
            'stage3/teacher_forcing_ratio': teacher_forcing_ratio,
            'stage3/val_ar_accuracy': val_metrics['ar_accuracy'],
            'stage3/val_ri_accuracy': val_metrics.get('ri_accuracy', 0.0),
            'stage3/val_mse': val_metrics['mse'],
        }
        if vq_steps > 0:
            wandb_log.update(
                {
                    "stage3/train_vq_loss": avg_vq_loss,
                    "stage3/train_vq_embed": avg_vq_embed,
                    "stage3/train_vq_commit": avg_vq_commit,
                    "stage3/train_vq_perplexity": avg_vq_perplexity,
                    "stage3/train_vq_usage": avg_vq_usage,
                }
            )
        if mirage_losses is not None and mirage_log:
            wandb_log.update({
                "stage3/mirage_dist_disc_acc": mirage_log.get("dist_disc_acc", 0),
                "stage3/mirage_output_act_acc": mirage_log.get("output_act_acc", 0),
                "stage3/mirage_output_id_acc": mirage_log.get("output_id_acc", 0),
                "stage3/mirage_gen_disc_loss": mirage_log.get("gen_disc_loss", 0),
                "stage3/mirage_output_act_loss": mirage_log.get("output_act_loss", 0),
                "stage3/mirage_output_id_adv_loss": mirage_log.get("output_id_adv_loss", 0),
                "stage3/mirage_contrastive_loss": mirage_log.get("output_contrastive_loss", 0),
                "stage3/mirage_ee_enhanced_loss": mirage_log.get("ee_enhanced_loss", 0),
            })
        
        # Add downstream AR metrics to wandb
        if downstream_ar_metrics:
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            wandb_log['stage3/downstream_ar_avg'] = avg_downstream_ar
            for model_name, acc in downstream_ar_metrics.items():
                wandb_log[f'stage3/downstream_ar_{model_name}'] = acc
        
        if not args.no_wandb:
            wandb.log(wandb_log)

        # Early stopping logic
        is_best = False
        
        if use_downstream_early_stop and downstream_ar_metrics:
            # Use downstream AR for early stopping
            avg_downstream_ar = sum(downstream_ar_metrics.values()) / len(downstream_ar_metrics)
            if avg_downstream_ar > best_downstream_ar:
                best_downstream_ar = avg_downstream_ar
                is_best = True
                epochs_without_improvement = 0
                print(f"  New best downstream AR: {best_downstream_ar:.4f}")
            else:
                epochs_without_improvement += 1
        else:
            # Use validation AR for early stopping
            if val_metrics['ar_accuracy'] > best_ar_accuracy:
                best_ar_accuracy = val_metrics['ar_accuracy']
                is_best = True
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Track best RI (lower is better)
        val_ri = val_metrics.get('ri_accuracy', 1.0)
        if val_ri < best_ri_accuracy:
            best_ri_accuracy = val_ri

        if (epoch + 1) % args.save_freq == 0 or is_best:
            from train_disentangled_tmr import save_checkpoint

            # Compute validation loss as combination of MSE and negative AR accuracy
            # Lower is better: high MSE = bad, low AR accuracy = bad
            val_loss = val_metrics['mse'] + (1.0 - val_metrics['ar_accuracy']) * 10.0

            # Prepare best metrics and training state for checkpoint
            best_metrics = {
                'best_ar_accuracy': best_ar_accuracy,
                'best_ri_accuracy': best_ri_accuracy,
                'best_downstream_ar': best_downstream_ar if use_downstream_early_stop else 0.0,
            }
            training_state = {
                'epochs_without_improvement': epochs_without_improvement,
                'patience': patience,
                'use_downstream_early_stop': use_downstream_early_stop,
                'downstream_eval_freq': downstream_eval_freq,
            }
            
            save_checkpoint(model, ar_classifier, None, optimizers, epoch, stage=3, 
                          args=args, is_best=is_best, val_loss=val_loss, 
                          best_metrics=best_metrics, training_state=training_state)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best AR accuracy: {best_ar_accuracy:.4f}")
            break

    print(f"\nStage 3 complete! Best AR accuracy: {best_ar_accuracy:.4f}")
    if ri_classifier is not None:
        print(f"  Best RI accuracy: {best_ri_accuracy:.4f}")
    return best_ar_accuracy, best_ri_accuracy


def validate_stage3(model, ar_classifier, recon_loss, physical_loss, physical_weights, frozen_sgn_loss, val_loader, args, ri_classifier=None):
    """Validate Stage 3"""
    model.eval()
    ar_classifier.eval()
    if ri_classifier is not None:
        ri_classifier.eval()

    ar_correct = 0
    ri_correct = 0
    total_samples = 0
    total_mse = 0.0
    total_physical = 0.0
    total_frozen_sgn_acc = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x1, x2, y1, y2, actors, actions = batch
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y1 = y1.to(args.device)
            y2 = y2.to(args.device)
            actors = actors.to(args.device)
            actions = actions.to(args.device)

            # Add M dimension
            x1 = x1.unsqueeze(-1)
            x2 = x2.unsqueeze(-1)
            y1 = y1.unsqueeze(-1)
            y2 = y2.unsqueeze(-1)

            action_labels = remap_action_labels((actions[:, 0] - 1).long(), args.dataset)
            identity_labels = (actors[:, 0] - 1).long()

            # Forward pass (inference-time setup)
            output, action_features, identity_features = model(x1, x2, target_motion=None, teacher_forcing_ratio=0.0)

            # Debug: log shapes on first batch to catch dimension mismatches early
            if total_samples == 0:
                print(f"  [validate_stage3] action_features: {action_features.shape}, identity_features: {identity_features.shape}")

            # AR accuracy
            ar_logits = ar_classifier(action_features)
            ar_preds = ar_logits.argmax(dim=1)
            ar_correct += (ar_preds == action_labels).sum().item()

            # RI accuracy (if classifier available)
            if ri_classifier is not None:
                # Handle temporal dimension if present (full-seq identity mode)
                id_feats = identity_features
                if id_feats.dim() == 3:
                    id_feats = id_feats.mean(dim=0)  # (T, B, D) -> (B, D)
                ri_logits = ri_classifier(id_feats)
                ri_preds = ri_logits.argmax(dim=1)
                ri_correct += (ri_preds == identity_labels).sum().item()

            # MSE
            # Note: output is (B, C, T-1, V, M), so slice target to match
            y2_target = y2[:, :, 1:, :, :]  # Remove first frame to match output shape
            loss, loss_dict = recon_loss.loss(output, y2_target, x1)
            mse_val = loss_dict.get('mse', 0.0)
            total_mse += mse_val.item() if isinstance(mse_val, torch.Tensor) else mse_val

            # Physical plausibility
            _, phys_dict = physical_loss(output, y2_target, physical_weights)
            total_physical += phys_dict['total_physical']

            # Frozen SGN accuracy (if enabled)
            if frozen_sgn_loss is not None:
                first_frame = x2[:, :, 0:1, :, :]
                output_padded = torch.cat([first_frame, output], dim=2)
                _, frozen_sgn_acc = frozen_sgn_loss(output_padded, action_labels)
                total_frozen_sgn_acc += frozen_sgn_acc

            total_samples += x1.size(0)

    return {
        'ar_accuracy': ar_correct / total_samples,
        'ri_accuracy': ri_correct / total_samples if ri_classifier is not None else 0.0,
        'mse': total_mse / len(val_loader),
        'physical': total_physical / len(val_loader),
        'frozen_sgn_acc': total_frozen_sgn_acc / len(val_loader) if frozen_sgn_loss is not None else 0.0
    }
