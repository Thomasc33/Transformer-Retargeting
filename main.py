import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import pickle
import os
import json
import time
import logging
from datetime import datetime, timedelta
from model.autoencoder import Model
# from train import Trainer  # Using OptimizedTrainer instead
from data import get_cross_data, load_data, optimize_data_loading, estimate_memory_usage
from util import init_seed
import argparse
import warnings
import yaml

# Suppress CUDNN warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings("ignore", message=".*CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

# Set CUDNN configuration for stability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_bool_env(varname, default=False):
    """
    Reads an environment variable (like 'true' / '1') as boolean.
    """
    val = os.environ.get(varname, str(default)).lower()
    return val in ['true', '1', 't', 'y', 'yes']

# Load hyperparameters from study results
def load_best_hyperparameters():
    """Load the best hyperparameters from the study results."""
    study_results_path = "experiments/hyperparameter/results/study_results_20250430_013324.json"
    if os.path.exists(study_results_path):
        try:
            with open(study_results_path, 'r') as f:
                study_data = json.load(f)
                return study_data['best_params']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load hyperparameters from {study_results_path}: {e}")
            return {}
    else:
        print(f"Warning: Study results file not found at {study_results_path}")
        return {}





def setup_logging(log_dir, rank):
    """Setup comprehensive logging system."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger, log_file
    else:
        return None, None

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, rank, is_best=False):
    """Save training checkpoint."""
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get model state dict (handle DDP case)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save epoch-specific checkpoint
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)

        # Save best checkpoint if specified
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_checkpoint_path)

        return checkpoint_path
    return None

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training checkpoint."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dict (handle DDP case)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))

        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

def process_batch(model, x1, x2, y1, y2, loss_fn, device, teacher_forcing_ratio, mixed_precision=False):
    """OPTIMIZED: Process a single batch with improved memory management and performance."""
    # PERFORMANCE: Use non-blocking transfer and pin memory for faster GPU transfer
    with torch.cuda.stream(torch.cuda.current_stream()):
        x1 = x1.float().to(device, non_blocking=True)
        x2 = x2.float().to(device, non_blocking=True)
        y1 = y1.float().to(device, non_blocking=True)
        y2 = y2.float().to(device, non_blocking=True)

    # OPTIMIZED: Faster NaN check using any() instead of all()
    if (torch.isnan(x1).any() or torch.isnan(x2).any() or
        torch.isnan(y1).any() or torch.isnan(y2).any()):
        print("⚠️  NaN detected in input data, returning zero loss")
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        zero_losses = {key: zero_loss.clone() for key in loss_fn.loss_weights.keys()}
        return zero_loss, zero_losses

    # Combine inputs and targets for batch processing
    inputs = torch.cat([x1, y1, x2, y2], dim=0)
    dummy = torch.cat([x2, y2, x1, y1], dim=0)
    targets = torch.cat([y2, x2, y1, x1], dim=0)

    # Free memory of individual tensors
    del x1, x2, y1, y2

    # Get dimensions
    N_total = inputs.size(0)
    T = inputs.size(1)
    M, V, C_in = 1, 25, 3

    # Reshape to (N, C, T, V, M) format
    source_motion = inputs.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
    dummy_skeleton = dummy.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
    target_motion = targets.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

    # Free memory of intermediate tensors
    del inputs, dummy, targets

    # CRITICAL FIX: Disable mixed precision for autoregressive mode OR multi-GPU
    # Mixed precision causes NaN gradients in:
    # 1. Autoregressive generation (teacher_forcing_ratio < 1.0)
    # 2. Multi-GPU training (DDP + mixed precision + attention = NaN)
    use_mixed_precision = mixed_precision and teacher_forcing_ratio >= 1.0

    # Log when mixed precision is disabled due to teacher forcing ratio
    if mixed_precision and teacher_forcing_ratio < 1.0:
        # Only log once per epoch to avoid spam
        if not hasattr(process_batch, '_logged_mp_disable'):
            print(f"🔧 INFO: Mixed precision disabled for teacher_forcing_ratio={teacher_forcing_ratio:.4f}")
            print(f"       Mixed precision only works reliably with pure teacher forcing (ratio=1.0)")
            process_batch._logged_mp_disable = True

    if use_mixed_precision:
        # Only use mixed precision for pure teacher forcing
        with torch.cuda.amp.autocast():
            output = model(source_motion, dummy_skeleton, target_motion=target_motion,
                          teacher_forcing_ratio=teacher_forcing_ratio)

            # Check for NaN/inf in model output
            if not torch.isfinite(output).all():
                print("⚠️  NaN/inf detected in model output, returning zero loss")
                zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                zero_losses = {key: zero_loss.clone() for key in loss_fn.loss_weights.keys()}
                del output, source_motion, dummy_skeleton, target_motion
                return zero_loss, zero_losses

            # Get ground truth next frames
            target = target_motion[:, :, 1:, :, :]
            # Calculate loss
            loss_val, losses_dict = loss_fn.loss(output, target, source_motion[:, :, 1:, :, :])
    else:
        output = model(source_motion, dummy_skeleton, target_motion=target_motion,
                      teacher_forcing_ratio=teacher_forcing_ratio)

        # Check for NaN/inf in model output
        if not torch.isfinite(output).all():
            print("⚠️  NaN/inf detected in model output, returning zero loss")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            zero_losses = {key: zero_loss.clone() for key in loss_fn.loss_weights.keys()}
            del output, source_motion, dummy_skeleton, target_motion
            return zero_loss, zero_losses

        # Get ground truth next frames
        target = target_motion[:, :, 1:, :, :]
        # Calculate loss
        loss_val, losses_dict = loss_fn.loss(output, target, source_motion[:, :, 1:, :, :])

    # Final check for NaN/inf in loss
    if not torch.isfinite(loss_val):
        print("⚠️  NaN/inf detected in loss computation, returning zero loss")
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        zero_losses = {key: zero_loss.clone() for key in loss_fn.loss_weights.keys()}
        del output, target, source_motion, dummy_skeleton, target_motion
        return zero_loss, zero_losses

    # Free memory
    del output, target, source_motion, dummy_skeleton, target_motion

    return loss_val, losses_dict

def train_model(model, optimizer, train_loader, test_loader, loss_fn, num_epochs, device, rank, world_size,
                teacher_forcing_ratio, start_epoch=0, best_loss=float('inf'), checkpoint_dir='checkpoints',
                save_every=1, mixed_precision=False, scaler=None, gradient_accumulation_steps=1,
                max_grad_norm=1.0, scheduler=None, wandb_enabled=False, validate_every=5):
    """
    Integrated training function with detailed progress reporting.
    """
    # Only print on rank 0 or if not distributed
    is_main_process = rank == 0 or world_size == 1

    if is_main_process:
        print("🚀 Starting Training (Autoregressive Motion Retargeting)...")
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {num_epochs} (starting from {start_epoch})")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(test_loader)}")
        print(f"   Teacher forcing ratio: {teacher_forcing_ratio}")
        print(f"   Device: {device}")
        print(f"   World size: {world_size}")
        if mixed_precision:
            print("   Using Automatic Mixed Precision (AMP)")
        if gradient_accumulation_steps > 1:
            print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
        print()



    model.train()

    for epoch in range(start_epoch, num_epochs):
        # Update teacher forcing ratio
        half_epochs = (num_epochs + 1) // 2
        if epoch < half_epochs:
            current_teacher_forcing = 1.0 - (epoch / half_epochs)
        else:
            current_teacher_forcing = 0.0

        # MEMORY FIX: Root cause fixed - gradient accumulation across time steps
        # Now autoregressive mode should use similar memory to teacher forcing
        original_batch_size = train_loader.batch_size
        adaptive_batch_size = original_batch_size  # No reduction needed with fix

        if is_main_process:
            print(f"📈 Epoch {epoch+1}/{num_epochs}")
            print(f"   Teacher forcing ratio: {current_teacher_forcing:.4f}")

            print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Reset logging flags for this epoch
        if hasattr(process_batch, '_logged_mp_disable'):
            delattr(process_batch, '_logged_mp_disable')

        epoch_start = time.time()
        running_loss = 0.0
        running_losses = {key: 0.0 for key in loss_fn.loss_weights.keys()}

        # OPTIMIZED: Progress tracking with reduced frequency for better performance
        total_batches = len(train_loader)
        report_interval = max(1, total_batches // 10)  # Report 10 times per epoch (reduced from 20)
        time_interval = 300  # Report every 5 minutes (reduced from 1 minute)
        last_time_report = time.time()

        # Initialize gradient accumulation
        optimizer.zero_grad(set_to_none=True)



        # Training loop with detailed progress
        for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(train_loader):
            try:
                # Process batch
                loss_val, losses_dict = process_batch(
                    model, x1, x2, y1, y2, loss_fn, device,
                    current_teacher_forcing, mixed_precision
                )

                # Check for NaN loss
                if not torch.isfinite(loss_val):
                    if is_main_process:
                        print(f"⚠️  NaN loss detected at batch {batch_idx}, skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Scale loss for gradient accumulation
                original_loss = loss_val.item()
                loss_val = loss_val / gradient_accumulation_steps
                scaled_loss = loss_val.item()



                # Backward pass
                if mixed_precision and scaler:
                    scaler.scale(loss_val).backward()
                else:
                    loss_val.backward()



                # Update running losses
                running_loss += loss_val.item() * gradient_accumulation_steps
                for key, value in losses_dict.items():
                    if torch.isfinite(value):
                        running_losses[key] += value.item()

                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if mixed_precision and scaler:


                        # Always unscale gradients first to check for NaN/inf before clipping
                        try:
                            scaler.unscale_(optimizer)
                        except RuntimeError as e:
                            if "already unscaled" in str(e):
                                # Gradients already unscaled, continue
                                pass
                            else:
                                raise e



                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        # Check for NaN gradients
                        has_nan_grads = any(
                            param.grad is not None and not torch.isfinite(param.grad).all()
                            for param in model.parameters()
                        )

                        if has_nan_grads:
                            if is_main_process:
                                print(f"⚠️  NaN gradients detected at batch {batch_idx}, skipping step...")
                            # Skip optimizer step but still update scaler
                            scaler.update()
                        else:
                            # Step optimizer and update scaler
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        # Non-mixed precision mode
                        # Check for NaN gradients
                        has_nan_grads = any(
                            param.grad is not None and not torch.isfinite(param.grad).all()
                            for param in model.parameters()
                        )

                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        if has_nan_grads:
                            if is_main_process:
                                print(f"⚠️  NaN gradients detected at batch {batch_idx}, skipping step...")
                            # Skip this optimization step
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        else:
                            # Step optimizer
                            optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                # Progress reporting
                current_time = time.time()
                should_report_batch = batch_idx % report_interval == 0 and batch_idx > 0
                should_report_time = (current_time - last_time_report) >= time_interval

                if is_main_process and (should_report_batch or should_report_time):
                    avg_loss = running_loss / (batch_idx + 1)
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    elapsed = current_time - epoch_start
                    eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)

                    print(f"   📊 Batch {batch_idx+1:5d}/{total_batches} ({progress_pct:5.1f}%) | "
                          f"Loss: {avg_loss:.6f} | "
                          f"Elapsed: {elapsed/60:.1f}m | "
                          f"ETA: {eta/60:.1f}m")

                    if should_report_time:
                        last_time_report = current_time

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if is_main_process:
                        print(f"💾 CUDA OOM at batch {batch_idx}, clearing cache...")
                        if batch_idx < 10:  # If OOM happens early, suggest further reduction
                            current_bs = train_loader.batch_size
                            if current_teacher_forcing < 1.0:
                                print(f"⚠️  Early OOM in autoregressive mode (TF ratio: {current_teacher_forcing:.3f})")
                                print(f"   Current adaptive batch size: {current_bs} (reduced from {original_batch_size})")
                                print(f"   Suggested: Further reduce to --batch-size {current_bs // 2}")
                            else:
                                print(f"⚠️  Early OOM in teacher forcing mode")
                                print(f"   Current batch size: {current_bs}")
                                print(f"   Suggested: --batch-size {current_bs // 2} --gradient-accumulation-steps {gradient_accumulation_steps * 2}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # OPTIMIZED: Less frequent cleanup to avoid performance bottlenecks
            if batch_idx % 200 == 0:  # Reduced frequency from every 50 to every 200 batches
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # REMOVED: Distributed barrier that was causing NCCL timeouts
                # DDP handles synchronization automatically during backward pass
                # Manual barriers are unnecessary and cause performance bottlenecks

                # Clear any cached CUDA operations (local only, no distributed sync)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Epoch completion
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        avg_loss = running_loss / len(train_loader)

        if is_main_process:
            print(f"   ✅ Training complete: {epoch_time/60:.1f}m | Avg Loss: {avg_loss:.6f}")

        # OPTIMIZED: Validation - only run every N epochs to improve performance
        should_validate = (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs

        if should_validate:
            if is_main_process:
                print(f"   🔍 Running validation (epoch {epoch+1})...")

            val_start = time.time()
            val_loss, val_losses = evaluate_model(model, test_loader, loss_fn, device, is_main_process)
            val_end = time.time()
            val_time = val_end - val_start
        else:
            # Skip validation for performance
            # Use a placeholder value to indicate validation was skipped
            val_loss = -1.0  # Placeholder to indicate validation was skipped
            val_losses = {}
            val_time = 0
            if is_main_process:
                print(f"   ⏭️  Skipping validation (will validate every {validate_every} epochs)")


        # Check if this is the best model
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save checkpoint (only on main process)
        if is_main_process:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, rank, is_best)

        # Print epoch summary
        if is_main_process:
            print(f"   📈 Epoch {epoch+1} Summary:")
            val_loss_str = "skipped" if val_loss == -1.0 else f"{val_loss:.6f}"
            best_loss_str = "inf" if best_loss == float('inf') else f"{best_loss:.6f}"
            print(f"      Train Loss: {avg_loss:.6f} | Val Loss: {val_loss_str}")
            print(f"      Train Time: {epoch_time/60:.1f}m | Val Time: {val_time/60:.1f}m")
            print(f"      Best Loss: {best_loss_str} {'🏆' if is_best else ''}")
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"      GPU Memory: {current_memory:.2f}GB")
                mp_status = "Yes" if mixed_precision and current_teacher_forcing >= 1.0 else "No"
                print(f"      Teacher Forcing: {current_teacher_forcing:.3f} | Mixed Precision: {mp_status}")
            print()

        # Log to wandb
        if wandb_enabled and is_main_process:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'teacher_forcing_ratio': current_teacher_forcing,
                    'epoch_time': epoch_time,
                    'val_time': val_time,
                    'best_loss': best_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"⚠️  wandb logging failed: {e}")
                wandb_enabled = False  # Disable further wandb logging

@torch.no_grad()
def evaluate_model(model, test_loader, loss_fn, device, is_main_process):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0.0
    losses = {key: 0.0 for key in loss_fn.loss_weights.keys()}

    total_batches = len(test_loader)

    for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(test_loader):
        try:
            # Process batch for evaluation (no teacher forcing)
            loss_val, losses_dict = process_batch(
                model, x1, x2, y1, y2, loss_fn, device,
                teacher_forcing_ratio=0.0, mixed_precision=False
            )

            # Accumulate losses
            total_loss += loss_val.item()
            for key, val in losses_dict.items():
                losses[key] += val.item()

            # Progress reporting for validation
            if is_main_process and batch_idx % max(1, total_batches // 10) == 0:
                progress_pct = (batch_idx + 1) / total_batches * 100
                print(f"      Val Batch {batch_idx+1:4d}/{total_batches} ({progress_pct:5.1f}%)")

        except RuntimeError as e:
            if "out of memory" in str(e):
                if is_main_process:
                    print(f"💾 CUDA OOM during validation at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # Periodic cleanup
        if batch_idx % 20 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # Calculate average losses
    avg_loss = total_loss / len(test_loader)
    losses = {key: val / len(test_loader) for key, val in losses.items()}

    model.train()
    return avg_loss, losses

def main():
    """Main training function."""
    # Get rank early to control printing
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print('Starting main.py execution...')

    # Load best hyperparameters first
    best_params = load_best_hyperparameters()

    # Only print on rank 0 to avoid duplicate output
    if rank == 0:
        if best_params:
            print(f"✅ Loaded optimized hyperparameters from study results:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  Using default hyperparameters (study results not found)")

    # Get dataset from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ntu', help='Dataset to use (ntu or ntu120 or etri)')
    parser.add_argument('--setting', type=str, default='cv', help='Evaluation setting (cs or cv)')
    parser.add_argument('--hpc', action='store_true', help='Enable HPC distributed mode.')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use for training (default: 4)')
    parser.add_argument('--batch-size', type=int, default=best_params.get('batch_size', 32), help='Batch size for training (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--lr', type=float, default=best_params.get('lr', 9.43062936149491e-05), help='Learning rate (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--train-samples', type=int, default=999999999, help='Number of training samples to use')
    parser.add_argument('--test-samples', type=int, default=10000, help='Number of test samples to use')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0,
                      help='Teacher forcing ratio (1.0=always use teacher forcing, 0.0=never use teacher forcing)')
    parser.add_argument('--teacher-forcing-decay', type=float, default=0.0,
                      help='Teacher forcing decay rate per epoch (0.0=no decay)')
    parser.add_argument('--data-path', type=str, default='data/ntu_cv_paired_comprehensive.pt',
                      help='Path to paired data file. If not provided, will use default naming convention.')
    parser.add_argument('--output-model-path', type=str, default='model.pth',
                      help='Path to save the trained model')
    parser.add_argument('--run-eval', action='store_true', help='Run evaluation after training')

    # Loss weight arguments
    parser.add_argument('--use-pretrained', action='store_true', help='Use pretrained encoder')
    parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false', help='Do not use pretrained encoder')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder parameters')
    parser.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false', help='Do not freeze encoder parameters')
    parser.set_defaults(use_pretrained=True, freeze_encoder=True)

    # Loss weight arguments - OPTIMIZED: Using hyperparameter tuning results (trial 17)
    parser.add_argument('--loss-mse', type=float, default=5.323284271000699, help='Weight for MSE loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-l1', type=float, default=0.0, help='Weight for L1 loss')
    parser.add_argument('--loss-smoothl1', type=float, default=0.0, help='Weight for Smooth L1 loss')
    parser.add_argument('--loss-kl', type=float, default=0.0, help='Weight for KL loss')
    parser.add_argument('--loss-ce', type=float, default=0.0, help='Weight for Cross Entropy loss')
    parser.add_argument('--loss-ee', type=float, default=4.7250245075017725, help='Weight for End-Effector loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-smoothing', type=float, default=0.28747697025246937, help='Weight for Smoothing loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-latent', type=float, default=0.0, help='Weight for Latent loss')
    parser.add_argument('--loss-triplet', type=float, default=0.0, help='Weight for Triplet loss')
    parser.add_argument('--loss-inception', type=float, default=0.30656068713914353, help='Weight for Inception loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-fid-vel', type=float, default=1.1304168895721447, help='Weight for FID Velocity loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-bone', type=float, default=0.6185377286837532, help='Weight for Bone Length loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-foot', type=float, default=0.0544125368059208, help='Weight for Foot Contact loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-joint-limit', type=float, default=0.01768344443841404, help='Weight for Joint Limit loss (OPTIMIZED from hyperparameter tuning)')
    parser.add_argument('--loss-weights', type=str, default=None, help='Comma-separated loss weights in format "mse:1.0,ee:2.0,..." (overrides individual loss arguments)')
    parser.add_argument('--decoder-dropout', type=float, default=best_params.get('decoder_dropout', 0.11551063114920847), help='Dropout rate for the decoder')
    parser.add_argument('--use-checkpoint', action='store_true', help='Use gradient checkpointing to save memory')
    parser.add_argument('--no-checkpoint', dest='use_checkpoint', action='store_false', help='Do not use gradient checkpointing')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--mixed-precision', action='store_true', help='Use automatic mixed precision training (DISABLED by default due to scaling issues causing NaN gradients)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='OPTIMIZED: Number of steps to accumulate gradients (increased default for better performance)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--nccl-timeout', type=int, default=3600, help='INCREASED: NCCL timeout in seconds (default: 60 minutes for large models)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save training logs')
    parser.add_argument('--validate-every', type=int, default=5, help='PERFORMANCE: Run validation every N epochs (default: 5 to reduce overhead)')
    parser.add_argument('--progress-every', type=int, default=100, help='PERFORMANCE: Print progress every N batches (default: 100)')
    parser.add_argument('--wandb-project', type=str, default=None, help='Weights & Biases project name (overrides default project naming)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (overrides command line arguments)')
    parser.set_defaults(use_checkpoint=True)
    args = parser.parse_args()

    # Load config file if specified
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)

            # Override args with config values (command line takes precedence)
            training_config = config.get('training', {})
            loss_weights = training_config.get('loss_weights', {})

            # Set loss weights from config (config overrides defaults, command line overrides config)
            for loss_name, config_value in loss_weights.items():
                arg_name = f'loss_{loss_name}'
                if hasattr(args, arg_name):
                    # Get the default value from the argument parser
                    parser_default = None
                    for action in parser._actions:
                        if action.dest == arg_name.replace('-', '_'):
                            parser_default = action.default
                            break

                    # If current value equals parser default, use config value
                    current_value = getattr(args, arg_name.replace('-', '_'))
                    if current_value == parser_default:
                        setattr(args, arg_name.replace('-', '_'), config_value)
                        print(f"  📊 Set {arg_name} = {config_value} from config (was {parser_default})")

            # Set other training parameters from config if not specified
            if args.batch_size == 32 and 'default_batch_size' in training_config:  # 32 is default
                args.batch_size = training_config['default_batch_size']
                print(f"  📦 Set batch_size = {args.batch_size} from config")

            if args.lr == 1e-06 and 'default_lr' in training_config:  # 1e-06 is default
                args.lr = training_config['default_lr']
                print(f"  📈 Set lr = {args.lr} from config")

        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {args.config}: {e}")
            print("   Continuing with command line arguments...")

    # If not passing --hpc, we can also read from an env var
    hpc = args.hpc or parse_bool_env("HPC_MODE", False)

    init_seed(42)

    # Parameters
    cache_samples = True
    save_samples = False
    batch_size = args.batch_size
    T = 64          # Number of frames (64)
    M = 1           # Number of persons
    V = 25          # Number of joints
    setting = args.setting
    dataset = args.dataset
    lr = args.lr
    train_samples = args.train_samples
    test_samples = args.test_samples
    teacher_forcing_ratio = args.teacher_forcing_ratio
    teacher_forcing_decay = args.teacher_forcing_decay

    # If HPC mode, read environment variables set by torchrun or srun
    # (One or more might be None if not set.)
    rank = 0
    world_size = 1
    local_rank = 0

    if hpc:
        # We check if WORLD_SIZE is set
        w = os.environ.get("WORLD_SIZE", None)
        if w is not None:
            world_size = int(w)
        # If you are using torchrun or the new torch.distributed.run script,
        # these should exist. Otherwise, you must export them manually in Slurm.
        r = os.environ.get("RANK", None)
        lrk = os.environ.get("LOCAL_RANK", None)
        if r is not None:
            rank = int(r)
        if lrk is not None:
            local_rank = int(lrk)

        # OPTIMIZED: Set NCCL timeout and communication settings for large models
        os.environ['NCCL_TIMEOUT'] = str(args.nccl_timeout * 2)  # Double the timeout for large models
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'  # Use async wait for better performance
        os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Use new env var name
        os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce debug verbosity for performance
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if causing issues
        os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P if causing issues

        # PERFORMANCE: Additional NCCL optimizations for large models
        os.environ['NCCL_BUFFSIZE'] = '8388608'  # 8MB buffer size for large gradients
        os.environ['NCCL_NTHREADS'] = '4'  # Increase NCCL threads
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Force tree algorithm for better scaling

        # Suppress CUDNN warnings and optimize for memory
        os.environ['CUDNN_DETERMINISTIC'] = '1'
        os.environ['CUDNN_BENCHMARK'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Reduce memory fragmentation

        # Check available GPUs before setting up distributed training
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        requested_gpus = args.gpus

        # In SLURM environment, check if CUDA_VISIBLE_DEVICES is set
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        visible_device_count = available_gpus
        if cuda_visible_devices:
            # Count visible devices
            visible_device_count = len([d for d in cuda_visible_devices.split(',') if d.strip()])

        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            print(f"🖥️  GPU Configuration:")
            print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
            print(f"   Available GPUs (torch.cuda.device_count()): {available_gpus}")
            print(f"   Visible GPU count: {visible_device_count}")
            print(f"   Requested GPUs: {requested_gpus}")
            print(f"   Environment world_size: {world_size}, local_rank: {local_rank}")

        # Use the available GPUs (should match world_size for proper distributed training)
        effective_gpus = available_gpus

        # Check for torchrun mismatch - if world_size > 1 but insufficient GPUs
        if world_size > 1 and available_gpus < world_size:
            if rank == 0:
                print(f"⚠️  WARNING: torchrun launched {world_size} processes but only {available_gpus} GPU(s) visible!")
                print(f"   This might be due to CUDA_VISIBLE_DEVICES configuration.")
                print(f"   Attempting to continue with available GPUs...")
            # Don't exit, let it try to continue

        # If world_size > 1, we do init_process_group
        if world_size > 1 and available_gpus >= world_size and effective_gpus >= world_size:
            print('Initializing distributed process group...')
            # Increase timeout significantly for large models and slow networks
            timeout_minutes = 30  # 30 minutes timeout
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(minutes=timeout_minutes)
            )
            print('Process group initialized.')

            # Ensure local_rank is within available GPU range
            if local_rank < available_gpus:
                torch.cuda.set_device(local_rank)
                device = torch.device('cuda', local_rank)
                print(f"[DDP Init] rank={rank}, local_rank={local_rank}, world_size={world_size}")
            else:
                print(f"Warning: local_rank {local_rank} >= available GPUs {available_gpus}, falling back to single GPU")
                torch.cuda.set_device(0)
                device = torch.device('cuda', 0)
                # Reset world_size and rank for single GPU fallback
                world_size = 1
                rank = 0
                local_rank = 0
        else:
            # Single-process mode (world_size == 1)
            if available_gpus > 0:
                torch.cuda.set_device(0)
                device = torch.device('cuda', 0)
                if rank == 0:
                    print("Single-GPU training mode.")
            else:
                device = torch.device('cpu')
                if rank == 0:
                    print("CPU training mode.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running in non-HPC mode on device {device}.")

    # Setup logging
    logger, log_file = setup_logging(args.log_dir, rank)

    # Load data (only print on rank 0 to avoid duplicate output)
    if cache_samples:
        # Try to load paired data from torch file
        if args.data_path:
            paired_file_path = args.data_path
        else:
            paired_file_path = f'data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt'

        # Check if file exists with new naming convention
        if not os.path.exists(paired_file_path):
            # Try legacy naming convention as fallback
            legacy_file_path = f'data/{dataset}_{setting}_paired.pt'
            if os.path.exists(legacy_file_path):
                if rank == 0:
                    print(f"Warning: Using legacy data file {legacy_file_path}")
                paired_file_path = legacy_file_path

        if os.path.exists(paired_file_path):
            if rank == 0:
                print(f"Loading paired data from {paired_file_path}")
            paired_data = torch.load(paired_file_path)
            paired_train = paired_data['train']
            paired_test = paired_data['test']

            # Check if this is the comprehensive dataset
            if rank == 0:
                if "comprehensive" in paired_file_path:
                    print(f"Loaded COMPREHENSIVE paired dataset with {len(paired_train.sampled_data)} training and {len(paired_test.sampled_data)} test samples")
                    print(f"Will sample {train_samples} training and {test_samples} test samples from this dataset")
                else:
                    print(f"Loaded paired data with {len(paired_train.sampled_data)} training and {len(paired_test.sampled_data)} test samples")
        else:
            if rank == 0:
                print(f"No paired data file found at {paired_file_path}, generating new data...")
            X = load_data(dataset, T)
            paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                       batch_size,
                                                       return_loader=False,
                                                       train_samples=train_samples,
                                                       test_samples=test_samples,
                                                       threads=1,
                                                       seg=T,
                                                       augment=True,
                                                       train_theta=0.3 if setting == 'cs' else 0.5)
            if save_samples and rank == 0:
                new_file_path = f'data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt'
                torch.save({'train': paired_train, 'test': paired_test}, new_file_path)
                print(f'Paired data saved to {new_file_path}')
    else:
        X = load_data(dataset, T)
        paired_train, paired_test = get_cross_data(X, dataset, setting,
                                                   batch_size,
                                                   return_loader=False,
                                                   train_samples=train_samples,
                                                   test_samples=test_samples,
                                                   threads=1,
                                                   seg=T,
                                                   augment=True,
                                                   train_theta=0.3 if setting == 'cs' else 0.5,
                                                   val_theta=0.3 if setting == 'cs' else 0.5)

    # Check if we should use all data (very large sample count indicates "use all")
    use_all_train = train_samples >= 999999
    use_all_test = test_samples >= 999999

    # For training data (only print on rank 0)
    if use_all_train:
        if rank == 0:
            print(f"Using ALL {len(paired_train.sampled_data)} available training samples")
    elif len(paired_train.sampled_data) > train_samples:
        if rank == 0:
            print(f"Trimming training data from {len(paired_train.sampled_data)} to {train_samples} samples")
        # Randomly sample to get a diverse subset
        import random
        random.seed(42)  # For reproducibility
        paired_train.sampled_data = random.sample(paired_train.sampled_data, train_samples)
    else:
        if rank == 0:
            print(f"Using all {len(paired_train.sampled_data)} available training samples")

    # For test data (only print on rank 0)
    if use_all_test:
        if rank == 0:
            print(f"Using ALL {len(paired_test.sampled_data)} available test samples")
    elif len(paired_test.sampled_data) > test_samples:
        if rank == 0:
            print(f"Trimming test data from {len(paired_test.sampled_data)} to {test_samples} samples")
        # Randomly sample to get a diverse subset
        import random
        random.seed(42)  # For reproducibility
        paired_test.sampled_data = random.sample(paired_test.sampled_data, test_samples)
    else:
        if rank == 0:
            print(f"Using all {len(paired_test.sampled_data)} available test samples")

    if save_samples:
        import pickle
        with open(f'data/{dataset}_{setting}_paired.pkl', 'wb') as f:
            pickle.dump({'train': paired_train, 'test': paired_test}, f)
            print('Paired data saved to pickle file')

    # Estimate memory usage for optimization
    if rank == 0:
        memory_stats = estimate_memory_usage(paired_train, batch_size)
        print(f"Memory usage estimates:")
        print(f"  Average item size: {memory_stats['avg_item_size_mb']:.2f} MB")
        print(f"  Batch memory: {memory_stats['batch_memory_mb']:.2f} MB")
        print(f"  Estimated peak memory: {memory_stats['estimated_peak_memory_gb']:.2f} GB")

    # Create optimized data loaders
    # Only use distributed if we actually have multiple GPUs and proper DDP setup
    use_distributed = hpc and world_size > 1 and torch.cuda.device_count() >= world_size
    train_loader, test_loader = optimize_data_loading(
        paired_train, paired_test, batch_size,
        distributed=use_distributed,
        rank=rank, world_size=world_size
    )

    # Initialize the model
    model = Model(num_class=120 if dataset == 'ntu120' else 60,
                  num_point=V, num_person=M, graph='graph.ntu_rgb_d.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  debug=False, dataset=dataset,
                  decoder_dropout=args.decoder_dropout,
                  use_checkpoint=args.use_checkpoint)

    # Load pretrained encoder weights if requested (only print on rank 0)
    if args.use_pretrained:
        # Check for comprehensive pretrained encoder first
        comprehensive_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder_{setting}_comprehensive.pth'
        setting_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder_{setting}.pth'
        default_encoder_path = f'eval/mixformer/pretrained/{args.dataset}/encoder.pth'

        # Try to load in this order: comprehensive, setting-specific, default
        if os.path.exists(comprehensive_encoder_path):
            pretrained_encoder_path = comprehensive_encoder_path
            if rank == 0:
                print(f"Loading COMPREHENSIVE pretrained encoder from {pretrained_encoder_path}")
        elif os.path.exists(setting_encoder_path):
            pretrained_encoder_path = setting_encoder_path
            if rank == 0:
                print(f"Loading {setting} pretrained encoder from {pretrained_encoder_path}")
        elif os.path.exists(default_encoder_path):
            pretrained_encoder_path = default_encoder_path
            if rank == 0:
                print(f"Loading default pretrained encoder from {pretrained_encoder_path}")
        else:
            pretrained_encoder_path = None
            if rank == 0:
                print(f"Warning: No pretrained encoder found. Training encoder from scratch.")

        # Load the encoder if found
        if pretrained_encoder_path:
            # Load onto the correct device directly
            encoder_state_dict = torch.load(pretrained_encoder_path, map_location=device)
            model.encoder.load_state_dict(encoder_state_dict)
            if rank == 0:
                print("Pretrained encoder loaded successfully.")
    else:
        if rank == 0:
            print("Using randomly initialized encoder (no pretraining).")

    # Freeze encoder parameters if requested (only print on rank 0)
    if args.freeze_encoder:
        if rank == 0:
            print("Freezing encoder parameters...")
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
        if rank == 0:
            print("Encoder parameters frozen.")
    else:
        if rank == 0:
            print("Encoder parameters will be trained (not frozen).")

    model = model.to(device) # Move model to device *after* loading state dict

    # Synchronize all processes before proceeding (for distributed training)
    if hpc and world_size > 1:
        dist.barrier()

    # --- Count parameters without verbose printing ---
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if rank == 0:
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("---------------------------------------------\n")

    # Apply gradient checkpointing to the model if requested
    if args.use_checkpoint:
        if rank == 0:
            print("Enabling gradient checkpointing for memory efficiency")
        # Enable gradient checkpointing for the encoder
        model.encoder.apply(lambda m: m.register_forward_hook(lambda module, _, output: output))

    # Wrap in DDP if multi-GPU AFTER loading and freezing
    if hpc and world_size > 1 and torch.cuda.device_count() >= world_size:
        # Use find_unused_parameters=True only if not using static graph
        use_static_graph = args.use_checkpoint

        # OPTIMIZED: Configure DDP with settings optimized for numerical stability
        # Mixed precision is disabled for multi-GPU, so use FP32-optimized settings
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False,  # Set to False for better performance
                   broadcast_buffers=True,   # CHANGED: Enable for better synchronization in FP32
                   bucket_cap_mb=100,        # REDUCED: Smaller buckets for better gradient sync
                   gradient_as_bucket_view=True,  # Optimize gradient memory usage
                   static_graph=use_static_graph)  # Enable static graph optimization

        # REMOVED: Static graph is now set in DDP constructor
        if use_static_graph and rank == 0:
            print("Using static graph for DDP with gradient checkpointing")
    elif hpc and world_size > 1:
        print(f"Warning: Skipping DDP initialization due to insufficient GPUs ({torch.cuda.device_count()} available, {world_size} requested)")

    # Define optimizer (only includes parameters where requires_grad is True)
    # Filter parameters *after* potential DDP wrapping if needed, though filtering before is usually fine.
    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Use standard optimizer settings - learning rate clipping removed
    optimizer = optim.Adam(params_to_optimize, lr=lr, eps=1e-8, weight_decay=1e-6,
                          betas=(0.9, 0.999), amsgrad=True)  # Use AMSGrad for better stability

    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-9)

    # Print count from rank 0
    if rank == 0:
        print(f"Optimizer initialized with {len(params_to_optimize)} trainable parameter tensors.")
        print(f"Using learning rate: {lr}")

        # Initialize model weights with standard initialization (removed overly conservative settings)
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Use standard Xavier initialization
                nn.init.xavier_uniform_(m.weight, gain=1.0)  # Standard gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize attention weights with standard gain
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight, gain=1.0)
                if hasattr(m, 'out_proj') and hasattr(m.out_proj, 'weight'):
                    nn.init.xavier_uniform_(m.out_proj.weight, gain=1.0)

        # Apply conservative initialization only to decoder (encoder is pretrained)
        if hasattr(model, 'decoder'):
            model.decoder.apply(init_weights)
            # Also initialize projection layers
            if hasattr(model, 'decoder_input_proj'):
                init_weights(model.decoder_input_proj)
            if hasattr(model, 'output_linear'):
                init_weights(model.output_linear)
            if hasattr(model, 'enc_dec_layer'):
                init_weights(model.enc_dec_layer)
            if hasattr(model, 'sty_tr_layer'):
                init_weights(model.sty_tr_layer)
        elif hasattr(model, 'module') and hasattr(model.module, 'decoder'):
            model.module.decoder.apply(init_weights)
            # Also initialize projection layers for DDP
            if hasattr(model.module, 'decoder_input_proj'):
                init_weights(model.module.decoder_input_proj)
            if hasattr(model.module, 'output_linear'):
                init_weights(model.module.output_linear)
            if hasattr(model.module, 'enc_dec_layer'):
                init_weights(model.module.enc_dec_layer)
            if hasattr(model.module, 'sty_tr_layer'):
                init_weights(model.module.sty_tr_layer)

    # Number of epochs
    num_epochs = args.epochs

    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')

    # Setup checkpoint directory
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')

    # Load checkpoint if resuming
    if args.resume_from:
        start_epoch, best_loss = load_checkpoint(args.resume_from, model, optimizer, device)

    # Initialize mixed precision scaler if using AMP
    scaler = None
    if args.mixed_precision:
        # CRITICAL: Check for DDP incompatibility
        if world_size > 1:
            if rank == 0:
                print("❌ CRITICAL: Mixed precision disabled for multi-GPU training")
                print("   🔧 Issue: DDP + Mixed Precision + Attention layers = NaN gradients")
                print("   📊 Evidence: module.decoder.layers.0.self_attn parameters produce NaN")
                print("   📊 Root cause: Gradient synchronization across GPUs with FP16 causes instability")
                print("   📊 Solution: Use FP32 for multi-GPU training")
                print("   💡 Single GPU training with mixed precision works fine")
            # Force disable mixed precision for multi-GPU
            args.mixed_precision = False
        else:
            # Single GPU: Mixed precision works fine
            scaler = torch.cuda.amp.GradScaler(
                init_scale=32.0,        # Conservative initial scale
                growth_factor=1.1,      # Very slow growth
                backoff_factor=0.95,    # Gentle backoff
                growth_interval=2000    # Very infrequent growth attempts
            )
            if rank == 0:
                print("✅ Using Automatic Mixed Precision (AMP) training (Single GPU)")
                print("   🔧 Conservative scaling (init_scale=32)")
                print("   📊 Mixed precision works reliably on single GPU")

    # Configure loss weights from arguments
    loss_weights = {}

    # Parse loss-weights string if provided (overrides individual arguments)
    if args.loss_weights:
        try:
            for pair in args.loss_weights.split(','):
                key, value = pair.strip().split(':')
                loss_weights[key.strip()] = float(value.strip())
            if rank == 0:
                print(f"✅ Parsed loss weights from --loss-weights: {loss_weights}")
        except (ValueError, IndexError) as e:
            if rank == 0:
                print(f"❌ Error parsing --loss-weights '{args.loss_weights}': {e}")
                print("   Expected format: 'mse:1.0,ee:2.0,smoothing:0.1'")
            raise ValueError(f"Invalid loss-weights format: {args.loss_weights}")
    else:
        # Use individual loss arguments (only add losses with non-zero weights)
        if args.loss_mse > 0: loss_weights['mse'] = args.loss_mse
        if args.loss_l1 > 0: loss_weights['l1'] = args.loss_l1
        if args.loss_smoothl1 > 0: loss_weights['smoothl1'] = args.loss_smoothl1
        if args.loss_kl > 0: loss_weights['kl'] = args.loss_kl
        if args.loss_ce > 0: loss_weights['ce'] = args.loss_ce
        if args.loss_ee > 0: loss_weights['ee'] = args.loss_ee
        if args.loss_smoothing > 0: loss_weights['smoothing'] = args.loss_smoothing
        if args.loss_latent > 0: loss_weights['latent'] = args.loss_latent
        if args.loss_triplet > 0: loss_weights['triplet'] = args.loss_triplet
        if args.loss_inception > 0: loss_weights['inception'] = args.loss_inception
        if args.loss_fid_vel > 0: loss_weights['fid_vel'] = args.loss_fid_vel
        if args.loss_bone > 0: loss_weights['bone'] = args.loss_bone
        if args.loss_foot > 0: loss_weights['foot'] = args.loss_foot
        if args.loss_joint_limit > 0: loss_weights['joint_limit'] = args.loss_joint_limit

    # Print loss weights for debugging (only on rank 0 or if not distributed)
    if rank == 0 or world_size == 1:
        print("\nUsing loss weights:")
        for k, v in loss_weights.items():
            print(f"  {k}: {v}")
        print()

    # Determine if we're using all data
    use_all_data = train_samples >= 999999 or test_samples >= 999999

    # Set wandb project name
    if args.wandb_project:
        wandb_project = args.wandb_project
    else:
        wandb_project = 'Motion Retargeting'
        if "comprehensive" in args.data_path:
            wandb_project += ' Comprehensive'
            if use_all_data:
                wandb_project += ' All'

    # Initialize loss function
    from src.training.loss import Loss
    # Get encoder from model (handle DDP case)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        encoder = model.module.encoder
    else:
        encoder = model.encoder
    loss_fn = Loss(loss_weights, device=device, dataset=dataset, encoder=encoder)

    # Initialize wandb if specified
    wandb_enabled = False
    if wandb_project and rank == 0:  # Only initialize on main process
        try:
            import wandb
            # Configure wandb with timeout and offline mode fallback
            wandb_settings = wandb.Settings(
                init_timeout=120,  # 2 minutes timeout
                start_method="thread"  # Use thread instead of fork for better HPC compatibility
            )

            # Try to initialize wandb with timeout
            print("🔄 Initializing wandb experiment tracking...")
            wandb.init(
                project=wandb_project,
                config={
                    'epochs': num_epochs,
                    'batch_size': batch_size,
                    'teacher_forcing_ratio': teacher_forcing_ratio,
                    'mixed_precision': args.mixed_precision,
                    'gradient_accumulation_steps': args.gradient_accumulation_steps,
                    'max_grad_norm': args.max_grad_norm,
                    'dataset': dataset,
                    'setting': setting,
                    'train_samples': train_samples,
                    'test_samples': test_samples
                },
                settings=wandb_settings
            )
            wandb.watch(model)
            wandb_enabled = True
            print("✅ wandb initialized successfully")
        except ImportError:
            print("⚠️  wandb not available, skipping experiment tracking")
        except Exception as e:
            print(f"⚠️  wandb initialization failed: {e}")
            print("   Continuing training without wandb tracking...")
            wandb_enabled = False

    # Train the model
    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device=device,
        rank=rank,
        world_size=world_size,
        teacher_forcing_ratio=teacher_forcing_ratio,
        start_epoch=start_epoch,
        best_loss=best_loss,
        checkpoint_dir=checkpoint_dir,
        save_every=args.save_every,
        mixed_precision=args.mixed_precision,
        scaler=scaler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        scheduler=scheduler,
        wandb_enabled=wandb_enabled,
        validate_every=args.validate_every
    )

    # Save model (only do so on rank 0)
    if (not hpc) or (hpc and rank == 0):
        # Determine if we're using all data
        use_all_data = train_samples >= 999999 or test_samples >= 999999

        # Set output model path
        output_model_path = args.output_model_path

        # If using all data and the path doesn't already include "all", add it
        if use_all_data and "all" not in output_model_path.lower():
            # Insert "all" before the file extension
            base, ext = os.path.splitext(output_model_path)
            output_model_path = f"{base}_all{ext}"

        torch.save(model.state_dict(), output_model_path)
        print(f"Model saved to {output_model_path}")

        # Run evaluation if requested
        if args.run_eval:
            print("\nRunning evaluation...")
            import subprocess
            eval_cmd = [
                'python', 'eval_model.py',
                f'--dataset={dataset}',
                f'--setting={setting}',
                f'--model_type=transformer',
                f'--transformer_model_path={output_model_path}',
                f'--eval_model=sgn',
                f'--test_samples={test_samples}'
            ]
            print(f"Executing: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd)

    # Cleanup
    if hpc and world_size > 1 and torch.cuda.device_count() >= world_size:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during process group cleanup: {e}")

if __name__ == '__main__':
    main()