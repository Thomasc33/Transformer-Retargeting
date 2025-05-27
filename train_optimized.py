import torch
from loss import Loss
import time
import os
import json
import logging
from datetime import datetime

class OptimizedTrainer:
    """
    Optimized Trainer class for skeleton motion retargeting models.

    Includes performance optimizations, robust checkpointing, and comprehensive logging.
    """

    def __init__(self, model, optimizer,
                 train_paired_loader, val_paired_loader,
                 num_epochs=10, wandb_project=None, device='cuda', dataset='ntu', rank=0,
                 teacher_forcing_ratio=1.0, teacher_forcing_decay=0.0, loss_weights=None,
                 start_epoch=0, best_loss=float('inf'), checkpoint_dir='checkpoints',
                 save_every=1, mixed_precision=False, scaler=None,
                 gradient_accumulation_steps=1, max_grad_norm=1.0, scheduler=None):
        """
        Initialize the optimized trainer with model, data, and training parameters.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_paired_loader = train_paired_loader
        self.val_paired_loader = val_paired_loader
        self.num_epochs = num_epochs
        self.wandb_project = wandb_project
        self.device = device
        self.dataset = dataset
        self.rank = rank
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay

        # Optimization parameters
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.mixed_precision = mixed_precision
        self.scaler = scaler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Training metrics tracking
        self.training_metrics = {
            'epoch_times': [],
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'memory_usage': []
        }

        # Configure loss function weights
        if loss_weights is None:
            loss_weights = {
                'mse': 7.0, 'ee': 5.0, 'smoothing': 0.075, 'inception': 0.05,
                'fid_vel': 1.0, 'bone': 10.0, 'foot': 3.0, 'joint_limit': 1.0
            }

        # Get encoder from model (handle DDP case)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            encoder = model.module.encoder
        else:
            encoder = model.encoder

        self.loss = Loss(loss_weights, device=device, dataset=dataset, encoder=encoder)

        # Initialize wandb for experiment tracking (if enabled)
        if self.wandb_project is not None and self.rank == 0:
            config = {
                'lr': self.optimizer.param_groups[0]['lr'],
                'num_epochs': self.num_epochs,
                'train_samples': len(self.train_paired_loader),
                'val_samples': len(self.val_paired_loader),
                'teacher_forcing_ratio': self.initial_teacher_forcing_ratio,
                'teacher_forcing_decay': self.teacher_forcing_decay,
                'mixed_precision': self.mixed_precision,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm,
                'start_epoch': self.start_epoch
            }
            import wandb
            wandb.init(project=self.wandb_project, config=config)
            wandb.watch(self.model)

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint."""
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Get model state dict (handle DDP case)
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'best_loss': self.best_loss,
                'teacher_forcing_ratio': self.teacher_forcing_ratio,
                'training_metrics': self.training_metrics,
                'timestamp': datetime.now().isoformat()
            }

            # Add scaler state if using mixed precision
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            # Save latest checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
            torch.save(checkpoint, checkpoint_path)

            # Save epoch-specific checkpoint
            if epoch % self.save_every == 0:
                epoch_checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, epoch_checkpoint_path)

            # Save best checkpoint if specified
            if is_best:
                best_checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
                torch.save(checkpoint, best_checkpoint_path)
                self.logger.info(f"New best model saved with loss {loss:.6f}")

            self.logger.info(f"Checkpoint saved: epoch {epoch}, loss {loss:.6f}")
            return checkpoint_path
        return None

    def save_training_log(self, epoch, train_loss, val_loss, epoch_time):
        """Save detailed training log in JSON format."""
        if self.rank == 0:
            log_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'teacher_forcing_ratio': self.teacher_forcing_ratio,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            }

            # Add memory usage if available
            if torch.cuda.is_available():
                log_data['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                log_data['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB

            # Save to JSON log file
            log_file = os.path.join(self.checkpoint_dir, 'training_log.jsonl')
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')

    def process_batch_optimized(self, x1, x2, y1, y2, is_training=True):
        """
        Optimized batch processing with memory management and mixed precision.
        """
        # Process input data with non-blocking transfer
        x1 = x1.float().to(self.device, non_blocking=True)
        x2 = x2.float().to(self.device, non_blocking=True)
        y1 = y1.float().to(self.device, non_blocking=True)
        y2 = y2.float().to(self.device, non_blocking=True)

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

        # Forward pass with mixed precision if enabled
        if self.mixed_precision and is_training:
            with torch.cuda.amp.autocast():
                if is_training:
                    output = self.model(source_motion, dummy_skeleton, target_motion=target_motion,
                                      teacher_forcing_ratio=self.teacher_forcing_ratio)
                else:
                    output = self.model(source_motion, dummy_skeleton, teacher_forcing_ratio=0.0)

                # Get ground truth next frames
                target = target_motion[:, :, 1:, :, :]

                # Calculate loss
                loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])
        else:
            if is_training:
                output = self.model(source_motion, dummy_skeleton, target_motion=target_motion,
                                  teacher_forcing_ratio=self.teacher_forcing_ratio)
            else:
                output = self.model(source_motion, dummy_skeleton, teacher_forcing_ratio=0.0)

            # Get ground truth next frames
            target = target_motion[:, :, 1:, :, :]

            # Calculate loss
            loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])

        # Free memory
        del output, target, source_motion, dummy_skeleton, target_motion

        return loss_val, losses_dict

    def train(self):
        """
        Optimized training loop with all performance enhancements.
        """
        if self.rank == 0:
            print("Starting Optimized Training (Autoregressive Motion Retargeting)...")
            print(f"Initial teacher forcing ratio: {self.teacher_forcing_ratio}")
            print(f"Starting from epoch {self.start_epoch}")
            if self.mixed_precision:
                print("Using Automatic Mixed Precision (AMP)")
            if self.gradient_accumulation_steps > 1:
                print(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps")

        self.model.train()

        for epoch in range(self.start_epoch, self.num_epochs):
            # Update teacher forcing ratio
            half_epochs = (self.num_epochs + 1) // 2
            if epoch < half_epochs:
                self.teacher_forcing_ratio = 1.0 - (epoch / half_epochs)
            else:
                self.teacher_forcing_ratio = 0.0

            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Teacher forcing ratio: {self.teacher_forcing_ratio:.4f}")

            epoch_start = time.time()
            running_loss = 0.0
            running_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

            # Initialize gradient accumulation
            self.optimizer.zero_grad(set_to_none=True)

            # Training loop
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.train_paired_loader):
                try:
                    # Process batch with optimizations
                    loss_val, losses_dict = self.process_batch_optimized(x1, x2, y1, y2, is_training=True)

                    # Check for NaN loss
                    if not torch.isfinite(loss_val):
                        if self.rank == 0:
                            print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                        # Zero gradients to prevent accumulation of invalid gradients
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    # Scale loss for gradient accumulation
                    loss_val = loss_val / self.gradient_accumulation_steps

                    # Backward pass with mixed precision if enabled
                    if self.mixed_precision:
                        self.scaler.scale(loss_val).backward()
                    else:
                        loss_val.backward()

                    # Update running losses
                    running_loss += loss_val.item() * self.gradient_accumulation_steps
                    for key, value in losses_dict.items():
                        if torch.isfinite(value):
                            running_losses[key] += value.item()
                        else:
                            if self.rank == 0:
                                print(f"Warning: NaN in {key} loss at batch {batch_idx}")

                    # Gradient accumulation step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.mixed_precision:
                            # Unscale gradients and clip
                            self.scaler.unscale_(self.optimizer)

                            # Check for NaN gradients
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            if not torch.isfinite(grad_norm):
                                if self.rank == 0:
                                    print(f"Warning: NaN gradients detected at batch {batch_idx}, skipping optimizer step...")
                                # Important: Still need to call step and update to reset scaler state
                                self.scaler.step(self.optimizer)  # This will be skipped internally due to NaN
                                self.scaler.update()
                                self.optimizer.zero_grad(set_to_none=True)
                            else:
                                # Update parameters normally
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad(set_to_none=True)
                        else:
                            # Check for NaN gradients and clip
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            if not torch.isfinite(grad_norm):
                                if self.rank == 0:
                                    print(f"Warning: NaN gradients detected at batch {batch_idx}, skipping optimizer step...")
                                self.optimizer.zero_grad(set_to_none=True)
                            else:
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)

                    # Progress reporting
                    if self.rank == 0 and batch_idx % 100 == 0:
                        current_loss = running_loss / (batch_idx + 1)
                        print(f"  Batch {batch_idx}/{len(self.train_paired_loader)}, Loss: {current_loss:.6f}")

                        # Debug: Print individual loss components every 1000 batches
                        if batch_idx % 1000 == 0 and batch_idx > 0:
                            print("  Individual loss components:")
                            for key, value in running_losses.items():
                                avg_loss = value / (batch_idx + 1)
                                print(f"    {key}: {avg_loss:.6f}")
                            print()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if self.rank == 0:
                            print(f"CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

                # Force garbage collection periodically
                if batch_idx % 50 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # Calculate epoch metrics
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            avg_loss = running_loss / len(self.train_paired_loader)
            epoch_losses = {key: value / len(self.train_paired_loader) for key, value in running_losses.items()}

            # Validation
            val_start = time.time()
            val_loss, val_losses = self.evaluate()
            val_end = time.time()
            val_time = val_end - val_start

            # Check if this is the best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # Save training log
            self.save_training_log(epoch, avg_loss, val_loss, epoch_time)

            # Update training metrics
            self.training_metrics['epoch_times'].append(epoch_time)
            self.training_metrics['epoch_losses'].append(avg_loss)
            self.training_metrics['val_losses'].append(val_loss)
            self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            if torch.cuda.is_available():
                self.training_metrics['memory_usage'].append(torch.cuda.memory_allocated() / 1024**3)

            # Print progress
            if self.rank == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}] Complete')
                print(f'  Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
                print(f'  Train Time: {epoch_time:.2f}s, Val Time: {val_time:.2f}s')
                print(f'  Best Loss: {self.best_loss:.6f}')
                if torch.cuda.is_available():
                    print(f'  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB')

            # Log to wandb
            if self.wandb_project is not None and self.rank == 0:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'teacher_forcing_ratio': self.teacher_forcing_ratio,
                    'epoch_time': epoch_time,
                    'val_time': val_time,
                    'best_loss': self.best_loss,
                    **{f'train_{k}': v for k, v in epoch_losses.items()},
                    **{f'val_{k}': v for k, v in val_losses.items()}
                })

    @torch.no_grad()
    def evaluate(self):
        """
        Optimized evaluation with memory management.
        """
        self.model.eval()
        total_loss = 0.0
        losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

        for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.val_paired_loader):
            try:
                # Process batch for evaluation
                loss_val, losses_dict = self.process_batch_optimized(x1, x2, y1, y2, is_training=False)

                # Accumulate losses
                total_loss += loss_val.item()
                for key, val in losses_dict.items():
                    losses[key] += val.item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.rank == 0:
                        print(f"CUDA OOM during validation at batch {batch_idx}, skipping...")
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
        avg_loss = total_loss / len(self.val_paired_loader)
        losses = {key: val / len(self.val_paired_loader) for key, val in losses.items()}

        self.model.train()
        return avg_loss, losses
