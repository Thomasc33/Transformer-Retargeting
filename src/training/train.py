import torch
from loss import Loss
import time
import os
import json
import logging
from datetime import datetime

class Trainer:
    """
    Trainer class for skeleton motion retargeting models.

    Handles the training loop, evaluation, and logging of metrics.
    """

    def __init__(self, model, optimizer,
                 train_paired_loader, val_paired_loader,
                 num_epochs=10, wandb_project=None, device='cuda', dataset='ntu', rank=0,
                 teacher_forcing_ratio=1.0, teacher_forcing_decay=0.0, loss_weights=None,
                 start_epoch=0, best_loss=float('inf'), checkpoint_dir='checkpoints',
                 save_every=1, mixed_precision=False, scaler=None,
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        """
        Initialize the trainer with model, data, and training parameters.

        Args:
            model: torch.nn.Module - The model to train
            optimizer: torch.optim.Optimizer - The optimizer for training
            train_paired_loader: DataLoader - Training data loader with paired samples
            val_paired_loader: DataLoader - Validation data loader with paired samples
            num_epochs: int - Number of training epochs
            wandb_project: str - Weights & Biases project name (None to disable)
            device: str - Device to use for training ('cuda' or 'cpu')
            dataset: str - Dataset name ('ntu', 'ntu120', 'etri')
            rank: int - Process rank for distributed training
            teacher_forcing_ratio: float - Initial teacher forcing ratio (1.0=always use teacher forcing)
            teacher_forcing_decay: float - Teacher forcing decay rate per epoch (0.0=no decay)
            start_epoch: int - Starting epoch (for resuming training)
            best_loss: float - Best loss so far (for resuming training)
            checkpoint_dir: str - Directory to save checkpoints
            save_every: int - Save checkpoint every N epochs
            mixed_precision: bool - Use automatic mixed precision training
            scaler: torch.cuda.amp.GradScaler - AMP scaler for mixed precision
            gradient_accumulation_steps: int - Number of steps to accumulate gradients
            max_grad_norm: float - Maximum gradient norm for clipping
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

        # New optimization parameters
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.mixed_precision = mixed_precision
        self.scaler = scaler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

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
        # Use provided loss weights or default values
        if loss_weights is None:
            loss_weights = {
                'mse': 7.0,
                'ee': 5.0,
                'smoothing': 0.075,
                'inception': 0.05,
                'fid_vel': 1.0,
                'bone': 10.0,
                'foot': 3.0,
                'joint_limit': 1.0
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

    def train(self):
        """
        Train the model for the specified number of epochs.

        Performs training and validation for each epoch, logs metrics,
        and prints progress information.
        """
        if self.rank == 0:
            print("Starting Training (Autoregressive Motion Retargeting)...")
            print(f"Initial teacher forcing ratio: {self.teacher_forcing_ratio}")
            print(f"Using linear decay to 0 over the first half of training, then 0 for the remainder")
            print(f"Starting from epoch {self.start_epoch}")
            if self.mixed_precision:
                print("Using Automatic Mixed Precision (AMP)")
            if self.gradient_accumulation_steps > 1:
                print(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps")

        self.model.train()

        for epoch in range(self.start_epoch, self.num_epochs):
            # Update teacher forcing ratio with new decay strategy
            half_epochs = (self.num_epochs + 1) // 2  # Ceiling of epochs/2
            if epoch < half_epochs:
                # Linear decay from 1.0 to 0.0 over half_epochs
                self.teacher_forcing_ratio = 1.0 - (epoch / half_epochs)
            else:
                # Zero for the second half
                self.teacher_forcing_ratio = 0.0

            if self.rank == 0:
                print(f"Epoch {epoch+1} teacher forcing ratio: {self.teacher_forcing_ratio:.4f}")

            start = time.time()
            running_loss = 0.0
            running_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

            # Initialize gradient accumulation
            accumulated_loss = 0.0
            accumulated_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

            # Training loop with optimizations
            for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.train_paired_loader):
                # Process input data - use non_blocking=True for asynchronous data transfer
                x1 = x1.float().to(self.device, non_blocking=True)  # P1 A1
                x2 = x2.float().to(self.device, non_blocking=True)  # P2 A2
                y1 = y1.float().to(self.device, non_blocking=True)  # P1 A2
                y2 = y2.float().to(self.device, non_blocking=True)  # P2 A1

                # Combine inputs and targets for batch processing
                inputs  = torch.cat([x1, y1, x2, y2], dim=0)
                dummy   = torch.cat([x2, y2, x1, y1], dim=0)
                targets = torch.cat([y2, x2, y1, x1], dim=0)

                # Free memory of individual tensors
                del x1, x2, y1, y2

                # Reshape data for model input
                N_total = inputs.size(0)
                T = inputs.size(1)
                M = 1
                V = 25
                C_in = 3

                # Split batch processing if batch size is too large
                # This helps reduce peak memory usage
                max_sub_batch = 16  # Process at most 16 samples at once

                # Process in smaller chunks if batch size is large
                if N_total > max_sub_batch:
                    # Calculate number of sub-batches
                    num_sub_batches = (N_total + max_sub_batch - 1) // max_sub_batch

                    # Initialize accumulated loss
                    accumulated_loss = 0.0
                    accumulated_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

                    # Zero gradients once before sub-batch processing
                    self.optimizer.zero_grad(set_to_none=True)

                    # Process each sub-batch
                    for sb in range(num_sub_batches):
                        # Calculate start and end indices
                        start_idx = sb * max_sub_batch
                        end_idx = min((sb + 1) * max_sub_batch, N_total)

                        # Extract sub-batch
                        inputs_sb = inputs[start_idx:end_idx]
                        dummy_sb = dummy[start_idx:end_idx]
                        targets_sb = targets[start_idx:end_idx]

                        # Get sub-batch size
                        N_sb = inputs_sb.size(0)

                        # Reshape to (N, C, T, V, M) format - use contiguous() to ensure memory layout is optimized
                        source_motion = inputs_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                        dummy_skeleton = dummy_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                        target_motion = targets_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                        # Free memory of intermediate tensors
                        del inputs_sb, dummy_sb, targets_sb

                        # Forward pass with current teacher forcing ratio
                        output = self.model(source_motion, dummy_skeleton, target_motion=target_motion, teacher_forcing_ratio=self.teacher_forcing_ratio)

                        # Compare output with ground truth next frames
                        target = target_motion[:, :, 1:, :, :]

                        # Calculate loss
                        loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])

                        # Scale loss by the ratio of sub-batch size to total batch size
                        # This ensures the gradients are properly scaled
                        loss_val = loss_val * (N_sb / N_total)

                        # Backward pass for this sub-batch
                        try:
                            loss_val.backward()

                            # Accumulate loss values for reporting
                            accumulated_loss += loss_val.item() * (N_total / N_sb)  # Scale back for reporting
                            for key, value in losses_dict.items():
                                accumulated_losses[key] += value.item() * (N_sb / N_total)

                        except RuntimeError as e:
                            print(f"Error during backward pass for sub-batch {sb}: {e}")
                            torch.cuda.synchronize()

                        # Explicitly free memory after each sub-batch
                        del output, target, source_motion, dummy_skeleton, target_motion
                        del loss_val, losses_dict

                        # Force garbage collection after each sub-batch
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                    # Update parameters once after processing all sub-batches
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # Update running losses with accumulated values
                    running_loss += accumulated_loss
                    for key, value in accumulated_losses.items():
                        running_losses[key] += value

                else:
                    # Process the entire batch at once for smaller batches or during evaluation
                    # Reshape to (N, C, T, V, M) format - use contiguous() to ensure memory layout is optimized
                    source_motion = inputs.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                    dummy_skeleton = dummy.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                    target_motion = targets.view(N_total, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                    # Free memory of intermediate tensors
                    del inputs, dummy, targets

                    # Forward pass with current teacher forcing ratio
                    output = self.model(source_motion, dummy_skeleton, target_motion=target_motion, teacher_forcing_ratio=self.teacher_forcing_ratio)

                    # Compare output with ground truth next frames
                    target = target_motion[:, :, 1:, :, :]

                    # Calculate loss
                    loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])

                    # Backpropagation
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient memory cleanup

                    # Use try-except to handle any backward pass errors
                    try:
                        loss_val.backward()

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                        # Update running losses
                        running_loss += loss_val.item()
                        for key, value in losses_dict.items():
                            running_losses[key] += value.item()
                    except RuntimeError as e:
                        print(f"Error during backward pass: {e}")
                        # Force synchronize CUDA to ensure all operations are complete
                        torch.cuda.synchronize()

                    # Explicitly free memory - do this regardless of backward success/failure
                    del output, target, source_motion, dummy_skeleton, target_motion
                    del loss_val, losses_dict

                # Force garbage collection after processing the batch
                import gc
                gc.collect()
                torch.cuda.empty_cache()  # Clear unused memory

            # Calculate epoch metrics
            end = time.time()
            avg_loss = running_loss / len(self.train_paired_loader)
            epoch_losses = {key: value / len(self.train_paired_loader) for key, value in running_losses.items()}

            # Print training progress
            if self.rank == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
                print(f'Time taken: {end - start:.2f} s')
                print(epoch_losses)

            # Validation
            vstart = time.time()
            val_loss, val_losses = self.evaluate()
            vend = time.time()
            if self.rank == 0:
                print(f'Validation Time taken: {vend - vstart:.2f} s')

            # Log metrics to wandb
            if self.wandb_project is not None and self.rank == 0:
                import wandb
                loss_info = {key: value for key, value in epoch_losses.items()}
                val_loss_info = {f"Val {key}": value for key, value in val_losses.items()}
                wandb.log({
                    'Loss': avg_loss,
                    'Val Loss': val_loss,
                    'Teacher Forcing Ratio': self.teacher_forcing_ratio,
                    **loss_info,
                    **val_loss_info,
                    'Train Time': end - start,
                    'Val Time': vend - vstart
                })

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on the validation dataset.

        Returns:
            tuple - (average_loss, component_losses)
        """
        self.model.eval()
        total_loss = 0.0
        losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

        # Set a smaller evaluation batch size to reduce memory usage
        max_eval_batch = 8  # Process at most 8 samples at once during evaluation

        for batch_idx, (x1, x2, y1, y2, actors, actions) in enumerate(self.val_paired_loader):
            # Process input data with non-blocking transfer
            x1 = x1.float().to(self.device, non_blocking=True)  # P1 A1
            x2 = x2.float().to(self.device, non_blocking=True)  # P2 A2
            y1 = y1.float().to(self.device, non_blocking=True)  # P1 A2
            y2 = y2.float().to(self.device, non_blocking=True)  # P2 A1

            # Combine inputs and targets
            inputs  = torch.cat([x1, y1, x2, y2], dim=0)
            dummy   = torch.cat([x2, y2, x1, y1], dim=0)
            targets = torch.cat([y2, x2, y1, x1], dim=0)

            # Free memory of individual tensors
            del x1, x2, y1, y2

            # Get dimensions
            N_total = inputs.size(0)
            T = inputs.size(1)
            M = 1
            V = 25
            C_in = 3

            # Process in smaller chunks to reduce memory usage
            batch_loss = 0.0
            batch_losses = {key: 0.0 for key in self.loss.loss_weights.keys()}

            # Calculate number of sub-batches
            num_sub_batches = (N_total + max_eval_batch - 1) // max_eval_batch

            for sb in range(num_sub_batches):
                # Calculate start and end indices
                start_idx = sb * max_eval_batch
                end_idx = min((sb + 1) * max_eval_batch, N_total)

                # Extract sub-batch
                inputs_sb = inputs[start_idx:end_idx]
                dummy_sb = dummy[start_idx:end_idx]
                targets_sb = targets[start_idx:end_idx]

                # Get sub-batch size
                N_sb = inputs_sb.size(0)

                # Reshape to (N, C, T, V, M) format
                source_motion = inputs_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                dummy_skeleton = dummy_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()
                target_motion = targets_sb.view(N_sb, T, M, V, C_in).permute(0, 4, 1, 3, 2).contiguous()

                # Free memory of intermediate tensors
                del inputs_sb, dummy_sb, targets_sb

                # Forward pass without teacher forcing
                output = self.model(source_motion, dummy_skeleton, teacher_forcing_ratio=0.0)

                # Get ground truth next frames
                target = target_motion[:, :, 1:, :, :]

                # Calculate loss
                loss_val, losses_dict = self.loss.loss(output, target, source_motion[:, :, 1:, :, :])

                # Accumulate loss values
                batch_loss += loss_val.item() * (N_sb / N_total)
                for key, val in losses_dict.items():
                    batch_losses[key] += val.item() * (N_sb / N_total)

                # Free memory aggressively
                del output, target, source_motion, dummy_skeleton, target_motion, loss_val, losses_dict

                # Force garbage collection after each sub-batch
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # Add batch losses to total
            total_loss += batch_loss
            for key, val in batch_losses.items():
                losses[key] += val

            # Free remaining memory
            del inputs, dummy, targets, batch_losses

            # Force garbage collection between batches
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate average losses
        avg_loss = total_loss / len(self.val_paired_loader)
        losses = {key: val / len(self.val_paired_loader) for key, val in losses.items()}

        # Print validation results
        if self.rank == 0:
            print(f'Validation Loss: {avg_loss:.4f}')
            print(losses)

        self.model.train()
        return avg_loss, losses
