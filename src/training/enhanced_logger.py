"""
Enhanced Logging and Monitoring Module

This module provides structured logging, progress tracking, and system monitoring
for TMR training with JSON output for automated analysis.

Features:
- Structured JSON logging for automated analysis
- Progress bars with ETA estimation using tqdm
- GPU memory usage monitoring
- Training speed tracking (samples/sec)
- Loss component tracking
- Hyperparameter logging

Requirements: 6.5
"""

import json
import time
import os
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import logging


class EnhancedLogger:
    """
    Enhanced logger for TMR training with structured output and monitoring.
    
    Features:
    - JSON structured logging for automated analysis
    - Progress tracking with ETA estimation
    - System resource monitoring (GPU memory, CPU usage)
    - Training speed metrics (samples/sec, batches/sec)
    - Loss component tracking over time
    - Hyperparameter logging
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "tmr_training",
        log_level: str = "INFO",
        enable_progress_bars: bool = True,
        monitor_resources: bool = True,
        log_frequency: int = 10
    ):
        """
        Initialize enhanced logger.
        
        Args:
            output_dir: Directory to save log files
            experiment_name: Name of the experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_progress_bars: Whether to show progress bars
            monitor_resources: Whether to monitor system resources
            log_frequency: Frequency of detailed logging (every N batches)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.enable_progress_bars = enable_progress_bars
        self.monitor_resources = monitor_resources
        self.log_frequency = log_frequency
        
        # Initialize logging
        self.setup_logging(log_level)
        
        # Initialize JSON log file
        self.json_log_path = self.output_dir / f"{experiment_name}_log.json"
        self.metrics_log_path = self.output_dir / f"{experiment_name}_metrics.json"
        
        # Training state tracking
        self.training_start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        self.total_samples_processed = 0
        self.current_stage = None
        self.current_epoch = None
        
        # Metrics tracking
        self.epoch_metrics = []
        self.batch_metrics = []
        self.loss_history = {}
        
        # Progress bars
        self.epoch_pbar = None
        self.batch_pbar = None
        
        # System monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
        
        self.logger.info(f"Enhanced logger initialized: {self.json_log_path}")
        
    def setup_logging(self, log_level: str):
        """Setup standard logging configuration."""
        self.logger = logging.getLogger(f"enhanced_logger_{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create file handler
        log_file = self.output_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_hyperparameters(self, args: Any):
        """Log hyperparameters and training configuration."""
        hyperparams = {}
        
        if hasattr(args, '__dict__'):
            hyperparams = vars(args)
        elif isinstance(args, dict):
            hyperparams = args
        else:
            hyperparams = {'args': str(args)}
        
        # Convert non-serializable objects to strings
        serializable_params = {}
        for key, value in hyperparams.items():
            try:
                json.dumps(value)
                serializable_params[key] = value
            except (TypeError, ValueError):
                serializable_params[key] = str(value)
        
        log_entry = {
            'timestamp': time.time(),
            'event_type': 'hyperparameters',
            'experiment_name': self.experiment_name,
            'hyperparameters': serializable_params,
            'system_info': self.get_system_info()
        }
        
        self.write_json_log(log_entry)
        self.logger.info(f"Hyperparameters logged: {len(serializable_params)} parameters")
    
    def start_training(self, total_stages: int = 3):
        """Mark the start of training."""
        self.training_start_time = time.time()
        
        log_entry = {
            'timestamp': self.training_start_time,
            'event_type': 'training_start',
            'experiment_name': self.experiment_name,
            'total_stages': total_stages,
            'system_info': self.get_system_info()
        }
        
        self.write_json_log(log_entry)
        self.logger.info(f"Training started: {total_stages} stages")
    
    def start_stage(self, stage: int, total_epochs: int, stage_name: str = None):
        """Mark the start of a training stage."""
        self.current_stage = stage
        stage_name = stage_name or f"Stage {stage}"
        
        log_entry = {
            'timestamp': time.time(),
            'event_type': 'stage_start',
            'experiment_name': self.experiment_name,
            'stage': stage,
            'stage_name': stage_name,
            'total_epochs': total_epochs
        }
        
        self.write_json_log(log_entry)
        self.logger.info(f"{stage_name} started: {total_epochs} epochs")
        
        # Initialize stage progress bar
        if self.enable_progress_bars:
            self.epoch_pbar = tqdm(
                total=total_epochs,
                desc=f"{stage_name}",
                unit="epoch",
                position=0,
                leave=True
            )
    
    def start_epoch(self, epoch: int, total_batches: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Initialize batch progress bar
        if self.enable_progress_bars:
            self.batch_pbar = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch+1}",
                unit="batch",
                position=1,
                leave=False
            )
    
    def log_batch(self, batch_idx: int, batch_size: int, loss_dict: Dict[str, float], 
                  metrics_dict: Optional[Dict[str, float]] = None):
        """Log batch-level metrics."""
        current_time = time.time()
        
        # Update progress bar
        if self.batch_pbar:
            self.batch_pbar.update(1)
            
            # Update progress bar description with key metrics
            if loss_dict:
                main_loss = loss_dict.get('total_loss', loss_dict.get('loss', 0.0))
                self.batch_pbar.set_postfix({'Loss': f'{main_loss:.4f}'})
        
        # Detailed logging every N batches
        if (batch_idx + 1) % self.log_frequency == 0:
            self.total_samples_processed += batch_size
            
            # Calculate training speed
            if self.epoch_start_time:
                epoch_elapsed = current_time - self.epoch_start_time
                samples_per_sec = self.total_samples_processed / epoch_elapsed if epoch_elapsed > 0 else 0
                batches_per_sec = (batch_idx + 1) / epoch_elapsed if epoch_elapsed > 0 else 0
            else:
                samples_per_sec = 0
                batches_per_sec = 0
            
            # Get system metrics
            system_metrics = self.get_system_metrics() if self.monitor_resources else {}
            
            log_entry = {
                'timestamp': current_time,
                'event_type': 'batch_log',
                'experiment_name': self.experiment_name,
                'stage': self.current_stage,
                'epoch': self.current_epoch,
                'batch': batch_idx,
                'batch_size': batch_size,
                'total_samples_processed': self.total_samples_processed,
                'samples_per_sec': samples_per_sec,
                'batches_per_sec': batches_per_sec,
                'losses': loss_dict,
                'metrics': metrics_dict or {},
                'system_metrics': system_metrics
            }
            
            self.write_json_log(log_entry)
            
            # Update loss history
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in self.loss_history:
                    self.loss_history[loss_name] = []
                self.loss_history[loss_name].append({
                    'timestamp': current_time,
                    'stage': self.current_stage,
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'value': loss_value
                })
    
    def end_epoch(self, epoch: int, epoch_metrics: Dict[str, float], 
                  validation_metrics: Optional[Dict[str, float]] = None,
                  is_best: bool = False):
        """Mark the end of an epoch and log epoch-level metrics."""
        current_time = time.time()
        
        # Close batch progress bar
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        # Update epoch progress bar
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            
            # Update description with key metrics
            if validation_metrics:
                val_acc = validation_metrics.get('ar_accuracy', validation_metrics.get('accuracy', 0.0))
                self.epoch_pbar.set_postfix({'Val Acc': f'{val_acc:.3f}'})
        
        # Calculate epoch duration
        epoch_duration = current_time - self.epoch_start_time if self.epoch_start_time else 0
        
        # Get system metrics
        system_metrics = self.get_system_metrics() if self.monitor_resources else {}
        
        log_entry = {
            'timestamp': current_time,
            'event_type': 'epoch_end',
            'experiment_name': self.experiment_name,
            'stage': self.current_stage,
            'epoch': epoch,
            'epoch_duration': epoch_duration,
            'is_best': is_best,
            'train_metrics': epoch_metrics,
            'validation_metrics': validation_metrics or {},
            'system_metrics': system_metrics
        }
        
        self.write_json_log(log_entry)
        self.epoch_metrics.append(log_entry)
        
        # Reset epoch tracking
        self.epoch_start_time = None
        self.total_samples_processed = 0
        
        self.logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")
    
    def end_stage(self, stage: int, stage_metrics: Dict[str, float]):
        """Mark the end of a training stage."""
        current_time = time.time()
        
        # Close epoch progress bar
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None
        
        log_entry = {
            'timestamp': current_time,
            'event_type': 'stage_end',
            'experiment_name': self.experiment_name,
            'stage': stage,
            'stage_metrics': stage_metrics
        }
        
        self.write_json_log(log_entry)
        self.logger.info(f"Stage {stage} completed")
    
    def end_training(self, final_metrics: Dict[str, float]):
        """Mark the end of training."""
        current_time = time.time()
        training_duration = current_time - self.training_start_time if self.training_start_time else 0
        
        log_entry = {
            'timestamp': current_time,
            'event_type': 'training_end',
            'experiment_name': self.experiment_name,
            'training_duration': training_duration,
            'final_metrics': final_metrics
        }
        
        self.write_json_log(log_entry)
        
        # Save complete metrics history
        self.save_metrics_summary()
        
        self.logger.info(f"Training completed in {training_duration:.2f}s")
    
    def log_checkpoint(self, checkpoint_path: str, is_best: bool = False, 
                      metrics: Optional[Dict[str, float]] = None):
        """Log checkpoint saving."""
        log_entry = {
            'timestamp': time.time(),
            'event_type': 'checkpoint_saved',
            'experiment_name': self.experiment_name,
            'stage': self.current_stage,
            'epoch': self.current_epoch,
            'checkpoint_path': str(checkpoint_path),
            'is_best': is_best,
            'metrics': metrics or {}
        }
        
        self.write_json_log(log_entry)
        self.logger.info(f"Checkpoint saved: {checkpoint_path} (best: {is_best})")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if self.gpu_available:
            info.update({
                'gpu_available': True,
                'gpu_device': self.gpu_device,
                'gpu_name': torch.cuda.get_device_name(self.gpu_device),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(self.gpu_device).total_memory / (1024**3)
            })
        else:
            info['gpu_available'] = False
        
        return info
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        }
        
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_stats(self.gpu_device)
                metrics.update({
                    'gpu_memory_allocated_gb': gpu_memory['allocated_bytes.all.current'] / (1024**3),
                    'gpu_memory_reserved_gb': gpu_memory['reserved_bytes.all.current'] / (1024**3),
                    'gpu_memory_percent': (gpu_memory['allocated_bytes.all.current'] / 
                                         torch.cuda.get_device_properties(self.gpu_device).total_memory) * 100
                })
            except Exception as e:
                self.logger.warning(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def write_json_log(self, log_entry: Dict[str, Any]):
        """Write log entry to JSON file."""
        try:
            with open(self.json_log_path, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")
    
    def save_metrics_summary(self):
        """Save complete metrics summary to file."""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': time.time(),
            'epoch_metrics': self.epoch_metrics,
            'loss_history': self.loss_history,
            'total_epochs': len(self.epoch_metrics),
            'training_duration': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        try:
            with open(self.metrics_log_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Metrics summary saved: {self.metrics_log_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics summary: {e}")
    
    def close(self):
        """Close progress bars and finalize logging."""
        if self.batch_pbar:
            self.batch_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()
        
        self.logger.info("Enhanced logger closed")


def create_enhanced_logger(args, experiment_name: str = None) -> EnhancedLogger:
    """
    Create enhanced logger from training arguments.
    
    Args:
        args: Training arguments
        experiment_name: Optional experiment name override
    
    Returns:
        EnhancedLogger instance
    """
    if experiment_name is None:
        experiment_name = f"tmr_{args.dataset}_{int(time.time())}"
    
    enable_progress = not getattr(args, 'no_progress_bars', False)
    monitor_resources = getattr(args, 'monitor_resources', True)
    log_frequency = getattr(args, 'log_freq', 10)
    
    logger = EnhancedLogger(
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        enable_progress_bars=enable_progress,
        monitor_resources=monitor_resources,
        log_frequency=log_frequency
    )
    
    # Log hyperparameters
    logger.log_hyperparameters(args)
    
    return logger