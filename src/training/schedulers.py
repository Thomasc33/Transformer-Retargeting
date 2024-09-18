"""
Learning rate schedulers with warmup and cosine annealing for optimized training
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of training epochs
        min_lr: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) 
                    for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler followed by constant learning rate
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def get_scheduler(optimizer, scheduler_type, warmup_epochs, max_epochs, min_lr=0):
    """
    Get learning rate scheduler based on type
    
    Args:
        optimizer: Wrapped optimizer
        scheduler_type: Type of scheduler ('cosine', 'linear', 'none')
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of training epochs
        min_lr: Minimum learning rate for cosine annealing
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingWarmupScheduler(optimizer, warmup_epochs, max_epochs, min_lr)
    elif scheduler_type == 'linear':
        return LinearWarmupScheduler(optimizer, warmup_epochs)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
