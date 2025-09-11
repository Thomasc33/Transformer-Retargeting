"""
Curriculum learning and algorithmic optimizations for autoregressive training.
"""

import torch
import numpy as np
from typing import Tuple, Optional

class CurriculumScheduler:
    """
    Implements curriculum learning strategies to accelerate training.
    """
    
    def __init__(self, 
                 initial_seq_length: int = 16,
                 target_seq_length: int = 64,
                 warmup_epochs: int = 5,
                 growth_strategy: str = "linear"):
        """
        Initialize curriculum scheduler.
        
        Args:
            initial_seq_length: Starting sequence length
            target_seq_length: Final sequence length
            warmup_epochs: Number of epochs to reach target length
            growth_strategy: How to grow sequence length ("linear", "exponential", "step")
        """
        self.initial_seq_length = initial_seq_length
        self.target_seq_length = target_seq_length
        self.warmup_epochs = warmup_epochs
        self.growth_strategy = growth_strategy
        
    def get_sequence_length(self, epoch: int) -> int:
        """Get the sequence length for the current epoch."""
        if epoch >= self.warmup_epochs:
            return self.target_seq_length
            
        progress = epoch / self.warmup_epochs
        
        if self.growth_strategy == "linear":
            length = self.initial_seq_length + progress * (self.target_seq_length - self.initial_seq_length)
        elif self.growth_strategy == "exponential":
            # Exponential growth: slower at start, faster at end
            exp_progress = progress ** 2
            length = self.initial_seq_length + exp_progress * (self.target_seq_length - self.initial_seq_length)
        elif self.growth_strategy == "step":
            # Step-wise growth
            steps = 4
            step_size = (self.target_seq_length - self.initial_seq_length) / steps
            step = int(progress * steps)
            length = self.initial_seq_length + step * step_size
        else:
            length = self.target_seq_length
            
        return max(self.initial_seq_length, min(int(length), self.target_seq_length))


class ScheduledSampling:
    """
    Implements scheduled sampling to bridge the gap between training and inference.
    """
    
    def __init__(self, 
                 initial_ratio: float = 1.0,
                 final_ratio: float = 0.0,
                 decay_strategy: str = "linear",
                 decay_epochs: int = 10):
        """
        Initialize scheduled sampling.
        
        Args:
            initial_ratio: Starting teacher forcing ratio
            final_ratio: Final teacher forcing ratio
            decay_strategy: How to decay ratio ("linear", "exponential", "inverse_sigmoid")
            decay_epochs: Number of epochs to decay from initial to final
        """
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_strategy = decay_strategy
        self.decay_epochs = decay_epochs
        
    def get_teacher_forcing_ratio(self, epoch: int, batch_idx: int = 0, total_batches: int = 1) -> float:
        """Get the teacher forcing ratio for the current training step."""
        # Calculate progress through training
        epoch_progress = min(epoch / self.decay_epochs, 1.0)
        
        if self.decay_strategy == "linear":
            ratio = self.initial_ratio - epoch_progress * (self.initial_ratio - self.final_ratio)
        elif self.decay_strategy == "exponential":
            # Exponential decay
            decay_rate = 0.1
            ratio = self.final_ratio + (self.initial_ratio - self.final_ratio) * np.exp(-decay_rate * epoch)
        elif self.decay_strategy == "inverse_sigmoid":
            # Inverse sigmoid decay - smooth transition
            k = 10  # Steepness parameter
            x = epoch_progress * 2 - 1  # Map to [-1, 1]
            sigmoid = 1 / (1 + np.exp(-k * x))
            ratio = self.initial_ratio - sigmoid * (self.initial_ratio - self.final_ratio)
        else:
            ratio = self.initial_ratio
            
        return max(self.final_ratio, min(ratio, self.initial_ratio))


class AdaptiveBatchSizing:
    """
    Implements adaptive batch sizing based on sequence length and memory usage.
    """
    
    def __init__(self, 
                 base_batch_size: int = 32,
                 memory_threshold: float = 0.8,
                 min_batch_size: int = 4):
        """
        Initialize adaptive batch sizing.
        
        Args:
            base_batch_size: Base batch size for shortest sequences
            memory_threshold: GPU memory threshold to trigger batch size reduction
            min_batch_size: Minimum allowed batch size
        """
        self.base_batch_size = base_batch_size
        self.memory_threshold = memory_threshold
        self.min_batch_size = min_batch_size
        
    def get_batch_size(self, sequence_length: int, teacher_forcing_ratio: float) -> int:
        """
        Get adaptive batch size based on sequence length and teacher forcing ratio.
        
        Autoregressive generation requires more memory, so reduce batch size accordingly.
        """
        # Base adjustment for sequence length
        length_factor = 64 / sequence_length  # Assume 64 is the reference length
        
        # Additional adjustment for autoregressive mode
        if teacher_forcing_ratio < 1.0:
            # Autoregressive mode uses more memory
            ar_factor = 0.5 + 0.5 * teacher_forcing_ratio  # Range: [0.5, 1.0]
        else:
            ar_factor = 1.0
            
        # Calculate adjusted batch size
        adjusted_batch_size = int(self.base_batch_size * length_factor * ar_factor)
        
        # Ensure minimum batch size
        return max(self.min_batch_size, adjusted_batch_size)
        
    def check_memory_usage(self) -> float:
        """Check current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return allocated / reserved if reserved > 0 else 0.0
        return 0.0


class TrainingOptimizer:
    """
    Combines all algorithmic optimizations for efficient training.
    """
    
    def __init__(self, 
                 curriculum_scheduler: Optional[CurriculumScheduler] = None,
                 scheduled_sampling: Optional[ScheduledSampling] = None,
                 adaptive_batching: Optional[AdaptiveBatchSizing] = None):
        """
        Initialize training optimizer with various strategies.
        """
        self.curriculum_scheduler = curriculum_scheduler or CurriculumScheduler()
        self.scheduled_sampling = scheduled_sampling or ScheduledSampling()
        self.adaptive_batching = adaptive_batching or AdaptiveBatchSizing()
        
    def get_training_params(self, epoch: int, batch_idx: int = 0, total_batches: int = 1) -> dict:
        """
        Get all training parameters for the current step.
        
        Returns:
            Dictionary with sequence_length, teacher_forcing_ratio, batch_size
        """
        sequence_length = self.curriculum_scheduler.get_sequence_length(epoch)
        teacher_forcing_ratio = self.scheduled_sampling.get_teacher_forcing_ratio(
            epoch, batch_idx, total_batches
        )
        batch_size = self.adaptive_batching.get_batch_size(sequence_length, teacher_forcing_ratio)
        
        return {
            'sequence_length': sequence_length,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'batch_size': batch_size,
            'memory_usage': self.adaptive_batching.check_memory_usage()
        }
        
    def should_adjust_batch_size(self) -> bool:
        """Check if batch size should be adjusted due to memory pressure."""
        memory_usage = self.adaptive_batching.check_memory_usage()
        return memory_usage > self.adaptive_batching.memory_threshold


# Default optimized training configuration
def get_default_training_optimizer() -> TrainingOptimizer:
    """Get a default training optimizer with reasonable settings."""
    curriculum = CurriculumScheduler(
        initial_seq_length=16,
        target_seq_length=64,
        warmup_epochs=3,
        growth_strategy="exponential"
    )
    
    sampling = ScheduledSampling(
        initial_ratio=1.0,
        final_ratio=0.0,
        decay_strategy="inverse_sigmoid",
        decay_epochs=8
    )
    
    batching = AdaptiveBatchSizing(
        base_batch_size=32,
        memory_threshold=0.85,
        min_batch_size=4
    )
    
    return TrainingOptimizer(curriculum, sampling, batching)
