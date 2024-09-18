"""
Wrapper for SkateFormer (Skeletal-Temporal Transformer).

Paper: Duan et al., "Skateformer: Skeletal-temporal transformer for human 
       action recognition", ECCV 2024
Performance: 92.4% (NTU60 X-View), 89.4% (NTU120 X-Set)
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base_wrapper import BaseModelWrapper


class SkateFormerWrapper(BaseModelWrapper):
    """
    Wrapper for SkateFormer model from external_repo_for_reference/SkateFormer/
    
    SkateFormer is a pure transformer architecture with spatial-temporal
    factorization for skeleton-based action recognition.
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
        depths: Depth of each stage (default: (2, 2, 2, 2))
        channels: Channels for each stage (default: (96, 192, 192, 192))
        embed_dim: Embedding dimension (default: 64)
        num_frames: Number of frames (default: 64)
        kernel_size: Kernel size for temporal convolution (default: 7)
        num_heads: Number of attention heads (default: 32)
        attn_drop: Attention dropout rate (default: 0.)
        head_drop: Head dropout rate (default: 0.)
        drop: Dropout rate (default: 0.)
        drop_path: Drop path rate (default: 0.)
        mlp_ratio: MLP expansion ratio (default: 4.)
        rel: Use relative position encoding (default: True)
        index_t: Use temporal indexing (default: False)
        global_pool: Global pooling type (default: 'avg')
        **kwargs: Additional model-specific arguments
    """
    
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        depths: tuple = (2, 2, 2, 2),
        channels: tuple = (96, 192, 192, 192),
        embed_dim: int = 96,  # SkateFormer default is 96, not 64
        num_frames: int = 64,
        kernel_size: int = 7,
        num_heads: int = 32,
        attn_drop: float = 0.,
        head_drop: float = 0.,
        drop: float = 0.,
        drop_path: float = 0.,
        mlp_ratio: float = 4.,
        rel: bool = True,
        index_t: bool = False,
        global_pool: str = 'avg',
        **kwargs
    ):
        super().__init__(
            model_name='SkateFormer',
            external_repo_name='SkateFormer',
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        # Import SkateFormer model from external repo
        try:
            SkateFormer = self._import_from_path('model.SkateFormer', 'SkateFormer', self.external_repo_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import SkateFormer model from {self.external_repo_path}. "
                f"Error: {e}\n"
                f"Make sure the external repository is properly cloned and contains model/SkateFormer.py"
            )
        
        # Create model instance
        # Note: SkateFormer was designed for num_points=50 (NW-UCLA dataset)
        # For NTU with 25 joints, we use num_points=25 and num_people=2 to get 50 total
        # This way the model architecture stays the same
        self.model = SkateFormer(
            in_channels=in_channels,
            depths=depths,
            channels=channels,
            num_classes=num_class,
            embed_dim=embed_dim,
            num_people=2,  # Use 2 to get 25*2=50 points
            num_frames=num_frames,
            num_points=num_point,  # 25 joints
            kernel_size=kernel_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            head_drop=head_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            rel=rel,
            index_t=index_t,
            global_pool=global_pool
        )
        
        self.original_num_point = num_point
        self.original_num_person = num_person
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SkateFormer.
        
        Args:
            x: Input tensor of shape (N, C, T, V, M)
               N: batch size
               C: number of channels (3 for x,y,z)
               T: number of frames (64)
               V: number of joints (25)
               M: number of persons (1)
        
        Returns:
            Output logits of shape (N, num_class)
        """
        # SkateFormer expects (N, C, T, V, M) with M=2 (we set num_people=2 in init)
        # We need to duplicate the M dimension to match
        N, C, T, V, M = x.size()
        
        if M == 1:
            # Duplicate the person dimension to match num_people=2
            x = x.repeat(1, 1, 1, 1, 2)  # (N, C, T, V, 1) -> (N, C, T, V, 2)
        
        # index_t is a boolean flag for temporal indexing (default: False)
        return self.model(x, index_t=False)
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from a checkpoint file.
        
        SkateFormer checkpoints may have different formats:
        - 'model_state_dict': Standard PyTorch checkpoint
        - 'state_dict': Alternative format
        - Direct state dict: Raw model weights
        
        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce key matching (default: False)
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict from different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {
            k.replace('module.', ''): v 
            for k, v in state_dict.items()
        }
        
        # Load weights
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=strict
            )
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
                
        except RuntimeError as e:
            if not strict:
                # Try loading with strict=False if num_classes mismatch
                print(f"Warning: Failed to load checkpoint strictly. Error: {e}")
                print("Attempting to load with strict=False...")
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise
