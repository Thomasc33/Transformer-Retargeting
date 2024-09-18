"""
Wrapper for MAMP (Masked Motion Predictors).

Paper: Mao et al., "Masked Motion Predictors are Strong 3D Action 
       Representation Learners", ICCV 2023
Performance: 93.0% (NTU60 X-Sub), 89.8% (NTU120 X-Sub)
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base_wrapper import BaseModelWrapper


class MAMPWrapper(BaseModelWrapper):
    """
    Wrapper for MAMP model from external_repo_for_reference/MAMP/
    
    MAMP uses masked autoencoder-style pre-training with motion prediction
    for skeleton-based action recognition.
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
        dim_feat: Feature dimension (default: 256)
        depth: Number of transformer blocks (default: 5)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP expansion ratio (default: 4)
        num_frames: Number of frames (default: 64)
        patch_size: Spatial patch size (default: 1)
        t_patch_size: Temporal patch size (default: 4)
        qkv_bias: Use bias in QKV projection (default: True)
        qk_scale: Scale for QK attention (default: None)
        drop_rate: Dropout rate (default: 0.)
        attn_drop_rate: Attention dropout rate (default: 0.)
        drop_path_rate: Drop path rate (default: 0.)
        protocol: Evaluation protocol ('linprobe' or 'finetune', default: 'finetune')
        **kwargs: Additional model-specific arguments
    """
    
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        dim_feat: int = 256,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        num_frames: int = 64,
        patch_size: int = 1,
        t_patch_size: int = 4,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        protocol: str = 'finetune',
        **kwargs
    ):
        super().__init__(
            model_name='MAMP',
            external_repo_name='MAMP',
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        # Import MAMP model from external repo
        try:
            MAMP = self._import_from_path('model.transformer', 'Transformer', self.external_repo_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import MAMP model from {self.external_repo_path}. "
                f"Error: {e}\n"
                f"Make sure the external repository is properly cloned and contains model/transformer.py"
            )
        
        # Create model instance
        self.model = MAMP(
            dim_in=in_channels,
            num_classes=num_class,
            dim_feat=dim_feat,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_frames=num_frames,
            num_joints=num_point,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            protocol=protocol
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MAMP.
        
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
        return self.model(x)
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from a checkpoint file.
        
        MAMP checkpoints may have different formats:
        - 'model': Standard MAMP checkpoint
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
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
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
