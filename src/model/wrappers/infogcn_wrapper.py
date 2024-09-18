"""
Wrapper for InfoGCN (Information Bottleneck Graph Convolution).

Paper: Chi et al., "Infogcn: Representation learning for human skeleton-based 
       action recognition", CVPR 2022
Performance: 93.0% (NTU60 X-View), 89.8% (NTU120 X-Set)
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base_wrapper import BaseModelWrapper


class InfoGCNWrapper(BaseModelWrapper):
    """
    Wrapper for InfoGCN model from external_repo_for_reference/infogcn/
    
    InfoGCN uses information bottleneck principle to learn discriminative
    skeleton representations for action recognition.
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
        graph: Graph type (default: 'graph.ntu_rgb_d.Graph')
        drop_out: Dropout rate (default: 0)
        num_head: Number of attention heads (default: 3)
        noise_ratio: Noise ratio for latent sampling (default: 0.1)
        k: Power of adjacency matrix (default: 0)
        gain: Gain for orthogonal initialization (default: 1)
        **kwargs: Additional model-specific arguments
    """
    
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        graph: str = 'graph.ntu_rgb_d.Graph',
        drop_out: float = 0,
        num_head: int = 3,
        noise_ratio: float = 0.1,
        k: int = 0,
        gain: int = 1,
        **kwargs
    ):
        super().__init__(
            model_name='InfoGCN',
            external_repo_name='infogcn',
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        # Import InfoGCN model from external repo
        try:
            InfoGCN = self._import_from_path('model.infogcn', 'InfoGCN', self.external_repo_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import InfoGCN model from {self.external_repo_path}. "
                f"Error: {e}\n"
                f"Make sure the external repository is properly cloned and contains model/infogcn.py"
            )
        
        # Ensure InfoGCN's repo is first in sys.path when creating the model
        # This prevents conflicts with other repos (e.g., CTR-GCN) that have similar module names
        import sys
        repo_str = str(self.external_repo_path)
        original_path = sys.path.copy()
        try:
            # Remove all other external repos from path temporarily
            sys.path = [p for p in sys.path if 'external_repo_for_reference' not in p or repo_str in p]
            # Ensure our repo is first
            if repo_str in sys.path:
                sys.path.remove(repo_str)
            sys.path.insert(0, repo_str)
            
            # Create model instance
            self.model = InfoGCN(
                num_class=num_class,
                num_point=num_point,
                num_person=num_person,
                graph=graph,
                in_channels=in_channels,
                drop_out=drop_out,
                num_head=num_head,
                noise_ratio=noise_ratio,
                k=k,
                gain=gain
            )
        finally:
            # Restore original sys.path
            sys.path = original_path
        
        # Convert model to float32 to avoid dtype issues
        # InfoGCN creates some tensors as float64 from numpy
        self.model = self.model.float()
        
        # Also convert A_vector specifically
        if hasattr(self.model, 'A_vector'):
            self.model.A_vector = self.model.A_vector.float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through InfoGCN.
        
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
        # InfoGCN returns (y_hat, z) tuple, we only need y_hat
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from a checkpoint file.
        
        InfoGCN checkpoints may have different formats:
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
