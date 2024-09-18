"""
Wrapper for HD-GCN (Hierarchically Decomposed Graph Convolution).

Paper: Lee et al., "Hierarchically Decomposed Graph Convolutional Networks 
       for Skeleton-Based Action Recognition", ICCV 2023
Performance: 90.1% (NTU60 X-View), 89.3% (NTU120 X-Set)
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base_wrapper import BaseModelWrapper


class HDGCNWrapper(BaseModelWrapper):
    """
    Wrapper for HD-GCN model from external_repo_for_reference/HD-GCN/
    
    HD-GCN uses hierarchical decomposition of graph structures for
    skeleton-based action recognition.
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
        graph: Graph type (default: 'graph.ntu_rgb_d_hierarchy.Graph')
        graph_args: Graph arguments (default: {})
        drop_out: Dropout rate (default: 0)
        adaptive: Use adaptive graph topology (default: True)
        **kwargs: Additional model-specific arguments
    """
    
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        graph: str = 'graph.ntu_rgb_d_hierarchy.Graph',
        graph_args: dict = None,
        drop_out: float = 0,
        adaptive: bool = True,
        **kwargs
    ):
        super().__init__(
            model_name='HD-GCN',
            external_repo_name='HD-GCN',
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        # Default graph arguments
        if graph_args is None:
            graph_args = {}
        
        # Import HD-GCN model from external repo
        try:
            # HD-GCN has circular import issues in graph/__init__.py
            # We need to import the graph module files directly first
            import sys
            import importlib.util
            
            # Load graph.tools directly
            tools_path = self.external_repo_path / 'graph' / 'tools.py'
            spec = importlib.util.spec_from_file_location('graph.tools', tools_path)
            tools_module = importlib.util.module_from_spec(spec)
            sys.modules['graph.tools'] = tools_module
            spec.loader.exec_module(tools_module)
            
            # Load graph.ntu_rgb_d_hierarchy directly
            hierarchy_path = self.external_repo_path / 'graph' / 'ntu_rgb_d_hierarchy.py'
            spec = importlib.util.spec_from_file_location('graph.ntu_rgb_d_hierarchy', hierarchy_path)
            hierarchy_module = importlib.util.module_from_spec(spec)
            sys.modules['graph.ntu_rgb_d_hierarchy'] = hierarchy_module
            spec.loader.exec_module(hierarchy_module)
            
            # Now import the model
            HDGCN = self._import_from_path('model.HDGCN', 'Model', self.external_repo_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import HD-GCN model from {self.external_repo_path}. "
                f"Error: {e}\n"
                f"Make sure the external repository is properly cloned and contains model/HDGCN.py"
            )
        
        # Create model instance
        self.model = HDGCN(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args,
            in_channels=in_channels,
            drop_out=drop_out,
            adaptive=adaptive
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HD-GCN.
        
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
        
        HD-GCN checkpoints may have different formats:
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
