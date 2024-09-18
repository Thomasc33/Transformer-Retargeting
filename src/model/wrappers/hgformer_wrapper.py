"""
Wrapper for HGformer (Autoregressive Adaptive Hypergraph Transformer).

Paper: "Autoregressive Adaptive Hypergraph Transformer for Skeleton-based 
       Activity Recognition", WACV 2025
Performance: TBD (WACV 2025)
"""

import torch
import torch.nn as nn
from pathlib import Path

from .base_wrapper import BaseModelWrapper


class HGformerWrapper(BaseModelWrapper):
    """
    Wrapper for HGformer model from external_repo_for_reference/AutoregAd-HGformer/
    
    HGformer uses hypergraph modeling with autoregressive adaptation for
    skeleton-based action recognition.
    
    Note: HGformer has a complex forward signature requiring additional inputs
    (y, joint_label, he_weight). For compatibility with our pipeline, we provide
    default values for these parameters.
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (default: 25 for NTU)
        num_person: Number of persons (default: 1 for single-actor)
        in_channels: Number of input channels (default: 3 for x,y,z)
        nf: Number of features (default: 64)
        oc: Output channels (default: 128)
        graph: Graph type (default: 'graph.ntu_rgb_d.Graph')
        graph_args: Graph arguments (default: {})
        drop_out: Dropout rate (default: 0)
        num_of_heads: Number of attention heads (default: 9)
        **kwargs: Additional model-specific arguments
    """
    
    def __init__(
        self,
        num_class: int,
        num_point: int = 25,
        num_person: int = 1,
        in_channels: int = 3,
        nf: int = 216,  # Must match 24*num_of_heads for architecture compatibility
        oc: int = 216,  # Should match nf (24 * num_of_heads)
        graph: str = 'graph.ntu_rgb_d.Graph',
        graph_args: dict = None,
        drop_out: float = 0,
        num_of_heads: int = 9,
        **kwargs
    ):
        super().__init__(
            model_name='HGformer',
            external_repo_name='AutoregAd-HGformer',
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            **kwargs
        )
        
        # Default graph arguments
        if graph_args is None:
            graph_args = {}
        
        # Import HGformer model from external repo
        try:
            HGformer = self._import_from_path('model.Hyperformer', 'final_model', self.external_repo_path)
        except Exception as e:
            raise ImportError(
                f"Failed to import HGformer model from {self.external_repo_path}. "
                f"Error: {e}\n"
                f"Note: HGformer has bugs in external code (num_points/super() issues)\n"
                f"Make sure the external repository is properly cloned and contains model/Hyperformer.py"
            )
        
        # Create model instance
        # final_model wraps the Model class and adds classification head
        
        self.model = HGformer(
            num_class=num_class,
            num_features=nf,
            out_channels=in_channels,  # Should be 3 (input channels), not oc
            n_points=num_point,
            n_person=num_person,
            graph=graph,
            graph_args=graph_args,
            drop_out=drop_out
        )
        
        # Store default parameters for forward pass
        self.num_point = num_point
        
        # Fix the weight_find module in hyp_gen to return correct shape
        # The WeightMlp returns (N, 1, 1, C) but we need (N, 1, 1, V)
        # We'll replace it with a module that transposes the output
        if hasattr(self.model.classifier_model, 'gen') and hasattr(self.model.classifier_model.gen, 'weight_find'):
            original_weight_find = self.model.classifier_model.gen.weight_find
            
            class FixedWeightModule(nn.Module):
                def __init__(self, original_module, num_points):
                    super().__init__()
                    self.original = original_module
                    self.num_points = num_points
                
                def forward(self, x):
                    # x: (N, C, T, V)
                    # We need to return (N, 1, 1, V) instead of (N, 1, 1, C)
                    # Simple approach: average over channels and time
                    N, C, T, V = x.shape
                    # Average over channels and time to get (N, V)
                    weights = x.mean(dim=(1, 2))  # (N, V)
                    # Normalize to get weights
                    weights = torch.sigmoid(weights)  # (N, V)
                    # Reshape to (N, 1, 1, V)
                    return weights.unsqueeze(1).unsqueeze(1)
            
            self.model.classifier_model.gen.weight_find = FixedWeightModule(original_weight_find, num_point)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HGformer.
        
        Note: HGformer's original forward requires (x, y, joint_label, he_weight).
        We provide default values for compatibility with our pipeline.
        
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
        # Create default parameters for HGformer's forward pass
        N, C, T, V, M = x.size()
        device = x.device
        
        # y: dummy target (not used during inference)
        y = torch.zeros(N, dtype=torch.long, device=device)
        
        # joint_label: list of joint group labels
        # Use a single group to avoid mismatches with dynamically generated groups
        # Format: [group_id_for_joint_0, group_id_for_joint_1, ..., group_id_for_joint_V-1]
        # All joints in group 0
        joint_label = [0] * V
        
        # he_weight: hyperedge weights
        # Must match the number of groups in joint_label
        # We use 1 group (all joints in group 0)
        # Expected shape: (num_groups,) - 1D tensor
        # The model will unsqueeze(1) to make it (num_groups, 1), then permute(1,0) to (1, num_groups),
        # then repeat(b,1) to (b, num_groups)
        num_groups = 1
        he_weight = torch.ones(num_groups, dtype=torch.float32, device=device)
        
        # final_model returns (op, inp, recon, joint_label, he_weight, qe)
        # We only need the classification output (op)
        op, _, _, _, _, _ = self.model(x, y, tr=False, jl=joint_label, he=he_weight)
        return op
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pre-trained weights from a checkpoint file.
        
        HGformer checkpoints may have different formats:
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
