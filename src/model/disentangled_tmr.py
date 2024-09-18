"""
Disentangled TMR - Transformer Motion Retargeting with Architectural Disentanglement

This model implements the redesigned TMR architecture with:
1. Action Encoder (temporal focus) - captures motion dynamics
2. Identity Encoder (spatial focus) - captures skeleton structure
3. Factorized Decoder - separate processing for action and identity
4. Strong disentanglement through architecture, not just losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .action_encoder import ActionEncoder
from .identity_encoder import IdentityEncoder
from .factorized_decoder import FactorizedDecoder
from .ske_mixf import import_class


class DisentangledTMR(nn.Module):
    """
    Disentangled Transformer Motion Retargeting
    
    Architecture:
        Input: source_motion (P1, A1), target_skeleton (P2)
        
        Action Encoder: source_motion -> action_features (A1)
        Identity Encoder: target_skeleton -> identity_features (P2)
        Decoder: action_features + identity_features -> output (P2, A1)
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (25 for NTU)
        num_person: Number of persons (1 for single-person)
        graph: Graph structure for skeleton
        graph_args: Graph arguments
        in_channels: Input channels (3 for x,y,z)
        d_action: Action feature dimension
        d_identity: Identity feature dimension
        d_model: Decoder model dimension
        nhead: Number of attention heads
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
        use_pretrained_action: Whether to include the action-recognition backbone architecture
                              (randomly initialized, no pretrained weights)
        dataset: Dataset name ('ntu', 'ntu120', 'etri')
        device: Device ('cuda' or 'cpu')
    """
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(),
                 in_channels=3, d_action=512, d_identity=128, d_model=320, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 use_pretrained_action=True, dataset='ntu', device='cuda',
                 use_temporal_convs=True, use_lstm=True, identity_use_full_sequence=False,
                 tokenizer_type=None, tokenizer_dim=256, token_fusion="add", use_codebook=False,
                 codebook_size=256, codebook_dim=256, codebook_distance="euclidean",
                 vq_commitment_weight: float = 0.25):
        super().__init__()
        
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.d_action = d_action
        self.d_identity = d_identity
        self.d_model = d_model
        self.dataset = dataset
        self.device = device
        
        # Resolve graph argument
        # ActionEncoder needs string (for import_class), 
        # but IdentityEncoder/Decoder need object (for structure)
        graph_obj = None
        if isinstance(graph, str):
            # Instantiate graph object for IdentityEncoder and Decoder
            try:
                Graph = import_class(graph)
                graph_obj = Graph(**graph_args)
            except Exception as e:
                print(f"Warning: Could not instantiate graph from string '{graph}': {e}")
        else:
            graph_obj = graph

        # Extract bone connections from graph if available
        bone_connections = None
        if graph_obj is not None and hasattr(graph_obj, 'inward') and graph_obj.inward:
             bone_connections = graph_obj.inward
        
        # Action Encoder (temporal focus)
        self.action_encoder = ActionEncoder(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            graph=graph,
            graph_args=graph_args,
            in_channels=in_channels,
            d_action=d_action,
            use_pretrained=use_pretrained_action,
            dataset=dataset,
            device=device,
            use_temporal_convs=use_temporal_convs,
            use_lstm=use_lstm,
            tokenizer_type=tokenizer_type,
            tokenizer_dim=tokenizer_dim,
            token_fusion=token_fusion,
            use_codebook=use_codebook,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            codebook_distance=codebook_distance,
            vq_commitment_weight=vq_commitment_weight,
        )
        
        # Identity Encoder (spatial focus)
        self.identity_encoder = IdentityEncoder(
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            d_identity=d_identity,
            graph=graph_obj,
            use_bone_lengths=True,
            dataset=dataset,
            use_full_sequence=identity_use_full_sequence
        )
        
        # Factorized Decoder
        self.decoder = FactorizedDecoder(
            d_model=d_model,
            d_action=d_action,
            d_identity=d_identity,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_joints=num_point,
            in_channels=in_channels,
            bone_connections=bone_connections,
        )
        
    def forward(
        self,
        source_motion,
        target_skeleton=None,
        target_motion=None,
        teacher_forcing_ratio=0.0,
        autoregressive_full_context: bool = True, # Use full context for better quality
    ):
        """
        Forward pass for Disentangled TMR
        
        Args:
            source_motion: (B, C, T, V, M) - Source motion sequence
            target_skeleton: (B, C, T, V, M) - Target skeleton (structure)
            target_motion: (B, C, T, V, M) - Target motion (ground truth for training)
            teacher_forcing_ratio: Probability of using ground truth autoregressively
            autoregressive_full_context: Whether to use full context during autoregressive generation
            
        Returns:
            output: (B, C, T-1, V, M) - Generated motion
            action_features: (B, D_action) or (T, B, D_action)
            identity_features: (B, D_identity) or (T, B, D_identity)
        """
        # Encode Action
        # Input: (B, C, T, V, M)
        # Output: (T, B, D_action)
        action_features = self.action_encoder(source_motion)
        
        # Encode Identity
        # Input: (B, C, T, V, M)
        # Output: (B, D_identity) or (T, B, D_identity)
        if target_skeleton is not None:
            identity_features = self.identity_encoder(target_skeleton)
        else:
            # During inference, we might not have a target skeleton separate from source
            # But usually we do. If None, use source (reconstruction task)
            identity_features = self.identity_encoder(source_motion)
            
        # Decode
        # Output: (B, C, T-1, V, M)
        output = self.decoder(
            action_features,
            identity_features,
            target_skeleton=target_skeleton,
            target_motion=target_motion,
            teacher_forcing_ratio=teacher_forcing_ratio,
            autoregressive_full_context=autoregressive_full_context
        )
        
        return output, action_features, identity_features
    
    def encode_action(self, motion):
        """Encode action features from motion"""
        return self.action_encoder(motion)
    
    def encode_identity(self, skeleton, temporal=False):
        """
        Encode identity features from skeleton.
        If the encoder is set to full-sequence mode, this pools over time unless temporal=True.
        """
        feats = self.identity_encoder(skeleton)
        if temporal:
            return feats
        if feats.dim() == 3:  # (T, B, D_identity) -> pool over time
            feats = feats.mean(dim=0)
        return feats
    
    def decode(
        self,
        action_features,
        identity_features,
        target_skeleton=None,
        teacher_forcing_ratio=0.0,
        target_motion=None,
        autoregressive_full_context: bool = False,
    ):
        """Decode from action and identity features"""
        return self.decoder(
            action_features,
            identity_features,
            target_skeleton=target_skeleton,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_motion=target_motion,
            autoregressive_full_context=autoregressive_full_context,
        )
    
    def freeze_action_encoder(self):
        """Freeze action encoder parameters"""
        for param in self.action_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_action_encoder(self):
        """Unfreeze action encoder parameters"""
        for param in self.action_encoder.parameters():
            param.requires_grad = True
    
    def freeze_identity_encoder(self):
        """Freeze identity encoder parameters"""
        for param in self.identity_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_identity_encoder(self):
        """Unfreeze identity encoder parameters"""
        for param in self.identity_encoder.parameters():
            param.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        self.unfreeze_action_encoder()
        self.unfreeze_identity_encoder()
        self.unfreeze_decoder()

    def get_num_params(self):
        """Get number of parameters for each component"""
        action_params = sum(p.numel() for p in self.action_encoder.parameters())
        identity_params = sum(p.numel() for p in self.identity_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = action_params + identity_params + decoder_params
        
        return {
            'action_encoder': action_params,
            'identity_encoder': identity_params,
            'decoder': decoder_params,
            'total': total_params
        }


def create_disentangled_tmr(dataset='ntu', num_class=60, device='cuda', **kwargs):
    """
    Factory function to create DisentangledTMR model
    
    Args:
        dataset: Dataset name ('ntu', 'ntu120', 'etri')
        num_class: Number of action classes
        device: Device ('cuda' or 'cpu')
        **kwargs: Additional arguments for DisentangledTMR
        
    Returns:
        model: DisentangledTMR instance
    """
    # Import graph based on dataset
    if dataset in ['ntu', 'ntu120', 'ntu_smoke', 'ntu_small']:
        from ..graph.ntu_rgb_d import Graph
    elif dataset == 'etri':
        from ..graph.ntu_rgb_d import Graph  # ETRI uses same skeleton structure
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    graph = Graph()
    
    # Create model
    graph_args = kwargs.pop('graph_args', {'labeling_mode': 'spatial'})
    
    model = DisentangledTMR(
        num_class=num_class,
        num_point=25,
        num_person=1,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args,
        in_channels=3,
        dataset=dataset,
        device=device,
        **kwargs
    )
    
    return model.to(device)


if __name__ == "__main__":
    # Test model
    print("Testing DisentangledTMR...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_disentangled_tmr(dataset='ntu', num_class=49, device=device)
    
    # Print model info
    params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Action Encoder: {params['action_encoder']:,}")
    print(f"  Identity Encoder: {params['identity_encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total: {params['total']:,}")
    
    # Test forward pass
    B, C, T, V, M = 2, 3, 64, 25, 1
    source_motion = torch.randn(B, C, T, V, M).to(device)
    target_skeleton = torch.randn(B, C, T, V, M).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Source motion: {source_motion.shape}")
    print(f"  Target skeleton: {target_skeleton.shape}")
    
    # Forward pass
    output, action_features, identity_features = model(source_motion, target_skeleton, teacher_forcing_ratio=1.0)
    
    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Action features: {action_features.shape}")
    print(f"  Identity features: {identity_features.shape}")
    
    print("\n✅ DisentangledTMR test passed!")
