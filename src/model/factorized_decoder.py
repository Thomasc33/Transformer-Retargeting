"""
Factorized Decoder - Strong Decoder with Separate Action and Identity Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusionLayer(nn.Module):
    """Adaptive fusion of action and identity features"""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, action_features, identity_features):
        """
        Args:
            action_features: (T, B, D) or (B, D)
            identity_features: (T, B, D) or (B, D)
        Returns:
            fused: (T, B, D) or (B, D)
        """
        # Concatenate
        combined = torch.cat([action_features, identity_features], dim=-1)
        
        # Compute gate
        gate = self.gate(combined)
        
        # Adaptive fusion
        fused = gate * action_features + (1 - gate) * identity_features
        
        # Normalize
        fused = self.norm(fused)
        
        return fused


class PhysicalConstraintLayer(nn.Module):
    """Apply physical constraints (bone lengths, joint limits)"""
    def __init__(self, num_joints=25, bone_connections=None):
        super().__init__()
        self.num_joints = num_joints
        
        # Define bone connections for NTU dataset
        if bone_connections is None:
            self.bone_connections = [
                (0, 1), (1, 20), (20, 2), (2, 3),  # Spine
                (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Right arm
                (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Left arm
                (0, 12), (12, 13), (13, 14), (14, 15),  # Right leg
                (0, 16), (16, 17), (17, 18), (18, 19),  # Left leg
            ]
        else:
            self.bone_connections = bone_connections
        
    def forward(self, x, target_bone_lengths=None):
        """
        Apply soft bone length constraints
        
        Args:
            x: (T, B, D) - decoder features (will be projected to skeleton)
            target_bone_lengths: Optional target bone lengths
            
        Returns:
            x: (T, B, D) - constrained features
        """
        # For now, just return x (constraints applied in loss function)
        # In future, could implement differentiable IK or bone length projection
        return x


class FactorizedDecoderLayer(nn.Module):
    """Single decoder layer with separate action and identity processing"""
    def __init__(self, d_model, d_action, d_identity, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_action = d_action
        self.d_identity = d_identity

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Separate cross-attention for action and identity
        self.action_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.identity_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Adaptive fusion
        self.fusion = AdaptiveFusionLayer(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, action_memory, identity_memory, tgt_mask=None, identity_is_temporal=False):
        """
        Args:
            tgt: (T, B, D_model) - target sequence
            action_memory: (T, B, D_action) - action features from encoder
            identity_memory: (B, D_identity) or (T, B, D_identity) - identity features from encoder
            tgt_mask: Causal mask for self-attention
            identity_is_temporal: Whether identity_memory is temporal (True) or static (False)
            
        Returns:
            output: (T, B, D_model)
        """
        T, B, _ = tgt.shape
        
        # 1. Self-attention with causal mask
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 2. Cross-attention with action features
        # NOTE: action_memory is already projected to d_model by FactorizedDecoder
        action_attn, _ = self.action_cross_attn(tgt, action_memory, action_memory)

        # 3. Cross-attention with identity features
        if identity_is_temporal:
            identity_memory_expanded = identity_memory  # (T, B, D_model)
        else:
            identity_memory_expanded = identity_memory.unsqueeze(0).expand(T, -1, -1)  # (T, B, D_model)
        identity_attn, _ = self.identity_cross_attn(tgt, identity_memory_expanded, identity_memory_expanded)
        
        # 4. Adaptive fusion of action and identity
        fused_attn = self.fusion(action_attn, identity_attn)
        tgt = self.norm2(tgt + self.dropout(fused_attn))
        
        # 5. Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        
        return tgt


class FactorizedDecoder(nn.Module):
    """
    Factorized Decoder with separate action and identity processing
    
    Key features:
    1. Separate cross-attention for action and identity
    2. Adaptive fusion layer
    3. Physical constraints
    4. Autoregressive generation with teacher forcing
    
    Args:
        d_model: Model dimension
        d_action: Action feature dimension
        d_identity: Identity feature dimension
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
        num_joints: Number of skeleton joints
        in_channels: Input channels (3 for x,y,z)
    """
    def __init__(self, d_model=320, d_action=256, d_identity=256, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, num_joints=25, in_channels=3, bone_connections=None):
        super().__init__()

        self.d_model = d_model
        self.d_action = d_action
        self.d_identity = d_identity
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Decoder layers
        self.layers = nn.ModuleList([
            FactorizedDecoderLayer(d_model, d_action, d_identity, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Input projection (skeleton to d_model)
        self.input_proj = nn.Linear(in_channels * num_joints, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 300, d_model) * 0.02)
        
        # Output projection (d_model to skeleton)
        self.output_proj = nn.Linear(d_model, in_channels * num_joints)
        
        # Physical constraints
        self.physical_constraint = PhysicalConstraintLayer(num_joints, bone_connections=bone_connections)
        
        # Projection layers for action and identity features
        self.action_proj = nn.Linear(d_action, d_model)
        self.identity_proj = nn.Linear(d_identity, d_model)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        action_features,
        identity_features,
        target_skeleton=None,
        teacher_forcing_ratio=1.0,
        target_motion=None,
        autoregressive_full_context: bool = False,
    ):
        """
        Forward pass with autoregressive generation
        
        Args:
            action_features: (T, B, D_action) - action features from encoder
            identity_features: (B, D_identity) or (T, B, D_identity) - identity features from encoder
            target_skeleton: (B, C, T, V, M) - target skeleton reference (identity + start frame)
            teacher_forcing_ratio: Probability of using teacher forcing
            target_motion: (B, C, T, V, M) - ground truth motion for teacher forcing (optional)
            
        Returns:
            output: (B, C, T-1, V, M) - generated skeleton sequence
        """
        T, B, _ = action_features.shape
        
        # Project action and identity features to d_model
        action_memory = self.action_proj(action_features)  # (T, B, D_model)
        if identity_features.dim() == 2:
            # Static identity
            identity_memory = self.identity_proj(identity_features)  # (B, D_model)
            identity_is_temporal = False
        elif identity_features.dim() == 3:
            # Temporal identity
            # identity_features: (T, B, D_identity)
            identity_memory = self.identity_proj(identity_features)  # (T, B, D_model)
            identity_is_temporal = True
        else:
            raise ValueError(f"Unexpected identity_features dim: {identity_features.dim()}")
        
        teacher_source = target_motion if target_motion is not None else target_skeleton

        # Determine if using teacher forcing
        use_teacher_forcing = (
            self.training
            and teacher_source is not None
            and torch.rand(1).item() < teacher_forcing_ratio
        )
        
        if use_teacher_forcing:
            # Teacher forcing: use ground truth as input
            # teacher_source: (B, C, T, V, M)
            tgt_input = teacher_source[:, :, :-1, :, :]  # (B, C, T-1, V, M)
            # Match inference: start from target_skeleton's first frame when target_motion is provided.
            if target_motion is not None and target_skeleton is not None:
                tgt_input = tgt_input.clone()
                tgt_input[:, :, 0:1, :, :] = target_skeleton[:, :, 0:1, :, :]
            tgt_input = tgt_input.permute(2, 0, 1, 3, 4).contiguous()  # (T-1, B, C, V, M)
            tgt_input = tgt_input.reshape(T-1, B, -1)  # (T-1, B, C*V*M) - use reshape
            tgt_input = self.input_proj(tgt_input)  # (T-1, B, D_model)
            
            # Add positional encoding
            tgt_input = tgt_input + self.pos_encoding[:, :T-1, :].permute(1, 0, 2)
            
            # Generate causal mask
            tgt_mask = self.generate_square_subsequent_mask(T-1).to(tgt_input.device)
            
            # Pass through decoder layers
            tgt = tgt_input
            for layer in self.layers:
                tgt = layer(
                    tgt,
                    action_memory[:-1],
                    identity_memory if identity_is_temporal else identity_memory,
                    tgt_mask,
                    identity_is_temporal=identity_is_temporal
                )
            
            # Apply physical constraints
            tgt = self.physical_constraint(tgt)

            # Project to output
            output = self.output_proj(tgt)  # (T-1, B, C*V*M)

        else:
            # Autoregressive generation
            # Start with first frame of target (or zeros)
            start_source = target_skeleton if target_skeleton is not None else teacher_source
            if start_source is not None:
                current_frame = start_source[:, :, 0, :, :]  # (B, C, V, M)
            else:
                current_frame = torch.zeros(B, self.in_channels, self.num_joints, 1, device=action_features.device)

            # Use action_memory[:-1] to match teacher forcing dimensions (T-1 frames)
            action_memory_for_gen = action_memory[:-1]  # (T-1, B, D_model)

            if autoregressive_full_context:
                # Full-context autoregressive decoding: feed the entire generated prefix
                # each step (matches teacher-forcing self-attention behavior).
                generated_frames = [current_frame]  # list of (B, C, V, 1)
                outputs = []

                for _t in range(T - 1):
                    L = len(generated_frames)
                    tgt_stack = torch.stack(generated_frames, dim=2)  # (B, C, L, V, 1)
                    tgt_input = tgt_stack.permute(2, 0, 1, 3, 4).contiguous().reshape(L, B, -1)  # (L, B, C*V)
                    tgt_input = self.input_proj(tgt_input)  # (L, B, D_model)
                    tgt_input = tgt_input + self.pos_encoding[:, :L, :].permute(1, 0, 2)

                    tgt_mask = self.generate_square_subsequent_mask(L).to(tgt_input.device)

                    tgt = tgt_input
                    for layer in self.layers:
                        tgt = layer(
                            tgt,
                            action_memory_for_gen,
                            identity_memory if identity_is_temporal else identity_memory,
                            tgt_mask=tgt_mask,
                            identity_is_temporal=identity_is_temporal,
                        )

                    tgt = self.physical_constraint(tgt)

                    output_token = self.output_proj(tgt[-1:])  # (1, B, C*V)
                    outputs.append(output_token)

                    current_frame = output_token.reshape(B, self.in_channels, self.num_joints, 1)
                    generated_frames.append(current_frame)

                output = torch.cat(outputs, dim=0)  # (T-1, B, C*V)
            else:
                # Fast autoregressive decoding: only feed the previous frame (1-token context).
                outputs = []

                for t in range(T - 1):
                    current_input = current_frame.reshape(B, -1)  # (B, C*V*M)
                    current_input = self.input_proj(current_input).unsqueeze(0)  # (1, B, D_model)
                    current_input = current_input + self.pos_encoding[:, t : t + 1, :].permute(1, 0, 2)

                    tgt = current_input
                    for layer in self.layers:
                        tgt = layer(
                            tgt,
                            action_memory_for_gen,
                            identity_memory if identity_is_temporal else identity_memory,
                            tgt_mask=None,
                            identity_is_temporal=identity_is_temporal,
                        )

                    tgt = self.physical_constraint(tgt)

                    output_frame = self.output_proj(tgt)  # (1, B, C*V*M)
                    outputs.append(output_frame)

                    current_frame = output_frame.reshape(B, self.in_channels, self.num_joints, 1)

                output = torch.cat(outputs, dim=0)  # (T-1, B, C*V*M)

        # Reshape to (B, C, T-1, V, M)
        output = output.reshape(T-1, B, self.in_channels, self.num_joints, 1)  # use reshape
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        
        return output
