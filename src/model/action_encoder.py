"""
Action Encoder - Temporal Focus

Captures motion dynamics (velocity, acceleration) for action representation and
optionally fuses a randomly initialized action-recognition backbone (no
pretrained weights) via a learned gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder as SkeletonMixFormer
from .tokenizers import PositionTokenizer, DynamicsTokenizer, SimpleCodebook


class MultiScaleTemporalConv(nn.Module):
    """Optimized multi-scale temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        stride = 1  # Define stride
        # Calculate channels per conv to ensure exact division
        channels_per_conv = out_channels // len(kernel_sizes)
        remainder = out_channels % len(kernel_sizes)
        
        for i, kernel_size in enumerate(kernel_sizes):
            # Distribute remainder to first few convs
            conv_out = channels_per_conv + (1 if i < remainder else 0)
            padding = (kernel_size - 1) // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, conv_out, kernel_size, stride, padding),
                nn.BatchNorm1d(conv_out),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ))
        self.fusion = nn.Conv1d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (B, C, T)
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        # Concatenate along channel dimension
        out = torch.cat(outputs, dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class TemporalAttention(nn.Module):
    """Temporal self-attention for capturing long-range dependencies (optimized)"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        # Try to use flash attention if available
        try:
            from flash_attn.flash_attention import FlashAttention
            self.attention = FlashAttention()
            self.use_flash = True
        except ImportError:
            self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
            self.use_flash = False
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (T, B, D)
        if self.use_flash:
            # Flash attention expects (B, T, D)
            x_batch = x.permute(1, 0, 2)  # (B, T, D)
            attn_out = self.attention(x_batch, x_batch, x_batch)
            attn_out = attn_out.permute(1, 0, 2)  # (T, B, D)
        else:
            attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class ActionEncoder(nn.Module):
    """
    Action Encoder with temporal focus
    
    Captures motion dynamics through:
    1. Velocity and acceleration computation
    2. Temporal convolutions
    3. Temporal attention
    4. Optional action-recognition backbone (architecture only, random init)
       fused via a learned gate
    
    Args:
        num_class: Number of action classes
        num_point: Number of skeleton joints (25 for NTU)
        num_person: Number of persons (1 for single-person)
        graph: Graph structure for skeleton
        graph_args: Graph arguments
        in_channels: Input channels (3 for x,y,z)
        d_action: Action feature dimension
        use_pretrained: Whether to include the action backbone architecture
                        (no pretrained weights are loaded)
        dataset: Dataset name ('ntu', 'ntu120', 'etri')
        device: Device ('cuda' or 'cpu')
    """
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(),
                 in_channels=3, d_action=512, use_pretrained=True, dataset='ntu', device='cuda',
                 use_temporal_convs=True, use_lstm=True,
                 tokenizer_type=None, tokenizer_dim=256,
                 token_fusion: str = "add",
                 use_codebook=False, codebook_size=256, codebook_dim=256,
                 codebook_distance: str = "euclidean",
                 vq_commitment_weight: float = 0.25):
        super().__init__()
        
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.d_action = d_action
        # Interpret use_pretrained as "use the AR backbone architecture", but never load pretrained weights
        self.use_action_backbone = use_pretrained
        self.use_temporal_convs = use_temporal_convs
        self.use_lstm = use_lstm
        self.tokenizer_type = tokenizer_type
        self.use_codebook = use_codebook
        self.token_fusion = token_fusion
        self._last_vq = None
        
        # Tokenizers
        if tokenizer_type == 'pos':
            self.tokenizer = PositionTokenizer(num_joints=num_point, in_channels=in_channels, token_dim=tokenizer_dim)
        elif tokenizer_type == 'dynamics':
            self.tokenizer = DynamicsTokenizer(num_joints=num_point, in_channels=in_channels, token_dim=tokenizer_dim)
        else:
            self.tokenizer = None
        if tokenizer_type is not None:
            self.token_proj = nn.Linear(tokenizer_dim, 256)
            if self.token_fusion not in {"add", "replace"}:
                raise ValueError(f"Unsupported token fusion mode: {self.token_fusion}")
            self.token_scale = nn.Parameter(torch.ones(1, 1, 256)) if self.token_fusion == "add" else None
        else:
            self.token_proj = None
            self.token_scale = None

        # Codebook
        if use_codebook and self.tokenizer is not None:
            self.codebook = SimpleCodebook(
                num_codes=codebook_size,
                code_dim=codebook_dim,
                distance=codebook_distance,
                commitment_weight=vq_commitment_weight,
            )
            self._token_to_codebook = nn.Linear(tokenizer_dim, codebook_dim) if codebook_dim != tokenizer_dim else None
            self._codebook_to_token = nn.Linear(codebook_dim, tokenizer_dim) if codebook_dim != tokenizer_dim else None
        else:
            self.codebook = None
            self._token_to_codebook = None
            self._codebook_to_token = None

        # Optional: Use Skeleton-MixFormer architecture (random init) as action backbone
        if self.use_action_backbone:
            self.base_encoder = SkeletonMixFormer(
                num_class=num_class, num_point=num_point, num_person=num_person,
                graph=graph, graph_args=graph_args, in_channels=in_channels,
                debug=False, dataset=dataset, load_pretrained=False, freeze_layers=False
            )
            base_dim = 320  # Skeleton-MixFormer output dimension
        else:
            self.base_encoder = None
            base_dim = 0
        
        # Velocity and acceleration processing (used when tokenizer is None)
        # Input: position (3 channels) + velocity (3 channels) + acceleration (3 channels) = 9 channels
        self.input_proj = nn.Linear(in_channels * 3 * num_point * num_person, 256)
        
        # Temporal convolution layers (optimized)
        if self.use_temporal_convs:
            self.temporal_conv = MultiScaleTemporalConv(256, 256, kernel_sizes=[3, 5, 7])
        else:
            self.temporal_conv = None
        
        # Temporal attention layers
        self.temporal_attn1 = TemporalAttention(256, nhead=8, dropout=0.1)
        self.temporal_attn2 = TemporalAttention(256, nhead=8, dropout=0.1)
        
        # LSTM for long-range temporal dependencies
        if self.use_lstm:
            self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=False, dropout=0.1, bidirectional=False)
        else:
            self.lstm = None
        
        # Gating + projection when using the action backbone
        if self.use_action_backbone:
            self.lstm_proj = nn.Linear(256, d_action)
            self.base_proj = nn.Linear(base_dim, d_action)
            self.gate = nn.Sequential(
                nn.Linear(d_action * 2, d_action),
                nn.Sigmoid()
            )
        else:
            self.lstm_proj = nn.Linear(256, d_action)
            self.base_proj = None
            self.gate = None
        
        # Final projection
        self.output_norm = nn.LayerNorm(d_action)

    def get_vq_info(self):
        """Return VQ/codebook losses + metrics from the most recent forward pass."""
        return self._last_vq
        
    def compute_velocity_acceleration(self, x):
        """
        Compute velocity and acceleration from position
        
        Args:
            x: (B, C, T, V, M) - skeleton positions
            
        Returns:
            velocity: (B, C, T-1, V, M)
            acceleration: (B, C, T-2, V, M)
        """
        # Velocity: v_t = x_t - x_{t-1}
        velocity = x[:, :, 1:] - x[:, :, :-1]
        
        # Acceleration: a_t = v_t - v_{t-1}
        acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
        
        return velocity, acceleration
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, C, T, V, M) - skeleton sequence
            
        Returns:
            action_features: (T, B, D_action) - temporal action features
        """
        B, C, T, V, M = x.shape

        self._last_vq = None

        # Baseline motion stream: position + velocity + acceleration
        base_features = None
        base_needed = (self.tokenizer is None) or (self.token_fusion != "replace")
        if base_needed:
            velocity, acceleration = self.compute_velocity_acceleration(x)

            # Pad to match temporal dimension
            velocity_padded = F.pad(velocity, (0, 0, 0, 0, 1, 0), mode='replicate')
            acceleration_padded = F.pad(acceleration, (0, 0, 0, 0, 2, 0), mode='replicate')

            # Concatenate position, velocity, acceleration (optimized)
            base_features = torch.cat([x, velocity_padded, acceleration_padded], dim=1)  # (B, 9, T, V, M)

            # Reshape more efficiently: (B, 9, T, V, M) -> (B, T, 9*V*M)
            base_features = base_features.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 9, V, M)
            base_features = base_features.reshape(B, T, -1)  # (B, T, 9*V*M)

            # Project to feature space
            base_features = self.input_proj(base_features)  # (B, T, 256)

        # Tokenization stream (optional): produces additive features instead of replacing by default
        if self.tokenizer is not None:
            tokens = self.tokenizer(x)  # (B, T, tokenizer_dim)
            if self.codebook is not None:
                tokens_cb = self._token_to_codebook(tokens) if self._token_to_codebook is not None else tokens
                tokens_cb, indices = self.codebook(tokens_cb)  # (B, T, codebook_dim)
                tokens = self._codebook_to_token(tokens_cb) if self._codebook_to_token is not None else tokens_cb

                vq_losses = self.codebook.last_losses() or {}
                vq_metrics = self.codebook.last_metrics() or {}
                self._last_vq = {
                    "embed": vq_losses.get("embed", None),
                    "commit": vq_losses.get("commit", None),
                    "total": vq_losses.get("total", None),
                    "perplexity": vq_metrics.get("perplexity", None),
                    "usage": vq_metrics.get("usage", None),
                    "indices": indices,
                }

            token_features = self.token_proj(tokens)  # (B, T, 256)
            if self.token_fusion == "replace":
                motion_features = token_features
            else:  # add
                motion_features = base_features + self.token_scale * token_features
        else:
            motion_features = base_features

        # Ensure batch-first (B, T, 256) before temporal blocks
        if motion_features.shape[:2] == (T, B):
            motion_features = motion_features.permute(1, 0, 2).contiguous()
        
        # Temporal convolutions (optional) - optimized
        if self.use_temporal_convs:
            temp_conv = motion_features.permute(0, 2, 1).contiguous()
            temp_conv = self.temporal_conv(temp_conv)
            # Back to (T, B, 256)
            temp_features = temp_conv.permute(2, 0, 1).contiguous()
        else:
            temp_features = motion_features.permute(1, 0, 2).contiguous()  # (T, B, 256)
        
        # Temporal attention
        temp_features = self.temporal_attn1(temp_features)
        temp_features = self.temporal_attn2(temp_features)
        
        # LSTM for long-range dependencies (optional)
        if self.use_lstm:
            lstm_out, _ = self.lstm(temp_features)
        else:
            lstm_out = temp_features
        
        # Optional: Fuse with action-recognition backbone (random init)
        if self.use_action_backbone and self.base_encoder is not None:
            # Get backbone features (architecture only, random init)
            base_features = self.base_encoder(x)  # (T_reduced, B, 320)
            
            # Upsample base_features to match temporal dimension if needed
            if base_features.size(0) != T:
                # Interpolate temporal dimension
                base_features = base_features.permute(1, 2, 0).contiguous()  # (B, 320, T_reduced)
                base_features = F.interpolate(base_features, size=T, mode='linear', align_corners=False)
                base_features = base_features.permute(2, 0, 1).contiguous()  # (T, B, 320)
            
            # Project to common space
            lstm_proj = self.lstm_proj(lstm_out)
            base_proj = self.base_proj(base_features)

            # Learned gate blends motion dynamics and backbone cues
            gate_input = torch.cat([lstm_proj, base_proj], dim=-1)
            gate = self.gate(gate_input)
            action_features = gate * lstm_proj + (1 - gate) * base_proj  # (T, B, D_action)
        else:
            action_features = self.lstm_proj(lstm_out)  # (T, B, D_action)
        
        # Normalize
        action_features = self.output_norm(action_features)
        
        return action_features
