"""
Identity Encoder - Spatial Focus
Captures skeleton structure (bone lengths, joint positions) for identity representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialGCN(nn.Module):
    """Spatial Graph Convolutional Network for skeleton structure"""
    def __init__(self, in_channels, out_channels, A, residual=True):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        
        if residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_conv = None
        
        # Adjacency matrix (skeleton graph)
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        
    def forward(self, x):
        # x: (B, C, V, 1)
        res = x
        
        # Graph convolution
        # Optimized implementation for (K, V, V) or (V, V) adjacency matrix
        N, C, V, T = x.size()
        A = self.A.to(x.device)
        
        # Support different A shapes
        if A.dim() == 2:
            # Single graph (V, V)
            # (N, C, V, T) -> (N, C, T, V)
            x = x.permute(0, 1, 3, 2).contiguous()
            x = torch.einsum('nctv,vw->nctw', (x, A))
            x = x.permute(0, 1, 3, 2).contiguous()
        elif A.dim() == 3:
            # K-partition graph (K, V, V) - typically used with multiple subsets of weights
            # This implementation assumes we average or sum over K if weights aren't K-specific
            # But standard GCN usually expects (K, V, V) A and (K, Out, In) weights.
            # Here self.gcn is just Conv2d(in, out, 1), effectively (Out, In, 1, 1).
            # We'll sum the contributions of the K partitions.
            
            # (N, C, V, T) -> (N, C, T, V)
            x_in = x.permute(0, 1, 3, 2).contiguous()
            x_out = 0
            for k in range(A.size(0)):
                x_out = x_out + torch.einsum('nctv,vw->nctw', (x_in, A[k]))
            x = x_out.permute(0, 1, 3, 2).contiguous()
            
        x = self.gcn(x)
        x = self.bn(x)
        
        # Residual connection
        if self.residual:
            if self.residual_conv is not None:
                res = self.residual_conv(res)
            x = x + res
        
        x = self.relu(x)
        return x


class BoneLengthEncoder(nn.Module):
    """Encode bone lengths as identity features"""
    def __init__(self, num_bones=24, d_model=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_bones, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, bone_lengths):
        # bone_lengths: (B, num_bones)
        return self.encoder(bone_lengths)


class IdentityEncoder(nn.Module):
    """
    Identity Encoder with spatial focus
    
    Captures skeleton structure through:
    1. Static pose (averaged or first frame)
    2. Bone lengths (identity-specific)
    3. Spatial graph convolutions
    4. Joint position encoding
    
    Args:
        num_point: Number of skeleton joints (25 for NTU)
        num_person: Number of persons (1 for single-person)
        in_channels: Input channels (3 for x,y,z)
        d_identity: Identity feature dimension
        graph: Graph structure for skeleton (adjacency matrix)
        use_bone_lengths: Whether to use bone length features
        dataset: Dataset name ('ntu', 'ntu120', 'etri')
    """
    def __init__(self, num_point=25, num_person=1, in_channels=3, d_identity=128,
                 graph=None, use_bone_lengths=True, dataset='ntu', use_full_sequence=False):
        super().__init__()
        
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.d_identity = d_identity
        self.use_bone_lengths = use_bone_lengths
        self.use_full_sequence = use_full_sequence
        
        # Define skeleton connections (bones)
        self.bone_connections = None
        
        # Adjacency matrix for GCN and bone connections from graph
        if graph is not None:
            A = graph.A if hasattr(graph, 'A') else np.eye(num_point)
            
            # Try to get bone connections from graph
            if hasattr(graph, 'inward') and graph.inward:
                self.bone_connections = graph.inward
        else:
            A = np.eye(num_point)
            
        # Fallback to hardcoded NTU connections if not available from graph
        if self.bone_connections is None:
            # Based on NTU RGB+D skeleton structure (25 joints)
            self.bone_connections = [
                (0, 1), (1, 20), (20, 2), (2, 3),  # Spine
                (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Right arm
                (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Left arm
                (0, 12), (12, 13), (13, 14), (14, 15),  # Right leg
                (0, 16), (16, 17), (17, 18), (18, 19),  # Left leg
            ]
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Spatial GCN layers
        self.gcn1 = SpatialGCN(64, 128, A, residual=True)
        self.gcn2 = SpatialGCN(128, 256, A, residual=True)
        self.gcn3 = SpatialGCN(256, 256, A, residual=True)
        
        # Bone length encoder
        if use_bone_lengths:
            num_bones = len(self.bone_connections)
            self.bone_encoder = BoneLengthEncoder(num_bones, d_model=128)
        else:
            self.bone_encoder = None
        
        # Spatial attention + global pooling
        self.spatial_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1, batch_first=True)
        
        # Fusion layer (pool + optional bone lengths)
        fused_input_dim = 256  # after pooling
        if use_bone_lengths:
            fused_input_dim += 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_identity)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_identity)
        
    def compute_bone_lengths(self, x):
        """
        Compute bone lengths from skeleton positions
        
        Args:
            x: (B, C, V, M) - skeleton positions (static pose)
            
        Returns:
            bone_lengths: (B, num_bones)
        """
        B, C, V, M = x.shape
        bone_lengths = []
        
        for joint1, joint2 in self.bone_connections:
            # Get joint positions
            p1 = x[:, :, joint1, 0]  # (B, C)
            p2 = x[:, :, joint2, 0]  # (B, C)
            
            # Compute Euclidean distance
            bone_length = torch.norm(p2 - p1, dim=1)  # (B,)
            bone_lengths.append(bone_length)
        
        # Stack to (B, num_bones)
        bone_lengths = torch.stack(bone_lengths, dim=1)
        return bone_lengths
    
    def get_static_pose(self, x, method='mean'):
        """
        Extract static pose from temporal sequence
        
        Args:
            x: (B, C, T, V, M) - skeleton sequence
            method: 'mean', 'first', 'last'
            
        Returns:
            static_pose: (B, C, V, M)
        """
        if method == 'mean':
            # Average over time
            static_pose = x.mean(dim=2)
        elif method == 'first':
            # Use first frame
            static_pose = x[:, :, 0, :, :]
        elif method == 'last':
            # Use last frame
            static_pose = x[:, :, -1, :, :]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return static_pose
    
    def forward(self, x, static_method='mean'):
        """
        Forward pass
        
        Args:
            x: (B, C, T, V, M) - skeleton sequence
            static_method: Method to extract static pose ('mean', 'first', 'last')
            
        Returns:
            identity_features: (B, D_identity) for static mode
                               or (T, B, D_identity) for full-sequence mode
        """
        B, C, T, V, M = x.shape
        
        if not self.use_full_sequence:
            # Extract static pose (identity doesn't change over time)
            static_pose = self.get_static_pose(x, method=static_method)  # (B, C, V, M)
            
            # Remove person dimension (assume M=1)
            static_pose = static_pose.squeeze(-1)  # (B, C, V)
            static_pose = static_pose.unsqueeze(-1)  # (B, C, V, 1) for Conv2d
            
            # Input projection
            spatial_features = self.input_proj(static_pose)  # (B, 64, V, 1)
            
            # Spatial GCN layers
            spatial_features = self.gcn1(spatial_features)  # (B, 128, V, 1)
            spatial_features = self.gcn2(spatial_features)  # (B, 256, V, 1)
            spatial_features = self.gcn3(spatial_features)  # (B, 256, V, 1)
            
            # Spatial attention
            # Reshape for attention: (B, V, 256)
            spatial_features = spatial_features.squeeze(-1).permute(0, 2, 1).contiguous()
            spatial_features, _ = self.spatial_attn(spatial_features, spatial_features, spatial_features)
            
            # Global average pool over joints
            pooled_spatial = spatial_features.mean(dim=1)  # (B, 256)
            
            # Optional: Add bone length features
            if self.use_bone_lengths and self.bone_encoder is not None:
                bone_lengths = self.compute_bone_lengths(static_pose)  # (B, num_bones)
                bone_features = self.bone_encoder(bone_lengths)  # (B, 128)
                combined_features = torch.cat([pooled_spatial, bone_features], dim=1)
            else:
                combined_features = pooled_spatial

            # Fusion to identity features (static)
            identity_features = self.fusion(combined_features)  # (B, D_identity)
            identity_features = self.output_norm(identity_features)
        else:
            # Full sequence identity: process every frame and keep temporal dimension
            # (B, C, T, V, M) -> (B*T, C, V, 1)
            seq = x.squeeze(-1).permute(0, 2, 1, 3).contiguous().view(B * T, C, V, 1)
            spatial_features = self.input_proj(seq)  # (B*T, 64, V, 1)
            spatial_features = self.gcn1(spatial_features)
            spatial_features = self.gcn2(spatial_features)
            spatial_features = self.gcn3(spatial_features)
            
            # Attention per frame
            spatial_features = spatial_features.squeeze(-1).permute(0, 2, 1).contiguous()  # (B*T, V, 256)
            spatial_features, _ = self.spatial_attn(spatial_features, spatial_features, spatial_features)
            pooled_spatial = spatial_features.mean(dim=1)  # (B*T, 256)
            
            # Optional bone lengths per frame
            if self.use_bone_lengths and self.bone_encoder is not None:
                frame_pose = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, V, M)
                frame_pose = frame_pose.squeeze(-1).unsqueeze(-1)
                bone_lengths = self.compute_bone_lengths(frame_pose)
                bone_features = self.bone_encoder(bone_lengths)  # (B*T, 128)
                combined_features = torch.cat([pooled_spatial, bone_features], dim=1)
            else:
                combined_features = pooled_spatial
        
            # Fusion per frame and restore time dimension
            identity_features = self.fusion(combined_features)  # (B*T, D_identity)
            identity_features = self.output_norm(identity_features)
            identity_features = identity_features.view(B, T, -1).permute(1, 0, 2).contiguous()  # (T, B, D_identity)
        
        return identity_features
