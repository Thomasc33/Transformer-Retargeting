"""
Tokenizers for skeleton sequences.

These modules convert raw skeleton tensors (B, C, T, V, M) into token sequences
for the action encoder. They are lightweight and do not rely on pretrained
weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionTokenizer(nn.Module):
    """
    Flatten raw positions per frame into tokens.
    Output shape: (B, T, D)
    """
    def __init__(self, num_joints=25, in_channels=3, token_dim=256):
        super().__init__()
        self.proj = nn.Linear(num_joints * in_channels, token_dim)

    def forward(self, x):
        # x: (B, C, T, V, M)
        B, C, T, V, M = x.shape
        x = x.squeeze(-1)  # (B, C, T, V)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V * C)
        tokens = self.proj(x)  # (B, T, token_dim)
        return tokens


class DynamicsTokenizer(nn.Module):
    """
    Use position + velocity + acceleration (+ optional bone lengths) as tokens.
    Output shape: (B, T, D)
    """
    def __init__(self, num_joints=25, in_channels=3, token_dim=256, include_bones=True):
        super().__init__()
        self.include_bones = include_bones
        base_dim = num_joints * in_channels * 3  # pos + vel + acc
        if include_bones:
            # 24 bones for NTU skeleton; include as scalar lengths
            base_dim += 24
        self.proj = nn.Linear(base_dim, token_dim)

        # Bone pairs for NTU
        self.bone_connections = [
            (0, 1), (1, 20), (20, 2), (2, 3),
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
            (0, 12), (12, 13), (13, 14), (14, 15),
            (0, 16), (16, 17), (17, 18), (18, 19),
        ]

    def compute_velocity_acceleration(self, x):
        v = x[:, :, 1:] - x[:, :, :-1]
        a = v[:, :, 1:] - v[:, :, :-1]
        v = F.pad(v, (0, 0, 0, 0, 1, 0), mode='replicate')
        a = F.pad(a, (0, 0, 0, 0, 2, 0), mode='replicate')
        return v, a

    def compute_bone_lengths(self, pose):
        # pose: (B, C, V, 1)
        lengths = []
        for j1, j2 in self.bone_connections:
            p1 = pose[:, :, j1, 0]
            p2 = pose[:, :, j2, 0]
            lengths.append(torch.norm(p2 - p1, dim=1))
        return torch.stack(lengths, dim=1)  # (B, num_bones)

    def forward(self, x):
        # x: (B, C, T, V, M)
        B, C, T, V, M = x.shape
        v, a = self.compute_velocity_acceleration(x)
        pose = x

        pose_flat = pose.permute(0, 2, 3, 1, 4).contiguous().view(B, T, V * C)
        v_flat = v.permute(0, 2, 3, 1, 4).contiguous().view(B, T, V * C)
        a_flat = a.permute(0, 2, 3, 1, 4).contiguous().view(B, T, V * C)

        if self.include_bones:
            # bone lengths from current pose per frame
            pose_per_frame = pose.squeeze(-1).permute(0, 2, 1, 3).contiguous().view(B * T, C, V, 1)
            bone_lengths = self.compute_bone_lengths(pose_per_frame).view(B, T, -1)
            tokens = torch.cat([pose_flat, v_flat, a_flat, bone_lengths], dim=-1)
        else:
            tokens = torch.cat([pose_flat, v_flat, a_flat], dim=-1)

        tokens = self.proj(tokens)  # (B, T, token_dim)
        return tokens


class SimpleCodebook(nn.Module):
    """
    Vector-quantization codebook (VQ-VAE style).

    Notes:
    - Quantization is non-differentiable w.r.t. the encoder outputs due to the
      argmin/argmax. We use the straight-through estimator so gradients flow to
      the encoder.
    - Codebook entries are trained via the standard VQ-VAE embedding loss.
    """
    def __init__(
        self,
        num_codes=256,
        code_dim=256,
        distance: str = "euclidean",
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        if distance not in {"euclidean", "cosine"}:
            raise ValueError(f"Unsupported codebook distance: {distance}")
        self.distance = distance
        self.commitment_weight = commitment_weight
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.1)
        self._last_losses = None
        self._last_metrics = None

    def forward(self, tokens):
        """
        tokens: (B, T, D)
        returns quantized_tokens with straight-through estimator
        """
        B, T, D = tokens.shape
        flat = tokens.view(B * T, D)

        if self.distance == "cosine":
            code_norm = F.normalize(self.codebook, dim=1)
            flat_norm = F.normalize(flat, dim=1)
            sims = torch.matmul(flat_norm, code_norm.t())  # (B*T, num_codes)
            indices = sims.argmax(dim=1)
        else:  # euclidean
            # Squared L2 distance: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
            flat_sq = (flat ** 2).sum(dim=1, keepdim=True)  # (B*T, 1)
            code_sq = (self.codebook ** 2).sum(dim=1).unsqueeze(0)  # (1, num_codes)
            distances = flat_sq + code_sq - 2.0 * torch.matmul(flat, self.codebook.t())
            indices = distances.argmin(dim=1)

        quantized = self.codebook[indices].view(B, T, D)

        # VQ-VAE losses
        embed_loss = F.mse_loss(quantized, tokens.detach())
        commit_loss = F.mse_loss(tokens, quantized.detach())
        total_loss = embed_loss + self.commitment_weight * commit_loss
        self._last_losses = {
            "embed": embed_loss,
            "commit": commit_loss,
            "total": total_loss,
        }

        with torch.no_grad():
            counts = torch.zeros(self.codebook.shape[0], device=tokens.device, dtype=torch.float)
            ones = torch.ones_like(indices, dtype=torch.float)
            counts.scatter_add_(0, indices, ones)
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            usage = (counts > 0).sum().to(dtype=torch.float)
            self._last_metrics = {
                "perplexity": perplexity,
                "usage": usage,
            }

        # Straight-through estimator (grad to encoder; codebook trained via embed_loss)
        quantized_st = tokens + (quantized - tokens).detach()
        return quantized_st, indices.view(B, T)

    def last_losses(self):
        return self._last_losses

    def last_metrics(self):
        return self._last_metrics
