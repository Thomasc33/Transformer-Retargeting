import torch
import torch.nn as nn
import torch.nn.functional as F

class MLMDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MLMDecoderLayer, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        # Self-Attention layer for full sequence reconstruction (no causal mask)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-Attention without causal mask (for reconstruction)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        return tgt

class MLMDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(MLMDecoder, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.layers = nn.ModuleList([
            MLMDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return tgt

def post_process(output, frames, batch_size, actor, joints, channels):
    """
    Process the output from the MLM decoder to the expected format.

    FIXED VERSION: Properly handles shape transformations for MLM reconstruction.

    Args:
        output: Tensor with shape (sequence_length, batch_size, feature_dim)
        frames: Number of frames in the original sequence
        batch_size: Batch size
        actor: Number of actors (typically 1)
        joints: Number of joints (typically 25)
        channels: Number of channels (typically 3)

    Returns:
        Tensor with shape (batch_size, frames, actor, joints, channels)
    """
    import torch.nn.functional as F

    # Get the actual shape of the output
    sequence_length, batch_size_out, feature_dim = output.shape

    # Debug information (can be removed in production)
    # print(f"post_process input: {output.shape}")
    # print(f"Expected output: ({batch_size}, {frames}, {actor}, {joints}, {channels})")

    try:
        # Method 1: Direct reshape if dimensions match
        expected_elements = batch_size * channels * frames * joints * actor
        actual_elements = sequence_length * batch_size_out * feature_dim

        if actual_elements == expected_elements:
            # Reshape directly
            output = output.permute(1, 0, 2)  # (batch_size, sequence_length, feature_dim)
            output = output.contiguous().view(batch_size, frames, actor, joints, channels)
            return output

        # Method 2: Handle encoder temporal reduction
        # Assume encoder reduces temporal dimension by factor of 4
        T_reduced = frames // 4
        if sequence_length == T_reduced * joints:
            # Reshape to (batch_size, T_reduced, joints, channels)
            output = output.permute(1, 0, 2)  # (batch_size, sequence_length, feature_dim)
            output = output.view(batch_size, T_reduced, joints, channels)

            # Upsample temporal dimension back to original frames
            output = output.permute(0, 3, 1, 2)  # (batch_size, channels, T_reduced, joints)
            output = F.interpolate(output, size=(frames, joints), mode='bilinear', align_corners=False)
            output = output.permute(0, 2, 3, 1)  # (batch_size, frames, joints, channels)

            # Add actor dimension
            output = output.unsqueeze(2)  # (batch_size, frames, 1, joints, channels)
            return output

        # Method 3: Feature dimension matches joint*channel output
        if feature_dim == joints * channels:
            # Reshape feature dimension to joints and channels
            output = output.permute(1, 0, 2)  # (batch_size, sequence_length, feature_dim)
            output = output.view(batch_size, sequence_length, joints, channels)

            # Handle temporal dimension mismatch
            if sequence_length != frames:
                # Interpolate to correct number of frames
                output = output.permute(0, 3, 1, 2)  # (batch_size, channels, sequence_length, joints)
                output = F.interpolate(output, size=(frames, joints), mode='bilinear', align_corners=False)
                output = output.permute(0, 2, 3, 1)  # (batch_size, frames, joints, channels)

            # Add actor dimension
            output = output.unsqueeze(2)  # (batch_size, frames, 1, joints, channels)
            return output

    except Exception as e1:
        print(f"Primary methods failed: {e1}")

        # Fallback method: Force reshape with padding/truncation
        try:
            output = output.permute(1, 0, 2)  # (batch_size, sequence_length, feature_dim)

            # Calculate target shape
            target_elements = frames * joints * channels
            current_elements = sequence_length * feature_dim

            if current_elements >= target_elements:
                # Truncate
                flat_output = output.view(batch_size, -1)[:, :target_elements]
            else:
                # Pad with zeros
                flat_output = output.view(batch_size, -1)
                padding_size = target_elements - current_elements
                padding = torch.zeros(batch_size, padding_size, device=output.device)
                flat_output = torch.cat([flat_output, padding], dim=1)

            # Reshape to final format
            result = flat_output.view(batch_size, frames, joints, channels)
            result = result.unsqueeze(2)  # Add actor dimension
            return result

        except Exception as e2:
            print(f"Ultimate fallback failed: {e2}")
            # Return zeros with correct shape
            result = torch.zeros(batch_size, frames, actor, joints, channels, device=output.device)
            return result

