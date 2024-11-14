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
    # output: (sequence_length, batch_size, channels)
    sequence_length, batch_size, channels = output.shape
    T_new = frames // 4  # Adjust based on your encoder's temporal reduction
    V_new = joints
    assert sequence_length == T_new * V_new, f"Expected sequence_length {T_new * V_new}, got {sequence_length}"
    
    # Reshape to (batch_size, T_new, V_new, channels)
    output = output.permute(1, 0, 2).contiguous()
    output = output.view(batch_size, T_new, V_new, channels)
    
    # Optional: Upsample temporal dimension back to frames if needed
    output = output.permute(0, 3, 1, 2)  # (batch_size, channels, T_new, V_new)
    output = F.interpolate(output, size=(frames, V_new), mode='bilinear', align_corners=False)
    output = output.permute(0, 2, 3, 1).contiguous()  # (batch_size, frames, V_new, channels)
    
    # Add actor dimension
    output = output.unsqueeze(2)  # (batch_size, frames, 1, V_new, channels)
    return output

