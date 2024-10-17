import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Self-Attention layer with causal mask
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Encoder-Decoder Attention layer
        self.enc_dec_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_prime, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, cache=None, use_cache=False):
        # Initialize cache if not provided
        if cache is None:
            cache = {}

        # Self-Attention with causal mask
        if 'self_attn' in cache and use_cache:
            # Retrieve cached keys and values
            prev_k, prev_v = cache['self_attn']
            # Concatenate with current inputs
            k = torch.cat([prev_k, tgt], dim=0)
            v = torch.cat([prev_v, tgt], dim=0)
        else:
            k = v = tgt
        # Update cache
        if use_cache:
            cache['self_attn'] = (k, v)
        
        tgt2 = self.self_attn(tgt, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Encoder-Decoder Attention
        # Cache encoder outputs since they are static during decoding
        if 'enc_dec_attn' in cache and use_cache:
            memory_k, memory_v = cache['enc_dec_attn']
        else:
            memory_k = memory_v = memory
            if use_cache:
                cache['enc_dec_attn'] = (memory_k, memory_v)
        tgt2 = self.enc_dec_attn(tgt, memory_k, memory_v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Cross-Attention (if memory_prime is used)
        if memory_prime is not None:
            if 'cross_attn' in cache and use_cache:
                memory_prime_k, memory_prime_v = cache['cross_attn']
            else:
                memory_prime_k = memory_prime_v = memory_prime
                if use_cache:
                    cache['cross_attn'] = (memory_prime_k, memory_prime_v)
            tgt2 = self.cross_attn(tgt, memory_prime_k, memory_prime_v)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm3(tgt)

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm4(tgt)
        
        return tgt, cache

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt, memory, memory_prime=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, cache=None, use_cache=False):
        # Initialize cache for each layer if not provided
        if cache is None:
            cache = [None] * len(self.layers)
        new_caches = []
        for i, layer in enumerate(self.layers):
            tgt, layer_cache = layer(
                tgt, memory, memory_prime, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, cache=cache[i], use_cache=use_cache
            )
            new_caches.append(layer_cache)
        return tgt, new_caches
