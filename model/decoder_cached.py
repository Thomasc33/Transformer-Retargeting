import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # Self-Attention layer with caching
        self.self_attn = CachedMultiheadAttention(d_model, nhead, dropout=dropout)
        # Encoder-Decoder Attention layer
        self.enc_dec_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-Attention layer with caching
        self.cross_attn = CachedMultiheadAttention(d_model, nhead, dropout=dropout)
        
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
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                self_cache=None, cross_attn_cache=None):
        # Self-Attention with caching
        tgt2, _, new_self_cache = self.self_attn(
            query=tgt,
            key=None,  # Will default to using query
            value=None,  # Will default to using query
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
            past_key_value=self_cache
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Encoder-Decoder Attention
        tgt2, _ = self.enc_dec_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Cross-Attention with caching
        tgt2, _, new_cross_attn_cache = self.cross_attn(
            query=tgt,
            key=memory_prime,
            value=memory_prime,
            need_weights=False,
            past_key_value=cross_attn_cache
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm4(tgt)
        
        return tgt, new_self_cache, new_cross_attn_cache



class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt, memory, memory_prime, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                self_caches=None, cross_attn_caches=None):
        if self_caches is None:
            self_caches = [None] * len(self.layers)
        if cross_attn_caches is None:
            cross_attn_caches = [None] * len(self.layers)
        
        new_self_caches = []
        new_cross_attn_caches = []

        for i, layer in enumerate(self.layers):
            tgt, new_self_cache, new_cross_attn_cache = layer(
                tgt, memory, memory_prime, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                self_cache=self_caches[i],
                cross_attn_cache=cross_attn_caches[i]
            )
            new_self_caches.append(new_self_cache)
            new_cross_attn_caches.append(new_cross_attn_cache)
        
        return tgt, new_self_caches, new_cross_attn_caches


class CachedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CachedMultiheadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    
    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None,
                need_weights=False, past_key_value=None):
        # query: (tgt_len, batch_size, embed_dim)
        # key, value: (src_len, batch_size, embed_dim)
        # past_key_value: (key, value), each of shape (past_len, batch_size, embed_dim)
        
        if past_key_value is not None:
            # Concatenate past key and value with current key and value
            if key is not None and value is not None:
                key = torch.cat([past_key_value[0], key], dim=0)
                value = torch.cat([past_key_value[1], value], dim=0)
            else:
                key = past_key_value[0]
                value = past_key_value[1]
        else:
            if key is None or value is None:
                key = query
                value = query
        
        # Save key and value for caching
        present_key_value = (key, value)
        
        # Adjust the attention mask
        if attn_mask is not None:
            # attn_mask shape should be (tgt_len, src_len)
            src_len = key.size(0)
            tgt_len = query.size(0)
            if attn_mask.size(0) != tgt_len or attn_mask.size(1) != src_len:
                # Need to adjust attn_mask to match the new src_len
                attn_mask = attn_mask[:, -src_len:]
        
        # Call nn.MultiheadAttention
        attn_output, attn_output_weights = self.mha(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights
        )
        
        return attn_output, attn_output_weights, present_key_value
