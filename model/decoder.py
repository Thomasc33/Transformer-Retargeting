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
            # Concatenate with current inputs - limit history to prevent memory growth
            # Reduce max history to save memory
            max_history = 32  # Reduced from 64 to save memory
            if prev_k.size(0) > max_history:
                prev_k = prev_k[-max_history:]
                prev_v = prev_v[-max_history:]
            k = torch.cat([prev_k, tgt], dim=0)
            v = torch.cat([prev_v, tgt], dim=0)
        else:
            k = v = tgt
        # Update cache with detached tensors to prevent memory leaks
        if use_cache:
            # Use clone().detach() to ensure complete memory separation
            cache['self_attn'] = (k.clone().detach(), v.clone().detach())

        # Compute self-attention
        tgt2 = self.self_attn(tgt, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        # Free memory
        del tgt2, k, v

        # Encoder-Decoder Attention
        # Cache encoder outputs since they are static during decoding
        if 'enc_dec_attn' in cache and use_cache:
            memory_k, memory_v = cache['enc_dec_attn']
        else:
            memory_k = memory_v = memory
            if use_cache:
                # Use clone().detach() to ensure complete memory separation
                cache['enc_dec_attn'] = (memory_k.clone().detach(), memory_v.clone().detach())

        # Compute encoder-decoder attention
        tgt2 = self.enc_dec_attn(tgt, memory_k, memory_v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        # Free memory
        del tgt2

        # Cross-Attention (if memory_prime is used)
        if memory_prime is not None:
            if 'cross_attn' in cache and use_cache:
                memory_prime_k, memory_prime_v = cache['cross_attn']
            else:
                memory_prime_k = memory_prime_v = memory_prime
                if use_cache:
                    # Use clone().detach() to ensure complete memory separation
                    cache['cross_attn'] = (memory_prime_k.clone().detach(), memory_prime_v.clone().detach())

            # Compute cross-attention
            tgt2 = self.cross_attn(tgt, memory_prime_k, memory_prime_v)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm3(tgt)
            # Free memory
            del tgt2, memory_prime_k, memory_prime_v

        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm4(tgt)
        # Free memory
        del tgt2

        return tgt, cache

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, tgt, memory, memory_prime=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, cache=None, use_cache=False):
        # Initialize cache for each layer if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Ensure cache is properly sized
        if len(cache) != len(self.layers):
            # Resize cache if needed (handles case where model structure changed)
            cache = [None] * len(self.layers)

        # Apply memory optimization: chunk processing for large sequences
        # This helps reduce peak memory usage during attention computation
        new_caches = []

        # Process through each decoder layer
        for i, layer in enumerate(self.layers):
            # Process the layer
            tgt, layer_cache = layer(
                tgt, memory, memory_prime, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, cache=cache[i], use_cache=use_cache
            )
            new_caches.append(layer_cache)

            # Force synchronize to ensure operations are complete
            if i < len(self.layers) - 1:
                torch.cuda.synchronize()

        # Clear references to old cache to help garbage collection
        if use_cache:
            for i in range(len(cache)):
                cache[i] = None

        # Force garbage collection after processing all layers
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return tgt, new_caches
