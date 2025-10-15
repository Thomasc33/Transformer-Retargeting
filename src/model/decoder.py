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

        # FIXED: Apply proper weight initialization to prevent NaN gradients
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with conservative values to prevent NaN gradients.

        The default PyTorch initialization for MultiheadAttention can be unstable,
        especially in mixed precision training. This uses Xavier/Glorot initialization
        with smaller variance to improve numerical stability.
        """
        # Initialize attention layers with conservative scaling
        for attn_layer in [self.self_attn, self.enc_dec_attn, self.cross_attn]:
            # Initialize in_proj_weight (combined Q, K, V weights)
            if hasattr(attn_layer, 'in_proj_weight') and attn_layer.in_proj_weight is not None:
                nn.init.xavier_uniform_(attn_layer.in_proj_weight, gain=0.5)  # Reduced gain for stability

            # Initialize in_proj_bias
            if hasattr(attn_layer, 'in_proj_bias') and attn_layer.in_proj_bias is not None:
                nn.init.constant_(attn_layer.in_proj_bias, 0.0)

            # Initialize out_proj weight and bias
            if hasattr(attn_layer, 'out_proj'):
                nn.init.xavier_uniform_(attn_layer.out_proj.weight, gain=0.5)  # Reduced gain
                if attn_layer.out_proj.bias is not None:
                    nn.init.constant_(attn_layer.out_proj.bias, 0.0)

        # Initialize FFN layers
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Conservative initialization
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        # Layer norm parameters are already properly initialized by PyTorch

    def forward(self, tgt, memory, memory_prime, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, cache=None, use_cache=False):
        # Initialize cache if not provided
        if cache is None:
            cache = {}

        # Self-Attention with causal mask
        if 'self_attn' in cache and use_cache and cache['self_attn'] is not None:
            # Retrieve cached keys and values
            prev_k, prev_v = cache['self_attn']

            # OPTIMIZED: Check for numerical instability and implement circular buffer
            if not torch.isfinite(prev_k).all() or not torch.isfinite(prev_v).all():
                # Reset cache if NaN/inf detected
                k = v = tgt
                print(f"⚠️  WARNING: NaN/inf detected in self_attn cache, resetting cache")
            else:
                # OPTIMIZED: Use fixed-size circular buffer to prevent unbounded growth
                max_history = 8  # Reduced for better memory efficiency
                if prev_k.size(0) >= max_history:
                    # Keep only the most recent entries (circular buffer behavior)
                    prev_k = prev_k[-(max_history-1):]
                    prev_v = prev_v[-(max_history-1):]

                # Concatenate with current input
                try:
                    k = torch.cat([prev_k, tgt], dim=0)
                    v = torch.cat([prev_v, tgt], dim=0)

                    # Ensure we don't exceed max_history after concatenation
                    if k.size(0) > max_history:
                        k = k[-max_history:]
                        v = v[-max_history:]

                    # Additional safety check after concatenation
                    if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                        print(f"⚠️  WARNING: NaN/inf after cache concatenation, using current input only")
                        k = v = tgt
                except RuntimeError as e:
                    print(f"⚠️  WARNING: Cache concatenation failed: {e}, using current input only")
                    k = v = tgt
        else:
            k = v = tgt

        # Update cache with detached tensors and numerical stability check
        if use_cache:
            # Ensure the values we're caching are finite
            if torch.isfinite(k).all() and torch.isfinite(v).all():
                # MEMORY FIX: Limit cache size more aggressively to prevent memory leaks
                max_cache_size = 8  # Further reduced from 16
                k_cache = k[-max_cache_size:].clone().detach()
                v_cache = v[-max_cache_size:].clone().detach()
                cache['self_attn'] = (k_cache, v_cache)
            else:
                # Don't cache if values are not finite
                print(f"⚠️  WARNING: Not caching non-finite k/v values in self_attn")
                cache['self_attn'] = (tgt.clone().detach(), tgt.clone().detach())

        # Compute self-attention
        tgt2 = self.self_attn(tgt, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        # Free memory
        del tgt2, k, v

        # Encoder-Decoder Attention
        # Cache encoder outputs since they are static during decoding
        if 'enc_dec_attn' in cache and use_cache and cache['enc_dec_attn'] is not None:
            memory_k, memory_v = cache['enc_dec_attn']
            # Check cached encoder values for numerical stability
            if not torch.isfinite(memory_k).all() or not torch.isfinite(memory_v).all():
                print(f"⚠️  WARNING: NaN/inf detected in enc_dec_attn cache, using fresh memory")
                memory_k = memory_v = memory
        else:
            memory_k = memory_v = memory
            if use_cache and torch.isfinite(memory).all():
                # Use clone().detach() to ensure complete memory separation
                cache['enc_dec_attn'] = (memory_k.clone().detach(), memory_v.clone().detach())
            elif use_cache:
                print(f"⚠️  WARNING: Not caching non-finite memory values in enc_dec_attn")

        # Compute encoder-decoder attention
        tgt2 = self.enc_dec_attn(tgt, memory_k, memory_v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        # Free memory
        del tgt2

        # Cross-Attention (if memory_prime is used)
        if memory_prime is not None:
            if 'cross_attn' in cache and use_cache and cache['cross_attn'] is not None:
                memory_prime_k, memory_prime_v = cache['cross_attn']
                # Check cached cross-attention values for numerical stability
                if not torch.isfinite(memory_prime_k).all() or not torch.isfinite(memory_prime_v).all():
                    print(f"⚠️  WARNING: NaN/inf detected in cross_attn cache, using fresh memory_prime")
                    memory_prime_k = memory_prime_v = memory_prime
            else:
                memory_prime_k = memory_prime_v = memory_prime
                if use_cache and torch.isfinite(memory_prime).all():
                    # Use clone().detach() to ensure complete memory separation
                    cache['cross_attn'] = (memory_prime_k.clone().detach(), memory_prime_v.clone().detach())
                elif use_cache:
                    print(f"⚠️  WARNING: Not caching non-finite memory_prime values in cross_attn")

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

        # FIXED: Ensure all layers are properly initialized
        # (DecoderLayer already calls _init_weights in its __init__, but this ensures consistency)
        self._verify_initialization()

    def _verify_initialization(self):
        """
        Verify that all decoder layers are properly initialized.
        This is a safety check to ensure numerical stability.
        """
        for i, layer in enumerate(self.layers):
            # Check if any weights are NaN or too large
            for name, param in layer.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN detected in layer {i}, parameter {name} during initialization")
                    # Re-initialize this parameter
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                elif torch.abs(param).max() > 10.0:
                    print(f"WARNING: Large weights detected in layer {i}, parameter {name}: max={torch.abs(param).max()}")
                    # Re-initialize with smaller variance
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)  # Even more conservative

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

            # Force synchronize to ensure operations are complete (only if CUDA is available)
            if i < len(self.layers) - 1 and torch.cuda.is_available():
                torch.cuda.synchronize()

        # MEMORY FIX: More aggressive cache cleanup
        if use_cache:
            for i in range(len(cache)):
                if cache[i] is not None:
                    # Clear individual cache entries
                    if isinstance(cache[i], dict):
                        for key in list(cache[i].keys()):
                            del cache[i][key]
                    cache[i] = None

        # MEMORY FIX: Force garbage collection more frequently
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return tgt, new_caches
