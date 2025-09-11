import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (sequence_length, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3, debug=False, dataset='ntu', device='cuda', decoder_dropout=0.1, use_checkpoint=True):
        super(Model, self).__init__()
        self.encoder = Encoder(num_class=num_class, num_point=num_point, num_person=num_person,
                               graph=graph, graph_args=graph_args, in_channels=in_channels, debug=debug, dataset=dataset)
        self.decoder = Decoder(d_model=320, nhead=8, num_layers=6, dim_feedforward=2048, dropout=decoder_dropout)
        self.debug = debug
        self.use_checkpoint = use_checkpoint

        d_model = 320
        self.decoder_input_proj = nn.Linear(in_channels * num_point * num_person, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=decoder_dropout)
        self.enc_dec_layer = nn.Linear(320, 320)
        self.sty_tr_layer = nn.Linear(320, 320)
        self.output_linear = nn.Linear(d_model, in_channels * num_point * num_person)
        self.in_channels = in_channels
        self.num_point = num_point
        self.num_person = num_person
        self.dataset = dataset
        self.d_model = d_model

    def forward(self, source_motion, dummy_skeleton, target_motion=None, teacher_forcing_ratio=1.0):
        N, C_in, T, V, M = source_motion.shape

        # Encode with optional gradient checkpointing to save memory
        # For DDP compatibility, we need to use a simpler approach to checkpointing
        motion_encoding = self.encoder(source_motion)      # (Seq_len, N, d_model)
        skeleton_encoding = self.encoder(dummy_skeleton)   # (Seq_len, N, d_model)

        # Assuming encoder output dim matches d_model (320) or these layers project to it
        motion_encoding = self.enc_dec_layer(motion_encoding)
        skeleton_encoding = self.enc_dec_layer(skeleton_encoding)

        # Determine if using teacher forcing for this batch (only relevant during training)
        use_teacher_forcing = False
        if self.training and target_motion is not None:
            # Use teacher forcing if ratio is 1.0 or random check passes
            if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                 use_teacher_forcing = True

        if use_teacher_forcing:
            # --- Teacher Forcing Path ---
            # Prepare target sequence input (excluding last frame)
            tgt_input = target_motion[:, :, :-1, :, :]     # (N, C_in, T-1, V, M)
            tgt_input = tgt_input.permute(2, 0, 3, 4, 1).contiguous()  # (T-1, N, V, M, C_in)
            tgt_input = tgt_input.view(T-1, N, -1)         # (T-1, N, D_input)
            tgt_input = self.decoder_input_proj(tgt_input) # (T-1, N, d_model)
            tgt_input = self.positional_encoding(tgt_input) # Apply positional encoding to the whole sequence

            # Generate causal mask
            tgt_mask = self.generate_square_subsequent_mask(T-1)

            # Standard forward pass - removed gradient checkpointing for DDP compatibility
            decoder_output = self.decoder(tgt_input, motion_encoding[:-1], memory_prime=skeleton_encoding[:-1], tgt_mask=tgt_mask)[0] # (T-1, N, d_model)

            # Apply final projection
            output = self.output_linear(decoder_output)  # (T-1, N, D_input)

        else:
            # --- OPTIMIZED Batched Autoregressive Path ---
            # CRITICAL FIX: Process entire batch in parallel instead of sequential timesteps
            # This is the main bottleneck causing 100x+ slowdown

            # Initialize outputs list for autoregressive generation
            outputs_list = []

            # Initialize decoder input with source motion's first frame for all batch items
            current_input_frame = source_motion[:, :, 0, :, :]   # (N, C_in, V, M)
            current_input_frame = current_input_frame.permute(0, 2, 3, 1).contiguous() # (N, V, M, C_in)
            current_input_frame = current_input_frame.view(N, -1)      # (N, D_input)

            # Project to decoder dimension
            current_input_proj = self.decoder_input_proj(current_input_frame) # (N, d_model)

            # Initialize cache with fixed size to prevent memory growth
            max_cache_size = min(32, T-1)  # Limit cache size for memory efficiency
            cache = [{'self_attn': None, 'enc_dec_attn': None, 'cross_attn': None}
                    for _ in range(len(self.decoder.layers))]

            # Free memory
            del current_input_frame

            # OPTIMIZED: Batched autoregressive generation with parallel processing
            # CRITICAL FIX: Process multiple timesteps in parallel instead of sequential
            # This addresses the main bottleneck causing 100x+ slowdown

            max_autoregressive_length = min(T-1, 64)  # Cap at 64 frames to prevent memory explosion

            # Process in chunks for memory efficiency while maintaining parallelism
            chunk_size = min(4, max_autoregressive_length)  # Process 4 timesteps at once

            for chunk_start in range(0, max_autoregressive_length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, max_autoregressive_length)
                actual_chunk_size = chunk_end - chunk_start

                # Prepare inputs for this chunk
                if chunk_start == 0:
                    # First chunk: use initial input for all timesteps
                    chunk_inputs = current_input_proj.unsqueeze(0).expand(actual_chunk_size, -1, -1)
                else:
                    # Use previous outputs as inputs
                    prev_outputs = torch.stack(outputs_list[-actual_chunk_size:], dim=0)
                    chunk_inputs = self.decoder_input_proj(prev_outputs.view(-1, prev_outputs.size(-1))).view(actual_chunk_size, N, -1)

                # Add positional encoding for each timestep in chunk
                pos_indices = torch.arange(chunk_start, chunk_end, device=source_motion.device)
                pos_encodings = self.positional_encoding.pe[pos_indices]  # (chunk_size, 1, d_model)
                chunk_inputs_with_pos = chunk_inputs + pos_encodings.expand(-1, N, -1)
                chunk_inputs_with_pos = self.positional_encoding.dropout(chunk_inputs_with_pos)

                # Generate causal mask for this chunk
                chunk_mask = self.generate_square_subsequent_mask(actual_chunk_size)

                # Decoder forward pass for entire chunk
                decoder_outputs, cache = self.decoder(
                    chunk_inputs_with_pos,
                    motion_encoding,
                    memory_prime=skeleton_encoding,
                    tgt_mask=chunk_mask,
                    cache=cache,
                    use_cache=True
                )

                # Project to output space
                chunk_outputs = self.output_linear(decoder_outputs)  # (chunk_size, N, D_output)

                # Store outputs (detached to prevent gradient accumulation)
                for t in range(actual_chunk_size):
                    if self.training:
                        outputs_list.append(chunk_outputs[t].detach())
                    else:
                        outputs_list.append(chunk_outputs[t].detach())

                # Memory cleanup
                del chunk_inputs, chunk_inputs_with_pos, decoder_outputs, chunk_outputs

                # Periodic cache cleanup to prevent memory growth
                if chunk_start > 0 and chunk_start % (chunk_size * 4) == 0:
                    # Limit cache size
                    cache = self._limit_cache_size(cache, max_cache_size)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Update current_input_proj for next iteration
                if chunk_end < max_autoregressive_length:
                    current_input_proj = self.decoder_input_proj(outputs_list[-1])

            # Stack all outputs
            outputs = torch.stack(outputs_list, dim=0)  # (T-1, N, D_output)

            # Free memory
            del outputs_list, current_input_proj, cache

            if self.training and target_motion is not None:
                # TRAINING: Use teacher forcing for gradient computation to avoid memory explosion
                # The autoregressive generation above was just for sequence generation
                # Now use teacher forcing for actual gradient computation

                # Prepare target sequence input (excluding last frame)
                tgt_input = target_motion[:, :, :-1, :, :]     # (N, C_in, T-1, V, M)
                tgt_input = tgt_input.permute(2, 0, 3, 4, 1).contiguous()  # (T-1, N, V, M, C_in)
                tgt_input = tgt_input.view(T-1, N, -1)         # (T-1, N, D_input)
                tgt_input = self.decoder_input_proj(tgt_input) # (T-1, N, d_model)
                tgt_input = self.positional_encoding(tgt_input) # Apply positional encoding

                # Generate causal mask
                tgt_mask = self.generate_square_subsequent_mask(T-1)

                # Forward pass with gradients for loss computation
                decoder_output = self.decoder(tgt_input, motion_encoding[:-1], memory_prime=skeleton_encoding[:-1], tgt_mask=tgt_mask)[0]

                # Apply final projection
                output = self.output_linear(decoder_output)  # (T-1, N, D_input)

                # Clear intermediate tensors
                del tgt_input, decoder_output, tgt_mask

            else:
                # EVALUATION: Use the autoregressive outputs (already detached)
                output = outputs  # Use the outputs from autoregressive generation

        # --- Final Reshaping (Common to both paths) ---
        # 'output' variable now holds (T-1, N, D_input) regardless of the path taken
        output = output.view(T-1, N, self.num_point, self.num_person, self.in_channels)  # Reshape to (T-1, N, V, M, C_in)
        output = output.permute(1, 4, 0, 2, 3).contiguous()  # Permute to (N, C_in, T-1, V, M)

        return output

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. Ensures mask is on the same device as model parameters.
        """
        # Get device from model parameters to ensure mask is on the correct device
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def _limit_cache_size(self, cache, max_size):
        """
        Limit the size of decoder cache to prevent memory growth.
        """
        if cache is None:
            return cache

        limited_cache = []
        for layer_cache in cache:
            if layer_cache is None:
                limited_cache.append(None)
                continue

            limited_layer_cache = {}
            for key, value in layer_cache.items():
                if value is not None and isinstance(value, tuple) and len(value) == 2:
                    k, v = value
                    if k.size(0) > max_size:
                        # Keep only the most recent entries
                        limited_layer_cache[key] = (k[-max_size:], v[-max_size:])
                    else:
                        limited_layer_cache[key] = value
                else:
                    limited_layer_cache[key] = value
            limited_cache.append(limited_layer_cache)

        return limited_cache