import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Encoder
from model.decoder import Decoder

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
            # --- Autoregressive Path (Evaluation or Training w/o Teacher Forcing) ---
            outputs_list = [] # Store projected outputs per step
            # Initialize cache for decoder layers
            cache = [None] * len(self.decoder.layers) if self.decoder.layers else None

            # Initial input from source motion's first frame
            current_input_frame = source_motion[:, :, 0, :, :]   # (N, C_in, V, M)
            current_input_frame = current_input_frame.permute(0, 2, 3, 1).contiguous() # (N, V, M, C_in)
            current_input_frame = current_input_frame.view(N, -1)      # (N, D_input)

            # Project initial frame to d_model
            current_input_proj = self.decoder_input_proj(current_input_frame) # (N, d_model)

            # Free memory
            del current_input_frame

            # MEMORY SAFETY: Limit maximum autoregressive sequence length
            max_autoregressive_length = min(T-1, 64)  # Cap at 64 frames to prevent memory explosion

            for t in range(max_autoregressive_length): # Generate frames with memory safety limit
                # CRITICAL MEMORY FIX: More aggressive cache clearing to prevent memory leaks
                # Clear cache every 4 steps instead of 8 to prevent memory accumulation
                if t > 0 and t % 4 == 0:
                    # Force garbage collection before clearing cache
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    cache = [None] * len(self.decoder.layers) if self.decoder.layers else None

                # Prepare input for this step: add sequence dim (1) and positional encoding for step t
                current_input_step = current_input_proj.unsqueeze(0) # (1, N, d_model)
                # Add positional encoding for the current time step 't'
                decoder_input_t = current_input_step + self.positional_encoding.pe[t]
                decoder_input_t = self.positional_encoding.dropout(decoder_input_t) # Apply dropout

                # Free memory immediately
                del current_input_step

                # Decoder output for this step using cache
                # Pass full memory encodings; cache handles efficiency
                output_t, cache = self.decoder(decoder_input_t, motion_encoding, memory_prime=skeleton_encoding, cache=cache, use_cache=True) # output_t shape (1, N, d_model)

                # Free memory immediately
                del decoder_input_t

                # Map decoder output (last step) to target space (D_input)
                output_frame = self.output_linear(output_t[-1, :, :]) # Get the prediction (N, D_input)

                # Free memory immediately
                del output_t

                # CRITICAL MEMORY FIX: Break gradient accumulation across time steps
                # The fundamental issue: autoregressive generation was accumulating gradients
                # across ALL time steps, creating exponentially growing computation graphs

                if self.training:
                    # FIXED: Always detach to prevent gradient accumulation across time steps
                    # Each time step should be independent for memory efficiency
                    output_frame_detached = output_frame.detach()
                    outputs_list.append(output_frame_detached)

                    # For next input: detach to break gradient chain completely
                    next_input_frame_raw = output_frame.detach()
                    current_input_proj = self.decoder_input_proj(next_input_frame_raw).detach()

                    # CRITICAL: Clear the computation graph for this step
                    del output_frame  # Remove reference to computation graph
                else:
                    # During evaluation: detach everything to save memory
                    output_frame_detached = output_frame.detach()
                    outputs_list.append(output_frame_detached)

                    next_input_frame_raw = output_frame.detach()
                    current_input_proj = self.decoder_input_proj(next_input_frame_raw).detach()
                    del output_frame

                # Free memory immediately
                del next_input_frame_raw

                # Light memory cleanup (root cause fixed, so less aggressive cleanup needed)
                if t % 8 == 0:
                    torch.cuda.empty_cache()

                # Force synchronize CUDA operations every few steps to prevent memory buildup
                if t % 10 == 0:
                    torch.cuda.synchronize()

                    # Optional: force garbage collection periodically
                    if t % 30 == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

            # CRITICAL FIX: Two-pass approach for autoregressive training
            # Pass 1: Generate sequence (detached, memory efficient)
            # Pass 2: Recompute with gradients for loss (only final outputs)

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
                if outputs_list:
                    output = torch.stack(outputs_list, dim=0) # (actual_frames, N, D_input)

                    # If we generated fewer frames than expected, pad with zeros
                    actual_frames = output.shape[0]
                    expected_frames = T - 1
                    if actual_frames < expected_frames:
                        padding_frames = expected_frames - actual_frames
                        padding = torch.zeros(padding_frames, N, output.shape[2],
                                            device=output.device, dtype=output.dtype)
                        output = torch.cat([output, padding], dim=0)
                else:
                    # Fallback: create zero output if no frames were generated
                    output = torch.zeros(T-1, N, V*M*C_in, device=device)

            # Clear outputs_list to free memory
            outputs_list.clear()

            # Clear the cache to free memory
            for i in range(len(cache)):
                cache[i] = None
            del cache

            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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