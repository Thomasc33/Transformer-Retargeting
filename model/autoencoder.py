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
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3, debug=False, dataset='ntu'):
        super(Model, self).__init__()
        self.encoder = Encoder(num_class=num_class, num_point=num_point, num_person=num_person,
                               graph=graph, graph_args=graph_args, in_channels=in_channels, debug=debug, dataset=dataset)
        self.decoder = Decoder(d_model=320, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1)
        self.debug = debug

        d_model = 320
        self.decoder_input_proj = nn.Linear(in_channels * num_point * num_person, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.enc_dec_layer = nn.Linear(320, 320)
        self.sty_tr_layer = nn.Linear(320, 320)
        self.output_linear = nn.Linear(d_model, in_channels * num_point * num_person)
        self.in_channels = in_channels
        self.num_point = num_point
        self.num_person = num_person
        self.dataset = dataset

    def forward(self, source_motion, dummy_skeleton, target_motion=None, teacher_forcing_ratio=1.0):
        N, C_in, T, V, M = source_motion.shape

        # Encode the source motion and dummy skeleton
        motion_encoding = self.encoder(source_motion)      # (Seq_len, N, d_model)
        skeleton_encoding = self.encoder(dummy_skeleton)   # (Seq_len, N, d_model)

        motion_encoding = self.enc_dec_layer(motion_encoding)
        skeleton_encoding = self.enc_dec_layer(skeleton_encoding)

        # Prepare decoder inputs
        if self.training and target_motion is not None:
            # Use ground truth previous frames (teacher forcing)
            tgt_input = target_motion[:, :, :-1, :, :]     # (N, C_in, T-1, V, M)
            tgt_input = tgt_input.permute(2, 0, 3, 4, 1).contiguous()  # (T-1, N, V, M, C_in)
            tgt_input = tgt_input.view(T-1, N, -1)         # (T-1, N, D_input)
            tgt_input = self.decoder_input_proj(tgt_input) # (T-1, N, d_model)
            tgt_input = self.positional_encoding(tgt_input)

            # Create causal mask
            tgt_mask = self.generate_square_subsequent_mask(T-1).to(tgt_input.device)

            # Decoder output
            output = self.decoder(tgt_input, motion_encoding[:-1], memory_prime=skeleton_encoding[:-1], tgt_mask=tgt_mask)[0]
        else:
            # During evaluation or when teacher forcing ratio is zero
            outputs = []
            decoder_input = source_motion[:, :, 0, :, :]   # First frame: (N, C_in, V, M)
            decoder_input = decoder_input.permute(0, 2, 3, 1).contiguous()  # (N, V, M, C_in)
            decoder_input = decoder_input.view(N, -1)      # (N, D_input)
            decoder_input = self.decoder_input_proj(decoder_input)  # (N, d_model)
            decoder_input = decoder_input.unsqueeze(0)     # (1, N, d_model)
            cache = None
            for t in range(T-1):
                # Apply positional encoding to the current input
                decoder_input_t = self.positional_encoding(decoder_input[-1:, :, :])  # (1, N, d_model)
                # Decoder output with cache
                output_t, cache = self.decoder(decoder_input_t, motion_encoding, memory_prime=skeleton_encoding, cache=cache, use_cache=True)
                output_t = output_t[-1, :, :]              # Get the last time step
                outputs.append(output_t)
                # Prepare next input
                if target_motion is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    # Use ground truth frame
                    next_input = target_motion[:, :, t+1, :, :]  # (N, C_in, V, M)
                    next_input = next_input.permute(0, 2, 3, 1).contiguous()
                    next_input = next_input.view(N, -1)          # (N, D_input)
                else:
                    # Use model's own prediction
                    next_input = self.output_linear(output_t)    # (N, D_input)
                next_input = self.decoder_input_proj(next_input) # (N, d_model)
                decoder_input = next_input.unsqueeze(0)  # Prepare for next time step
            output = torch.stack(outputs, dim=0)  # (T-1, N, d_model)

        # Map output to target space
        output = self.output_linear(output)  # (T-1, N, D_input)
        output = output.view(T-1, N, self.num_point, self.num_person, self.in_channels)  # (T-1, N, V, M, C_in)
        output = output.permute(1, 4, 0, 2, 3).contiguous()  # (N, C_in, T-1, V, M)

        return output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask