import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Encoder
from model.decoder import Decoder

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3, debug=False):
        super(Model, self).__init__()
        self.encoder = Encoder(num_class=num_class, num_point=num_point, num_person=num_person,
                               graph=graph, graph_args=graph_args, in_channels=in_channels, debug=debug)
        self.decoder = Decoder(d_model=320, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1)
        self.debug = debug

        # Linear layer to map from d_model (320) to C_in (3)
        self.output_linear = nn.Linear(320, in_channels)
        self.in_channels = in_channels

    def forward(self, x, dummy=None):
        if self.debug: print('input x shape:', x.shape)
        
        # Pass input through the encoder
        memory = self.encoder(x)  # Shape: (sequence_length, batch_size, d_model)

        # Pass dummy through the encoder
        if dummy is not None:
            memory_prime = self.encoder(dummy)
        else:
            memory_prime = memory

        if self.debug: print('memory shape:', memory.shape)
        if self.debug and dummy is not None: print('memory_prime shape:', memory_prime.shape)

        # Initialize tgt as zeros matching the shape of memory
        tgt = torch.zeros_like(memory)  # Shape: (sequence_length, batch_size, d_model)

        # Pass through the decoder
        output = self.decoder(tgt, memory, memory_prime)
        if self.debug: print('decoded output shape:', output.shape)  # Shape: (sequence_length, batch_size, d_model)

        # Permute and reshape
        N = output.size(1)
        d_model = output.size(2)
        output = output.permute(1, 0, 2).contiguous()  # (N, seq_len, d_model)

        T_new = 16
        V_new = 25
        output = output.view(N, T_new, V_new, d_model)  # (N, T_new, V_new, d_model)

        # Map d_model to C_in
        output = output.view(-1, d_model)  # (N * T_new * V_new, d_model)
        output = self.output_linear(output)  # (N * T_new * V_new, C_in)
        output = output.view(N, T_new, V_new, self.in_channels)  # (N, T_new, V_new, C_in)

        # Permute to (N, V_new, C_in, T_new)
        output = output.permute(0, 2, 3, 1).contiguous()  # (N, V_new, C_in, T_new)

        # Reshape to (N * V_new * C_in, T_new)
        output = output.view(-1, T_new)  # (N * V_new * C_in, T_new)

        # Add dimension for interpolation
        output = output.unsqueeze(1)  # (N * V_new * C_in, 1, T_new)

        # Perform interpolation
        output = F.interpolate(output, size=64, mode='linear', align_corners=False)  # (N * V_new * C_in, 1, 64)

        # Remove added dimension
        output = output.squeeze(1)  # (N * V_new * C_in, 64)

        # Reshape back to (N, V_new, C_in, T)
        output = output.view(N, V_new, self.in_channels, 64)  # (N, V_new, C_in, 64)

        # Permute back to (N, C_in, T, V_new)
        output = output.permute(0, 2, 3, 1).contiguous()  # (N, C_in, 64, V_new)

        # Add person dimension
        output = output.unsqueeze(-1)  # (N, C_in, 64, V_new, 1)

        if self.debug: print('Final output shape:', output.shape)

        return output  # Shape: (N, C_in, 64, V_new, M)