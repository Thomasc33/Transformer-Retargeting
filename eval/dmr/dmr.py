#!/usr/bin/env python3
"""
DMR (Disentangled Motion Retargeting) Model

Restored from git history (commit 8acd08e) for loading pretrained checkpoints.
Uses 2D convolution encoders with encoded_channels = (256, 32).

Input format: (B, T, 25, 3) where T=75
Output format: (B, T, 75) where 75 = 25 joints * 3 coords

For retargeting:
    forward(x_action_source, x_identity_target)
    - static_encoder extracts action/motion from x_action_source
    - dynamic_encoder extracts identity/structure from x_identity_target
    - decoder combines them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# DMR architecture parameters
T = 75
dmr_encoded_channels = (256, 32)


class DMR_Encoder2D(nn.Module):
    """DMR 2D Encoder. Input: (B, T, 25, 3). Output: (B, 256, 32)."""

    def __init__(self):
        super(DMR_Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3, 3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=dmr_encoded_channels[0], kernel_size=(3, 3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(dmr_encoded_channels[0], dmr_encoded_channels[0] * dmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, *dmr_encoded_channels)

        return x


class DMR_Decoder1D(nn.Module):
    """DMR 1D Decoder. Input: (B, 512, 32). Output: (B, T, 75)."""

    def __init__(self):
        super(DMR_Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=dmr_encoded_channels[0] * 2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)

        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest')

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref2(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref3(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref4(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)

        return x


class DMR(nn.Module):
    """
    DMR (Disentangled Motion Retargeting) Model.

    For retargeting: forward(x_action_source, x_identity_target)
      - static_encoder extracts motion/action features from x_action_source
      - dynamic_encoder extracts identity/structure features from x_identity_target
      - decoder fuses them to produce retargeted output

    Input: (B, T, 25, 3) tensors
    Output: (B, T, 75) tensor (flattened joints*coords)
    """

    def __init__(self, use_adv=False):
        super(DMR, self).__init__()

        # Core autoencoder components
        self.static_encoder = DMR_Encoder2D()
        self.dynamic_encoder = DMR_Encoder2D()
        self.decoder = DMR_Decoder1D()

    def forward(self, x, x_rot):
        """
        Retargeting forward pass.

        Args:
            x: action source tensor (B, T, 25, 3) -- extract motion from this
            x_rot: identity target tensor (B, T, 25, 3) -- extract skeleton from this

        Returns:
            x_hat: retargeted output (B, T, 75)
        """
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x)
        x_hat = self.decoder(torch.cat((dyn, sta), dim=1))
        return x_hat
