#!/usr/bin/env python3
"""
PMR (Privacy-preserving Motion Retargeting) Model

This module implements the PMR model for skeleton-based motion anonymization.
PMR is similar to DMR but uses a different architecture approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

# Configuration parameters (similar to DMR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 75  # Number of frames
V = 25  # Number of joints
C = 3   # Number of channels (x, y, z)

# PMR specific parameters (matching saved model architecture)
pmr_encoded_channels = [128, 16]  # Matching saved model
one_dimension_conv = False  # Use 2D convolutions like the saved model
util_classifier_alpha = 1.0
priv_classifier_alpha = 1.0

class PMR_Encoder2D(nn.Module):
    """PMR 2D Encoder matching saved model architecture."""

    def __init__(self):
        super(PMR_Encoder2D, self).__init__()

        # Encoder layers matching the saved model dimensions
        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=pmr_encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        # Reflection padding
        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        # Global pooling and fully connected
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(pmr_encoded_channels[0], pmr_encoded_channels[0] * pmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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
        x = x.view(-1, *pmr_encoded_channels)

        return x

class PMR_Encoder1D(nn.Module):
    """PMR 1D Encoder for motion encoding."""
    
    def __init__(self):
        super(PMR_Encoder1D, self).__init__()
        
        # Encoder layers - different architecture from DMR
        self.enc1 = nn.Conv1d(in_channels=T, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=384, out_channels=pmr_encoded_channels[0], kernel_size=3, stride=1, padding=1)
        
        # Reflection padding layers
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        
        # Pooling and activation
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.acti = nn.LeakyReLU(0.2)
        
        # Global pooling and fully connected
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(pmr_encoded_channels[0], pmr_encoded_channels[0] * pmr_encoded_channels[1])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)
        
        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)
        
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = x.view(-1, *pmr_encoded_channels)
        
        return x

class PMR_Decoder1D(nn.Module):
    """PMR 1D Decoder matching saved model architecture."""

    def __init__(self):
        super(PMR_Decoder1D, self).__init__()

        # Decoder layers matching saved model dimensions
        self.dec1 = nn.ConvTranspose1d(in_channels=pmr_encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)
        
        # Reflection padding
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        
        # Upsampling and activation
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest')
        self.acti = nn.LeakyReLU(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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

class PMR(nn.Module):
    """
    PMR (Privacy-preserving Motion Retargeting) Model
    
    This model performs motion anonymization while preserving utility.
    """
    
    def __init__(self, adv_lr=1e-4, use_adv=False, dataset='ntu', batch_size=32, datasets={}):
        super(PMR, self).__init__()
        self.batch_size = batch_size
        
        # Dataset Info
        if dataset in datasets:
            self.utility_classes = datasets[dataset]['num_class']
            self.privacy_classes = datasets[dataset]['num_actor']
        else:
            # Default values for NTU dataset
            self.utility_classes = 60
            self.privacy_classes = 40
        
        # AutoEncoder Models (using 2D encoders to match saved model)
        self.static_encoder = PMR_Encoder2D()
        self.dynamic_encoder = PMR_Encoder2D()
        self.decoder = PMR_Decoder1D()
        
        # Loss Functions
        self.triplet_loss = nn.TripletMarginLoss()
        self.bce_loss = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # End effectors for loss calculation
        self.end_effectors = torch.tensor([19, 15, 23, 24, 21, 22, 3]).to(device) * 3
        self.chain_lengths = torch.tensor([5, 5, 8, 8, 8, 8, 5]).to(device)
        
        # Loss weights (different from DMR)
        self.lambda_rec = 1.5
        self.lambda_cross = 0.2
        self.lambda_ee = 0.8
        self.lambda_smoothing = 2.5
        self.lambda_trip = 0.8
        self.lambda_latent = 8
        self.lambda_adv_util_coop = util_classifier_alpha
        self.lambda_adv_priv_coop = priv_classifier_alpha
        self.lambda_adv_util_adv = util_classifier_alpha
        self.lambda_adv_priv_adv = priv_classifier_alpha
        self.lambda_adv_disc = 0.8
    
    def forward(self, x, x_rot=None):
        """Forward pass through PMR model."""
        if x_rot is None:
            x_rot = x
        
        # Encode motion
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x_rot)
        
        # Decode motion
        x_hat = self.decoder(torch.cat((dyn, sta), dim=1))
        
        return x_hat
    
    def set_eval(self, eval_mode=True):
        """Set model to evaluation mode."""
        if eval_mode:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()
    
    def reconstruction_loss(self, x, y):
        """Calculate reconstruction loss."""
        return torch.square(torch.norm(x - y, dim=1)).mean()
    
    def end_effector_loss(self, x, y):
        """Calculate end effector loss."""
        x_ee = x[:, self.end_effectors]
        y_ee = y[:, self.end_effectors]
        return F.mse_loss(x_ee, y_ee)
    
    def smoothing_loss(self, x):
        """Calculate temporal smoothing loss."""
        diff = x[:, 1:] - x[:, :-1]
        return torch.mean(torch.norm(diff, dim=-1))
