"""
Adversarial Motion Loss Module

This module implements adversarial training for motion realism using a temporal CNN discriminator.
The discriminator distinguishes between real (ground truth) and fake (TMR-generated) motion sequences.

Components:
1. MotionDiscriminator: Temporal CNN for motion discrimination
2. AdversarialMotionLoss: Adversarial training loop with gradient penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class MotionDiscriminator(nn.Module):
    """
    Temporal CNN discriminator for motion sequences.
    
    Architecture:
    - 3 temporal convolutional layers for feature extraction
    - 2 fully connected layers for classification
    - Binary classification head (real vs fake)
    - Optional conditional discrimination on action labels
    
    Input: (N, C, T, V, M) motion sequences
    Output: (N, 1) real/fake probabilities
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 num_joints: int = 25,
                 num_frames: int = 64,
                 num_actions: int = None,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize motion discriminator.
        
        Args:
            in_channels: Input channels (3 for x,y,z coordinates)
            num_joints: Number of skeleton joints (25 for NTU)
            num_frames: Number of temporal frames (64)
            num_actions: Number of action classes for conditional discrimination
            hidden_dim: Hidden dimension for FC layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Flatten spatial dimensions (C * V * M) for temporal convolution
        # Input shape: (N, C*V*M, T) = (N, 3*25*1, 64) = (N, 75, 64)
        conv_in_channels = in_channels * num_joints
        
        # Temporal CNN layers
        self.conv_layers = nn.Sequential(
            # Conv1: (N, 75, 64) -> (N, 128, 32)
            nn.Conv1d(conv_in_channels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            # Conv2: (N, 128, 32) -> (N, 256, 16)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            # Conv3: (N, 256, 16) -> (N, 512, 8)
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        
        # Calculate flattened feature size after convolutions
        # After 3 conv layers with stride=2: 64 -> 32 -> 16 -> 8
        conv_output_size = 512 * 8  # 4096
        
        # Add action embedding if conditional
        if num_actions is not None:
            self.action_embedding = nn.Embedding(num_actions, 64)
            fc_input_size = conv_output_size + 64
        else:
            self.action_embedding = None
            fc_input_size = conv_output_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            # Binary classification head
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, motion: torch.Tensor, action_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            motion: Motion sequences (N, C, T, V, M)
            action_labels: Action labels for conditional discrimination (N,)
            
        Returns:
            logits: Real/fake logits (N, 1)
        """
        batch_size = motion.size(0)
        
        # Reshape for temporal convolution: (N, C, T, V, M) -> (N, C*V*M, T)
        # (N, 3, 64, 25, 1) -> (N, 75, 64)
        motion_flat = motion.view(batch_size, -1, self.num_frames)
        
        # Temporal CNN feature extraction
        conv_features = self.conv_layers(motion_flat)  # (N, 512, 8)
        conv_features = conv_features.view(batch_size, -1)  # (N, 4096)
        
        # Add action conditioning if available
        if self.action_embedding is not None and action_labels is not None:
            action_emb = self.action_embedding(action_labels)  # (N, 64)
            features = torch.cat([conv_features, action_emb], dim=1)  # (N, 4160)
        else:
            features = conv_features
        
        # Classification
        logits = self.fc_layers(features)  # (N, 1)
        
        return logits


class AdversarialMotionLoss(nn.Module):
    """
    Adversarial training loss for motion realism.
    
    Implements WGAN-GP style training with:
    - Discriminator loss: Wasserstein distance with gradient penalty
    - Generator loss: Adversarial loss to fool discriminator
    - Alternating updates: 5 discriminator steps, then 1 generator step
    """
    
    def __init__(self,
                 discriminator: MotionDiscriminator,
                 lambda_gp: float = 10.0,
                 discriminator_steps: int = 5,
                 generator_steps: int = 1,
                 device: str = 'cuda'):
        """
        Initialize adversarial motion loss.
        
        Args:
            discriminator: Motion discriminator network
            lambda_gp: Gradient penalty weight
            discriminator_steps: Number of discriminator updates per generator update
            generator_steps: Number of generator updates (typically 1)
            device: Device for computation
        """
        super().__init__()
        
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.device = device
        
        # Optimizers for discriminator and generator
        self.discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        
        # Training state
        self.step_count = 0
        self.discriminator_losses = []
        self.generator_losses = []
    
    def compute_gradient_penalty(self, 
                                real_motion: torch.Tensor,
                                fake_motion: torch.Tensor,
                                action_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP training stability.
        
        Args:
            real_motion: Real motion sequences (N, C, T, V, M)
            fake_motion: Generated motion sequences (N, C, T, V, M)
            action_labels: Action labels (N,)
            
        Returns:
            gradient_penalty: Gradient penalty loss
        """
        batch_size = real_motion.size(0)
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        interpolated = alpha * real_motion + (1 - alpha) * fake_motion
        interpolated.requires_grad_(True)
        
        # Discriminator output on interpolated samples
        d_interpolated = self.discriminator(interpolated, action_labels)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def update_discriminator(self,
                           real_motion: torch.Tensor,
                           fake_motion: torch.Tensor,
                           action_labels: Optional[torch.Tensor] = None) -> float:
        """
        Update discriminator with Wasserstein loss and gradient penalty.
        
        Args:
            real_motion: Real motion sequences (N, C, T, V, M)
            fake_motion: Generated motion sequences (N, C, T, V, M)
            action_labels: Action labels (N,)
            
        Returns:
            discriminator_loss: Discriminator loss value
        """
        self.discriminator_optimizer.zero_grad()
        
        # Discriminator outputs
        d_real = self.discriminator(real_motion, action_labels)
        d_fake = self.discriminator(fake_motion.detach(), action_labels)
        
        # Wasserstein loss
        wasserstein_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_motion, fake_motion, action_labels)
        
        # Total discriminator loss
        d_loss = wasserstein_loss + self.lambda_gp * gradient_penalty
        
        d_loss.backward()
        self.discriminator_optimizer.step()
        
        return d_loss.item()
    
    def compute_generator_loss(self,
                             fake_motion: torch.Tensor,
                             action_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute generator adversarial loss.
        
        Args:
            fake_motion: Generated motion sequences (N, C, T, V, M)
            action_labels: Action labels (N,)
            
        Returns:
            generator_loss: Adversarial loss for generator
        """
        # Generator wants discriminator to classify fake as real
        d_fake = self.discriminator(fake_motion, action_labels)
        generator_loss = -d_fake.mean()
        
        return generator_loss
    
    def forward(self,
                real_motion: torch.Tensor,
                fake_motion: torch.Tensor,
                action_labels: Optional[torch.Tensor] = None,
                update_discriminator: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with alternating discriminator/generator updates.
        
        Args:
            real_motion: Real motion sequences (N, C, T, V, M)
            fake_motion: Generated motion sequences (N, C, T, V, M)
            action_labels: Action labels (N,)
            update_discriminator: Whether to update discriminator this step
            
        Returns:
            generator_loss: Loss for generator (to be added to total loss)
            loss_dict: Dictionary of individual losses for logging
        """
        loss_dict = {}
        
        # Update discriminator every discriminator_steps
        if update_discriminator and (self.step_count % (self.discriminator_steps + self.generator_steps) < self.discriminator_steps):
            d_loss = self.update_discriminator(real_motion, fake_motion, action_labels)
            loss_dict['discriminator_loss'] = d_loss
            self.discriminator_losses.append(d_loss)
            
            # Return zero generator loss during discriminator updates
            generator_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Update generator
            generator_loss = self.compute_generator_loss(fake_motion, action_labels)
            loss_dict['generator_loss'] = generator_loss.item()
            self.generator_losses.append(generator_loss.item())
        
        self.step_count += 1
        
        return generator_loss, loss_dict
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {}
        
        if self.discriminator_losses:
            stats['avg_discriminator_loss'] = np.mean(self.discriminator_losses[-100:])
        
        if self.generator_losses:
            stats['avg_generator_loss'] = np.mean(self.generator_losses[-100:])
        
        stats['step_count'] = self.step_count
        
        return stats
    
    def reset_stats(self):
        """Reset training statistics."""
        self.discriminator_losses = []
        self.generator_losses = []
        self.step_count = 0


def create_motion_discriminator(num_actions: Optional[int] = None,
                              conditional: bool = True,
                              **kwargs) -> MotionDiscriminator:
    """
    Factory function to create motion discriminator.
    
    Args:
        num_actions: Number of action classes for conditional discrimination
        conditional: Whether to use conditional discrimination
        **kwargs: Additional arguments for MotionDiscriminator
        
    Returns:
        discriminator: Initialized motion discriminator
    """
    if conditional and num_actions is None:
        raise ValueError("num_actions must be specified for conditional discrimination")
    
    if not conditional:
        num_actions = None
    
    discriminator = MotionDiscriminator(
        num_actions=num_actions,
        **kwargs
    )
    
    return discriminator


def create_adversarial_loss(discriminator: MotionDiscriminator,
                          lambda_gp: float = 10.0,
                          discriminator_steps: int = 5,
                          device: str = 'cuda') -> AdversarialMotionLoss:
    """
    Factory function to create adversarial motion loss.
    
    Args:
        discriminator: Motion discriminator network
        lambda_gp: Gradient penalty weight
        discriminator_steps: Number of discriminator updates per generator update
        device: Device for computation
        
    Returns:
        adversarial_loss: Initialized adversarial motion loss
    """
    return AdversarialMotionLoss(
        discriminator=discriminator,
        lambda_gp=lambda_gp,
        discriminator_steps=discriminator_steps,
        device=device
    )