"""
Cycle Consistency Loss for Motion Retargeting
Ensures that A→B→A retargeting recovers the original motion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss for motion retargeting
    Forces the model to be invertible: A→B→A ≈ A
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for forward cycle loss
        self.beta = beta    # Weight for backward cycle loss
        
    def forward(self, x1, x2, y1, y2, model):
        """
        Args:
            x1: (B, C, T, V, M) - Source skeleton (P1, A1)
            x2: (B, C, T, V, M) - Target skeleton (P2, A2)
            y1: (B, C, T, V, M) - Cross skeleton (P1, A2)
            y2: (B, C, T, V, M) - Cross skeleton (P2, A1)
            model: DisentangledTMR model
            
        Returns:
            cycle_loss: Combined cycle consistency loss
            cycle_forward: Forward cycle loss (x1→x2→x1')
            cycle_backward: Backward cycle loss (x2→x1→x2')
        """
        # Forward cycle: x1 → x2 → x1_reconstructed
        # First retarget: P1,A1 → P2,A1 (should look like y2)
        output_forward, _, _ = model(x1, x2, y2, teacher_forcing_ratio=0.0)
        
        # Retarget back: P2,A1 → P1,A1 (should recover x1)
        # Use output_forward as source skeleton with x2's identity
        output_cycle, _, _ = model(output_forward, x1, x1[:, :, 1:, :, :], teacher_forcing_ratio=0.0)
        
        # Backward cycle: x2 → x1 → x2_reconstructed
        # First retarget: P2,A2 → P1,A2 (should look like y1)
        output_backward, _, _ = model(x2, x1, y1, teacher_forcing_ratio=0.0)
        
        # Retarget back: P1,A2 → P2,A2 (should recover x2)
        output_cycle_back, _, _ = model(output_backward, x2, x2[:, :, 1:, :, :], teacher_forcing_ratio=0.0)
        
        # Compute cycle losses
        # Skip first frame for teacher forcing alignment
        x1_target = x1[:, :, 1:, :, :]
        x2_target = x2[:, :, 1:, :, :]
        
        cycle_forward = F.mse_loss(output_cycle, x1_target)
        cycle_backward = F.mse_loss(output_cycle_back, x2_target)
        
        # Combined loss
        cycle_loss = self.alpha * cycle_forward + self.beta * cycle_backward
        
        return cycle_loss, cycle_forward, cycle_backward


class IdentityPreservationLoss(nn.Module):
    """
    Ensures that retargeting preserves identity-specific characteristics
    when action is the same, and vice versa
    """
    def __init__(self, identity_encoder, action_encoder, temperature=0.1):
        super().__init__()
        self.identity_encoder = identity_encoder
        self.action_encoder = action_encoder
        self.temperature = temperature
        
        # Freeze encoders
        for param in self.identity_encoder.parameters():
            param.requires_grad = False
        for param in self.action_encoder.parameters():
            param.requires_grad = False
            
        self.identity_encoder.eval()
        self.action_encoder.eval()
    
    def forward(self, x1, x2, y1, y2, model):
        """
        Args:
            x1: (B, C, T, V, M) - Source skeleton (P1, A1)
            x2: (B, C, T, V, M) - Target skeleton (P2, A2)
            y1: (B, C, T, V, M) - Cross skeleton (P1, A2)
            y2: (B, C, T, V, M) - Cross skeleton (P2, A1)
            model: DisentangledTMR model
            
        Returns:
            identity_loss: Identity preservation loss
        """
        with torch.no_grad():
            # Get ground truth features
            id_x1 = self.identity_encoder(x1)
            id_x2 = self.identity_encoder(x2)
            act_x1 = self.action_encoder(x1)
            act_x2 = self.action_encoder(x2)
        
        # Generate retargeted outputs
        output_y2, _, _ = model(x1, x2, y2, teacher_forcing_ratio=0.0)  # P2,A1
        output_y1, _, _ = model(x2, x1, y1, teacher_forcing_ratio=0.0)  # P1,A2
        
        # Get features from retargeted outputs
        id_output_y2 = self.identity_encoder(output_y2)
        id_output_y1 = self.identity_encoder(output_y1)
        act_output_y2 = self.action_encoder(output_y2)
        act_output_y1 = self.action_encoder(output_y1)
        
        # Identity should be preserved in cross-retargeting
        # output_y2 should have same identity as x2
        # output_y1 should have same identity as x1
        id_loss_1 = F.mse_loss(id_output_y2, id_x2)
        id_loss_2 = F.mse_loss(id_output_y1, id_x1)
        
        # Action should be preserved in cross-retargeting
        # output_y2 should have same action as x1
        # output_y1 should have same action as x2
        act_loss_1 = F.mse_loss(act_output_y2, act_x1)
        act_loss_2 = F.mse_loss(act_output_y1, act_x2)
        
        identity_loss = id_loss_1 + id_loss_2 + act_loss_1 + act_loss_2
        
        return identity_loss
