"""
Auxiliary Task Losses for Enhanced TMR Training

These losses force the model to learn better spatial-temporal representations:
1. Denoising Loss: Recover clean motion from noisy input
2. Masked Prediction Loss: Reconstruct masked joints

Based on recommendations from Gemini Deep Research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def denoising_loss(model, clean_motion, noise_std=0.01, teacher_forcing_ratio=1.0):
    """
    Denoising loss: Train model to recover clean motion from noisy input.
    
    This forces the model to learn robust spatial-temporal structure by
    removing noise from the input motion.
    
    Args:
        model: TMR model (autoencoder)
        clean_motion: (B, C, T, V, M) clean skeleton sequence
        noise_std: Standard deviation of Gaussian noise to add
        teacher_forcing_ratio: Teacher forcing ratio for decoder
    
    Returns:
        loss: MSE between predicted and clean motion
    """
    # Add Gaussian noise to input
    noise = torch.randn_like(clean_motion) * noise_std
    noisy_motion = clean_motion + noise
    
    # Clamp to reasonable range to prevent extreme values
    noisy_motion = torch.clamp(noisy_motion, min=-10.0, max=10.0)
    
    # Get dummy skeleton (use first frame of clean motion as target skeleton)
    dummy_skeleton = clean_motion[:, :, :1, :, :].expand_as(clean_motion)
    
    # Forward pass: try to recover clean motion from noisy input
    # Note: We pass clean_motion as target for teacher forcing
    predicted_motion = model(noisy_motion, dummy_skeleton, 
                            target_motion=clean_motion,
                            teacher_forcing_ratio=teacher_forcing_ratio)
    
    # Compute MSE loss between predicted and clean motion
    # predicted_motion is (B, C, T-1, V, M) due to autoregressive generation
    # So we compare with clean_motion[:, :, 1:, :, :]
    target = clean_motion[:, :, 1:, :, :]
    
    loss = F.mse_loss(predicted_motion, target)
    
    return loss


def masked_prediction_loss(model, motion, mask_ratio=0.15, teacher_forcing_ratio=1.0):
    """
    Masked prediction loss: Train model to reconstruct masked joints.
    
    This forces the model to learn joint dependencies by predicting
    masked joint coordinates from visible joints.
    
    Args:
        model: TMR model (autoencoder)
        motion: (B, C, T, V, M) skeleton sequence
        mask_ratio: Ratio of joints to mask (default 0.15 = 15%)
        teacher_forcing_ratio: Teacher forcing ratio for decoder
    
    Returns:
        loss: MSE between predicted and original motion at masked joints
    """
    B, C, T, V, M = motion.shape
    
    # Create random mask for joints
    # mask shape: (B, V) - same mask for all time steps and channels
    mask = torch.rand(B, V, device=motion.device) < mask_ratio  # (B, V)
    
    # Expand mask to match motion shape
    mask_expanded = mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)  # (B, 1, 1, V, 1)
    mask_expanded = mask_expanded.expand(B, C, T, V, M)  # (B, C, T, V, M)
    
    # Create masked motion (set masked joints to zero)
    masked_motion = motion.clone()
    masked_motion[mask_expanded] = 0.0
    
    # Get dummy skeleton (use first frame as target skeleton)
    dummy_skeleton = motion[:, :, :1, :, :].expand_as(motion)
    
    # Forward pass: try to reconstruct original motion from masked input
    predicted_motion = model(masked_motion, dummy_skeleton,
                            target_motion=motion,
                            teacher_forcing_ratio=teacher_forcing_ratio)
    
    # Compute MSE loss only at masked joints
    # predicted_motion is (B, C, T-1, V, M)
    target = motion[:, :, 1:, :, :]
    mask_target = mask_expanded[:, :, 1:, :, :]
    
    # Only compute loss at masked positions
    if mask_target.sum() > 0:
        loss = F.mse_loss(predicted_motion[mask_target], target[mask_target])
    else:
        # If no joints were masked, return zero loss
        loss = torch.tensor(0.0, device=motion.device)
    
    return loss


def combined_auxiliary_loss(model, motion, noise_std=0.01, mask_ratio=0.15, 
                           teacher_forcing_ratio=1.0, denoising_weight=0.5, 
                           masking_weight=0.5):
    """
    Combined auxiliary loss: Denoising + Masked Prediction.
    
    Args:
        model: TMR model
        motion: (B, C, T, V, M) skeleton sequence
        noise_std: Noise level for denoising
        mask_ratio: Masking ratio for masked prediction
        teacher_forcing_ratio: Teacher forcing ratio
        denoising_weight: Weight for denoising loss
        masking_weight: Weight for masking loss
    
    Returns:
        total_loss: Weighted sum of auxiliary losses
        loss_dict: Dictionary with individual losses for logging
    """
    # Compute individual losses
    loss_denoise = denoising_loss(model, motion, noise_std, teacher_forcing_ratio)
    loss_mask = masked_prediction_loss(model, motion, mask_ratio, teacher_forcing_ratio)
    
    # Weighted sum
    total_loss = denoising_weight * loss_denoise + masking_weight * loss_mask
    
    # Return both total and individual losses for logging
    loss_dict = {
        'auxiliary_total': total_loss.item(),
        'auxiliary_denoise': loss_denoise.item(),
        'auxiliary_mask': loss_mask.item()
    }
    
    return total_loss, loss_dict


def test_auxiliary_losses():
    """
    Test function to verify auxiliary losses work correctly.
    """
    print("Testing Auxiliary Losses...")
    
    # Create dummy model (simple passthrough for testing)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        
        def forward(self, source, target_skel, target_motion=None, teacher_forcing_ratio=1.0):
            # Simple passthrough that returns T-1 frames
            return source[:, :, 1:, :, :]
    
    model = DummyModel()
    model.eval()
    
    # Create dummy input
    batch_size = 4
    motion = torch.randn(batch_size, 3, 64, 25, 1)
    
    # Test denoising loss
    with torch.no_grad():
        loss_denoise = denoising_loss(model, motion, noise_std=0.01)
    
    assert not torch.isnan(loss_denoise), "Denoising loss is NaN"
    assert not torch.isinf(loss_denoise), "Denoising loss is Inf"
    assert loss_denoise >= 0, "Denoising loss is negative"
    
    print(f"✅ Denoising loss: {loss_denoise.item():.6f}")
    
    # Test masked prediction loss
    with torch.no_grad():
        loss_mask = masked_prediction_loss(model, motion, mask_ratio=0.15)
    
    assert not torch.isnan(loss_mask), "Masked prediction loss is NaN"
    assert not torch.isinf(loss_mask), "Masked prediction loss is Inf"
    assert loss_mask >= 0, "Masked prediction loss is negative"
    
    print(f"✅ Masked prediction loss: {loss_mask.item():.6f}")
    
    # Test combined loss
    with torch.no_grad():
        total_loss, loss_dict = combined_auxiliary_loss(model, motion)
    
    assert not torch.isnan(total_loss), "Combined loss is NaN"
    assert not torch.isinf(total_loss), "Combined loss is Inf"
    assert total_loss >= 0, "Combined loss is negative"
    
    print(f"✅ Combined auxiliary loss: {total_loss.item():.6f}")
    print(f"   - Denoising: {loss_dict['auxiliary_denoise']:.6f}")
    print(f"   - Masking: {loss_dict['auxiliary_mask']:.6f}")
    
    print("\n✅ All auxiliary loss tests passed!")


if __name__ == "__main__":
    test_auxiliary_losses()

