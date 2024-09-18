"""
Action Preservation Loss for Decoder Training
Ensures that decoded skeletons maintain discriminative action features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionPreservationLoss(nn.Module):
    """
    Loss to preserve action information in decoded skeletons
    Forces the decoder to maintain action discriminability
    """
    def __init__(self, action_encoder, ar_classifier, temperature=0.1):
        super().__init__()
        self.action_encoder = action_encoder
        self.ar_classifier = ar_classifier
        self.temperature = temperature
        
        # Freeze encoders - we only want to train the decoder
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        for param in self.ar_classifier.parameters():
            param.requires_grad = False
            
        self.action_encoder.eval()
        self.ar_classifier.eval()
    
    def forward(self, decoded_skeletons, original_actions):
        """
        Args:
            decoded_skeletons: (B, C, T, V, M) - decoded motion
            original_actions: (B,) - action labels (0-indexed)
        
        Returns:
            loss: Action preservation loss
            accuracy: AR accuracy on decoded skeletons
        """
        with torch.no_grad():
            # Encode original skeletons to get reference features
            # Note: We don't have original skeletons here, so we'll use decoded ones
            pass
        
        # Pad to 64 frames if needed (Decoder produces T-1 frames)
        if decoded_skeletons.shape[2] < 64:
            pad_len = 64 - decoded_skeletons.shape[2]
            last_frame = decoded_skeletons[:, :, -1:, :, :]
            decoded_skeletons = torch.cat([decoded_skeletons, last_frame.repeat(1, 1, pad_len, 1, 1)], dim=2)

        # Encode decoded skeletons
        decoded_features = self.action_encoder(decoded_skeletons)  # (T, B, D) or (B, D)
        
        # Get AR predictions
        ar_logits = self.ar_classifier(decoded_features)  # (B, num_classes)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(ar_logits, original_actions)
        
        # Compute accuracy
        predictions = ar_logits.argmax(dim=1)
        accuracy = (predictions == original_actions).float().mean()
        
        return loss, accuracy


class FeatureConsistencyLoss(nn.Module):
    """
    Maintains consistency between action features before and after decoding
    """
    def __init__(self, action_encoder, temperature=0.1):
        super().__init__()
        self.action_encoder = action_encoder
        self.temperature = temperature
        
        # Freeze encoder
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        self.action_encoder.eval()
    
    def forward(self, original_features, decoded_skeletons):
        """
        Args:
            original_features: (T, B, D) - action features before decoding
            decoded_skeletons: (B, C, T, V, M) - decoded motion
        
        Returns:
            loss: Feature consistency loss
        """
        # Pad to 64 frames if needed (Decoder produces T-1 frames)
        # print(f"DEBUG: decoded_skeletons shape before padding: {decoded_skeletons.shape}")
        if decoded_skeletons.shape[2] < 64:
            pad_len = 64 - decoded_skeletons.shape[2]
            last_frame = decoded_skeletons[:, :, -1:, :, :]
            decoded_skeletons = torch.cat([decoded_skeletons, last_frame.repeat(1, 1, pad_len, 1, 1)], dim=2)
        # print(f"DEBUG: decoded_skeletons shape after padding: {decoded_skeletons.shape}")

        # Encode decoded skeletons
        decoded_features = self.action_encoder(decoded_skeletons)  # (T, B, D) or (B, D)
        
        # Temporal average if needed
        if decoded_features.dim() == 3:
            decoded_features = decoded_features.mean(dim=0)  # (B, D)
        if original_features.dim() == 3:
            original_features = original_features.mean(dim=0)  # (B, D)
        
        # Normalize features
        orig_norm = F.normalize(original_features, dim=1)
        decoded_norm = F.normalize(decoded_features, dim=1)
        
        # Cosine similarity loss
        similarity = F.cosine_similarity(orig_norm, decoded_norm, dim=1)
        loss = 1 - similarity.mean()
        
        return loss


class MotionDynamicsLoss(nn.Module):
    """
    Preserves motion dynamics (velocity, acceleration) in decoded skeletons
    """
    def __init__(self):
        super().__init__()
    
    def compute_velocity(self, x):
        """Compute velocity: x[t+1] - x[t]"""
        return x[:, :, 1:] - x[:, :, :-1]
    
    def compute_acceleration(self, x):
        """Compute acceleration: v[t+1] - v[t]"""
        velocity = self.compute_velocity(x)
        return velocity[:, :, 1:] - velocity[:, :, :-1]
    
    def forward(self, decoded_skeletons, target_skeletons):
        """
        Args:
            decoded_skeletons: (B, C, T, V, M)
            target_skeletons: (B, C, T, V, M)
        
        Returns:
            loss: Motion dynamics loss
        """
        # Reshape for easier computation
        B, C, T, V, M = decoded_skeletons.shape
        decoded_flat = decoded_skeletons.reshape(B, C * V * M, T)
        target_flat = target_skeletons.reshape(B, C * V * M, T)
        
        # Compute velocity and acceleration
        decoded_vel = self.compute_velocity(decoded_flat)
        target_vel = self.compute_velocity(target_flat)
        decoded_acc = self.compute_acceleration(decoded_flat)
        target_acc = self.compute_acceleration(target_flat)
        
        # MSE loss on dynamics
        vel_loss = F.mse_loss(decoded_vel, target_vel)
        acc_loss = F.mse_loss(decoded_acc, target_acc)
        
        return vel_loss + acc_loss


class ContrastiveActionLoss(nn.Module):
    """
    Contrastive loss to maintain action separability in decoded skeletons
    """
    def __init__(self, action_encoder, temperature=0.1):
        super().__init__()
        self.action_encoder = action_encoder
        self.temperature = temperature
        
        # Freeze encoder
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        self.action_encoder.eval()
    
    def forward(self, decoded_skeletons, actions):
        """
        Args:
            decoded_skeletons: (B, C, T, V, M)
            actions: (B,) - action labels
        
        Returns:
            loss: Contrastive loss
        """
        # Encode decoded skeletons
        features = self.action_encoder(decoded_skeletons)  # (T, B, D) or (B, D)
        
        # Temporal average if needed
        if features.dim() == 3:
            features = features.mean(dim=0)  # (B, D)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (same action = positive, different action = negative)
        labels = actions.unsqueeze(0) == actions.unsqueeze(1)
        labels = labels.float()
        
        # Compute InfoNCE loss
        # Remove diagonal (self-similarity)
        mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(-1)
        similarity_matrix = similarity_matrix[~mask].view(-1)
        
        # Cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(similarity_matrix, labels)
        
        return loss
