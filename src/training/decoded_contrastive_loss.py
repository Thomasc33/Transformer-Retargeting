"""
Contrastive Learning Loss for Decoded Skeletons
Improves action separability in the output space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodedContrastiveLoss(nn.Module):
    """
    Contrastive loss on decoded skeletons to improve action separability
    Pulls same-action samples together, pushes different actions apart
    """
    def __init__(self, action_encoder, temperature=0.1, hard_negatives=True):
        super().__init__()
        self.action_encoder = action_encoder
        self.temperature = temperature
        self.hard_negatives = hard_negatives
        
        # Freeze encoder
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        self.action_encoder.eval()
    
    def forward(self, decoded_skeletons, actions):
        """
        Args:
            decoded_skeletons: (B, C, T, V, M) - Decoded motion
            actions: (B,) - Action labels
            
        Returns:
            contrastive_loss: InfoNCE contrastive loss
        """
        # Extract features from decoded skeletons
        features = self.action_encoder(decoded_skeletons)  # (T, B, D) or (B, D)
        
        # Temporal average if needed
        if features.dim() == 3:
            features = features.mean(dim=0)  # (B, D)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (positive pairs are same action)
        labels = actions.unsqueeze(0) == actions.unsqueeze(1)
        labels = labels.float()
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(-1)
        similarity_matrix = similarity_matrix[~mask].view(-1)
        
        # Compute InfoNCE loss
        contrastive_loss = F.binary_cross_entropy_with_logits(similarity_matrix, labels)
        
        return contrastive_loss


class MotionStyleContrastiveLoss(nn.Module):
    """
    Contrastive loss that separates motion style from action
    Ensures identity information doesn't leak into action features
    """
    def __init__(self, action_encoder, identity_encoder, temperature=0.1):
        super().__init__()
        self.action_encoder = action_encoder
        self.identity_encoder = identity_encoder
        self.temperature = temperature
        
        # Freeze encoders
        for param in self.action_encoder.parameters():
            param.requires_grad = False
        for param in self.identity_encoder.parameters():
            param.requires_grad = False
            
        self.action_encoder.eval()
        self.identity_encoder.eval()
    
    def forward(self, decoded_skeletons, actions, identities):
        """
        Args:
            decoded_skeletons: (B, C, T, V, M) - Decoded motion
            actions: (B,) - Action labels
            identities: (B,) - Identity labels
            
        Returns:
            style_loss: Loss encouraging action/identity disentanglement
        """
        # Extract features
        action_features = self.action_encoder(decoded_skeletons)
        identity_features = self.identity_encoder(decoded_skeletons)
        
        # Temporal average if needed
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)
        if identity_features.dim() == 3:
            identity_features = identity_features.mean(dim=0)
        
        # Normalize features
        action_features = F.normalize(action_features, dim=1)
        identity_features = F.normalize(identity_features, dim=1)
        
        # Action features should be similar for same actions, regardless of identity
        action_sim = torch.matmul(action_features, action_features.T) / self.temperature
        action_labels = (actions.unsqueeze(0) == actions.unsqueeze(1)).float()
        
        # Identity features should be similar for same identities, regardless of action
        identity_sim = torch.matmul(identity_features, identity_features.T) / self.temperature
        identity_labels = (identities.unsqueeze(0) == identities.unsqueeze(1)).float()
        
        # Remove diagonals
        mask = torch.eye(action_labels.size(0), dtype=torch.bool, device=action_labels.device)
        action_labels = action_labels[~mask].view(-1)
        action_sim = action_sim[~mask].view(-1)
        identity_labels = identity_labels[~mask].view(-1)
        identity_sim = identity_sim[~mask].view(-1)
        
        # Compute losses
        action_loss = F.binary_cross_entropy_with_logits(action_sim, action_labels)
        identity_loss = F.binary_cross_entropy_with_logits(identity_sim, identity_labels)
        
        # Encourage orthogonality between action and identity features
        orthogonality_loss = torch.mean(torch.abs(torch.sum(action_features * identity_features, dim=1)))
        
        style_loss = action_loss + identity_loss + 0.1 * orthogonality_loss
        
        return style_loss
