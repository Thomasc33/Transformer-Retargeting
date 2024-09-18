"""
Disentanglement Losses for TMR

Implements various losses to enforce disentanglement of action and identity:
1. Contrastive Loss (InfoNCE) - same action should have similar features
2. Adversarial Loss - identity discriminator can't predict identity from action features
3. Orthogonality Loss - action and identity features should be independent
4. Mutual Information Loss - minimize MI between action and identity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss for action features
    
    Positive pairs: Same action, different identity
    Negative pairs: Different actions
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, action_features, action_labels):
        """
        Args:
            action_features: (B, D) or (T, B, D) - action representations
            action_labels: (B,) - action class labels
            
        Returns:
            loss: scalar
        """
        # If temporal dimension exists, average over time
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)  # (B, D)
        
        # Normalize features
        # Use larger eps for stability in mixed precision and handle potential zeros
        action_features = F.normalize(action_features, dim=1, eps=1e-6)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(action_features, action_features.T) / self.temperature  # (B, B)
        
        # Create positive mask (same action, different sample)
        action_mask = action_labels.unsqueeze(0) == action_labels.unsqueeze(1)  # (B, B)
        action_mask.fill_diagonal_(False)  # Exclude self
        
        # Check if there are any positive pairs
        if not action_mask.any():
            # No positive pairs in this batch, return zero loss
            return torch.tensor(0.0, device=action_features.device, requires_grad=True)
        
        # InfoNCE loss
        # Use log_softmax for numerical stability instead of exp().sum()
        # loss = -log(exp(pos) / sum(exp(all))) = -pos + log(sum(exp(all)))
        #       = -pos + LogSumExp(all)
        
        # We need to compute this for each positive pair
        # But standard InfoNCE often treats the batch as: 1 positive, N-1 negatives
        # Here we might have multiple positives per anchor.
        # Let's stick to the stable implementation of:
        # log_prob = logits - logsumexp(logits)
        # loss = - (mask * log_prob).sum() / mask.sum()
        
        logits = sim_matrix
        log_probs = F.log_softmax(logits, dim=1)
        
        # Sum log_probs for positive pairs
        loss = -(log_probs * action_mask.float()).sum() / action_mask.sum()
        
        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial Loss for identity removal from action features
    
    Train a discriminator to predict identity from action features,
    then train encoder to fool the discriminator.
    """
    def __init__(self, d_action, num_identities, hidden_dim=512):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(d_action, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_identities)
        )
        
    def forward_discriminator(self, action_features, identity_labels):
        """
        Train discriminator to predict identity from action features
        
        Args:
            action_features: (B, D) or (T, B, D) - action representations (detached)
            identity_labels: (B,) - identity labels
            
        Returns:
            loss: scalar
        """
        # If temporal dimension exists, average over time
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)  # (B, D)
        
        # Predict identity
        logits = self.discriminator(action_features.detach())
        loss = F.cross_entropy(logits, identity_labels)
        
        return loss
    
    def forward_encoder(self, action_features, identity_labels):
        """
        Train encoder to fool discriminator (adversarial loss)
        
        Args:
            action_features: (B, D) or (T, B, D) - action representations
            identity_labels: (B,) - identity labels
            
        Returns:
            loss: scalar (negative cross-entropy to fool discriminator)
        """
        # If temporal dimension exists, average over time
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)  # (B, D)
        
        # Predict identity
        logits = self.discriminator(action_features)
        
        # Adversarial loss: maximize entropy (fool discriminator)
        # We want uniform distribution over identities
        uniform_target = torch.ones_like(logits) / logits.size(1)
        loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_target, reduction='batchmean')
        
        return loss
    
    def get_accuracy(self, action_features, identity_labels):
        """Get discriminator accuracy (for monitoring)"""
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)
        
        with torch.no_grad():
            logits = self.discriminator(action_features)
            preds = logits.argmax(dim=1)
            accuracy = (preds == identity_labels).float().mean()
        
        return accuracy.item()


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Loss - action and identity features should be independent
    
    Minimizes cosine similarity between action and identity features
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, action_features, identity_features):
        """
        Args:
            action_features: (B, D_action) or (T, B, D_action)
            identity_features: (B, D_identity) or (T, B, D_identity)
            
        Returns:
            loss: scalar
        """
        # If temporal dimension exists, average over time
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)  # (B, D_action)
        if identity_features.dim() == 3:
            identity_features = identity_features.mean(dim=0)  # (B, D_identity)
        
        # Normalize features
        action_norm = F.normalize(action_features, dim=1, eps=1e-6)
        identity_norm = F.normalize(identity_features, dim=1, eps=1e-6)
        
        # Compute cosine similarity (should be close to 0)
        # If dimensions don't match, project to common space
        if action_norm.size(1) != identity_norm.size(1):
            # Project to smaller dimension
            min_dim = min(action_norm.size(1), identity_norm.size(1))
            action_proj = action_norm[:, :min_dim]
            identity_proj = identity_norm[:, :min_dim]
        else:
            action_proj = action_norm
            identity_proj = identity_norm
        
        similarity = (action_proj * identity_proj).sum(dim=1)
        
        # Minimize absolute similarity
        loss = similarity.abs().mean()
        
        return loss


class MutualInformationLoss(nn.Module):
    """
    Mutual Information Loss - minimize MI between action and identity features
    
    Approximates MI minimization using correlation
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, action_features, identity_features):
        """
        Args:
            action_features: (B, D_action) or (T, B, D_action)
            identity_features: (B, D_identity) or (T, B, D_identity)
            
        Returns:
            loss: scalar
        """
        # If temporal dimension exists, flatten
        if action_features.dim() == 3:
            T, B, D_action = action_features.shape
            action_features = action_features.permute(1, 0, 2).contiguous()  # (B, T, D_action)
            action_features = action_features.reshape(B, -1)  # (B, T*D_action)
        
        # If identity is temporal, average over time to keep feature dimension manageable
        if identity_features.dim() == 3:
            identity_features = identity_features.mean(dim=0)  # (B, D_identity)
        
        # Check batch size > 1 for correlation computation
        if action_features.size(0) < 2:
            return torch.tensor(0.0, device=action_features.device, requires_grad=True)

        # Normalize features (zero mean, unit variance)
        action_mean = action_features.mean(dim=0, keepdim=True)
        action_std = action_features.std(dim=0, keepdim=True) + 1e-8
        action_norm = (action_features - action_mean) / action_std
        
        identity_mean = identity_features.mean(dim=0, keepdim=True)
        identity_std = identity_features.std(dim=0, keepdim=True) + 1e-8
        identity_norm = (identity_features - identity_mean) / identity_std
        
        # Compute correlation matrix
        # If dimensions don't match, project to common space
        if action_norm.size(1) != identity_norm.size(1):
            min_dim = min(action_norm.size(1), identity_norm.size(1))
            action_proj = action_norm[:, :min_dim]
            identity_proj = identity_norm[:, :min_dim]
        else:
            action_proj = action_norm
            identity_proj = identity_norm
        
        # Correlation: (D, D)
        correlation = torch.matmul(action_proj.T, identity_proj) / action_proj.size(0)

        # Minimize Frobenius norm of correlation (want independence)
        # Scale by number of dimensions to keep loss magnitude reasonable
        loss = torch.norm(correlation, p='fro') ** 2
        loss = loss / (action_proj.size(1) * identity_proj.size(1))

        return loss


class DisentanglementLosses:
    """
    Container for all disentanglement losses
    """
    def __init__(self, d_action=256, d_identity=256, num_identities=40, 
                 temperature=0.07, device='cuda'):
        self.contrastive = ContrastiveLoss(temperature=temperature).to(device)
        self.adversarial = AdversarialLoss(d_action, num_identities).to(device)
        self.orthogonality = OrthogonalityLoss().to(device)
        self.mutual_info = MutualInformationLoss().to(device)
        self.device = device
        
    def compute_all(self, action_features, identity_features, action_labels, identity_labels,
                   train_discriminator=False):
        """
        Compute all disentanglement losses
        
        Args:
            action_features: (T, B, D_action) or (B, D_action)
            identity_features: (B, D_identity)
            action_labels: (B,) - action class labels
            identity_labels: (B,) - identity labels
            train_discriminator: Whether to train discriminator (True) or encoder (False)
            
        Returns:
            losses: dict of losses
        """
        losses = {}
        
        # Contrastive loss
        losses['contrastive'] = self.contrastive(action_features, action_labels)
        
        # Adversarial loss
        if train_discriminator:
            losses['adversarial'] = self.adversarial.forward_discriminator(action_features, identity_labels)
        else:
            losses['adversarial'] = self.adversarial.forward_encoder(action_features, identity_labels)
        
        # Orthogonality loss
        losses['orthogonality'] = self.orthogonality(action_features, identity_features)
        
        # Mutual information loss
        losses['mutual_info'] = self.mutual_info(action_features, identity_features)
        
        # Discriminator accuracy (for monitoring)
        losses['disc_accuracy'] = self.adversarial.get_accuracy(action_features, identity_labels)
        
        return losses
    
    def get_discriminator_params(self):
        """Get discriminator parameters for optimizer"""
        return self.adversarial.discriminator.parameters()
