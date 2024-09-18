"""
Simple MLP classifiers for action and identity features
"""

import torch
import torch.nn as nn


class ActionClassifier(nn.Module):
    """
    Simple MLP classifier for action recognition from action features
    """
    def __init__(self, d_action, num_classes, dropout=0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_action, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, action_features):
        """
        Args:
            action_features: (B, D_action) or (T, B, D_action)
        
        Returns:
            logits: (B, num_classes)
        """
        # If temporal dimension exists, average over time
        if action_features.dim() == 3:
            action_features = action_features.mean(dim=0)  # (B, D_action)
        
        return self.classifier(action_features)


class IdentityClassifier(nn.Module):
    """
    Simple MLP classifier for re-identification from identity features
    """
    def __init__(self, d_identity, num_identities, dropout=0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_identity, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_identities)
        )
    
    def forward(self, identity_features):
        """
        Args:
            identity_features: (B, D_identity)
        
        Returns:
            logits: (B, num_identities)
        """
        return self.classifier(identity_features)


if __name__ == "__main__":
    # Test
    d_action = 256
    d_identity = 256
    num_classes = 49
    num_identities = 40
    batch_size = 8
    T = 64
    
    # Create classifiers
    ar_classifier = ActionClassifier(d_action, num_classes)
    ri_classifier = IdentityClassifier(d_identity, num_identities)
    
    # Test action classifier
    action_features = torch.randn(T, batch_size, d_action)
    ar_logits = ar_classifier(action_features)
    print(f"Action features: {action_features.shape}")
    print(f"AR logits: {ar_logits.shape}")
    assert ar_logits.shape == (batch_size, num_classes)
    
    # Test identity classifier
    identity_features = torch.randn(batch_size, d_identity)
    ri_logits = ri_classifier(identity_features)
    print(f"Identity features: {identity_features.shape}")
    print(f"RI logits: {ri_logits.shape}")
    assert ri_logits.shape == (batch_size, num_identities)
    
    print("\n✅ All tests passed!")

